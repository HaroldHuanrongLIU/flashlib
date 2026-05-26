"""Per-component time breakdown for flash_knn across multiple D buckets.

Workload axis: **D (vector dimension)** at fixed Q=M=100K, K=10, bf16, build
mode (self-kNN). The heuristic's BN/BM/D_INNER/NUM_SPLITS choices shift with
D, so this sweep exercises three qualitatively different regimes:

  * D=8   — sub-16 zero-pad path (Triton ``tl.dot`` requires K>=16; the
            input is padded to D=16 internally and the kernel runs with a
            narrow ``D_INNER=16`` GEMM tile).
  * D=128 — the headline shape; the heuristic picks BN=BM=64 single-pass
            insert with NUM_SPLITS=1, and the bf16 GEMM tile saturates HBM.
  * D=512 — large-D regime; ``D_INNER`` caps at 128 so the kernel
            D-splits 4 ways and the heuristic flips to BN=128 +
            NUM_SPLITS=2 to keep all 132 SMs fed.

Body is INLINED here (rather than calling ``flash_knn(...)``) so each
kernel call can sit in its own ``Stage`` context — exactly the same
arguments and dispatch logic as ``flash_knn_dispatch`` +
``flash_knn_triton._run`` + ``triton_knn_gather_sqdist``.
"""
from __future__ import annotations

import math

import torch

from flashlib.primitives.knn.triton.dispatch import _heuristic_config
from flashlib.primitives.knn.triton.insert import _flash_knn_insert_kernel
from flashlib.kernels.distance.triton.knn_gather_l2sq import (
    triton_knn_gather_sqdist,
)

from ._common import (
    StageGroup, free_gpu, run_multi_shape, write_multi_shape_md,
)

M_FIXED, Q_FIXED, K_FIXED = 100_000, 100_000, 10
_KNN_MIN_D = 16  # mirrors flashlib.primitives.knn.impl._KNN_MIN_D

SHAPES = [
    ("D=8",   {"D": 8}),
    ("D=128", {"D": 128}),
    ("D=512", {"D": 512}),
]
STAGES = ["data_prep", "main_knn", "gather"]


def prepare(D: int) -> dict:
    torch.manual_seed(0)
    device = "cuda"
    Xc32 = torch.randn(M_FIXED, D, device=device, dtype=torch.float32)
    return {"Xc32": Xc32, "D": D}


def run(stg: StageGroup, ctx: dict) -> None:
    device = "cuda"
    D = ctx["D"]
    Xc32 = ctx["Xc32"]
    M = M_FIXED
    Q = Q_FIXED
    K = K_FIXED

    with stg["data_prep"]:
        Xc = Xc32.to(torch.bfloat16)
        if D < _KNN_MIN_D:
            Xc_pad = torch.zeros(M, _KNN_MIN_D, device=device,
                                 dtype=torch.bfloat16)
            Xc_pad[:, :D] = Xc
            Xc = Xc_pad
        Xq = Xc  # self-kNN reuses the cast buffer
        x_p = Xq[None].contiguous()
        c_p = Xc[None].contiguous()
        B, N_q, D_eff = x_p.shape
        M_c = c_p.shape[1]
        assert (B, N_q) == (1, Q) and M_c == M
        assert D_eff == max(D, _KNN_MIN_D)

    with stg["main_knn"]:
        cfg = _heuristic_config(B, N_q, M_c, D_eff, K, force_path=None)
        bn = cfg["BN"]
        bm = cfg["BM"]
        d_inner = cfg["D_INNER"]
        topk_pad = cfg["TOPK_PAD"]
        mps = cfg["M_PER_SPLIT"]
        num_splits = cfg["NUM_SPLITS"]
        nw = cfg["num_warps"]
        ns_pipe = cfg.get("NUM_STAGES_PIPE", 2)
        kernel_mode = cfg["kernel_mode"]
        assert kernel_mode == "insert", (
            f"breakdown expects insert at every shape (D={D}); "
            f"got {kernel_mode}"
        )

        partial_vals = torch.empty(
            (B, N_q, num_splits, K), device=device, dtype=torch.float32,
        )
        partial_idxs = torch.empty(
            (B, N_q, num_splits, K), device=device, dtype=torch.int32,
        )
        grid = (num_splits, math.ceil(N_q / bn), B)
        pv_s0, pv_s1, pv_s2, pv_s3 = partial_vals.stride()
        pi_s0, pi_s1, pi_s2, pi_s3 = partial_idxs.stride()
        max_steps = min(K, bm)
        _flash_knn_insert_kernel[grid](
            x_p, c_p, partial_vals, partial_idxs,
            x_p.stride(0), x_p.stride(1), x_p.stride(2),
            c_p.stride(0), c_p.stride(1), c_p.stride(2),
            pv_s0, pv_s2, pv_s1, pv_s3,
            pi_s0, pi_s2, pi_s1, pi_s3,
            N=N_q, M=M_c, D=D_eff, K=K, M_PER_SPLIT=mps,
            BN=bn, BM=bm, D_INNER=d_inner,
            TOPK_PAD=topk_pad, MAX_STEPS=max_steps,
            NUM_STAGES_PIPE=ns_pipe,
            num_warps=nw,
        )
        if num_splits == 1:
            idxs = partial_idxs[:, :, 0, :].contiguous()
        else:
            pv = partial_vals.view(B, N_q, -1)
            pi = partial_idxs.view(B, N_q, -1)
            _, sel = pv.topk(K, dim=-1, largest=False, sorted=True)
            idxs = pi.gather(-1, sel.to(torch.int64)).to(torch.int32)

    with stg["gather"]:
        vals = triton_knn_gather_sqdist(x_p, c_p, idxs)
        _ = (vals, idxs)  # consumer


def main() -> None:
    print(f"[breakdown:knn] sweeping D at Q=M={Q_FIXED:,}, K={K_FIXED}, bf16")
    results = run_multi_shape(SHAPES, prepare, run, STAGES,
                                warmup=1, repeat=3)

    write_multi_shape_md(
        prim="knn",
        shape_axis=(f"D (vector dim) at Q=M={Q_FIXED:,}, K={K_FIXED}, "
                    "bf16, self-kNN build"),
        results=results,
        stage_names=STAGES,
        notes=(
            "data_prep = fp32 -> bf16 cast + (D<16) zero-pad to D=16 + "
            ".contiguous(). main_knn = _heuristic_config + "
            "_flash_knn_insert_kernel; when the heuristic picks "
            "NUM_SPLITS>1 the Stage-2 reduce (`pv.topk` + `pi.gather`) "
            "is included in this stage too. gather = "
            "triton_knn_gather_sqdist (per-neighbour true squared L2)."
        ),
        sensitivity=(
            "**At D=8** the input is zero-padded to D=16 (Triton's "
            "`tl.dot` requires the inner dim >= 16; zeros contribute 0 "
            "to squared L2 so the gather is bit-exact). The heuristic picks "
            "BN=BM=64 single-pass insert; `main_knn` runs faster than the "
            "D=128 baseline in proportion to the smaller GEMM K-dim "
            "(D_INNER=16 vs 128 = 8x less FMA per inner-loop step). "
            "**At D=128** (the headline) the bf16 GEMM saturates HBM in "
            "a SINGLE-pass insert kernel (BN=BM=64, NUM_SPLITS=1, "
            "ctas_no_split=1563 already 12x-oversubscribes 132 SMs), so "
            "`main_knn` dominates the wall and `gather` becomes a tiny "
            "tail. **At D=512** the dispatcher hits two boundaries at "
            "once: D_INNER caps at 128 (so the kernel D-splits 4x per "
            "row, reloading C from HBM each chunk), and the heuristic "
            "flips BN to 128 -- which halves `ctas_no_split` to 782 and "
            "triggers a 2-way M-split with a Stage-2 reduce. `main_knn` "
            "therefore rises super-linearly in D (≈4x vs D=128, not 4x "
            "from FLOPs alone): the extra HBM traffic from D-split is "
            "the L2-pressure boundary the autotune comments warn about. "
            "`gather` also scales with D (its per-neighbour distance "
            "compute reloads BLOCK_D from HBM)."
        ),
    )
    free_gpu()


if __name__ == "__main__":
    main()
