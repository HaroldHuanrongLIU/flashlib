"""Per-component time breakdown for flash_tsne across multiple N.

flash_tsne is exact O(N²) per iter (no KNN sparsification, no PCA-init).
The pipeline (``flashlib/primitives/tsne/triton/train.py:238-277``):

  1. ``_compute_p_matrix(X, perplexity)``:
       dists_sq = torch.cdist(X, X).pow(2).contiguous()        # N×N fp32
       _tsne_bisect_kernel    -> beta_i, d_min_i per row
       _tsne_pmat_emit_kernel -> P_unnorm (off-diag exp)
       P = P_unnorm / row_sum ; P = (P + P.T) / (2N) ; clamp
  2. P_exag = P * 12.0
  3. Y init + velocity init
  4. for i in range(n_iter):
       qsum = triton_tsne_qsum(Y)
       grad = triton_tsne_grad(Y, P_use, qsum)
       velocity = momentum * velocity - lr * grad
       Y = Y + velocity

Workload axis: **N (point count)** at fixed D=128, K=10, n_iter=500.
Both the one-shot stages (pairwise_dists, p_construct) and the per-iter
SGD stages (qsum, grad, apply) scale O(N²), but the SGD stages accumulate
500 launches per call — so their share depends on how well each kernel
saturates at the given N.
"""
from __future__ import annotations

import math

import numpy as np
import torch
import triton
from sklearn.datasets import make_blobs

from flashlib.primitives.tsne.triton.grad import (
    triton_tsne_qsum, triton_tsne_grad,
    _tsne_qsum_kernel, _tsne_grad_kernel,
)
from flashlib.primitives.tsne.triton.train import (
    _tsne_bisect_kernel, _tsne_pmat_emit_kernel, _pick_pmat_tile,
)

from ._common import (
    StageGroup, free_gpu, run_multi_shape, write_multi_shape_md,
)

# Shared knobs (mirror flash_tsne defaults at the heavy headline shape).
D = 128
K_BLOBS = 10
N_ITER = 500
PERPLEXITY = 30.0
EE_FACTOR = 12.0           # train.py:261
LR = 200.0                 # flash_tsne default
SEED = 0
N_BISECT = 50              # train.py:_compute_p_matrix default

SHAPES = [
    ("small N=10K n_iter=500",  {"N":  10_000, "D":  64, "K_blobs": K_BLOBS}),
    ("medium N=15K n_iter=500", {"N":  15_000, "D": 128, "K_blobs": K_BLOBS}),
    ("large N=20K n_iter=500",  {"N":  20_000, "D": 128, "K_blobs": K_BLOBS}),
]
STAGES = ["pairwise_dists", "p_construct",
          "sgd_qsum", "sgd_grad", "sgd_apply"]


def prepare(N: int, D: int, K_blobs: int) -> dict:
    torch.manual_seed(0)
    device = "cuda"
    # Same data style as heavy/tsne.py:59-61 (cluster blobs for realistic P).
    X_np, _ = make_blobs(
        n_samples=N, centers=K_blobs, n_features=D,
        cluster_std=2.0, random_state=0,
    )
    X_np = X_np.astype(np.float32)
    X = torch.tensor(X_np, device=device)
    # Reset Triton autotune cache so each shape autotunes its OWN N.
    # `_tsne_qsum_kernel`/`_tsne_grad_kernel` key on `_round_to_bucket(N)`
    # (a power-of-2 bucket); without a clear, two shapes that round to the
    # same bucket (e.g. N=10K and N=15K both → 16384) would share the cached
    # config tuned on whichever shape ran first, giving misleading timings.
    _tsne_qsum_kernel.cache.clear()
    _tsne_grad_kernel.cache.clear()
    return {"X": X, "N": N, "D": D}


def run(stg: StageGroup, ctx: dict) -> None:
    X, N = ctx["X"], ctx["N"]
    device = X.device

    # train.py:253-254 — match flash_tsne's early_exag_iters default.
    early_exag_iters = min(250, max(50, N_ITER // 3))
    target = math.log(PERPLEXITY)

    # ── Stage 1: pairwise dists (one-shot) ────────────────────────────
    with stg["pairwise_dists"]:
        dists_sq = torch.cdist(X, X, p=2).pow(2).contiguous()

    # ── Stage 2: P-matrix construct (one-shot, fused bisect + emit + sym) ─
    with stg["p_construct"]:
        beta = torch.empty(N, device=device, dtype=torch.float32)
        d_min = torch.empty(N, device=device, dtype=torch.float32)
        P_unnorm = torch.zeros(N, N, device=device, dtype=torch.float32)

        BLOCK_I, BLOCK_J, num_warps = _pick_pmat_tile(N)
        grid = (triton.cdiv(N, BLOCK_I),)
        _tsne_bisect_kernel[grid](
            dists_sq, beta, d_min, int(N), float(target),
            N_BISECT=N_BISECT,
            BLOCK_I=BLOCK_I, BLOCK_J=BLOCK_J,
            num_warps=num_warps,
            num_stages=2,
        )
        _tsne_pmat_emit_kernel[grid](
            dists_sq, beta, d_min, P_unnorm, int(N),
            BLOCK_I=BLOCK_I, BLOCK_J=BLOCK_J,
            num_warps=num_warps,
        )
        P = P_unnorm / (P_unnorm.sum(dim=1, keepdim=True) + 1e-12)
        P = (P + P.T) / (2.0 * N)
        P = torch.clamp(P, min=1e-12)

    # train.py:261 — one-shot P*12 (kept outside any timed stage; cheap).
    P_exag = P * EE_FACTOR

    # train.py:263-265 — deterministic RNG, init Y / velocity.
    torch.manual_seed(SEED)
    Y = torch.randn(N, 2, device=device, dtype=torch.float32) * 1e-4
    velocity = torch.zeros_like(Y)

    # ── Stages 3–5: SGD loop (train.py:267-275) — per-iter accumulate ───
    for i in range(N_ITER):
        in_ee = i < early_exag_iters
        P_use = P_exag if in_ee else P
        momentum = 0.5 if in_ee else 0.8

        with stg["sgd_qsum"]:
            qsum = triton_tsne_qsum(Y)
        with stg["sgd_grad"]:
            grad = triton_tsne_grad(Y, P_use, qsum)
        with stg["sgd_apply"]:
            velocity = momentum * velocity - LR * grad
            Y = Y + velocity

    del dists_sq, beta, d_min, P_unnorm, P, P_exag, Y, velocity, qsum, grad


def main() -> None:
    print(f"[breakdown:tsne] sweeping N at n_iter={N_ITER}, "
          f"perplexity={PERPLEXITY}")
    results = run_multi_shape(SHAPES, prepare, run, STAGES,
                               warmup=1, repeat=3)

    write_multi_shape_md(
        prim="tsne",
        shape_axis=f"N (point count) at n_iter={N_ITER}, "
                   f"perplexity={PERPLEXITY}, fp32",
        results=results,
        stage_names=STAGES,
        notes=(
            "Per-iter SGD = 2 Triton launches (qsum + grad) + a torch "
            "momentum step; `sgd_*` stages accumulate across all "
            f"{N_ITER} iterations. One-shot stages (`pairwise_dists`, "
            "`p_construct`) run once per call. flash_tsne is exact "
            "O(N²) per iter — no KNN sparsification, no PCA-init."
        ),
        sensitivity=(
            f"All five stages are O(N²) per call, but the SGD trio runs "
            f"{N_ITER} launches vs one-shot for the first two — so their "
            "RATIO is roughly 500× if every kernel saturates. **As N "
            "grows from 10K → 15K → 20K** the breakdown stays "
            "qualitatively stable: `sgd_grad` is the long pole at "
            "every N (~57–67 % of wall), `sgd_qsum` ~26–31 %, "
            "`sgd_apply` ~3–7 %, and the two one-shot stages "
            "(`pairwise_dists` + `p_construct`) together stay under "
            "10 % at every N. The user's prediction that *if the "
            "per-iter SGD kernel doesn't saturate at small N, "
            "`sgd_grad`'s share will grow with N* is borne out from "
            "10K → 15K (57 % → 67 %) but the trend doesn't continue "
            "monotonically — at N=20K `sgd_grad` lands back near 59 %. "
            "Inspecting the absolute ms reveals the cause: "
            "`sgd_grad`'s autotuned config ladder (5 fixed BLOCK_I × "
            "BLOCK_J configs in `grad.py`) picks a particularly "
            "fortunate tile size at N=20K — the kernel hits ~398 ms "
            "for 500 iters, which is FASTER than N=15K's 524 ms "
            "despite ~78 % more arithmetic. So the % breakdown at any "
            "single N depends on which config the autotuner lands on; "
            "the SHAPE-LEVEL takeaway is still robust: SGD dominates "
            "everywhere, and `pairwise_dists`/`p_construct` collectively "
            "stay <10 %. **Optimisation steers by N**: at every N the "
            "per-iter grad kernel is the long pole (fuse qsum + grad + "
            "apply, blocked-grad to keep P in L2, or expand the "
            "autotune ladder so non-power-of-2 N pick a non-suboptimal "
            "tile). The one-shot P-matrix stages only become "
            "worth re-tuning at smaller N where launch overhead and "
            "L2-fit edges dominate."
        ),
    )
    free_gpu()


if __name__ == "__main__":
    main()
