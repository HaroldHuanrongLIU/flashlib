"""KNN: ``flash_knn`` (Triton / CuteDSL FA3) vs ``cuml.neighbors.NearestNeighbors``.

Each shape is reported in TWO precisions:
  * fp32 input  -- inside the kernel ``tl.dot`` defaults to TF32 (Hopper);
                   distances are TF32-grade (~1e-3 rel err) but candidate
                   indices match cuml fp32 IEEE recall@K >= 99.8%.
  * bf16 input  -- native bf16 GEMM with fp32 accumulator. Matches
                   FA3 sweet spot; ``cutedsl-FA3`` actually engages here
                   (for fp32 input it raises NotImplementedError, since
                   WGMMA does not lower fp32 K>=16 on Hopper).

The Triton path exposes a single ``triton-auto`` row whose internal
small-N / large-N branch is shape-driven.

Reference baselines (always fp32, GPU-resident):
  * sklearn brute (CPU fp32 IEEE) -- recall@K ground truth.
  * cuml brute (GPU fp32, cupy view of torch buffer) -- the bar to beat.

Per-row metrics:
  * time(ms)   -- median of repeats with warmup.
  * TFLOPS     -- 2 * Q * M * D / time. Right metric for the all-pairs
                  ``build`` shapes (compute-bound).
  * GB/s       -- ((Q + M) * D * sizeof(dtype) + Q * K * 8) / time.
                  Right metric for the small-Q ``search`` shapes
                  (HBM-bound; we load M*D once per N tile).
  * recall@K   -- vs sklearn fp32 brute.
  * vs cuml    -- speedup against the SAME cuml fp32 timing for the
                  shape (so the bf16 columns honestly say "look how
                  much extra you get from going bf16").
"""
from benchmarks.vs_cuml._common import (
    cap_threads, cuml_shim, time_gpu, time_cpu, title,
    recall_at_k, header, fmt_table,
)
cap_threads(); cuml_shim()

import warnings; warnings.filterwarnings("ignore")
import numpy as np
import torch
import cupy as cp

from sklearn.neighbors import NearestNeighbors as skNN
from cuml.neighbors import NearestNeighbors as cuNN
from flashlib.primitives.knn import flash_knn


SHAPES = [
    # (label,                          M,      Q,      D,   K)
    # Search workloads unchanged (latency / bandwidth regime).
    ("search small-Q",                 50_000,   1_024,  64,  10),
    ("search medium-Q",                100_000,  4_096,  64,  10),
    # Build workloads — larger Q×M so launch / top-K tail is amortised and
    # effective TFLOPS approaches what you'd expect on a big fused GEMM.
    ("build  K=10",                    96_000,  96_000,  64,  10),
    ("build  K=32",                    56_000,  56_000,  64,  32),
    ("build  K=10 D=128",              64_000,  64_000, 128,  10),
    ("build  K=64",                    48_000,  48_000,  64,  64),
    # Optional peak tile — very large N×M; uncomment if you want one row
    # that saturates HBM/compute (~6e11 mul-adds for bf16):
    # ("build  K=10 peak",           120_000, 120_000,  64,  10),
]

DTYPES = [
    # (label, torch_dtype, sizeof_bytes)
    ("fp32", torch.float32, 4),
    ("bf16", torch.bfloat16, 2),
]


def _torch_to_cupy(t: torch.Tensor) -> "cp.ndarray":
    """Zero-copy view of a CUDA torch tensor as a cupy ndarray."""
    return cp.from_dlpack(t)


def _flops_tf(Q, M, D, t_ms) -> float:
    """KNN GEMM TFLOPS: 2 * Q * M * D mul-adds."""
    return (2.0 * Q * M * D) / 1e12 / (t_ms / 1000.0)


def _bw_gb(Q, M, D, K, t_ms, sz) -> float:
    """Useful HBM bandwidth (input + output)."""
    inp = (Q + M) * D * sz
    out = Q * K * (4 + 4)  # vals fp32 + idxs int32
    return (inp + out) / 1e9 / (t_ms / 1000.0)


def run_one(idx, label, M, Q, D, K):
    title(f"KNN  {label}  (M={M:,}, Q={Q:,}, D={D}, K={K})")

    rng = np.random.RandomState(0)
    Xc_np = rng.randn(M, D).astype(np.float32)
    Xq_np = rng.randn(Q, D).astype(np.float32) if Q != M else Xc_np

    # --- sklearn brute (CPU, fp32 IEEE) -- recall ground truth ---
    sk_nn = skNN(n_neighbors=K, algorithm="brute", metric="euclidean").fit(Xc_np)
    sk_idx = sk_nn.kneighbors(Xq_np, return_distance=False)
    t_sk = time_cpu(lambda: sk_nn.kneighbors(Xq_np, return_distance=False), repeat=1)

    # --- cuml brute (GPU fp32) -- single baseline reused across dtype tables ---
    Xc_t32 = torch.tensor(Xc_np, device="cuda")
    Xq_t32 = torch.tensor(Xq_np, device="cuda")
    Xc_cp = _torch_to_cupy(Xc_t32); Xq_cp = _torch_to_cupy(Xq_t32)
    cu_nn = cuNN(n_neighbors=K, algorithm="brute", metric="euclidean").fit(Xc_cp)
    cu_out = cu_nn.kneighbors(Xq_cp, return_distance=False)
    cu_idx = cu_out.get() if hasattr(cu_out, "get") else np.asarray(cu_out)
    t_cu = time_gpu(lambda: cu_nn.kneighbors(Xq_cp, return_distance=False),
                    repeat=10, warmup=3)

    rows = []
    rows.append(("fp32", "sklearn (CPU)", f"{t_sk:8.2f}",
                 f"{_flops_tf(Q, M, D, t_sk):6.1f}",
                 f"{_bw_gb(Q, M, D, K, t_sk, 4):5.1f}",
                 f"{recall_at_k(sk_idx, sk_idx, K):.4f}", "1.00x"))
    rows.append(("fp32", "cuml",          f"{t_cu:8.2f}",
                 f"{_flops_tf(Q, M, D, t_cu):6.1f}",
                 f"{_bw_gb(Q, M, D, K, t_cu, 4):5.1f}",
                 f"{recall_at_k(cu_idx, sk_idx, K):.4f}", "1.00x"))

    for dlabel, dtype, sz in DTYPES:
        Xc_t = Xc_t32.to(dtype); Xq_t = Xq_t32.to(dtype)
        Xq_b = Xq_t.unsqueeze(0); Xc_b = Xc_t.unsqueeze(0)

        backends = [
            ("flashlib triton-auto",   {"backend": "triton"}),
        ]
        # FA3 engages for bf16/fp16 only (fp32 raises inside the CuteDSL
        # path). Uses heuristic compile (``autotune=False`` default) --
        # pair with ``benchmarks.tune.knn`` for the full autotune sweep.
        if dtype is not torch.float32:
            backends.append(
                ("flashlib cutedsl-FA3", {"backend": "cutedsl"})
            )

        for blabel, bkw in backends:
            try:
                out_idx = flash_knn(Xq_b, Xc_b, K, **bkw)[1].squeeze(0).cpu().numpy()
                # CuteDSL FA3 has a multi-minute first-call autotune; allow
                # extra warmups so we measure the steady-state kernel.
                warmup = 5 if "cutedsl" in blabel else 3
                t = time_gpu(lambda: flash_knn(Xq_b, Xc_b, K, **bkw),
                             repeat=10, warmup=warmup)
                rows.append((
                    dlabel, blabel, f"{t:8.2f}",
                    f"{_flops_tf(Q, M, D, t):6.1f}",
                    f"{_bw_gb(Q, M, D, K, t, sz):5.1f}",
                    f"{recall_at_k(out_idx, sk_idx, K):.4f}",
                    f"{t_cu / t:.2f}x",
                ))
            except Exception as e:
                msg = str(e).splitlines()[0][:32]
                rows.append((dlabel, blabel, "  SKIP", "  --  ", "  -- ", msg, "  --  "))

    print(fmt_table(rows, ["dtype", "engine", "time(ms)", "TFLOPS", "GB/s",
                            "recall@K", "vs cuml"]))


def main():
    header()
    for i, s in enumerate(SHAPES):
        run_one(i, *s)
    print()


if __name__ == "__main__":
    main()
