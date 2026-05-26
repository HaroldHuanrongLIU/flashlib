"""Heavy-workload KMeans: flash_kmeans (bf16) vs cuml.cluster.KMeans (fp32).

The headline ``benchmarks/vs_cuml/kmeans.py`` tops out at N=500K, K=1024 -- a
regime where cuML is already well-tuned and the flash_kmeans win is modest
(1.5-2x). Real industrial KMeans workloads are an order of magnitude larger:
billion-scale recsys clustering, KNN-graph quantization, and bulk vector
quantization routinely use N >= 10M and K >= 10K.

This script sweeps (N, K) up to (100M, 100K) at D=64 so the asymptotic
behaviour shows up:

* flashlib's fused assign keeps the (N x K) distance tile in registers; the
  per-iter cost is bound by the bf16 GEMM (N x D x K flops on the centroid
  side), and the centroid-update step never touches a materialised
  (N, K) matrix.
* cuML's RAFT KMeans does a chunked fp32 GEMM for the assign; at fp32 the
  TF ceiling is ~1/15 of bf16 dense, so cuML pays both the precision tax
  AND a wider tile that misses L2.

Data is generated on GPU (CPU ``make_blobs`` is impractical for K=100K),
identical init centroids are shared between engines, and the per-iter budget
is fixed at ``max_iter=5`` (Lloyd converges slowly in this regime; what we
want to measure is per-iter throughput, not convergence).

Inertia is reported as the canonical convergence-quality metric; ARI is
omitted because computing it on 1e8 labels is itself ~minutes on CPU.

Run:
    python benchmarks/vs_cuml/kmeans_heavy.py            # full sweep (~15 min)
    python benchmarks/vs_cuml/kmeans_heavy.py --quick    # skip the 100M point
"""
from benchmarks.vs_cuml._common import (
    cap_threads, cuml_shim, time_gpu, title, header, fmt_table,
)
cap_threads(); cuml_shim()

import argparse
import warnings; warnings.filterwarnings("ignore")
import gc
import time
import torch
import cupy as cp

from cuml.cluster import KMeans as cuKMeans
from flashlib.primitives.kmeans import flash_kmeans


# (label, N, D, K, max_iter)
SHAPES = [
    ("medium",     10_000_000, 64,  10_000, 5),
    ("large",      30_000_000, 64,  30_000, 5),
    ("xlarge",     50_000_000, 64,  50_000, 5),
    ("xxlarge",   100_000_000, 64, 100_000, 5),
]


def chunked_inertia(X: torch.Tensor, C: torch.Tensor, ids: torch.Tensor,
                    chunk: int = 1_000_000) -> float:
    """Sum of squared distances to assigned centroid, in O(N*D) extra mem.

    ``X`` (N, D) any float dtype; ``C`` (K, D); ``ids`` (N,) int.
    Computation done in fp32 to match cuML's inertia_ convention.
    """
    Cf = C.to(torch.float32)
    total = torch.zeros((), device=X.device, dtype=torch.float64)
    N = X.shape[0]
    for s in range(0, N, chunk):
        e = min(s + chunk, N)
        diff = X[s:e].to(torch.float32) - Cf[ids[s:e]]
        total += diff.pow(2).sum().to(torch.float64)
    return float(total.item())


def run_one(label: str, N: int, D: int, K: int, max_iter: int):
    title(
        f"KMeans {label}  (N={N:,}, D={D}, K={K:,}, max_iter={max_iter}, "
        f"flash=bf16, cuml=fp32)"
    )

    KM_TOL = 1e-6

    # Build inputs on GPU (CPU make_blobs is impractical for K >= 10K).
    torch.manual_seed(0)
    X32 = torch.randn(N, D, device="cuda", dtype=torch.float32)
    init_idx = torch.randperm(N, device="cuda")[:K]
    init32 = X32[init_idx].contiguous()
    Xbf = X32.to(torch.bfloat16)
    init_bf = Xbf[init_idx].contiguous()

    rows = []

    init_cp = cp.from_dlpack(init32)
    X_cp = cp.from_dlpack(X32)

    def cu_fit():
        km = cuKMeans(n_clusters=K, init=init_cp, n_init=1,
                       max_iter=max_iter, tol=KM_TOL)
        km.fit(X_cp)
        return km

    km_cu = cu_fit()
    cu_inertia = float(km_cu.inertia_)
    t_cu = time_gpu(cu_fit, repeat=1, warmup=0)  # one extra (post-warmup)
    rows.append(("fp32", "cuml",     f"{t_cu:9.1f}",
                 f"{cu_inertia:.3e}", "1.00x"))
    del km_cu; gc.collect(); torch.cuda.empty_cache()

    init_bf_b = init_bf.unsqueeze(0)
    fl_ids, fl_C, _ = flash_kmeans(Xbf, K, max_iters=max_iter,
                                     init_centroids=init_bf_b)
    fl_inertia = chunked_inertia(X32, fl_C.squeeze(0), fl_ids.squeeze(0))
    t_fl = time_gpu(
        lambda: flash_kmeans(Xbf, K, max_iters=max_iter,
                              init_centroids=init_bf_b),
        repeat=1, warmup=0,
    )
    rows.append(("bf16", "flashlib", f"{t_fl:9.1f}",
                 f"{fl_inertia:.3e}", f"{t_cu / t_fl:.2f}x"))

    print(fmt_table(rows, ["dtype", "engine", "time(ms)",
                            "inertia", "vs cuml"]))

    # Free before next shape.
    del X32, Xbf, init32, init_bf, init_bf_b, X_cp, init_cp, fl_ids, fl_C
    gc.collect(); torch.cuda.empty_cache()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true",
                    help="skip the 100M-point case (~5 min cuML)")
    args = ap.parse_args()

    header()
    shapes = SHAPES[:-1] if args.quick else SHAPES
    t0 = time.time()
    for s in shapes:
        run_one(*s)
    print(f"\ntotal wall time: {(time.time()-t0)/60:.1f} min")


if __name__ == "__main__":
    main()
