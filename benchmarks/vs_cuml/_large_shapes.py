"""Large-shape KNN + DBSCAN sweep for headroom-vs-cuml check.

Standalone script (not part of run_all). Use for the "is our speedup
gap big enough at scale?" question. cuml's brute KNN/DBSCAN run on the
same buffers (zero-copy via cupy.from_dlpack for KNN, .cpu() for
DBSCAN because cuml.cluster.DBSCAN doesn't accept cupy fp32 input on
the installed 25.10 build).
"""
from benchmarks.vs_cuml._common import (
    cap_threads, cuml_shim, time_gpu, title,
    recall_at_k, ari, header, fmt_table, cluster_count,
)
cap_threads(); cuml_shim()

import warnings; warnings.filterwarnings("ignore")
import numpy as np
import torch
import cupy as cp

from sklearn.datasets import make_blobs
from cuml.neighbors import NearestNeighbors as cuNN
from cuml.cluster import DBSCAN as cuDBSCAN
from flashlib.primitives.knn import flash_knn
from flashlib.primitives.dbscan import flash_dbscan


# (label,                       M,        Q,        D,   K)
KNN_SHAPES = [
    ("build  128K x 128K  D=64",   128_000,  128_000,  64,  10),
    ("build  200K x 200K  D=64",   200_000,  200_000,  64,  10),
    ("build  100K x 100K  D=128",  100_000,  100_000, 128,  10),
    ("build  256K x 256K  D=32",   256_000,  256_000,  32,  10),
    ("build   64K x  64K  K=64",    64_000,   64_000,  64,  64),
]

# (label,        N,         D,    eps,  min_samples, n_centers)
DBSCAN_SHAPES = [
    ("100K  D=16",   100_000,  16,  3.5,  5,  8),
    ("200K  D=16",   200_000,  16,  3.5,  5, 12),
    ("500K  D=16",   500_000,  16,  3.5,  5, 16),
    ("100K  D=32",   100_000,  32,  6.0,  5, 10),
    ("100K  D=64",   100_000,  64,  8.0,  5, 12),
    ("1M    D=16", 1_000_000,  16,  3.5,  5, 20),
]


def _torch_to_cupy(t):
    return cp.from_dlpack(t)


def _flops(Q, M, D, t_ms):
    return (2.0 * Q * M * D) / 1e12 / (t_ms / 1000.0)


def _bw(Q, M, D, K, t_ms, sz):
    return ((Q + M) * D * sz + Q * K * 8) / 1e9 / (t_ms / 1000.0)


def knn_run_one(label, M, Q, D, K):
    title(f"KNN  {label}  (M={M:,}, Q={Q:,}, D={D}, K={K})")
    rng = np.random.RandomState(0)
    Xc_np = rng.randn(M, D).astype(np.float32)
    Xq_np = rng.randn(Q, D).astype(np.float32) if Q != M else Xc_np

    Xc32 = torch.tensor(Xc_np, device="cuda")
    Xq32 = torch.tensor(Xq_np, device="cuda")

    cu_nn = cuNN(n_neighbors=K, algorithm="brute", metric="euclidean").fit(
        _torch_to_cupy(Xc32))
    cu_idx = cu_nn.kneighbors(_torch_to_cupy(Xq32), return_distance=False)
    cu_idx = cu_idx.get() if hasattr(cu_idx, "get") else np.asarray(cu_idx)
    t_cu = time_gpu(
        lambda: cu_nn.kneighbors(_torch_to_cupy(Xq32), return_distance=False),
        repeat=3, warmup=1)

    rows = [("fp32", "cuml", f"{t_cu:9.2f}",
             f"{_flops(Q, M, D, t_cu):6.1f}",
             f"{_bw(Q, M, D, K, t_cu, 4):6.1f}",
             "1.0000", "1.00x")]

    for dlabel, dtype, sz in [("fp32", torch.float32, 4),
                              ("bf16", torch.bfloat16, 2)]:
        Xc = Xc32.to(dtype); Xq = Xq32.to(dtype)
        out = flash_knn(Xq[None], Xc[None], K, backend="triton")
        idx = out[1].squeeze(0).cpu().numpy()
        t = time_gpu(
            lambda: flash_knn(Xq[None], Xc[None], K, backend="triton"),
            repeat=3, warmup=1)
        rows.append((
            dlabel, "flashlib triton-auto", f"{t:9.2f}",
            f"{_flops(Q, M, D, t):6.1f}",
            f"{_bw(Q, M, D, K, t, sz):6.1f}",
            f"{recall_at_k(idx, cu_idx, K):.4f}",
            f"{t_cu / t:.2f}x"))

    print(fmt_table(rows, ["dtype", "engine", "time(ms)", "TFLOPS",
                            "GB/s", "recall@K", "vs cuml"]))


def dbscan_run_one(label, N, D, eps, min_samples, n_centers):
    title(f"DBSCAN  {label}  (N={N:,}, D={D}, eps={eps}, k={min_samples})")
    X_np, _ = make_blobs(n_samples=N, centers=n_centers, n_features=D,
                          cluster_std=1.0, random_state=0)
    X_np = X_np.astype(np.float32)
    X32 = torch.tensor(X_np, device="cuda")

    cu_lbl = np.asarray(cuDBSCAN(eps=eps, min_samples=min_samples).fit_predict(X_np))
    t_cu = time_gpu(
        lambda: cuDBSCAN(eps=eps, min_samples=min_samples).fit_predict(X_np),
        repeat=2, warmup=1)

    fl_lbl = flash_dbscan(X32, eps=eps, min_samples=min_samples).cpu().numpy()
    t_fl = time_gpu(
        lambda: flash_dbscan(X32, eps=eps, min_samples=min_samples),
        repeat=2, warmup=1)

    rows = [
        ("cuml",     f"{t_cu:9.2f}", f"{cluster_count(cu_lbl):d}", "1.00x"),
        ("flashlib", f"{t_fl:9.2f}", f"{cluster_count(fl_lbl):d}",
                     f"{t_cu / t_fl:.2f}x"),
    ]
    print(fmt_table(rows, ["engine", "time(ms)", "#cl", "fl/cuml"]))
    print(f"  ARI(flashlib vs cuml) = {ari(cu_lbl, fl_lbl):.4f}")


def main():
    header()
    print("\n========== KNN large-shape sweep ==========")
    for s in KNN_SHAPES:
        knn_run_one(*s)
    print("\n========== DBSCAN large-shape sweep ==========")
    for s in DBSCAN_SHAPES:
        dbscan_run_one(*s)
    print()


if __name__ == "__main__":
    main()
