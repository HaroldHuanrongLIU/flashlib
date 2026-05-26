"""HDBSCAN: ``flash_hdbscan`` (exact) vs ``cuml.cluster.HDBSCAN``.

Ground truth: scikit-learn's HDBSCAN (CPU). The interesting comparison
here is the dense-MRD path -- flashlib uses a fully-fused triton MRD
kernel; cuml uses RAFT's MST + condensed-tree.
"""
from benchmarks.vs_cuml._common import (
    cap_threads, cuml_shim, time_gpu, time_cpu, title,
    ari, header, fmt_table, cluster_count,
)
cap_threads(); cuml_shim()

import warnings; warnings.filterwarnings("ignore")
import numpy as np
import torch

from sklearn.cluster import HDBSCAN as skHDBSCAN
from sklearn.datasets import make_blobs
from cuml.cluster import HDBSCAN as cuHDBSCAN
from flashlib.primitives.hdbscan import flash_hdbscan


SHAPES = [
    # (label,  N,      D,   mcs,  ms, n_centers)
    ("small",  10_000, 16,  20,   5,  6),
    ("medium", 30_000, 16,  30,   5,  8),
    ("larger", 50_000, 32,  50,   5,  8),
]


def run_one(label, N, D, mcs, ms, n_centers):
    title(f"HDBSCAN {label}  (N={N:,}, D={D}, min_cluster_size={mcs}, min_samples={ms})")
    X_np, _ = make_blobs(n_samples=N, centers=n_centers, n_features=D,
                         cluster_std=1.0, random_state=0)
    X_np = X_np.astype(np.float32)

    # --- sklearn ---
    sk_lbl = skHDBSCAN(min_cluster_size=mcs, min_samples=ms).fit_predict(X_np)
    t_sk = time_cpu(lambda: skHDBSCAN(min_cluster_size=mcs, min_samples=ms).fit_predict(X_np), repeat=1)

    # --- flashlib (exact) ---
    X32 = torch.tensor(X_np, device="cuda")
    fl_lbl = flash_hdbscan(X32, min_cluster_size=mcs, min_samples=ms)
    fl_lbl = fl_lbl if isinstance(fl_lbl, np.ndarray) else np.asarray(fl_lbl)
    t_fl = time_gpu(lambda: flash_hdbscan(X32, min_cluster_size=mcs, min_samples=ms),
                    repeat=3, warmup=1)

    # --- cuml ---
    cu_lbl = np.asarray(cuHDBSCAN(min_cluster_size=mcs, min_samples=ms).fit_predict(X_np))
    t_cu = time_gpu(lambda: cuHDBSCAN(min_cluster_size=mcs, min_samples=ms).fit_predict(X_np),
                    repeat=3, warmup=1)

    rows = [
        ("sklearn (CPU)", f"{t_sk:9.2f}", "1.0000",  f"{cluster_count(sk_lbl):d}", "1.00x"),
        ("flashlib",      f"{t_fl:9.2f}", f"{ari(sk_lbl, fl_lbl):.4f}",
                          f"{cluster_count(fl_lbl):d}", f"{t_cu / t_fl:.2f}x"),
        ("cuml",          f"{t_cu:9.2f}", f"{ari(sk_lbl, cu_lbl):.4f}",
                          f"{cluster_count(cu_lbl):d}", "1.00x"),
    ]
    print(fmt_table(rows, ["engine", "time(ms)", "ARI vs sk", "#cl", "fl/cuml"]))


def main():
    header()
    for s in SHAPES:
        run_one(*s)
    print()


if __name__ == "__main__":
    main()
