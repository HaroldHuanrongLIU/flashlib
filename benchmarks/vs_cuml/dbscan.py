"""DBSCAN: ``flash_dbscan`` (exact) vs ``cuml.cluster.DBSCAN``.

Ground truth: scikit-learn DBSCAN. Both flashlib and cuml currently
return ARI = 1.0 vs sklearn on the included shapes; the interesting
axis is wall-clock time.

Sweeps low-D (where sklearn / cuml can use a spatial index) and
high-D (where everyone falls back to dense distance computation).
"""
from benchmarks.vs_cuml._common import (
    cap_threads, cuml_shim, time_gpu, time_cpu, title,
    ari, header, fmt_table, cluster_count,
)
cap_threads(); cuml_shim()

import warnings; warnings.filterwarnings("ignore")
import numpy as np
import torch

from sklearn.cluster import DBSCAN as skDBSCAN
from sklearn.datasets import make_blobs
from cuml.cluster import DBSCAN as cuDBSCAN
from flashlib.primitives.dbscan import flash_dbscan


SHAPES = [
    # (label,             N,      D,   eps,  min_samples, n_centers)
    ("low-D",             20_000,  8,  1.5,  5,           4),
    ("medium-D",          20_000, 32,  6.0,  5,           6),
    ("high-D",            20_000, 64,  8.0,  5,          10),
    ("large N",           50_000, 16,  3.5,  5,           8),
]


def run_one(label, N, D, eps, min_samples, n_centers):
    title(f"DBSCAN {label}  (N={N:,}, D={D}, eps={eps}, min_samples={min_samples})")
    X_np, _ = make_blobs(n_samples=N, centers=n_centers, n_features=D,
                         cluster_std=1.0, random_state=0)
    X_np = X_np.astype(np.float32)

    # --- sklearn ground truth ---
    sk_lbl = skDBSCAN(eps=eps, min_samples=min_samples).fit_predict(X_np)
    t_sk = time_cpu(lambda: skDBSCAN(eps=eps, min_samples=min_samples).fit_predict(X_np), repeat=1)

    # --- flashlib (exact) ---
    X32 = torch.tensor(X_np, device="cuda")
    fl_lbl = flash_dbscan(X32, eps=eps, min_samples=min_samples).cpu().numpy()
    t_fl = time_gpu(lambda: flash_dbscan(X32, eps=eps, min_samples=min_samples),
                    repeat=3, warmup=1)

    # --- cuml ---
    cu_lbl = np.asarray(cuDBSCAN(eps=eps, min_samples=min_samples).fit_predict(X_np))
    t_cu = time_gpu(lambda: cuDBSCAN(eps=eps, min_samples=min_samples).fit_predict(X_np),
                    repeat=3, warmup=1)

    rows = [
        ("sklearn (CPU)", f"{t_sk:8.2f}", "1.0000",          f"{cluster_count(sk_lbl):d}", "1.00x"),
        ("flashlib",      f"{t_fl:8.2f}", f"{ari(sk_lbl, fl_lbl):.4f}",
                          f"{cluster_count(fl_lbl):d}",     f"{t_cu / t_fl:.2f}x"),
        ("cuml",          f"{t_cu:8.2f}", f"{ari(sk_lbl, cu_lbl):.4f}",
                          f"{cluster_count(cu_lbl):d}",     "1.00x"),
    ]
    print(fmt_table(rows, ["engine", "time(ms)", "ARI vs sk", "#cl", "fl/cuml"]))


def main():
    header()
    for s in SHAPES:
        run_one(*s)
    print()


if __name__ == "__main__":
    main()
