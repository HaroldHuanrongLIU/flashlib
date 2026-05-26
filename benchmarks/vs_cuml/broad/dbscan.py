"""broad/dbscan — N x D x eps sweep.

Carefully chosen shapes that complete on the cuML side — at large N
combined with permissive eps cuML hangs in its CPU-bound label-merge
loop, so we stay in the "completes in < 60s" envelope for each cell.
"""
from benchmarks.vs_cuml.broad._common import (
    cap_threads, cuml_shim, run_grid, free_gpu,
)
cap_threads(); cuml_shim()

import warnings; warnings.filterwarnings("ignore")
import numpy as np
import torch
import cupy as cp

from sklearn.datasets import make_blobs
from cuml.cluster import DBSCAN as cuDBSCAN
from flashlib.primitives.dbscan import flash_dbscan

PRIM = "dbscan"

# (label, N, D, eps, n_centers) - hand-picked to avoid the cuML CPU-merge hang
GRID = [
    # D=2 grid path (small N to avoid cuML hang)
    ("D=2 N=50K  eps=0.3",  50_000, 2, 0.3, 5),
    ("D=2 N=100K eps=0.3", 100_000, 2, 0.3, 5),
    ("D=2 N=200K eps=0.3", 200_000, 2, 0.3, 5),
    # D=16 brute path
    ("D=16 N=50K  eps=3.0",  50_000, 16, 3.0, 8),
    ("D=16 N=100K eps=3.0", 100_000, 16, 3.0, 8),
    ("D=16 N=200K eps=3.0", 200_000, 16, 3.0, 8),
    # D=32 brute path
    ("D=32 N=50K  eps=5.0",  50_000, 32, 5.0, 8),
    ("D=32 N=100K eps=5.0", 100_000, 32, 5.0, 8),
    # D=64 brute path
    ("D=64 N=30K  eps=7.0",  30_000, 64, 7.0, 8),
    ("D=64 N=50K  eps=7.0",  50_000, 64, 7.0, 8),
    # D=128 brute path
    ("D=128 N=10K eps=10.0", 10_000, 128, 10.0, 8),
    ("D=128 N=30K eps=10.0", 30_000, 128, 10.0, 8),
    ("D=128 N=50K eps=10.0", 50_000, 128, 10.0, 8),
]


def _setup(N, D, eps, n_centers):
    def setup():
        X_np, _ = make_blobs(n_samples=N, centers=n_centers,
                              n_features=D, cluster_std=1.0,
                              random_state=0)
        X_np = X_np.astype(np.float32)
        X_t = torch.from_numpy(X_np).cuda()
        X_cp = cp.from_dlpack(X_t)

        def cu_fn():
            cuDBSCAN(eps=eps, min_samples=5,
                       output_type="cupy").fit(X_cp)

        def fl_fn():
            flash_dbscan(X_t, eps=eps, min_samples=5)

        def teardown():
            nonlocal X_t, X_cp
            del X_t, X_cp
            free_gpu()
        return cu_fn, fl_fn, teardown
    return setup


def build_cells():
    cells = []
    for label, N, D, eps, n_centers in GRID:
        cells.append({
            "label": label,
            "axes": {"N": N, "D": D, "eps": eps},
            "dtype": "fp32",
            "setup": _setup(N, D, eps, n_centers),
            "repeat": 1,
            "warmup": 1,
            "cuml_repeat": 1,
            "notes": "dense brute-force (cuML) vs grid/kNN-radius (flashlib)",
        })
    return cells


if __name__ == "__main__":
    run_grid(PRIM, build_cells())
