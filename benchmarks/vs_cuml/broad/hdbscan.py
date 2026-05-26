"""broad/hdbscan — N x D sweep, dense-MRD path."""
from benchmarks.vs_cuml.broad._common import (
    cap_threads, cuml_shim, run_grid, free_gpu,
)
cap_threads(); cuml_shim()

import warnings; warnings.filterwarnings("ignore")
import numpy as np
import torch

from sklearn.datasets import make_blobs
from cuml.cluster import HDBSCAN as cuHDBSCAN
from flashlib.primitives.hdbscan import flash_hdbscan

PRIM = "hdbscan"

# (N, D, mcs) - HDBSCAN dense-MRD is O(N^2), so N capped at 50K
GRID = [
    ( 5_000,  8,  10),
    ( 5_000, 32,  10),
    ( 5_000,128,  10),
    (10_000,  8,  20),
    (10_000, 32,  20),
    (10_000, 64,  20),
    (10_000,128,  20),
    (20_000, 16,  20),
    (20_000, 32,  20),
    (20_000, 64,  20),
    (30_000, 16,  30),
    (30_000, 32,  30),
    (50_000, 16,  50),
    (50_000, 32,  50),
]


def _setup(N, D, mcs):
    def setup():
        X_np, _ = make_blobs(n_samples=N, centers=max(5, N // 1000),
                              n_features=D, cluster_std=1.0,
                              random_state=0)
        X_np = X_np.astype(np.float32)
        X32 = torch.tensor(X_np, device="cuda")

        def cu_fn():
            cuHDBSCAN(min_cluster_size=mcs, min_samples=5).fit_predict(X_np)

        def fl_fn():
            flash_hdbscan(X32, min_cluster_size=mcs, min_samples=5)

        def teardown():
            nonlocal X32
            del X32
            free_gpu()
        return cu_fn, fl_fn, teardown
    return setup


def build_cells():
    cells = []
    for N, D, mcs in GRID:
        cells.append({
            "label": f"N={N//1000}K D={D} mcs={mcs}",
            "axes": {"N": N, "D": D, "mcs": mcs},
            "dtype": "fp32",
            "setup": _setup(N, D, mcs),
            "repeat": 2,
            "warmup": 1,
            "cuml_repeat": 1,
            "notes": "dense-MRD path",
        })
    return cells


if __name__ == "__main__":
    run_grid(PRIM, build_cells())
