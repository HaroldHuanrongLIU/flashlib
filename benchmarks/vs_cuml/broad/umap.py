"""broad/umap — N x D sweep."""
from benchmarks.vs_cuml.broad._common import (
    cap_threads, cuml_shim, run_grid, free_gpu,
)
cap_threads(); cuml_shim()

import warnings; warnings.filterwarnings("ignore")
import numpy as np
import torch

from sklearn.datasets import make_blobs
from cuml.manifold import UMAP as cuUMAP
from flashlib.primitives.umap import flash_umap

PRIM = "umap"

# (N, D, NN, epochs)
GRID = [
    (10_000,  32, 15, 200),
    (10_000, 128, 15, 200),
    (30_000,  32, 15, 200),
    (30_000, 128, 15, 200),
    (50_000,  32, 15, 200),
    (50_000,  64, 15, 200),
    (50_000, 128, 15, 200),
    (50_000, 256, 15, 200),
    (100_000, 32, 15, 200),
    (100_000, 64, 15, 200),
    (100_000, 128, 15, 200),
    (200_000, 64, 15, 200),
]


def _setup(N, D, NN, epochs):
    def setup():
        X_np, _ = make_blobs(n_samples=N, centers=10, n_features=D,
                              cluster_std=2.0, random_state=0)
        X_np = X_np.astype(np.float32)
        X32 = torch.tensor(X_np, device="cuda")

        def cu_fn():
            cuUMAP(n_components=2, n_neighbors=NN,
                    n_epochs=epochs, random_state=0).fit_transform(X_np)

        def fl_fn():
            flash_umap(X32, n_neighbors=NN, n_components=2,
                         n_epochs=epochs, tol=None, seed=0)

        def teardown():
            nonlocal X32
            del X32
            free_gpu()
        return cu_fn, fl_fn, teardown
    return setup


def build_cells():
    cells = []
    for N, D, NN, epochs in GRID:
        cells.append({
            "label": f"N={N//1000}K D={D} NN={NN}",
            "axes": {"N": N, "D": D, "NN": NN, "epochs": epochs},
            "dtype": "fp32",
            "setup": _setup(N, D, NN, epochs),
            "repeat": 1,
            "warmup": 1,
            "cuml_repeat": 1,
            "notes": "fp32 exact KNN distances",
        })
    return cells


if __name__ == "__main__":
    run_grid(PRIM, build_cells())
