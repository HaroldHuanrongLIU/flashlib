"""broad/tsne — N x D sweep, exact-vs-exact O(N^2) comparison."""
from benchmarks.vs_cuml.broad._common import (
    cap_threads, cuml_shim, run_grid, free_gpu,
)
cap_threads(); cuml_shim()

import warnings; warnings.filterwarnings("ignore")
import numpy as np
import torch

from sklearn.datasets import make_blobs
from cuml.manifold import TSNE as cuTSNE
from flashlib.primitives.tsne import flash_tsne

PRIM = "tsne"

# (N, D, perplexity, n_iter)
GRID = [
    (2_000,  32, 30.0, 300),
    (3_000,  64, 30.0, 300),
    (5_000,  64, 30.0, 300),
    (5_000, 128, 30.0, 300),
    (8_000,  64, 30.0, 300),
    (10_000, 64, 30.0, 300),
    (10_000, 128, 30.0, 300),
    (15_000, 128, 30.0, 300),
]


def _setup(N, D, perplexity, n_iter):
    def setup():
        X_np, _ = make_blobs(n_samples=N, centers=max(5, N // 1000),
                              n_features=D, cluster_std=2.0,
                              random_state=0)
        X_np = X_np.astype(np.float32)
        X32 = torch.tensor(X_np, device="cuda")

        def cu_fn():
            cuTSNE(n_components=2, perplexity=perplexity,
                    n_iter=n_iter, random_state=0,
                    method="exact").fit_transform(X_np)

        def fl_fn():
            flash_tsne(X32, n_iter=n_iter,
                         perplexity=perplexity, seed=0)

        def teardown():
            nonlocal X32
            del X32
            free_gpu()
        return cu_fn, fl_fn, teardown
    return setup


def build_cells():
    cells = []
    for N, D, perplexity, n_iter in GRID:
        cells.append({
            "label": f"N={N//1000}K D={D} iter={n_iter}",
            "axes": {"N": N, "D": D, "iter": n_iter},
            "dtype": "fp32",
            "setup": _setup(N, D, perplexity, n_iter),
            "repeat": 1,
            "warmup": 1,
            "cuml_repeat": 1,
            "notes": "method=exact O(N^2) on both sides",
        })
    return cells


if __name__ == "__main__":
    run_grid(PRIM, build_cells())
