"""broad/spectral_clustering — N x D sweep, sklearn (CPU) baseline.

**EXCLUDED FROM THE BROAD SWEEP** (see broad/run_all.py): cuML 25.10
has no SpectralClustering peer. The baseline here is sklearn-CPU,
which would inflate the "vs cuML" headline misleadingly (sklearn-CPU
is hundreds of times slower than any GPU implementation, so the
ratio is largely an artefact of the CPU/GPU gap, not flashlib's
algorithmic advantage over a GPU peer).

This script is preserved for use if cuML adds SpectralClustering in
a future release. The flashlib SpectralClustering's algorithmic
win is documented in ``heavy/spectral_clustering.{md,json}`` (against
its real-world alternative, sklearn-CPU).
"""
from benchmarks.vs_cuml.broad._common import (
    cap_threads, cuml_shim, run_grid, free_gpu,
)
cap_threads(); cuml_shim()

import warnings; warnings.filterwarnings("ignore")
import numpy as np
import torch

from sklearn.cluster import SpectralClustering as skSpectral
from sklearn.datasets import make_blobs
from flashlib.primitives.spectral_clustering import flash_spectral_clustering

PRIM = "spectral_clustering"

# (N, D, K, NN)
GRID = [
    ( 2_000,  16,  5, 15),
    ( 3_000,  32,  5, 15),
    ( 5_000,  32,  8, 15),
    ( 5_000,  64,  8, 15),
    (10_000,  32,  8, 15),
    (10_000,  64, 10, 20),
    (15_000,  32,  8, 15),
    (15_000,  64, 10, 20),
    (20_000,  64, 10, 20),
]


def _setup(N, D, K, NN):
    def setup():
        X_np, _ = make_blobs(n_samples=N, centers=K, n_features=D,
                              cluster_std=1.5, random_state=0)
        X_np = X_np.astype(np.float32)
        X32 = torch.tensor(X_np, device="cuda")

        def cu_fn():
            skSpectral(n_clusters=K, n_neighbors=NN,
                          affinity="nearest_neighbors", random_state=0,
                          assign_labels="kmeans").fit_predict(X_np)

        def fl_fn():
            flash_spectral_clustering(X32, n_clusters=K,
                                          n_neighbors=NN, seed=0)

        def teardown():
            nonlocal X32
            del X32
            free_gpu()
        return cu_fn, fl_fn, teardown
    return setup


def build_cells():
    cells = []
    for N, D, K, NN in GRID:
        cells.append({
            "label": f"N={N//1000}K D={D} K={K} NN={NN}",
            "axes": {"N": N, "D": D, "K": K, "NN": NN},
            "dtype": "fp32",
            "setup": _setup(N, D, K, NN),
            "repeat": 1,
            "warmup": 1,
            "cuml_repeat": 1,
            "cuml_kind": "cpu",  # sklearn is CPU
            "notes": "sklearn baseline (no cuML peer in 25.10)",
        })
    return cells


if __name__ == "__main__":
    run_grid(PRIM, build_cells())
