"""broad/random_forest — N x D x trees x depth sweep (fit timing)."""
from benchmarks.vs_cuml.broad._common import (
    cap_threads, cuml_shim, run_grid, free_gpu,
)
cap_threads(); cuml_shim()

import warnings; warnings.filterwarnings("ignore")
import numpy as np
import torch

from cuml.ensemble import RandomForestClassifier as cuRF
from flashlib.primitives.random_forest import FlashRandomForestClassifier

PRIM = "random_forest"

# (N, D, C, trees, depth)
GRID = [
    (50_000,  32,  4,  50,  8),
    (50_000,  32,  4, 100,  8),
    (50_000,  32,  4, 100, 12),
    (50_000,  64,  4, 100, 12),
    (100_000, 32,  4, 100, 12),
    (100_000, 64,  4, 100, 12),
    (100_000, 128, 6, 100, 12),
    (100_000, 32,  4, 200, 12),
    (200_000, 64,  6, 100, 12),
    (200_000, 64,  6, 200, 12),
    (300_000, 64,  6, 100, 14),
    (500_000, 64,  8, 100, 14),
]


def _gen_data(N, D, C):
    """GPU-resident class-conditioned synthetic features."""
    torch.manual_seed(0)
    y_t = torch.randint(0, C, (N,), device="cuda", dtype=torch.int64)
    centers = torch.randn(C, D, device="cuda") * 1.5
    X_t = torch.randn(N, D, device="cuda") + centers[y_t]
    return X_t, y_t


def _setup(N, D, C, n_est, depth):
    def setup():
        X_t, y_t = _gen_data(N, D, C)
        X_np = X_t.cpu().numpy().astype(np.float32)
        y_np_int32 = y_t.cpu().numpy().astype(np.int32)
        y_t_int32 = y_t.to(torch.int32).contiguous()

        def cu_fn():
            cuRF(n_estimators=n_est, max_depth=depth,
                  random_state=0).fit(X_np, y_np_int32)

        def fl_fn():
            FlashRandomForestClassifier(
                n_estimators=n_est, max_depth=depth,
                max_features=None, seed=0,
            ).fit(X_t, y_t_int32)

        def teardown():
            nonlocal X_t, y_t, y_t_int32, X_np, y_np_int32
            del X_t, y_t, y_t_int32, X_np, y_np_int32
            free_gpu()
        return cu_fn, fl_fn, teardown
    return setup


def build_cells():
    cells = []
    for N, D, C, n_est, depth in GRID:
        cells.append({
            "label": f"N={N//1000}K D={D} trees={n_est} depth={depth}",
            "axes": {"N": N, "D": D, "C": C,
                      "trees": n_est, "depth": depth},
            "dtype": "fp32",
            "setup": _setup(N, D, C, n_est, depth),
            "repeat": 1,
            "warmup": 1,
            "cuml_repeat": 1,
            "notes": "fit timing, max_features=None on both",
        })
    return cells


if __name__ == "__main__":
    run_grid(PRIM, build_cells())
