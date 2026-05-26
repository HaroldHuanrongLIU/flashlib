"""broad/ridge — N x D x T sweep."""
from benchmarks.vs_cuml.broad._common import (
    cap_threads, cuml_shim, run_grid, free_gpu,
)
cap_threads(); cuml_shim()

import warnings; warnings.filterwarnings("ignore")
import torch
import cupy as cp

from cuml.linear_model import Ridge as cuRidge
from flashlib.primitives.ridge import flash_ridge_regression

PRIM = "ridge"

# (N, D, T)
# cuML's Ridge only accepts 1-D y; multi-target cells skip the cuML side
# automatically below.
GRID = [
    (300_000,  128,  1),
    (300_000,  512,  1),
    (300_000, 2048,  1),
    (1_000_000,  128,  1),
    (1_000_000,  512,  1),
    (1_000_000, 2048,  1),
    (3_000_000,  128,  1),
    (3_000_000,  512,  1),
    (3_000_000, 2048,  1),
    (300_000,  4096, 1),
    (1_000_000,  4096, 1),
    (1_000_000, 1024, 1),
]


def _setup(N, D, T):
    def setup():
        torch.manual_seed(0)
        X = torch.randn(N, D, device="cuda", dtype=torch.float32)
        w = 0.1 * torch.randn(D, T, device="cuda", dtype=torch.float32)
        Y = X @ w + 0.05 * torch.randn(N, T, device="cuda",
                                          dtype=torch.float32)
        if T == 1:
            Y = Y.squeeze(-1)
        X_cp = cp.from_dlpack(X)
        Y_cp = cp.from_dlpack(Y)

        def cu_fn():
            cuRidge(alpha=1.0).fit(X_cp, Y_cp)

        def fl_fn():
            flash_ridge_regression(X, Y, alpha=1.0, tol=None)

        def teardown():
            nonlocal X, Y, w, X_cp, Y_cp
            del X, Y, w, X_cp, Y_cp
            free_gpu()
        return cu_fn, fl_fn, teardown
    return setup


def build_cells():
    cells = []
    for N, D, T in GRID:
        cells.append({
            "label": f"N={N//1000}K D={D} T={T}",
            "axes": {"N": N, "D": D, "T": T},
            "dtype": "fp32",
            "setup": _setup(N, D, T),
            "repeat": 2,
            "warmup": 1,
            "cuml_repeat": 1,
            "notes": "fp32 closed-form vs cuML default",
        })
    return cells


if __name__ == "__main__":
    run_grid(PRIM, build_cells())
