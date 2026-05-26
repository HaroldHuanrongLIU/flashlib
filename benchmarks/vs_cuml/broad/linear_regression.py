"""broad/linear_regression — N x D sweep, fp32 normal-eq vs cuML default."""
from benchmarks.vs_cuml.broad._common import (
    cap_threads, cuml_shim, run_grid, free_gpu,
)
cap_threads(); cuml_shim()

import warnings; warnings.filterwarnings("ignore")
import torch
import cupy as cp

from cuml.linear_model import LinearRegression as cuLinReg
from flashlib.primitives.linear_regression import flash_linear_regression

PRIM = "linear_regression"

NS = [100_000, 300_000, 1_000_000, 3_000_000]
DS = [64, 256, 1024, 4096]


def _setup(N, D):
    def setup():
        torch.manual_seed(0)
        X = torch.randn(N, D, device="cuda", dtype=torch.float32)
        w = 0.1 * torch.randn(D, device="cuda", dtype=torch.float32)
        y = X @ w + 0.05 * torch.randn(N, device="cuda",
                                          dtype=torch.float32)
        X_cp = cp.from_dlpack(X)
        y_cp = cp.from_dlpack(y)

        def cu_fn():
            cuLinReg().fit(X_cp, y_cp)

        def fl_fn():
            flash_linear_regression(X, y, tol=None)

        def teardown():
            nonlocal X, y, X_cp, y_cp, w
            del X, y, X_cp, y_cp, w
            free_gpu()
        return cu_fn, fl_fn, teardown
    return setup


def build_cells():
    cells = []
    for N in NS:
        for D in DS:
            if N * D > 6 * 10**9:
                continue
            cells.append({
                "label": f"N={N//1000}K D={D}",
                "axes": {"N": N, "D": D},
                "dtype": "fp32",
                "setup": _setup(N, D),
                "repeat": 2,
                "warmup": 1,
                "cuml_repeat": 1,
                "notes": "fp32 normal-eq + Cholesky vs cuML lstsqEig",
            })
    return cells


if __name__ == "__main__":
    run_grid(PRIM, build_cells())
