"""broad/logistic_regression — N x D sweep, binary."""
from benchmarks.vs_cuml.broad._common import (
    cap_threads, cuml_shim, run_grid, free_gpu,
)
cap_threads(); cuml_shim()

import warnings; warnings.filterwarnings("ignore")
import torch
import cupy as cp

from cuml.linear_model import LogisticRegression as cuLogReg
from flashlib.primitives.logistic_regression import flash_logistic_regression

PRIM = "logistic_regression"

NS = [100_000, 300_000, 1_000_000, 3_000_000]
DS = [128, 512, 2048]
MAX_ITER = 50


def _setup(N, D):
    def setup():
        torch.manual_seed(0)
        y = (torch.rand(N, device="cuda") < 0.5).float()
        sign = (2 * y - 1).unsqueeze(1)
        X = torch.randn(N, D, device="cuda", dtype=torch.float32)
        X[:, :D // 2] += 0.30 * sign
        X_cp = cp.from_dlpack(X)
        y_cp = cp.from_dlpack(y)

        def cu_fn():
            cuLogReg(C=1.0, max_iter=MAX_ITER, tol=1e-4).fit(X_cp, y_cp)

        def fl_fn():
            flash_logistic_regression(X, y, C=1.0,
                                         n_iter=MAX_ITER, gtol=1e-4)

        def teardown():
            nonlocal X, y, X_cp, y_cp
            del X, y, X_cp, y_cp
            free_gpu()
        return cu_fn, fl_fn, teardown
    return setup


def build_cells():
    cells = []
    for N in NS:
        for D in DS:
            if N * D > 4 * 10**9:
                continue
            cells.append({
                "label": f"N={N//1000}K D={D}",
                "axes": {"N": N, "D": D, "C": 2},
                "dtype": "fp32",
                "setup": _setup(N, D),
                "repeat": 1,
                "warmup": 1,
                "cuml_repeat": 1,
                "notes": "binary L-BFGS, fp32",
            })
    return cells


if __name__ == "__main__":
    run_grid(PRIM, build_cells())
