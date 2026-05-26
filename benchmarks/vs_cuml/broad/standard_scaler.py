"""broad/standard_scaler — N x D sweep.

**EXCLUDED FROM THE BROAD SWEEP** (see broad/run_all.py): cuML 25.10
has no native GPU StandardScaler; it just re-exports
``sklearn.preprocessing.StandardScaler`` (CPU). Including this module
in a "vs cuML" plot would be misleading because the baseline isn't
cuML — it's either:

* the real cuML user-visible path (cupy → numpy → sklearn → cupy,
  showing 100-300x slowdown for an apples-to-CPU compare), or
* a cupy-primitive emulation (mean + var + subtract + divide) as a
  hypothetical native GPU implementation.

Neither is a fair "vs cuML" cell. This script is preserved for use if
cuML adds a GPU StandardScaler in a future release. The flashlib
StandardScaler's actual win (single-pass shifted-sum + fused
transform vs cupy's 4-pass primitives) is documented in
``heavy/standard_scaler.{md,json}`` instead.
"""
from benchmarks.vs_cuml.broad._common import (
    cap_threads, cuml_shim, run_grid, free_gpu,
)
cap_threads(); cuml_shim()

import warnings; warnings.filterwarnings("ignore")
import torch
import cupy as cp

from flashlib.primitives.standard_scaler import (
    flash_standard_scaler_fit_transform,
)

PRIM = "standard_scaler"

NS = [100_000, 300_000, 1_000_000, 5_000_000]
DS = [64, 512, 4_096]


def _setup(N, D):
    def setup():
        torch.manual_seed(0)
        X = torch.randn(N, D, device="cuda", dtype=torch.float32)
        X_cp = cp.from_dlpack(X)

        def cu_emulation():
            mu = X_cp.mean(axis=0)
            std = X_cp.std(axis=0)
            (X_cp - mu) / std

        def fl_fn():
            flash_standard_scaler_fit_transform(X)

        def teardown():
            nonlocal X, X_cp
            del X, X_cp
            free_gpu()
        return cu_emulation, fl_fn, teardown
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
                "repeat": 3,
                "warmup": 1,
                "cuml_repeat": 2,
                "notes": ("cuML baseline = cupy mean+var+subtract+divide; "
                          "fl = single-pass shifted-sum + fused transform"),
            })
    return cells


if __name__ == "__main__":
    run_grid(PRIM, build_cells())
