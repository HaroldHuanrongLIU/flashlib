"""broad/multinomial_nb — N x V x C sweep, fit+predict combined."""
from benchmarks.vs_cuml.broad._common import (
    cap_threads, cuml_shim, run_grid, free_gpu,
)
cap_threads(); cuml_shim()

import warnings; warnings.filterwarnings("ignore")
import torch
import cupy as cp

from cuml.naive_bayes import MultinomialNB as cuMNB
from flashlib.primitives.multinomial_nb import flash_multinomial_nb

PRIM = "multinomial_nb"

# (N, V, C). Kept small to avoid the cuML illegal-memory regime at C >= 50.
GRID = [
    (100_000,   500, 10),
    (100_000,  1000, 10),
    (100_000,  2000, 10),
    (300_000,   500, 10),
    (300_000,  1000, 10),
    (300_000,  2000, 10),
    (500_000,  1000, 10),
    (500_000,  2000, 10),
    (500_000,  1000, 20),
    (500_000,  2000, 20),
    (1_000_000, 1000, 20),
    (1_000_000, 2000, 20),
]


def _setup(N, V, C):
    def setup():
        torch.manual_seed(0)
        y = torch.randint(0, C, (N,), device="cuda", dtype=torch.int64)
        base = torch.rand(C, V, device="cuda") * 8.0
        lam = base[y]
        X = torch.poisson(lam).to(torch.float32)
        n_test = max(4096, N // 20)
        Xtr = X[:-n_test].contiguous()
        Xte = X[-n_test:].contiguous()
        ytr = y[:-n_test].contiguous()
        Xtr_cp = cp.from_dlpack(Xtr)
        ytr_cp = cp.from_dlpack(ytr)
        Xte_cp = cp.from_dlpack(Xte)

        def cu_fn():
            cuMNB(alpha=1.0).fit(Xtr_cp, ytr_cp).predict(Xte_cp)

        def fl_fn():
            flash_multinomial_nb(Xtr, ytr, Xte, n_classes=C,
                                    alpha=1.0, tol=None)

        def teardown():
            nonlocal X, y, Xtr, Xte, ytr, Xtr_cp, ytr_cp, Xte_cp, base, lam
            del X, y, Xtr, Xte, ytr, Xtr_cp, ytr_cp, Xte_cp, base, lam
            free_gpu()
        return cu_fn, fl_fn, teardown
    return setup


def build_cells():
    cells = []
    for N, V, C in GRID:
        cells.append({
            "label": f"N={N//1000}K V={V} C={C}",
            "axes": {"N": N, "V": V, "C": C},
            "dtype": "fp32",
            "setup": _setup(N, V, C),
            "repeat": 2,
            "warmup": 1,
            "cuml_repeat": 1,
            "notes": "fit+predict end-to-end",
        })
    return cells


if __name__ == "__main__":
    run_grid(PRIM, build_cells())
