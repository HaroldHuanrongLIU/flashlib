"""broad/truncated_svd — sweep across aspect ratios x K."""
from benchmarks.vs_cuml.broad._common import (
    cap_threads, cuml_shim, run_grid, free_gpu,
)
cap_threads(); cuml_shim()

import warnings; warnings.filterwarnings("ignore")
import torch
import cupy as cp

from cuml.decomposition import TruncatedSVD as cuTSVD
from flashlib.primitives.truncated_svd import flash_truncated_svd

PRIM = "truncated_svd"

GRID = [
    ("tall  N=500K  D=128  K=16",   500_000,    128,  16),
    ("tall  N=500K  D=128  K=64",   500_000,    128,  64),
    ("tall  N=2M    D=256  K=32",  2_000_000,   256,  32),
    ("tall  N=2M    D=256  K=128", 2_000_000,   256, 128),
    ("tall  N=5M    D=256  K=64",  5_000_000,   256,  64),
    ("sq    N=300K  D=512  K=32",   300_000,    512,  32),
    ("sq    N=500K  D=1K   K=64",   500_000,  1_024,  64),
    ("sq    N=1M    D=2K   K=128", 1_000_000,  2_048, 128),
    ("wide  N=30K   D=2K   K=32",     30_000,  2_048,  32),
    ("wide  N=20K   D=4K   K=32",     20_000,  4_000,  32),
    ("wide  N=10K   D=8K   K=32",     10_000,  8_000,  32),
    ("wide  N=5K    D=16K  K=32",      5_000, 16_000,  32),
]


def _setup(N, D, K):
    def setup():
        torch.manual_seed(0)
        X32 = torch.randn(N, D, device="cuda", dtype=torch.float32)
        X_cp = cp.from_dlpack(X32)

        def cu_fn():
            cuTSVD(n_components=K).fit(X_cp)

        def fl_fn():
            flash_truncated_svd(X32, K)

        def teardown():
            nonlocal X32, X_cp
            del X32, X_cp
            free_gpu()
        return cu_fn, fl_fn, teardown
    return setup


def build_cells():
    cells = []
    for label, N, D, K in GRID:
        cells.append({
            "label": label,
            "axes": {"N": N, "D": D, "K": K},
            "dtype": "fp32",
            "setup": _setup(N, D, K),
            "repeat": 2,
            "warmup": 1,
            "cuml_repeat": 1,
            "notes": "fp32 exact vs cuML default",
        })
    return cells


if __name__ == "__main__":
    run_grid(PRIM, build_cells())
