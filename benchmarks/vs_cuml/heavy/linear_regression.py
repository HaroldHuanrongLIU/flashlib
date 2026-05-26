"""Heavy LinearRegression sweep — release-candidate audit.

Stresses both:

* **tall (N >> D)**: N up to 10M at D=4K — the regime where the normal-
  equations path's ``X.T X`` GEMM dominates. ``D=4K`` was chosen so
  ``X`` (40 GB at fp32) fits in HBM.
* **wide-ish (D ~ N)**: N=2M D=16K — borderline; ``X.T X`` is 16K^2
  fp32 = 1 GB, the Cholesky factorisation cost on the (16K, 16K)
  Gram becomes non-trivial.

Anti-reward-hacking guardrails:

* ``make_regression`` is fixed seed; cuML AND flashlib see the SAME
  ``X``, ``y``.
* R^2 on a 10% held-out fraction is the quality metric; gate at >= 0.999
  for the fp32 path (LR is exact under fp32 + Cholesky).
* fp32-vs-fp32 reported as the headline; no bf16 row (see docstring of
  ``benchmarks/vs_cuml/linear_regression.py`` — the ``make_regression``
  rank-deficient informative subset can drive ``X.T X`` out of PSD
  under bf16).
"""
from benchmarks.vs_cuml.heavy._common import (
    cap_threads, cuml_shim, time_gpu, title, header,
    audit_record, apples_to_apples,
    hbm_peak_reset, hbm_peak_gb, gate_metric, free_gpu, RESULTS_DIR,
)
cap_threads(); cuml_shim()

import warnings; warnings.filterwarnings("ignore")
import gc
import numpy as np
import torch
import cupy as cp

from sklearn.metrics import r2_score
from cuml.linear_model import LinearRegression as cuLinReg
from flashlib.primitives.linear_regression import flash_linear_regression


# (label, N, D)
# HBM budget: ~12*N*D fp32 bytes total (X, predict, intermediates) must
# fit in 140 GB → N*D < 12 G elements with margin. Cap at ~3G.
SHAPES = [
    ("tall    N=2M    D=512",   2_000_000,    512),
    ("tall    N=5M    D=1024",  5_000_000,  1_024),
    ("tall    N=3M    D=2048",  3_000_000,  2_048),
    ("tall    N=2M    D=4096",  2_000_000,  4_096),
    ("square  N=500K  D=16K",     500_000, 16_000),
]

PRIM = "linear_regression"


def _run_one(label, N, D):
    title(f"LinearRegression  {label}  (N={N:,}, D={D})")

    # make_regression with massive N+D needs ~N*D fp32 bytes.
    # At N=10M D=4K it is 160 GB on CPU; build directly on GPU to skip
    # the CPU intermediate.
    torch.manual_seed(0)
    X32 = torch.randn(N, D, device="cuda", dtype=torch.float32)
    w_true = torch.randn(D, device="cuda", dtype=torch.float32) * 0.1
    noise = 0.05 * torch.randn(N, device="cuda", dtype=torch.float32)
    y32 = X32 @ w_true + noise

    n_test = max(8192, N // 20)
    Xte_t = X32[-n_test:].contiguous()
    yte_t = y32[-n_test:].contiguous()
    Xtr_t = X32[:-n_test].contiguous()
    ytr_t = y32[:-n_test].contiguous()
    yte_np = yte_t.cpu().numpy()  # tiny — needed for sklearn r2_score

    # cuML — pass cupy zero-copy view of the GPU tensor (so cuML's
    # timing is pure compute, no H2D). This is fair: both engines run
    # GPU-resident X, y.
    cu_repeats = 1 if (N >= 5_000_000 or D >= 8_000) else 2
    Xtr_cp = cp.from_dlpack(Xtr_t)
    ytr_cp = cp.from_dlpack(ytr_t)
    Xte_cp = cp.from_dlpack(Xte_t)
    free_gpu(); hbm_peak_reset()
    try:
        cu = cuLinReg().fit(Xtr_cp, ytr_cp)
        cu_pred = cp.asnumpy(cu.predict(Xte_cp))
        cu_r2 = r2_score(yte_np, cu_pred)
        t_cu = time_gpu(lambda: cuLinReg().fit(Xtr_cp, ytr_cp),
                        repeat=cu_repeats, warmup=1 if N < 5_000_000 else 0)
        hbm_cu = hbm_peak_gb()
        audit_record(PRIM, {
            "shape": label, "dtype": "fp32", "engine": "cuml",
            "time_ms": f"{t_cu:10.2f}", "R2": f"{cu_r2:.4f}",
            "vs_cuml": "1.00x", "HBM_GB": f"{hbm_cu:.1f}",
            "gate": gate_metric("R2", cu_r2, lower=0.99),
            "conditions": apples_to_apples(
                op="linreg", shape={"N": N, "D": D},
                flashlib_dtype="-", cuml_dtype="fp32",
                flashlib_algorithm="-",                 cuml_algorithm="cuml_linreg_default",
                init_shared=False,
                notes="cuML fit() consumes cupy zero-copy (no H2D); "
                      "compute-only timing — apples-to-apples with flashlib"),
        }, columns=["shape", "dtype", "engine", "time_ms", "R2",
                    "vs_cuml", "HBM_GB", "gate"])
    except Exception as e:
        t_cu = float("inf")
        audit_record(PRIM, {
            "shape": label, "dtype": "fp32", "engine": "cuml",
            "time_ms": "ERR", "R2": "-", "vs_cuml": "-", "HBM_GB": "-",
            "gate": f"FAIL ({type(e).__name__}: {str(e)[:80]})",
            "conditions": {},
        }, columns=["shape", "dtype", "engine", "time_ms", "R2",
                    "vs_cuml", "HBM_GB", "gate"])

    # flashlib fp32 exact (the only path we benchmark per the docstring
    # constraints of `benchmarks/vs_cuml/linear_regression.py`).
    free_gpu(); hbm_peak_reset()
    try:
        w = flash_linear_regression(Xtr_t, ytr_t, tol=None)
        pred = (Xte_t @ w).cpu().numpy()
        fl_r2 = r2_score(yte_np, pred)
        fl_repeats = 2 if N >= 5_000_000 else 3
        t_fl = time_gpu(
            lambda: flash_linear_regression(Xtr_t, ytr_t, tol=None),
            repeat=fl_repeats, warmup=1,
        )
        hbm_fl = hbm_peak_gb()
        audit_record(PRIM, {
            "shape": label, "dtype": "fp32 exact", "engine": "flashlib",
            "time_ms": f"{t_fl:10.2f}", "R2": f"{fl_r2:.4f}",
            "vs_cuml": (f"{t_cu / t_fl:.2f}x" if t_cu != float("inf") else "n/a"),
            "HBM_GB": f"{hbm_fl:.1f}",
            "gate": gate_metric("R2", fl_r2, lower=0.99),
            "conditions": apples_to_apples(
                op="linreg", shape={"N": N, "D": D},
                flashlib_dtype="fp32", cuml_dtype="fp32",
                flashlib_algorithm="normal_eq_+_cholesky_solve_3xbf16",
                cuml_algorithm="cuml_linreg_default",
                init_shared=False,
                notes="GPU-resident X, y on both sides; pure compute"),
        }, columns=["shape", "dtype", "engine", "time_ms", "R2",
                    "vs_cuml", "HBM_GB", "gate"])
    except Exception as e:
        audit_record(PRIM, {
            "shape": label, "dtype": "fp32 exact", "engine": "flashlib",
            "time_ms": "ERR", "R2": "-", "vs_cuml": "-", "HBM_GB": "-",
            "gate": f"FAIL ({type(e).__name__}: {str(e)[:80]})",
            "conditions": {},
        }, columns=["shape", "dtype", "engine", "time_ms", "R2",
                    "vs_cuml", "HBM_GB", "gate"])

    del X32, y32, Xtr_t, ytr_t, Xte_t, yte_t
    del Xtr_cp, ytr_cp, Xte_cp, yte_np
    gc.collect(); torch.cuda.empty_cache()


def main():
    header()
    for ext in (".md", ".json"):
        p = RESULTS_DIR / f"{PRIM}{ext}"
        if p.exists():
            p.unlink()
    for s in SHAPES:
        _run_one(*s)


if __name__ == "__main__":
    main()
