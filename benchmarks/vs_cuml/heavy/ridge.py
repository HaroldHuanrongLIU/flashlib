"""Heavy Ridge sweep — release-candidate audit.

Same shape family as ``heavy/linear_regression.py`` plus a multi-
target row (T=64) that exercises the batched Cholesky-solve path.

Anti-reward-hacking guardrails:

* Same fixed-seed inputs across engines.
* R^2 on a 10% held-out slice. Multi-target row averages R^2 over the
  64 targets.
* fp32 only (bf16 storage is unsafe at the make_regression rank-
  deficient design).
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
from cuml.linear_model import Ridge as cuRidge
from flashlib.primitives.ridge import flash_ridge_regression


# (label, N, D, T, alpha)
# HBM budget: 8*N*D + 4*N*T fp32 bytes must fit in 140 GB.
SHAPES = [
    ("tall   N=2M    D=512    T=1   a=1",   2_000_000,    512,  1,  1.0),
    ("tall   N=5M    D=1024   T=1   a=1",   5_000_000,  1_024,  1,  1.0),
    ("tall   N=3M    D=2048   T=1   a=1",   3_000_000,  2_048,  1, 10.0),
    ("tall   N=2M    D=4096   T=1   a=10",  2_000_000,  4_096,  1, 10.0),
    ("multi  N=2M    D=512    T=64  a=1",   2_000_000,    512, 64,  1.0),
    ("multi  N=2M    D=1024   T=64  a=1",   2_000_000,  1_024, 64,  1.0),
    ("wide   N=500K  D=16K    T=1   a=10",    500_000, 16_000,  1, 10.0),
]

PRIM = "ridge"


def _r2_multi(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Average R^2 across targets (per-target then mean)."""
    if y_true.ndim == 1:
        return float(r2_score(y_true, y_pred))
    return float(np.mean([r2_score(y_true[:, t], y_pred[:, t])
                          for t in range(y_true.shape[1])]))


def _run_one(label, N, D, T, alpha):
    title(f"Ridge  {label}  (N={N:,}, D={D}, T={T}, alpha={alpha})")

    torch.manual_seed(0)
    X32 = torch.randn(N, D, device="cuda", dtype=torch.float32)
    w_true = torch.randn(D, T, device="cuda", dtype=torch.float32) * 0.1
    noise = 0.05 * torch.randn(N, T, device="cuda", dtype=torch.float32)
    Y32 = X32 @ w_true + noise
    if T == 1:
        Y32 = Y32.squeeze(-1)

    n_test = max(8192, N // 20)
    Xte_t = X32[-n_test:].contiguous()
    Yte_t = Y32[-n_test:].contiguous()
    Xtr_t = X32[:-n_test].contiguous()
    Ytr_t = Y32[:-n_test].contiguous()
    Yte_np = Yte_t.cpu().numpy()

    # cuML — cupy zero-copy.
    cu_repeats = 1 if (N >= 5_000_000 or D >= 8_000) else 2
    Xtr_cp = cp.from_dlpack(Xtr_t)
    Ytr_cp = cp.from_dlpack(Ytr_t)
    Xte_cp = cp.from_dlpack(Xte_t)
    free_gpu(); hbm_peak_reset()
    try:
        cu = cuRidge(alpha=alpha).fit(Xtr_cp, Ytr_cp)
        cu_pred = cp.asnumpy(cu.predict(Xte_cp))
        cu_r2 = _r2_multi(Yte_np, cu_pred)
        t_cu = time_gpu(lambda: cuRidge(alpha=alpha).fit(Xtr_cp, Ytr_cp),
                        repeat=cu_repeats, warmup=1 if N < 5_000_000 else 0)
        hbm_cu = hbm_peak_gb()
        audit_record(PRIM, {
            "shape": label, "dtype": "fp32", "engine": "cuml",
            "time_ms": f"{t_cu:10.2f}", "R2_mean": f"{cu_r2:.4f}",
            "vs_cuml": "1.00x", "HBM_GB": f"{hbm_cu:.1f}",
            "gate": gate_metric("R2", cu_r2, lower=0.99),
            "conditions": apples_to_apples(
                op="ridge", shape={"N": N, "D": D, "T": T, "alpha": alpha},
                flashlib_dtype="-", cuml_dtype="fp32",
                flashlib_algorithm="-",                 cuml_algorithm="cuml_ridge_default",
                init_shared=False,
                notes="cuML fit consumes cupy zero-copy (no H2D); "
                      "compute-only timing"),
        }, columns=["shape", "dtype", "engine", "time_ms", "R2_mean",
                    "vs_cuml", "HBM_GB", "gate"])
    except Exception as e:
        t_cu = float("inf")
        audit_record(PRIM, {
            "shape": label, "dtype": "fp32", "engine": "cuml",
            "time_ms": "ERR", "R2_mean": "-", "vs_cuml": "-", "HBM_GB": "-",
            "gate": f"FAIL ({type(e).__name__}: {str(e)[:80]})",
            "conditions": {},
        }, columns=["shape", "dtype", "engine", "time_ms", "R2_mean",
                    "vs_cuml", "HBM_GB", "gate"])

    # flashlib fp32 exact.
    free_gpu(); hbm_peak_reset()
    try:
        w = flash_ridge_regression(Xtr_t, Ytr_t, alpha=alpha, tol=None)
        pred = (Xte_t @ w).cpu().numpy()
        fl_r2 = _r2_multi(Yte_np, pred)
        fl_repeats = 2 if N >= 5_000_000 else 3
        t_fl = time_gpu(
            lambda: flash_ridge_regression(Xtr_t, Ytr_t, alpha=alpha, tol=None),
            repeat=fl_repeats, warmup=1,
        )
        hbm_fl = hbm_peak_gb()
        audit_record(PRIM, {
            "shape": label, "dtype": "fp32 exact", "engine": "flashlib",
            "time_ms": f"{t_fl:10.2f}", "R2_mean": f"{fl_r2:.4f}",
            "vs_cuml": (f"{t_cu / t_fl:.2f}x" if t_cu != float("inf") else "n/a"),
            "HBM_GB": f"{hbm_fl:.1f}",
            "gate": gate_metric("R2", fl_r2, lower=0.99),
            "conditions": apples_to_apples(
                op="ridge", shape={"N": N, "D": D, "T": T, "alpha": alpha},
                flashlib_dtype="fp32", cuml_dtype="fp32",
                flashlib_algorithm="normal_eq_diag(alpha)_+_cholesky_solve_3xbf16",
                cuml_algorithm="cuml_ridge_default",
                init_shared=False,
                notes="GPU-resident X, Y on both sides; pure compute"),
        }, columns=["shape", "dtype", "engine", "time_ms", "R2_mean",
                    "vs_cuml", "HBM_GB", "gate"])
    except Exception as e:
        audit_record(PRIM, {
            "shape": label, "dtype": "fp32 exact", "engine": "flashlib",
            "time_ms": "ERR", "R2_mean": "-", "vs_cuml": "-", "HBM_GB": "-",
            "gate": f"FAIL ({type(e).__name__}: {str(e)[:80]})",
            "conditions": {},
        }, columns=["shape", "dtype", "engine", "time_ms", "R2_mean",
                    "vs_cuml", "HBM_GB", "gate"])

    del X32, Y32, w_true, noise, Xtr_cp, Ytr_cp, Xte_cp
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
