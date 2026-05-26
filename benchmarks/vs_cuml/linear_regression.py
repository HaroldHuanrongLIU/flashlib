"""LinearRegression: ``flash_linear_regression`` vs ``cuml.LinearRegression``.

flashlib runs normal equations (``X.T @ X`` cuBLAS GEMM) + Cholesky;
``tol=None`` is exact (fp32 GEMM with one iterative refinement step).
Pass ``tol=1e-3`` to cast to bf16 storage in the GEMM.

Correctness signal:
* R^2 on held-out fraction of the same dataset.
"""
from benchmarks.vs_cuml._common import (
    cap_threads, cuml_shim, time_gpu, time_cpu, title, header, fmt_table,
)
cap_threads(); cuml_shim()

import warnings; warnings.filterwarnings("ignore")
import numpy as np
import torch

from sklearn.linear_model import LinearRegression as skLinReg
from sklearn.datasets import make_regression
from sklearn.metrics import r2_score
from cuml.linear_model import LinearRegression as cuLinReg
from flashlib.primitives.linear_regression import flash_linear_regression


SHAPES = [
    ("tall  N=100K D=64",  100_000,   64, True),
    ("tall  N=500K D=128", 500_000,  128, False),
    ("tall  N=1M   D=256", 1_000_000, 256, False),
    ("wide  N=2K   D=8K",     2_000, 8_000, False),
]


def run_one(label, N, D, use_sklearn_cpu: bool):
    title(f"LinearRegression {label}  (N={N:,}, D={D})")

    X_np, y_np = make_regression(n_samples=N, n_features=D, noise=1.0,
                                  random_state=0)
    X_np = X_np.astype(np.float32)
    y_np = y_np.astype(np.float32)

    # Hold out 10% for R^2.
    n_test = max(1024, N // 10)
    Xtr, Xte = X_np[:-n_test], X_np[-n_test:]
    ytr, yte = y_np[:-n_test], y_np[-n_test:]

    rows = []
    if use_sklearn_cpu:
        sk = skLinReg().fit(Xtr, ytr)
        sk_r2 = r2_score(yte, sk.predict(Xte))
        t_sk = time_cpu(lambda: skLinReg().fit(Xtr, ytr), repeat=1)
        rows.append(("fp32", "sklearn (CPU)", f"{t_sk:7.2f}",
                     f"{sk_r2:.4f}", "1.00x"))

    cu = cuLinReg().fit(Xtr, ytr)
    cu_r2 = r2_score(yte, np.asarray(cu.predict(Xte)))
    t_cu = time_gpu(lambda: cuLinReg().fit(Xtr, ytr), repeat=3, warmup=1)
    rows.append(("fp32", "cuml", f"{t_cu:7.2f}",
                 f"{cu_r2:.4f}", "1.00x"))

    Xtr_t = torch.tensor(Xtr, device="cuda")
    ytr_t = torch.tensor(ytr, device="cuda")
    Xte_t = torch.tensor(Xte, device="cuda")
    yte_t = torch.tensor(yte, device="cuda")
    # Linear regression: only the fp32 path is reported here. The bf16
    # storage opt-in (tol=1e-3) is bf16-only on the dominant ``X.T @ X``
    # GEMM and can drive XtX out of PSD on the conditioning of
    # ``make_regression`` (which is by design rank-deficient on the
    # informative subset). ``flash_linear_regression`` already keeps the
    # accumulator in fp32; the cuBLAS TF32 default is the headline path.
    variants = [
        ("fp32 exact", torch.float32, None),
    ]
    for dlabel, dtype, tol in variants:
        X = Xtr_t.to(dtype)
        y = ytr_t.to(dtype)
        Xte_dt = Xte_t.to(dtype)
        w = flash_linear_regression(X, y, tol=tol)
        pred = (Xte_dt.float() @ w.float()).cpu().numpy()
        fl_r2 = r2_score(yte, pred)
        t_fl = time_gpu(
            lambda: flash_linear_regression(X, y, tol=tol),
            repeat=5, warmup=2,
        )
        rows.append((dlabel, "flashlib", f"{t_fl:7.2f}",
                     f"{fl_r2:.4f}", f"{t_cu / t_fl:.2f}x"))

    print(fmt_table(rows, ["dtype", "engine", "time(ms)",
                            "R^2", "vs cuml"]))


def main():
    header()
    for s in SHAPES:
        run_one(*s)
    print()


if __name__ == "__main__":
    main()
