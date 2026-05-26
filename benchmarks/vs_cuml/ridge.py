"""Ridge: ``flash_ridge_regression`` vs ``cuml.linear_model.Ridge``.

flashlib runs normal equations + Cholesky with the L2 penalty added to
the diagonal; ``tol=None`` is exact (fp32 cuBLAS GEMM + iterative
refinement). Pass ``tol=1e-3`` to cast to bf16 storage in the GEMM.

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

from sklearn.linear_model import Ridge as skRidge
from sklearn.datasets import make_regression
from sklearn.metrics import r2_score
from cuml.linear_model import Ridge as cuRidge
from flashlib.primitives.ridge import flash_ridge_regression


# (label, N, D, alpha, use_sklearn_cpu)
SHAPES = [
    ("tall  N=100K D=64   a=1",  100_000,   64,  1.0, True),
    ("tall  N=500K D=128  a=1",  500_000,  128,  1.0, False),
    ("tall  N=1M   D=256  a=10", 1_000_000, 256, 10.0, False),
    ("wide  N=2K   D=8K   a=1",    2_000, 8_000,  1.0, False),
]


def run_one(label, N, D, alpha, use_sklearn_cpu: bool):
    title(f"Ridge {label}  (N={N:,}, D={D}, alpha={alpha})")

    X_np, y_np = make_regression(n_samples=N, n_features=D, noise=1.0,
                                  random_state=0)
    X_np = X_np.astype(np.float32)
    y_np = y_np.astype(np.float32)

    n_test = max(1024, N // 10)
    Xtr, Xte = X_np[:-n_test], X_np[-n_test:]
    ytr, yte = y_np[:-n_test], y_np[-n_test:]

    rows = []
    if use_sklearn_cpu:
        sk = skRidge(alpha=alpha).fit(Xtr, ytr)
        sk_r2 = r2_score(yte, sk.predict(Xte))
        t_sk = time_cpu(lambda: skRidge(alpha=alpha).fit(Xtr, ytr),
                         repeat=1)
        rows.append(("fp32", "sklearn (CPU)", f"{t_sk:7.2f}",
                     f"{sk_r2:.4f}", "1.00x"))

    cu = cuRidge(alpha=alpha).fit(Xtr, ytr)
    cu_r2 = r2_score(yte, np.asarray(cu.predict(Xte)))
    t_cu = time_gpu(lambda: cuRidge(alpha=alpha).fit(Xtr, ytr),
                    repeat=3, warmup=1)
    rows.append(("fp32", "cuml", f"{t_cu:7.2f}",
                 f"{cu_r2:.4f}", "1.00x"))

    Xtr_t = torch.tensor(Xtr, device="cuda")
    ytr_t = torch.tensor(ytr, device="cuda")
    Xte_t = torch.tensor(Xte, device="cuda")
    # See ``linear_regression.py`` for why bf16 storage is gated off:
    # ``make_regression``'s informative-subset design can drive XtX out
    # of PSD under bf16. Ridge's L2 diagonal saves it for moderate
    # ``alpha`` but the gain is marginal; report only the fp32 path.
    variants = [
        ("fp32 exact", torch.float32, None),
    ]
    for dlabel, dtype, tol in variants:
        X = Xtr_t.to(dtype)
        y = ytr_t.to(dtype)
        Xte_dt = Xte_t.to(dtype)
        w = flash_ridge_regression(X, y, alpha=alpha, tol=tol)
        pred = (Xte_dt.float() @ w.float()).cpu().numpy()
        fl_r2 = r2_score(yte, pred)
        t_fl = time_gpu(
            lambda: flash_ridge_regression(X, y, alpha=alpha, tol=tol),
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
