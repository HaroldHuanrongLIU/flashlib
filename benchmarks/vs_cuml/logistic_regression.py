"""LogisticRegression: ``flash_logistic_regression`` vs cuML / sklearn.

flashlib runs L-BFGS with closed-form iter-0 step + fused Triton
sigmoid/residual/loss kernel. ``tol=None`` keeps the dominant
``X @ w`` / ``X.T @ r`` GEMVs in input dtype (exact); ``tol=1e-3``
casts ``X`` to bf16 storage for ~3-5x GEMV speedup.

Correctness signal:
* Accuracy on held-out fraction of the same dataset.

The signature uses ``gtol`` (sklearn's ``tol`` rename) for the
convergence tolerance, freeing ``tol`` for the precision lever.
"""
from benchmarks.vs_cuml._common import (
    cap_threads, cuml_shim, time_gpu, time_cpu, title, header, fmt_table,
)
cap_threads(); cuml_shim()

import warnings; warnings.filterwarnings("ignore")
import numpy as np
import torch

from sklearn.linear_model import LogisticRegression as skLogReg
from sklearn.datasets import make_classification
from sklearn.metrics import accuracy_score
from cuml.linear_model import LogisticRegression as cuLogReg
from flashlib.primitives.logistic_regression import flash_logistic_regression


# (label, N, D, use_sklearn_cpu)
SHAPES = [
    ("tall  N=100K D=64",   100_000,   64, True),
    ("tall  N=500K D=128",  500_000,  128, False),
    ("tall  N=1M   D=256", 1_000_000, 256, False),
]

MAX_ITER = 100
C = 1.0
GTOL = 1e-4  # sklearn's `tol`


def run_one(label, N, D, use_sklearn_cpu: bool):
    title(f"LogisticRegression {label}  (N={N:,}, D={D}, "
          f"C={C}, max_iter={MAX_ITER})")

    X_np, y_np = make_classification(
        n_samples=N, n_features=D, n_informative=D // 2,
        n_redundant=0, random_state=0,
    )
    X_np = X_np.astype(np.float32)
    y_np = y_np.astype(np.float32)

    n_test = max(1024, N // 10)
    Xtr, Xte = X_np[:-n_test], X_np[-n_test:]
    ytr, yte = y_np[:-n_test], y_np[-n_test:]

    rows = []
    if use_sklearn_cpu:
        sk = skLogReg(C=C, max_iter=MAX_ITER, tol=GTOL,
                      solver="lbfgs").fit(Xtr, ytr)
        sk_acc = accuracy_score(yte, sk.predict(Xte))
        t_sk = time_cpu(
            lambda: skLogReg(C=C, max_iter=MAX_ITER, tol=GTOL,
                              solver="lbfgs").fit(Xtr, ytr),
            repeat=1,
        )
        rows.append(("fp32", "sklearn (CPU)", f"{t_sk:7.2f}",
                     f"{sk_acc:.4f}", "1.00x"))

    cu = cuLogReg(C=C, max_iter=MAX_ITER, tol=GTOL).fit(Xtr, ytr)
    cu_acc = accuracy_score(yte, np.asarray(cu.predict(Xte)))
    t_cu = time_gpu(
        lambda: cuLogReg(C=C, max_iter=MAX_ITER, tol=GTOL).fit(Xtr, ytr),
        repeat=3, warmup=1,
    )
    rows.append(("fp32", "cuml", f"{t_cu:7.2f}",
                 f"{cu_acc:.4f}", "1.00x"))

    Xtr_t = torch.tensor(Xtr, device="cuda")
    ytr_t = torch.tensor(ytr, device="cuda")
    Xte_t = torch.tensor(Xte, device="cuda")
    # bf16 storage (tol=1e-3) is intentionally not benchmarked here:
    # L-BFGS at ``gtol=1e-4`` requires more precision than bf16 GEMVs
    # provide, so the loop oscillates and either re-iterates 3x or
    # diverges. The exact fp32 path is the headline win.
    variants = [
        ("fp32 exact", torch.float32, None),
    ]
    for dlabel, dtype, tol in variants:
        X = Xtr_t.to(dtype)
        for attr in ("_flash_lr_storage_cache",):
            if hasattr(X, attr):
                delattr(X, attr)

        w, b = flash_logistic_regression(
            X, ytr_t, n_iter=MAX_ITER, C=C, gtol=GTOL, tol=tol)
        logits = (Xte_t @ w.float() + b.float()).cpu().numpy()
        fl_acc = accuracy_score(yte, (logits > 0).astype(np.float32))

        t_fl = time_gpu(
            lambda: flash_logistic_regression(
                X, ytr_t, n_iter=MAX_ITER, C=C, gtol=GTOL, tol=tol),
            repeat=5, warmup=2,
        )
        rows.append((dlabel, "flashlib", f"{t_fl:7.2f}",
                     f"{fl_acc:.4f}", f"{t_cu / t_fl:.2f}x"))

    print(fmt_table(rows, ["dtype", "engine", "time(ms)",
                            "accuracy", "vs cuml"]))


def main():
    header()
    for s in SHAPES:
        run_one(*s)
    print()


if __name__ == "__main__":
    main()
