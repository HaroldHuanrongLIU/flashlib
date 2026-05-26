"""MultinomialNB: ``flash_multinomial_nb`` vs cuML / sklearn.

flashlib fits per-class log-prob tables with a Triton kernel, then
predicts via a single ``X_test @ logp.T`` GEMM. ``tol=None`` keeps
the GEMM in input dtype (exact); ``tol=1e-3`` casts to bf16 (~2-3x
on the GEMM, ~1e-3 rel-err on the joint-log-likelihood; safe for
argmax).

Correctness signal:
* Accuracy on held-out fraction of the same dataset.
"""
from benchmarks.vs_cuml._common import (
    cap_threads, cuml_shim, time_gpu, time_cpu, title, header, fmt_table,
)
cap_threads(); cuml_shim()

import warnings; warnings.filterwarnings("ignore")
import numpy as np
import torch

from sklearn.naive_bayes import MultinomialNB as skMNB
from sklearn.datasets import make_classification
from sklearn.metrics import accuracy_score
from cuml.naive_bayes import MultinomialNB as cuMNB
from flashlib.primitives.multinomial_nb import flash_multinomial_nb


# (label, N, D, n_classes, use_sklearn_cpu)
SHAPES = [
    ("medium  N=100K D=512   C=10",   100_000,   512, 10, True),
    ("large   N=500K D=1024  C=10",   500_000, 1_024, 10, False),
    ("xlarge  N=1M   D=2048  C=20", 1_000_000, 2_048, 20, False),
]

ALPHA = 1.0


def _gen_counts(N, D, n_classes, rng):
    """Generate sparse non-negative integer count features per class."""
    X, y = make_classification(
        n_samples=N, n_features=D, n_informative=min(D, 64),
        n_redundant=0, n_classes=n_classes, n_clusters_per_class=1,
        random_state=0,
    )
    # Shift to non-negative + discretise. MultinomialNB requires counts.
    X = np.clip(np.round(X * 5.0 + 5.0), 0, None).astype(np.float32)
    return X, y.astype(np.int64)


def run_one(label, N, D, n_classes, use_sklearn_cpu: bool):
    title(f"MultinomialNB {label}  (N={N:,}, D={D}, C={n_classes}, "
          f"alpha={ALPHA})")

    rng = np.random.RandomState(0)
    X_np, y_np = _gen_counts(N, D, n_classes, rng)

    n_test = max(1024, N // 10)
    Xtr, Xte = X_np[:-n_test], X_np[-n_test:]
    ytr, yte = y_np[:-n_test], y_np[-n_test:]

    rows = []
    if use_sklearn_cpu:
        sk = skMNB(alpha=ALPHA).fit(Xtr, ytr)
        sk_acc = accuracy_score(yte, sk.predict(Xte))
        t_sk = time_cpu(
            lambda: skMNB(alpha=ALPHA).fit(Xtr, ytr).predict(Xte),
            repeat=1,
        )
        rows.append(("fp32", "sklearn (CPU)", f"{t_sk:7.2f}",
                     f"{sk_acc:.4f}", "1.00x"))

    # Pre-stage on GPU so cuml's timing reflects compute, not the H2D
    # copy implicit in fit(numpy)/predict(numpy).
    import cupy as cp
    Xtr_cp = cp.asarray(Xtr)
    ytr_cp = cp.asarray(ytr)
    Xte_cp = cp.asarray(Xte)
    cu = cuMNB(alpha=ALPHA).fit(Xtr_cp, ytr_cp)
    cu_acc = accuracy_score(yte, cp.asnumpy(cu.predict(Xte_cp)))
    t_cu = time_gpu(
        lambda: cuMNB(alpha=ALPHA).fit(Xtr_cp, ytr_cp).predict(Xte_cp),
        repeat=3, warmup=1,
    )
    rows.append(("fp32", "cuml", f"{t_cu:7.2f}",
                 f"{cu_acc:.4f}", "1.00x"))

    Xtr_t = torch.tensor(Xtr, device="cuda")
    ytr_t = torch.tensor(ytr, device="cuda")
    Xte_t = torch.tensor(Xte, device="cuda")
    # bf16 storage (``tol=1e-3``) is dropped from the bench: at large C
    # the per-class log-prob spread is small (sum-to-zero), so the bf16
    # predict GEMM loses enough precision to flip argmax — measured
    # 0.05 accuracy at (D=2048, C=20) vs 0.40 fp32 on this data. The
    # opt-in is left in the public API for callers who know their
    # spread is safe.
    variants = [
        ("fp32 exact", torch.float32, None),
    ]
    for dlabel, dtype, tol in variants:
        Xtr_d = Xtr_t.to(dtype)
        Xte_d = Xte_t.to(dtype)
        labels = flash_multinomial_nb(
            Xtr_d, ytr_t, Xte_d, n_classes=n_classes,
            alpha=ALPHA, tol=tol,
        )
        fl_acc = accuracy_score(yte, labels.cpu().numpy())
        t_fl = time_gpu(
            lambda: flash_multinomial_nb(
                Xtr_d, ytr_t, Xte_d, n_classes=n_classes,
                alpha=ALPHA, tol=tol),
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
