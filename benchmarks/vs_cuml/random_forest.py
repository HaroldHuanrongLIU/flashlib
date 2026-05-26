"""RandomForestClassifier: ``FlashRandomForestClassifier`` vs cuML / sklearn.

flashlib runs:
  * uint8 quantile binning of features (one-shot at fit-start).
  * batched level-wise BFS tree growth via fused Triton histogram /
    best-split / partition kernels.
  * inference: vectorised torch tree traversal across all trees.

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

from sklearn.ensemble import RandomForestClassifier as skRF
from sklearn.datasets import make_classification
from sklearn.metrics import accuracy_score
from cuml.ensemble import RandomForestClassifier as cuRF
from flashlib.primitives.random_forest import FlashRandomForestClassifier


# (label, N, D, n_classes, n_estimators, max_depth, use_sklearn_cpu)
SHAPES = [
    ("small  N=10K   D=32  trees=100 depth=10",  10_000,  32, 4, 100, 10, True),
    ("medium N=100K  D=64  trees=100 depth=12", 100_000,  64, 6, 100, 12, False),
    ("large  N=500K  D=128 trees=100 depth=14", 500_000, 128, 8, 100, 14, False),
]


def _fit_predict_one(label, N, D, n_classes, n_estimators, max_depth,
                     use_sklearn_cpu: bool):
    title(f"RandomForest {label}  (N={N:,}, D={D}, C={n_classes}, "
          f"trees={n_estimators}, depth={max_depth})")

    X_np, y_np = make_classification(
        n_samples=N, n_features=D, n_informative=min(D, 16),
        n_redundant=0, n_classes=n_classes, n_clusters_per_class=2,
        random_state=0,
    )
    X_np = X_np.astype(np.float32)
    y_np = y_np.astype(np.int32)

    n_test = max(1024, N // 10)
    Xtr, Xte = X_np[:-n_test], X_np[-n_test:]
    ytr, yte = y_np[:-n_test], y_np[-n_test:]

    rows = []
    if use_sklearn_cpu:
        sk = skRF(n_estimators=n_estimators, max_depth=max_depth,
                  n_jobs=8, random_state=0).fit(Xtr, ytr)
        sk_acc = accuracy_score(yte, sk.predict(Xte))
        t_sk = time_cpu(
            lambda: skRF(n_estimators=n_estimators, max_depth=max_depth,
                          n_jobs=8, random_state=0).fit(Xtr, ytr),
            repeat=1,
        )
        rows.append(("fp32", "sklearn (CPU)", f"{t_sk:7.2f}",
                     f"{sk_acc:.4f}", "1.00x"))

    cu = cuRF(n_estimators=n_estimators, max_depth=max_depth,
              random_state=0).fit(Xtr, ytr)
    cu_acc = accuracy_score(yte, np.asarray(cu.predict(Xte)))
    t_cu = time_gpu(
        lambda: cuRF(n_estimators=n_estimators, max_depth=max_depth,
                      random_state=0).fit(Xtr, ytr),
        repeat=3, warmup=1,
    )
    rows.append(("fp32", "cuml", f"{t_cu:7.2f}",
                 f"{cu_acc:.4f}", "1.00x"))

    Xtr_t = torch.tensor(Xtr, device="cuda")
    ytr_t = torch.tensor(ytr, device="cuda", dtype=torch.int32)
    Xte_t = torch.tensor(Xte, device="cuda")
    # ``max_features=None`` (use all features per split) is set
    # explicitly: the default ``'sqrt'`` subsamples features per split
    # and on these synthetic shapes loses ~25 absolute accuracy points
    # vs cuML — an under-training mode in the level-wise BFS split
    # selector at small subsample counts. ``None`` matches cuML's
    # accuracy at parity / better on these shapes and isolates the
    # kernel speedup, which is the headline number.
    fl = FlashRandomForestClassifier(
        n_estimators=n_estimators, max_depth=max_depth,
        max_features=None, seed=0,
    ).fit(Xtr_t, ytr_t)
    fl_acc = accuracy_score(yte, fl.predict(Xte_t).cpu().numpy())
    t_fl = time_gpu(
        lambda: FlashRandomForestClassifier(
            n_estimators=n_estimators, max_depth=max_depth,
            max_features=None, seed=0,
        ).fit(Xtr_t, ytr_t),
        repeat=3, warmup=1,
    )
    rows.append(("fp32", "flashlib", f"{t_fl:7.2f}",
                 f"{fl_acc:.4f}", f"{t_cu / t_fl:.2f}x"))

    print(fmt_table(rows, ["dtype", "engine", "time(ms)",
                            "accuracy", "vs cuml"]))


def main():
    header()
    for s in SHAPES:
        _fit_predict_one(*s)
    print()


if __name__ == "__main__":
    main()
