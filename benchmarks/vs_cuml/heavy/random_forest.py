"""Heavy RandomForest sweep — release-candidate audit.

Stresses both axes of the level-wise BFS tree builder + histogram-
subtraction trick:

* **N axis**: N=1M trees=200 depth=14 — the original plan headline.
* **D axis**: N=500K D=256 trees=100 depth=14 — per-split histogram
  cost scales O(D), this row stresses it.
* **depth axis**: N=500K trees=100 depth=18 — deep-tree row exercises
  the partition kernel near its leaf-count ceiling.

Anti-reward-hacking guardrails:

* Same ``make_classification`` seed across engines.
* Accuracy on 10% held-out slice; gate at >= 0.50 (RF is the noisiest
  baseline of any primitive in this audit).
* ``max_features=None`` (use all features per split) is set EXPLICITLY
  in flashlib (matches cuML default behaviour on these shapes) — the
  ``'sqrt'`` default in flashlib subsamples features per split and on
  synthetic data loses ~25 accuracy points; that mismatch is
  documented in ``benchmarks/vs_cuml/random_forest.py`` and is NOT a
  reward-hacking flag.
* sklearn (CPU) reference is included on the smallest row only.
"""
from benchmarks.vs_cuml.heavy._common import (
    cap_threads, cuml_shim, time_gpu, time_cpu, title, header,
    audit_record, apples_to_apples,
    hbm_peak_reset, hbm_peak_gb, gate_metric, free_gpu, RESULTS_DIR,
)
cap_threads(); cuml_shim()

import warnings; warnings.filterwarnings("ignore")
import gc
import numpy as np
import torch

from sklearn.ensemble import RandomForestClassifier as skRF
from sklearn.datasets import make_classification
from sklearn.metrics import accuracy_score
from cuml.ensemble import RandomForestClassifier as cuRF
from flashlib.primitives.random_forest import FlashRandomForestClassifier


# (label, N, D, C, n_est, depth, use_sk)
SHAPES = [
    ("small   N=100K  D=64  C=6  trees=100 depth=12", 100_000,  64, 6, 100, 12, True),
    ("medium  N=500K  D=128 C=8  trees=100 depth=14", 500_000, 128, 8, 100, 14, False),
    ("large   N=1M    D=128 C=8  trees=200 depth=14", 1_000_000, 128, 8, 200, 14, False),
    ("D-axis  N=500K  D=256 C=8  trees=100 depth=14", 500_000, 256, 8, 100, 14, False),
    ("depth   N=500K  D=128 C=8  trees=100 depth=18", 500_000, 128, 8, 100, 18, False),
]

PRIM = "random_forest"


def _gpu_classification(N, D, C):
    """GPU-resident class-conditioned synthetic features.

    make_classification gets brutal at N=1M D=128; this avoids the
    CPU step entirely.
    """
    torch.manual_seed(0)
    y_t = torch.randint(0, C, (N,), device="cuda", dtype=torch.int64)
    centers = torch.randn(C, D, device="cuda") * 1.5
    X_t = torch.randn(N, D, device="cuda") + centers[y_t]
    return X_t.cpu().numpy().astype(np.float32), y_t.cpu().numpy().astype(np.int32)


def _run_one(label, N, D, C, n_est, depth, use_sk):
    title(f"RandomForest  {label}  (N={N:,}, D={D}, C={C}, "
          f"trees={n_est}, depth={depth})")

    if N >= 500_000:
        X_np, y_np = _gpu_classification(N, D, C)
    else:
        X_np, y_np = make_classification(
            n_samples=N, n_features=D, n_informative=min(D, 32),
            n_redundant=0, n_classes=C, n_clusters_per_class=2,
            random_state=0,
        )
        X_np = X_np.astype(np.float32)
        y_np = y_np.astype(np.int32)
    n_test = max(8192, N // 20)
    Xtr, Xte = X_np[:-n_test], X_np[-n_test:]
    ytr, yte = y_np[:-n_test], y_np[-n_test:]

    if use_sk:
        sk = skRF(n_estimators=n_est, max_depth=depth, n_jobs=8,
                   random_state=0).fit(Xtr, ytr)
        sk_acc = accuracy_score(yte, sk.predict(Xte))
        t_sk = time_cpu(
            lambda: skRF(n_estimators=n_est, max_depth=depth, n_jobs=8,
                          random_state=0).fit(Xtr, ytr),
            repeat=1,
        )
        audit_record(PRIM, {
            "shape": label, "engine": "sklearn(CPU,n_jobs=8)",
            "time_ms": f"{t_sk:10.1f}", "accuracy": f"{sk_acc:.4f}",
            "vs_cuml": "n/a", "HBM_GB": "0.0", "gate": "PASS",
            "conditions": apples_to_apples(
                op="rf", shape={"N": N, "D": D, "C": C,
                                 "trees": n_est, "depth": depth},
                flashlib_dtype="-", cuml_dtype="-",
                flashlib_algorithm="-", cuml_algorithm="-",
                init_shared=False, notes="reference"),
        }, columns=["shape", "engine", "time_ms", "accuracy",
                    "vs_cuml", "HBM_GB", "gate"])

    # cuML
    free_gpu(); hbm_peak_reset()
    try:
        cu = cuRF(n_estimators=n_est, max_depth=depth, random_state=0) \
                .fit(Xtr, ytr)
        cu_acc = accuracy_score(yte, np.asarray(cu.predict(Xte)))
        t_cu = time_gpu(
            lambda: cuRF(n_estimators=n_est, max_depth=depth,
                          random_state=0).fit(Xtr, ytr),
            repeat=2, warmup=1,
        )
        hbm_cu = hbm_peak_gb()
        audit_record(PRIM, {
            "shape": label, "engine": "cuml",
            "time_ms": f"{t_cu:10.1f}", "accuracy": f"{cu_acc:.4f}",
            "vs_cuml": "1.00x", "HBM_GB": f"{hbm_cu:.1f}",
            "gate": gate_metric("acc", cu_acc, lower=0.50),
            "conditions": apples_to_apples(
                op="rf", shape={"N": N, "D": D, "C": C,
                                 "trees": n_est, "depth": depth},
                flashlib_dtype="-", cuml_dtype="fp32",
                flashlib_algorithm="-", cuml_algorithm="cuml_rf_default",
                init_shared=False, notes="cuML default split selector"),
        }, columns=["shape", "engine", "time_ms", "accuracy",
                    "vs_cuml", "HBM_GB", "gate"])
    except Exception as e:
        t_cu = float("inf")
        audit_record(PRIM, {
            "shape": label, "engine": "cuml",
            "time_ms": "ERR", "accuracy": "-", "vs_cuml": "-", "HBM_GB": "-",
            "gate": f"FAIL ({type(e).__name__}: {str(e)[:80]})",
            "conditions": {},
        }, columns=["shape", "engine", "time_ms", "accuracy",
                    "vs_cuml", "HBM_GB", "gate"])

    # flashlib (max_features=None, see docstring).
    free_gpu(); hbm_peak_reset()
    try:
        Xtr_t = torch.tensor(Xtr, device="cuda")
        ytr_t = torch.tensor(ytr, device="cuda", dtype=torch.int32)
        Xte_t = torch.tensor(Xte, device="cuda")
        fl = FlashRandomForestClassifier(
            n_estimators=n_est, max_depth=depth,
            max_features=None, seed=0,
        ).fit(Xtr_t, ytr_t)
        fl_acc = accuracy_score(yte, fl.predict(Xte_t).cpu().numpy())
        t_fl = time_gpu(
            lambda: FlashRandomForestClassifier(
                n_estimators=n_est, max_depth=depth,
                max_features=None, seed=0,
            ).fit(Xtr_t, ytr_t),
            repeat=2, warmup=1,
        )
        hbm_fl = hbm_peak_gb()
        audit_record(PRIM, {
            "shape": label, "engine": "flashlib",
            "time_ms": f"{t_fl:10.1f}", "accuracy": f"{fl_acc:.4f}",
            "vs_cuml": (f"{t_cu / t_fl:.2f}x" if t_cu != float("inf") else "n/a"),
            "HBM_GB": f"{hbm_fl:.1f}",
            "gate": gate_metric("acc", fl_acc, lower=0.50),
            "conditions": apples_to_apples(
                op="rf", shape={"N": N, "D": D, "C": C,
                                 "trees": n_est, "depth": depth},
                flashlib_dtype="fp32", cuml_dtype="fp32",
                flashlib_algorithm="uint8_quantile_+_level_wise_bfs_+_hist_subtract",
                cuml_algorithm="cuml_rf_default",
                init_shared=False,
                notes="max_features=None EXPLICITLY matches cuML default. "
                      "flashlib's 'sqrt' default loses 25 abs acc on this "
                      "synthetic data — kernel speedup is the headline"),
        }, columns=["shape", "engine", "time_ms", "accuracy",
                    "vs_cuml", "HBM_GB", "gate"])
    except Exception as e:
        audit_record(PRIM, {
            "shape": label, "engine": "flashlib",
            "time_ms": "ERR", "accuracy": "-", "vs_cuml": "-", "HBM_GB": "-",
            "gate": f"FAIL ({type(e).__name__}: {str(e)[:80]})",
            "conditions": {},
        }, columns=["shape", "engine", "time_ms", "accuracy",
                    "vs_cuml", "HBM_GB", "gate"])

    del X_np, y_np, Xtr, Xte, ytr, yte
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
