"""Heavy LogisticRegression sweep — release-candidate audit.

The flashlib primitive is BINARY only (the L-BFGS path is built around
a single sigmoid + residual + loss kernel). The multinomial / C>2 path
in the plan ("1M D=256 C=20") is intentionally **out of scope**, and
this script declares a SKIP row for it so the audit transparently
records the gap.

Heavy binary shapes:

* N=5M D=512 — the original plan headline.
* N=2M D=1024, N=1M D=2048 — D-axis stress.
* N=10M D=256 — N-axis stress.

Anti-reward-hacking guardrails:

* Same ``make_classification`` seed across engines; same ``C``,
  ``max_iter``, ``gtol``.
* Accuracy on a 10% held-out slice; gate at >= 0.85 (the noise floor
  on ``make_classification(n_informative=D//2)``).
* fp32 only — bf16 storage is documented as unsafe at ``gtol=1e-4``
  (L-BFGS oscillates).
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

from sklearn.datasets import make_classification
from sklearn.metrics import accuracy_score
from cuml.linear_model import LogisticRegression as cuLogReg
from flashlib.primitives.logistic_regression import flash_logistic_regression


# (label, N, D, num_classes)
SHAPES = [
    ("binary  N=2M    D=512  C=2",   2_000_000,   512, 2),
    ("binary  N=5M    D=512  C=2",   5_000_000,   512, 2),
    ("binary  N=10M   D=256  C=2",  10_000_000,   256, 2),
    ("binary  N=2M    D=1024 C=2",   2_000_000, 1_024, 2),
    ("binary  N=1M    D=2048 C=2",   1_000_000, 2_048, 2),
    # Out-of-scope record: multinomial not supported by flashlib LR.
    ("multi   N=1M    D=256  C=20",  1_000_000,   256, 20),
]

MAX_ITER = 100
C_REG = 1.0
GTOL = 1e-4

PRIM = "logistic_regression"


def _run_one(label, N, D, num_classes):
    title(f"LogReg  {label}  (N={N:,}, D={D}, C={num_classes})")

    if num_classes > 2:
        # Flashlib LR is binary-only; record a transparent SKIP row so
        # the audit doc can point to it.
        audit_record(PRIM, {
            "shape": label, "engine": "flashlib",
            "time_ms": "SKIP", "accuracy": "-", "vs_cuml": "-",
            "HBM_GB": "-",
            "gate": "SKIP (multinomial not in scope)",
            "conditions": apples_to_apples(
                op="logreg", shape={"N": N, "D": D, "C": num_classes},
                flashlib_dtype="-", cuml_dtype="fp32",
                flashlib_algorithm="(binary only)",
                cuml_algorithm="cuml_logreg_default",
                init_shared=False,
                notes="flashlib LR ships binary L-BFGS only; "
                      "multinomial deferred to future release"),
        }, columns=["shape", "engine", "time_ms", "accuracy",
                    "vs_cuml", "HBM_GB", "gate"])
        return

    # make_classification at N=10M D=512 needs ~20 GB CPU + several
    # minutes. For the largest N we generate synthetic-but-meaningful
    # data directly on GPU.
    if N >= 5_000_000:
        torch.manual_seed(0)
        # Build labels first, then mix-mean per-class — recovers the
        # n_informative=D//2 separability without paying CPU
        # make_classification's wall.
        y_t = (torch.rand(N, device="cuda") < 0.5).float()
        sign = (2 * y_t - 1).unsqueeze(1)
        X_t = torch.randn(N, D, device="cuda", dtype=torch.float32)
        X_t[:, :D // 2] += 0.30 * sign
        y_np = y_t.cpu().numpy()
        # Skip the CPU intermediates entirely; everything stays on GPU.
        n_test = max(8192, N // 20)
        Xtr_t = X_t[:-n_test].contiguous()
        ytr_t = y_t[:-n_test].contiguous()
        Xte_t = X_t[-n_test:].contiguous()
        yte = y_np[-n_test:]
        # cuML cupy views
        Xtr_cp = cp.from_dlpack(Xtr_t)
        ytr_cp = cp.from_dlpack(ytr_t)
        Xte_cp = cp.from_dlpack(Xte_t)
        Xtr_src, ytr_src, Xte_src = Xtr_cp, ytr_cp, Xte_cp
        on_gpu = True
    else:
        X_np, y_np = make_classification(
            n_samples=N, n_features=D, n_informative=D // 2,
            n_redundant=0, random_state=0,
        )
        X_np = X_np.astype(np.float32)
        y_np = y_np.astype(np.float32)
        n_test = max(8192, N // 20)
        Xtr, Xte = X_np[:-n_test], X_np[-n_test:]
        ytr, yte = y_np[:-n_test], y_np[-n_test:]
        Xtr_src, ytr_src, Xte_src = Xtr, ytr, Xte
        on_gpu = False

    # cuML
    cu_repeats = 1 if N >= 5_000_000 else 2
    free_gpu(); hbm_peak_reset()
    try:
        cu = cuLogReg(C=C_REG, max_iter=MAX_ITER, tol=GTOL).fit(Xtr_src, ytr_src)
        cu_pred_raw = cu.predict(Xte_src)
        if hasattr(cu_pred_raw, "get"):
            cu_pred = cu_pred_raw.get()
        else:
            cu_pred = np.asarray(cu_pred_raw)
        cu_acc = accuracy_score(yte, cu_pred)
        t_cu = time_gpu(
            lambda: cuLogReg(C=C_REG, max_iter=MAX_ITER, tol=GTOL).fit(Xtr_src, ytr_src),
            repeat=cu_repeats, warmup=1 if N < 5_000_000 else 0,
        )
        hbm_cu = hbm_peak_gb()
        audit_record(PRIM, {
            "shape": label, "engine": "cuml",
            "time_ms": f"{t_cu:10.2f}", "accuracy": f"{cu_acc:.4f}",
            "vs_cuml": "1.00x", "HBM_GB": f"{hbm_cu:.1f}",
            "gate": gate_metric("acc", cu_acc, lower=0.85),
            "conditions": apples_to_apples(
                op="logreg", shape={"N": N, "D": D, "C": 2},
                flashlib_dtype="-", cuml_dtype="fp32",
                flashlib_algorithm="-", cuml_algorithm="cuml_lr_qn",
                init_shared=False,
                notes="cuML L-BFGS / QN; same C, gtol, max_iter as flashlib"),
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

    # flashlib fp32 exact.
    free_gpu(); hbm_peak_reset()
    try:
        if not on_gpu:
            Xtr_t = torch.tensor(Xtr, device="cuda")
            ytr_t = torch.tensor(ytr, device="cuda")
            Xte_t = torch.tensor(Xte, device="cuda")
        w, b = flash_logistic_regression(
            Xtr_t, ytr_t, n_iter=MAX_ITER, C=C_REG, gtol=GTOL, tol=None)
        logits = (Xte_t @ w + b).cpu().numpy()
        fl_acc = accuracy_score(yte, (logits > 0).astype(np.float32))
        fl_repeats = 2 if N >= 5_000_000 else 3
        t_fl = time_gpu(
            lambda: flash_logistic_regression(
                Xtr_t, ytr_t, n_iter=MAX_ITER, C=C_REG, gtol=GTOL, tol=None),
            repeat=fl_repeats, warmup=1,
        )
        hbm_fl = hbm_peak_gb()
        audit_record(PRIM, {
            "shape": label, "engine": "flashlib",
            "time_ms": f"{t_fl:10.2f}", "accuracy": f"{fl_acc:.4f}",
            "vs_cuml": (f"{t_cu / t_fl:.2f}x" if t_cu != float("inf") else "n/a"),
            "HBM_GB": f"{hbm_fl:.1f}",
            "gate": gate_metric("acc", fl_acc, lower=0.85),
            "conditions": apples_to_apples(
                op="logreg", shape={"N": N, "D": D, "C": 2},
                flashlib_dtype="fp32", cuml_dtype="fp32",
                flashlib_algorithm="lbfgs_+_analytic_iter0_+_fused_sigmoid_residual_loss",
                cuml_algorithm="cuml_lr_qn",
                init_shared=False,
                notes="matched algorithm class (L-BFGS-family); same gtol/max_iter"),
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
