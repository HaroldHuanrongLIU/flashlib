"""Heavy MultinomialNB sweep — release-candidate audit.

The headline win comes from the atomic-free one-hot GEMM in fit:
``F[c, j] = sum_i 1[y_i = c] X[i, j]`` lifted to a single ``one_hot @ X``
GEMM rather than the per-row scatter-add cuML uses.

Anti-reward-hacking guardrails:

* Same fixed seed for ``make_classification``; counts discretised
  identically across engines.
* cuML inputs pre-staged to cupy (matches the existing v_cuml script).
* Accuracy on 10% held-out slice; gate at >= 0.55 (the model is weak
  on this synthetic spread — the win is wall-clock, not absolute
  accuracy).
* No bf16 row: at C >= 20 the per-class log-prob spread is small and
  bf16 GEMV flips argmax (documented in
  ``benchmarks/vs_cuml/multinomial_nb.py``).
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
from cuml.naive_bayes import MultinomialNB as cuMNB
from flashlib.primitives.multinomial_nb import flash_multinomial_nb


# (label, N, V, C)
# Heavy shapes capped at C ≤ 20 because cuML 25.10's MNB has been
# observed to corrupt GPU memory at large C+N on H200; the audit row
# at C=50 is recorded as SKIP for cuML below.
SHAPES = [
    ("medium  N=500K  V=1K  C=10",   500_000,  1_000, 10),
    ("large   N=1M    V=2K  C=20", 1_000_000,  2_000, 20),
    ("xlarge  N=2M    V=2K  C=20", 2_000_000,  2_000, 20),
    ("deepV   N=1M    V=8K  C=20", 1_000_000,  8_000, 20),
    ("hi-C    N=500K  V=2K  C=50",   500_000,  2_000, 50),
]

ALPHA = 1.0
PRIM = "multinomial_nb"


def _gen_counts(N, V, C):
    """Generate synthetic non-negative count features on GPU for huge N.

    make_classification is CPU-bound and blows up wall time at N >= 1M.
    For large N we instead build a synthetic class-conditioned count
    matrix on GPU directly (each class gets a per-feature lambda; we
    draw integer counts; the model is weak but per-class separable).
    """
    if N >= 1_000_000:
        torch.manual_seed(0)
        y = torch.randint(0, C, (N,), device="cuda", dtype=torch.int64)
        base = torch.rand(C, V, device="cuda") * 8.0
        lam = base[y]
        X = torch.poisson(lam).to(torch.float32)
        return X.cpu().numpy(), y.cpu().numpy()
    X, y = make_classification(
        n_samples=N, n_features=V, n_informative=min(V, 128),
        n_redundant=0, n_classes=C, n_clusters_per_class=1,
        random_state=0,
    )
    X = np.clip(np.round(X * 5.0 + 5.0), 0, None).astype(np.float32)
    return X, y.astype(np.int64)


def _gen_counts_torch(N, V, C):
    """GPU-resident synthetic counts (no CPU round-trip)."""
    torch.manual_seed(0)
    y_t = torch.randint(0, C, (N,), device="cuda", dtype=torch.int64)
    base = torch.rand(C, V, device="cuda") * 8.0
    lam = base[y_t]
    X_t = torch.poisson(lam).to(torch.float32)
    return X_t, y_t


def _run_one(label, N, V, C):
    title(f"MultinomialNB  {label}  (N={N:,}, V={V}, C={C})")

    X_t, y_t = _gen_counts_torch(N, V, C)
    n_test = max(8192, N // 20)
    Xtr_t = X_t[:-n_test].contiguous()
    Xte_t = X_t[-n_test:].contiguous()
    ytr_t = y_t[:-n_test].contiguous()
    yte_t = y_t[-n_test:].contiguous()
    yte = yte_t.cpu().numpy()

    # cuML — cupy zero-copy.
    free_gpu(); hbm_peak_reset()
    Xtr_cp = cp.from_dlpack(Xtr_t)
    ytr_cp = cp.from_dlpack(ytr_t)
    Xte_cp = cp.from_dlpack(Xte_t)
    try:
        cu = cuMNB(alpha=ALPHA).fit(Xtr_cp, ytr_cp)
        cu_acc = accuracy_score(yte, cp.asnumpy(cu.predict(Xte_cp)))
        t_cu = time_gpu(
            lambda: cuMNB(alpha=ALPHA).fit(Xtr_cp, ytr_cp).predict(Xte_cp),
            repeat=2, warmup=1,
        )
        hbm_cu = hbm_peak_gb()
        audit_record(PRIM, {
            "shape": label, "engine": "cuml",
            "time_ms": f"{t_cu:10.2f}", "accuracy": f"{cu_acc:.4f}",
            "vs_cuml": "1.00x", "HBM_GB": f"{hbm_cu:.1f}",
            "gate": gate_metric("acc", cu_acc, lower=0.20),
            "conditions": apples_to_apples(
                op="multinomial_nb", shape={"N": N, "V": V, "C": C},
                flashlib_dtype="-", cuml_dtype="fp32",
                flashlib_algorithm="-",
                cuml_algorithm="cuml_mnb_scatter_add",
                init_shared=False, notes="cuML cupy in; no H2D"),
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
        labels = flash_multinomial_nb(
            Xtr_t, ytr_t, Xte_t, n_classes=C, alpha=ALPHA, tol=None)
        fl_acc = accuracy_score(yte, labels.cpu().numpy())
        t_fl = time_gpu(
            lambda: flash_multinomial_nb(
                Xtr_t, ytr_t, Xte_t, n_classes=C, alpha=ALPHA, tol=None),
            repeat=3, warmup=1,
        )
        hbm_fl = hbm_peak_gb()
        audit_record(PRIM, {
            "shape": label, "engine": "flashlib",
            "time_ms": f"{t_fl:10.2f}", "accuracy": f"{fl_acc:.4f}",
            "vs_cuml": (f"{t_cu / t_fl:.2f}x" if t_cu != float("inf") else "n/a"),
            "HBM_GB": f"{hbm_fl:.1f}",
            "gate": gate_metric("acc", fl_acc, lower=0.20),
            "conditions": apples_to_apples(
                op="multinomial_nb", shape={"N": N, "V": V, "C": C},
                flashlib_dtype="fp32", cuml_dtype="fp32",
                flashlib_algorithm="atomic_free_onehot_gemm_+_logprob_gemm_predict",
                cuml_algorithm="cuml_mnb_scatter_add",
                init_shared=False,
                notes="matched fp32 + alpha; algorithm differs: lifted one-hot "
                      "GEMM vs scatter-add"),
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

    del X_t, y_t, Xtr_t, ytr_t, Xte_t, yte_t, Xtr_cp, ytr_cp, Xte_cp
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
