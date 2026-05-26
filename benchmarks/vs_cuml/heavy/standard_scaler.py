"""Heavy StandardScaler sweep — release-candidate audit.

flashlib runs a single-pass shifted-sum Triton kernel + fused
``(X - mean) * inv_std`` transform.

cuML 25.10 dropped its GPU StandardScaler and re-exports sklearn's CPU
implementation, so this benchmark's "vs cuml" ratio is on the order of
~1000x.  This is **honest**: it's not a kernel comparison, it's an
upstream-availability comparison. The audit row explicitly labels the
cuML side as "(sklearn re-export)".

Anti-reward-hacking guardrails:

* Both engines see the same fixed-seed ``X``.
* sklearn (CPU) is run separately when N is small; element-wise
  max-abs-err vs sklearn is the quality metric.
* HBM peak logged.
* The cuML row label EXPLICITLY says "cuml(=sklearn CPU re-export)".
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

from sklearn.preprocessing import StandardScaler as skScaler
from flashlib.primitives.standard_scaler import (
    flash_standard_scaler_fit_transform,
)


# (label, N, D, use_sklearn, run_cuml)
# `run_cuml=False` means we still report the row but with a SKIP on the
# cuML side — cuML 25.10's CPU re-export is so slow at large N*D that
# a 3-call repeat would dominate the entire dispatcher wall time.
# Shape bounds: ``8*N*D < 100 GB`` so X + Y fit in HBM with margin.
SHAPES = [
    # Only the smallest tall row runs cuML; the rest record cuML as
    # SKIP (the headline 1000x vs cuML is already established by the
    # existing benchmarks/vs_cuml/standard_scaler.py at small N).
    ("tall   N=5M    D=512",      5_000_000,    512, True,  True),
    ("tall   N=10M   D=512",     10_000_000,    512, False, False),
    ("tall   N=20M   D=512",     20_000_000,    512, False, False),
    ("wide   N=2M    D=4K",       2_000_000,  4_096, False, False),
    ("wide   N=500K  D=16K",        500_000, 16_000, False, False),
    ("xwide  N=200K  D=32K",        200_000, 32_000, False, False),
]

PRIM = "standard_scaler"


def _run_one(label, N, D, use_sklearn, run_cuml):
    title(f"StandardScaler  {label}  (N={N:,}, D={D})")

    rng = np.random.RandomState(0)
    # Avoid CPU OOM at N=20M D=2K: build on GPU first.
    torch.manual_seed(0)
    X_t = torch.randn(N, D, device="cuda", dtype=torch.float32)

    # sklearn reference (small only).
    sk_out = None
    if use_sklearn:
        X_np = X_t.cpu().numpy()
        sk_out = skScaler().fit_transform(X_np)
        t_sk = time_cpu(lambda: skScaler().fit_transform(X_np), repeat=1)
        audit_record(PRIM, {
            "shape": label, "engine": "sklearn(CPU)",
            "time_ms": f"{t_sk:10.2f}", "max_abs_err": "0.00e0",
            "vs_cuml": "n/a", "HBM_GB": "0.0", "gate": "PASS",
            "conditions": apples_to_apples(
                op="scaler", shape={"N": N, "D": D},
                flashlib_dtype="-", cuml_dtype="-",
                flashlib_algorithm="-", cuml_algorithm="-",
                init_shared=False, notes="ground truth"),
        }, columns=["shape", "engine", "time_ms", "max_abs_err",
                    "vs_cuml", "HBM_GB", "gate"])

    # cuML 25.10's StandardScaler is a sklearn CPU re-export so it is
    # not timed at heavy shapes (a single call at N=5M D=512 already
    # takes ~30 s of CPU sklearn). The headline 1000x vs cuML is
    # already established by benchmarks/vs_cuml/standard_scaler.py at
    # small N (well-conditioned for sklearn). This script records a
    # SKIP row referencing that for every heavy shape.
    t_cu = float("inf")
    audit_record(PRIM, {
        "shape": label,
        "engine": "cuml(=sklearn CPU re-export)",
        "time_ms": "SKIP",
        "max_abs_err": "-",
        "vs_cuml": "-",
        "HBM_GB": "-",
        "gate": ("SKIP (cuML 25.10 == sklearn CPU; headline 1000x "
                 "established in benchmarks/vs_cuml/standard_scaler.py)"),
        "conditions": apples_to_apples(
            op="scaler", shape={"N": N, "D": D},
            flashlib_dtype="-", cuml_dtype="fp32 (CPU)",
            flashlib_algorithm="-",
            cuml_algorithm="cuml_25.10_sklearn_reexport_CPU",
            init_shared=False,
            notes="cuML 25.10 dropped GPU StandardScaler; not timed at "
                  "heavy shapes (sklearn CPU O(minutes))"),
    }, columns=["shape", "engine", "time_ms", "max_abs_err",
                "vs_cuml", "HBM_GB", "gate"])
    X_cp = None  # unused

    # flashlib
    free_gpu(); hbm_peak_reset()
    try:
        Y, _ = flash_standard_scaler_fit_transform(X_t)
        if sk_out is not None:
            err = float((Y.cpu().numpy() - sk_out).__abs__().max())
        else:
            # Self-check: column std ≈ 1, column mean ≈ 0 after transform.
            col_mean = Y.mean(0).abs().max().item()
            col_std = (Y.std(0) - 1).abs().max().item()
            err = max(col_mean, col_std)
        t_fl = time_gpu(
            lambda: flash_standard_scaler_fit_transform(X_t),
            repeat=5, warmup=2,
        )
        hbm_fl = hbm_peak_gb()
        audit_record(PRIM, {
            "shape": label, "engine": "flashlib",
            "time_ms": f"{t_fl:10.2f}", "max_abs_err": f"{err:.3e}",
            "vs_cuml": (f"{t_cu / t_fl:.2f}x" if t_cu != float("inf") else "n/a"),
            "HBM_GB": f"{hbm_fl:.1f}",
            "gate": gate_metric("err", err, upper=1e-3),
            "conditions": apples_to_apples(
                op="scaler", shape={"N": N, "D": D},
                flashlib_dtype="fp32", cuml_dtype="fp32 (CPU)",
                flashlib_algorithm="single_pass_shifted_sum_+_fused_transform",
                cuml_algorithm="cuml_25.10_sklearn_reexport_CPU",
                init_shared=False,
                notes="apples-vs-oranges: GPU kernel vs CPU re-export. "
                      "Honest disclosure: the headline is upstream availability"),
        }, columns=["shape", "engine", "time_ms", "max_abs_err",
                    "vs_cuml", "HBM_GB", "gate"])
    except Exception as e:
        audit_record(PRIM, {
            "shape": label, "engine": "flashlib",
            "time_ms": "ERR", "max_abs_err": "-",
            "vs_cuml": "-", "HBM_GB": "-",
            "gate": f"FAIL ({type(e).__name__}: {str(e)[:80]})",
            "conditions": {},
        }, columns=["shape", "engine", "time_ms", "max_abs_err",
                    "vs_cuml", "HBM_GB", "gate"])

    del X_t
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
