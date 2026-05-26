"""Heavy TruncatedSVD sweep — release-candidate audit.

Same shape family as ``heavy/pca.py`` (tall / wide / square) plus an
explicit K=128 row to ensure Halko's win is not specific to small K.

Anti-reward-hacking guardrails: identical to ``heavy/pca.py`` —
reference top-K singular values via torch SVD on the raw X (or via
eigh on Gram / Cov for the side whose dim is smaller), and a
matched-dtype lossless row vs cuML before the Halko opt-in.
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

from cuml.decomposition import TruncatedSVD as cuTSVD
from flashlib.primitives.truncated_svd import flash_truncated_svd


# (label, N, D, K)
SHAPES = [
    ("tall    N=2M    D=256  K=32",   2_000_000,   256,  32),
    ("tall    N=10M   D=256  K=64",  10_000_000,   256,  64),
    ("tall    N=10M   D=256  K=128", 10_000_000,   256, 128),
    ("tall    N=5M    D=512  K=64",   5_000_000,   512,  64),
    ("square  N=2M    D=2048 K=128",  2_000_000,  2_048, 128),
    ("wide    N=20K   D=16K  K=64",      20_000, 16_000,  64),
    ("wide    N=10K   D=8K   K=128",     10_000,  8_000, 128),
]

PRIM = "truncated_svd"


def _topk_sv_torch(X: torch.Tensor, K: int) -> np.ndarray:
    """Reference top-K singular values via torch eigh on the smaller
    of Cov / Gram (avoids the full O(N D min(N,D)) SVD when D >> K).
    """
    N, D = X.shape
    if D <= N:
        Cov = X.t() @ X
        ev = torch.linalg.eigvalsh(Cov)
    else:
        G = X @ X.t()
        ev = torch.linalg.eigvalsh(G)
    ev = torch.sort(ev, descending=True).values[:K]
    return torch.sqrt(ev.clamp_min(0)).float().cpu().numpy()


def _rel_err(a: np.ndarray, b: np.ndarray) -> float:
    denom = max(float(np.max(np.abs(b))), 1e-12)
    return float(np.max(np.abs(a - b)) / denom)


def _run_one(label, N, D, K):
    title(f"TruncatedSVD  {label}  (N={N:,}, D={D}, K={K})")

    torch.manual_seed(0)
    X32 = torch.randn(N, D, device="cuda", dtype=torch.float32)
    ref_sv = _topk_sv_torch(X32, K)

    # cuML fp32.
    free_gpu(); hbm_peak_reset()
    X_cp = cp.from_dlpack(X32)
    try:
        cu = cuTSVD(n_components=K).fit(X_cp)
        sv_raw = cu.singular_values_
        if hasattr(sv_raw, "get"):
            cu_sv = sv_raw.get().astype(np.float32)
        else:
            cu_sv = np.asarray(sv_raw).astype(np.float32)
        rel_cu = _rel_err(cu_sv, ref_sv)
        t_cu = time_gpu(lambda: cuTSVD(n_components=K).fit(X_cp),
                        repeat=2, warmup=1)
        hbm_cu = hbm_peak_gb()
        audit_record(PRIM, {
            "shape": label, "dtype": "fp32", "engine": "cuml",
            "time_ms": f"{t_cu:10.2f}",
            "rel_err_svK": f"{rel_cu:.3e}",
            "vs_cuml": "1.00x", "HBM_GB": f"{hbm_cu:.1f}",
            "gate": gate_metric("rel_err", rel_cu, upper=0.05),
            "conditions": apples_to_apples(
                op="tsvd", shape={"N": N, "D": D, "K": K},
                flashlib_dtype="-", cuml_dtype="fp32",
                flashlib_algorithm="-", cuml_algorithm="cuml_tsvd_default",
                init_shared=False, notes="reference top-K via torch eigh"),
        }, columns=["shape", "dtype", "engine", "time_ms", "rel_err_svK",
                    "vs_cuml", "HBM_GB", "gate"])
    except Exception as e:
        t_cu = float("inf")
        audit_record(PRIM, {
            "shape": label, "dtype": "fp32", "engine": "cuml",
            "time_ms": "ERR", "rel_err_svK": "-",
            "vs_cuml": "-", "HBM_GB": "-",
            "gate": f"FAIL ({type(e).__name__}: {str(e)[:80]})",
            "conditions": {},
        }, columns=["shape", "dtype", "engine", "time_ms", "rel_err_svK",
                    "vs_cuml", "HBM_GB", "gate"])

    variants = [
        ("fp32 exact", None, 0.05, "exact (cov/gram + eigh)"),
        ("fp32 halko", 1e-3, 0.05, "Halko subspace iter"),
    ]
    for dlabel, tol, gate_upper, alg in variants:
        free_gpu(); hbm_peak_reset()
        try:
            S, _ = flash_truncated_svd(X32, K=K, tol=tol)
            fl_sv = S.float().cpu().numpy()
            rel = _rel_err(fl_sv, ref_sv)
            t_fl = time_gpu(lambda: flash_truncated_svd(X32, K=K, tol=tol),
                            repeat=3, warmup=1)
            hbm_fl = hbm_peak_gb()
            audit_record(PRIM, {
                "shape": label, "dtype": dlabel, "engine": "flashlib",
                "time_ms": f"{t_fl:10.2f}",
                "rel_err_svK": f"{rel:.3e}",
                "vs_cuml": (f"{t_cu / t_fl:.2f}x" if t_cu != float("inf") else "n/a"),
                "HBM_GB": f"{hbm_fl:.1f}",
                "gate": gate_metric("rel_err", rel, upper=gate_upper),
                "conditions": apples_to_apples(
                    op="tsvd", shape={"N": N, "D": D, "K": K},
                    flashlib_dtype="fp32", cuml_dtype="fp32",
                    flashlib_algorithm=alg, cuml_algorithm="cuml_tsvd_default",
                    init_shared=False,
                    notes=("matched dtype + algorithm" if tol is None
                           else "matched dtype, Halko vs cuML default — "
                                "algorithmic step-down disclosed via tol=1e-3")),
            }, columns=["shape", "dtype", "engine", "time_ms", "rel_err_svK",
                        "vs_cuml", "HBM_GB", "gate"])
        except Exception as e:
            audit_record(PRIM, {
                "shape": label, "dtype": dlabel, "engine": "flashlib",
                "time_ms": "ERR", "rel_err_svK": "-",
                "vs_cuml": "-", "HBM_GB": "-",
                "gate": f"FAIL ({type(e).__name__}: {str(e)[:80]})",
                "conditions": {},
            }, columns=["shape", "dtype", "engine", "time_ms", "rel_err_svK",
                        "vs_cuml", "HBM_GB", "gate"])

    del X32, X_cp
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
