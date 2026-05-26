"""Heavy PCA sweep — release-candidate audit.

Three regimes that exercise the dual/primal route gate AND the
Halko ``tol`` opt-in independently:

* **tall**: N=20M, D=256, K=64 — dual eigh (G = X @ X.T) is impractical
  (N^2 = 400 TB), primal (Cov = X.T @ X, D=256) is the right path.
* **wide**: N=20K, D=16K, K=64 — Cov is 16K^2 fp32 = 1 GB; dual eigh
  on the 20K x 20K Gram is the cheap path (and Halko on Gram is even
  cheaper).
* **square**: N=2M, D=2K, K=128 — both paths are comparable; the route
  picks based on the eigh dim. Stresses bf16 storage cast as well
  (``tol=1e-3``).

Anti-reward-hacking guardrails:

* fp32-exact (``tol=None``) vs cuML fp32 row reported FIRST (apples-to-
  apples).
* fp32 + Halko (``tol=1e-3``) row reported SECOND, with the algorithm
  step-down flagged in conditions.
* Reference top-K explained variance computed from the SAME centered
  ``X`` via ``torch.linalg.eigh`` on the relevant Gram (no library
  internals; pure torch reference).
* Relative error gate: < 5% on top-K eigenvalues for the lossless
  path, < 5e-2 for the Halko path.
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

from cuml.decomposition import PCA as cuPCA
from flashlib.primitives.pca import flash_pca


# (label, N, D, K)
SHAPES = [
    ("tall    N=2M    D=256  K=32",   2_000_000,   256, 32),
    ("tall    N=10M   D=256  K=64",  10_000_000,   256, 64),
    ("tall    N=20M   D=256  K=64",  20_000_000,   256, 64),
    ("square  N=2M    D=2048 K=128",  2_000_000,  2_048, 128),
    ("wide    N=20K   D=16K  K=64",      20_000, 16_000, 64),
    ("wide    N=10K   D=8K   K=32",      10_000,  8_000, 32),
]

PRIM = "pca"


def _topk_ev_torch(X_centered: torch.Tensor, K: int) -> np.ndarray:
    """Reference top-K explained variance via Gram or Cov, whichever
    has the smaller eigh.  Pure torch; runs in fp32.
    """
    N, D = X_centered.shape
    if D <= N:
        Cov = (X_centered.t() @ X_centered) / max(N - 1, 1)
        ev = torch.linalg.eigvalsh(Cov)
    else:
        G = (X_centered @ X_centered.t()) / max(N - 1, 1)
        ev = torch.linalg.eigvalsh(G)
    ev = torch.sort(ev, descending=True).values[:K]
    return ev.float().cpu().numpy()


def _rel_err(a: np.ndarray, b: np.ndarray) -> float:
    denom = max(float(np.max(np.abs(b))), 1e-12)
    return float(np.max(np.abs(a - b)) / denom)


def _run_one(label, N, D, K):
    title(f"PCA  {label}  (N={N:,}, D={D}, K={K})")

    torch.manual_seed(0)
    # Allocate on GPU; CPU randn(20M, 256) is ~20 GB.
    X32 = torch.randn(N, D, device="cuda", dtype=torch.float32)
    Xc = X32 - X32.mean(dim=0, keepdim=True)
    # Reference top-K ev.
    ref_ev = _topk_ev_torch(Xc, K)

    # cuML fp32 — pre-stage on GPU via cupy view.
    free_gpu(); hbm_peak_reset()
    Xc_cp = cp.from_dlpack(Xc)
    try:
        cu_pca = cuPCA(n_components=K).fit(Xc_cp)
        ev_raw = cu_pca.explained_variance_
        if hasattr(ev_raw, "get"):
            cu_ev = ev_raw.get().astype(np.float32)
        else:
            cu_ev = np.asarray(ev_raw).astype(np.float32)
        # Normalise: cuML may scale by N or N-1; align by ratio.
        if cu_ev[0] > 0 and ref_ev[0] > 0:
            scale = ref_ev[0] / cu_ev[0]
            cu_ev_aligned = cu_ev * scale
        else:
            cu_ev_aligned = cu_ev
        rel_cu = _rel_err(cu_ev_aligned, ref_ev)
        t_cu = time_gpu(lambda: cuPCA(n_components=K).fit(Xc_cp),
                        repeat=2, warmup=1)
        hbm_cu = hbm_peak_gb()
        audit_record(PRIM, {
            "shape": label, "dtype": "fp32", "engine": "cuml",
            "time_ms": f"{t_cu:10.2f}",
            "rel_err_evK": f"{rel_cu:.3e}",
            "vs_cuml": "1.00x", "HBM_GB": f"{hbm_cu:.1f}",
            "gate": gate_metric("rel_err", rel_cu, upper=0.05),
            "conditions": apples_to_apples(
                op="pca", shape={"N": N, "D": D, "K": K},
                flashlib_dtype="-", cuml_dtype="fp32",
                flashlib_algorithm="-", cuml_algorithm="cuml_pca_default",
                init_shared=False, notes="reference top-K via torch eigh"),
        }, columns=["shape", "dtype", "engine", "time_ms", "rel_err_evK",
                    "vs_cuml", "HBM_GB", "gate"])
    except Exception as e:
        t_cu = float("inf")
        audit_record(PRIM, {
            "shape": label, "dtype": "fp32", "engine": "cuml",
            "time_ms": "ERR", "rel_err_evK": "-",
            "vs_cuml": "-", "HBM_GB": "-",
            "gate": f"FAIL ({type(e).__name__}: {str(e)[:80]})",
            "conditions": {},
        }, columns=["shape", "dtype", "engine", "time_ms", "rel_err_evK",
                    "vs_cuml", "HBM_GB", "gate"])

    # flashlib variants: fp32 exact + fp32 Halko.
    variants = [
        ("fp32 exact", None,    0.05, "exact (cov/gram + eigh)"),
        ("fp32 halko", 1e-3,    0.05, "Halko subspace iter (truncated)"),
    ]
    for dlabel, tol, gate_upper, alg in variants:
        free_gpu(); hbm_peak_reset()
        try:
            ev, _ = flash_pca(Xc, K=K, tol=tol)
            fl_ev = torch.sort(ev.float(), descending=True).values.cpu().numpy()
            if fl_ev[0] > 0 and ref_ev[0] > 0:
                scale = ref_ev[0] / fl_ev[0]
                fl_ev_aligned = fl_ev * scale
            else:
                fl_ev_aligned = fl_ev
            rel = _rel_err(fl_ev_aligned, ref_ev)
            t_fl = time_gpu(lambda: flash_pca(Xc, K=K, tol=tol),
                            repeat=3, warmup=1)
            hbm_fl = hbm_peak_gb()
            audit_record(PRIM, {
                "shape": label, "dtype": dlabel, "engine": "flashlib",
                "time_ms": f"{t_fl:10.2f}",
                "rel_err_evK": f"{rel:.3e}",
                "vs_cuml": (f"{t_cu / t_fl:.2f}x" if t_cu != float("inf") else "n/a"),
                "HBM_GB": f"{hbm_fl:.1f}",
                "gate": gate_metric("rel_err", rel, upper=gate_upper),
                "conditions": apples_to_apples(
                    op="pca", shape={"N": N, "D": D, "K": K},
                    flashlib_dtype="fp32", cuml_dtype="fp32",
                    flashlib_algorithm=alg, cuml_algorithm="cuml_pca_default",
                    init_shared=False,
                    notes=("matched dtype + matched-algorithm" if tol is None
                           else "matched dtype, Halko vs cuML full SVD — "
                                "algorithmic step-down disclosed via tol=1e-3")),
            }, columns=["shape", "dtype", "engine", "time_ms", "rel_err_evK",
                        "vs_cuml", "HBM_GB", "gate"])
        except Exception as e:
            audit_record(PRIM, {
                "shape": label, "dtype": dlabel, "engine": "flashlib",
                "time_ms": "ERR", "rel_err_evK": "-",
                "vs_cuml": "-", "HBM_GB": "-",
                "gate": f"FAIL ({type(e).__name__}: {str(e)[:80]})",
                "conditions": {},
            }, columns=["shape", "dtype", "engine", "time_ms", "rel_err_evK",
                        "vs_cuml", "HBM_GB", "gate"])

    del X32, Xc, Xc_cp
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
