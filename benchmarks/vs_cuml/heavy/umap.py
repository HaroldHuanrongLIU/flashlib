"""Heavy UMAP sweep — release-candidate audit.

flashlib pipeline:
  1. KNN graph via :func:`flash_knn` (so ``tol`` propagates: bf16 KNN).
  2. Fuzzy simplicial set construction on GPU.
  3. Deterministic-negative SGD epochs in Triton.

Anti-reward-hacking guardrails:

* Same blob inputs across engines; same ``n_neighbors``, ``n_epochs``,
  ``random_state``.
* Trustworthiness at k=12 (subsampled to 5K for large N).
* Both fp32 exact AND bf16 ``tol=1e-3`` rows reported so the precision
  step-down vs cuML fp32 is visible.
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

from sklearn.datasets import make_blobs
from sklearn.manifold import trustworthiness
from cuml.manifold import UMAP as cuUMAP
from flashlib.primitives.umap import flash_umap


# (label, N, D, K, n_neighbors, n_epochs)
SHAPES = [
    ("medium  N=100K  D=128 K=10 NN=15", 100_000, 128, 10, 15, 200),
    ("large   N=200K  D=64  K=10 NN=15", 200_000,  64, 10, 15, 200),
    ("xlarge  N=500K  D=32  K=10 NN=15", 500_000,  32, 10, 15, 200),
    ("D=256   N=100K  D=256 K=10 NN=15", 100_000, 256, 10, 15, 200),
]

PRIM = "umap"


def _run_one(label, N, D, K, n_neighbors, n_epochs):
    title(f"UMAP  {label}  (N={N:,}, D={D}, K={K}, "
          f"NN={n_neighbors}, n_epochs={n_epochs})")

    X_np, _ = make_blobs(n_samples=N, centers=K, n_features=D,
                          cluster_std=2.0, random_state=0)
    X_np = X_np.astype(np.float32)

    # Subsample for trustworthiness (otherwise it's O(N^2) for the
    # ground-truth high-D neighbour list).
    n_tw = min(N, 5_000)
    rng = np.random.RandomState(0)
    tw_idx = rng.choice(N, size=n_tw, replace=False)

    # cuML
    free_gpu(); hbm_peak_reset()
    try:
        cu_emb = np.asarray(
            cuUMAP(n_components=2, n_neighbors=n_neighbors,
                    n_epochs=n_epochs, random_state=0).fit_transform(X_np)
        )
        cu_tw = trustworthiness(X_np[tw_idx], cu_emb[tw_idx], n_neighbors=12)
        t_cu = time_gpu(
            lambda: cuUMAP(n_components=2, n_neighbors=n_neighbors,
                            n_epochs=n_epochs, random_state=0).fit_transform(X_np),
            repeat=1, warmup=1,
        )
        hbm_cu = hbm_peak_gb()
        audit_record(PRIM, {
            "shape": label, "dtype": "fp32", "engine": "cuml",
            "time_ms": f"{t_cu:10.1f}", "trustworthiness": f"{cu_tw:.4f}",
            "vs_cuml": "1.00x", "HBM_GB": f"{hbm_cu:.1f}",
            "gate": gate_metric("tw", cu_tw, lower=0.70),
            "conditions": apples_to_apples(
                op="umap", shape={"N": N, "D": D, "K": K,
                                   "NN": n_neighbors, "epochs": n_epochs},
                flashlib_dtype="-", cuml_dtype="fp32",
                flashlib_algorithm="-", cuml_algorithm="cuml_umap_default",
                init_shared=False, notes="cuML default; same n_neighbors/n_epochs/seed"),
        }, columns=["shape", "dtype", "engine", "time_ms", "trustworthiness",
                    "vs_cuml", "HBM_GB", "gate"])
    except Exception as e:
        t_cu = float("inf")
        audit_record(PRIM, {
            "shape": label, "dtype": "fp32", "engine": "cuml",
            "time_ms": "ERR", "trustworthiness": "-",
            "vs_cuml": "-", "HBM_GB": "-",
            "gate": f"FAIL ({type(e).__name__}: {str(e)[:80]})",
            "conditions": {},
        }, columns=["shape", "dtype", "engine", "time_ms", "trustworthiness",
                    "vs_cuml", "HBM_GB", "gate"])

    # flashlib fp32 exact + bf16 tol.
    X32 = torch.tensor(X_np, device="cuda")
    variants = [
        ("fp32 exact", None, "exact in input dtype"),
        ("bf16 tol",   1e-3, "bf16-cached KNN distances"),
    ]
    for dlabel, tol, alg_note in variants:
        free_gpu(); hbm_peak_reset()
        try:
            fl_emb_t = flash_umap(
                X32, n_neighbors=n_neighbors, n_components=2,
                n_epochs=n_epochs, tol=tol, seed=0,
            )
            fl_emb = fl_emb_t.float().cpu().numpy()
            fl_tw = trustworthiness(X_np[tw_idx], fl_emb[tw_idx], n_neighbors=12)
            t_fl = time_gpu(
                lambda: flash_umap(
                    X32, n_neighbors=n_neighbors, n_components=2,
                    n_epochs=n_epochs, tol=tol, seed=0,
                ),
                repeat=1, warmup=1,
            )
            hbm_fl = hbm_peak_gb()
            audit_record(PRIM, {
                "shape": label, "dtype": dlabel, "engine": "flashlib",
                "time_ms": f"{t_fl:10.1f}", "trustworthiness": f"{fl_tw:.4f}",
                "vs_cuml": (f"{t_cu / t_fl:.2f}x" if t_cu != float("inf") else "n/a"),
                "HBM_GB": f"{hbm_fl:.1f}",
                "gate": gate_metric("tw", fl_tw, lower=0.70),
                "conditions": apples_to_apples(
                    op="umap", shape={"N": N, "D": D, "K": K,
                                       "NN": n_neighbors, "epochs": n_epochs},
                    flashlib_dtype=("fp32" if tol is None else "bf16(KNN)+fp32(SGD)"),
                    cuml_dtype="fp32",
                    flashlib_algorithm=("flash_knn_+_fuzzy_set_+_det_negative_sgd"),
                    cuml_algorithm="cuml_umap_default",
                    init_shared=False,
                    notes=alg_note),
            }, columns=["shape", "dtype", "engine", "time_ms",
                        "trustworthiness", "vs_cuml", "HBM_GB", "gate"])
        except Exception as e:
            audit_record(PRIM, {
                "shape": label, "dtype": dlabel, "engine": "flashlib",
                "time_ms": "ERR", "trustworthiness": "-",
                "vs_cuml": "-", "HBM_GB": "-",
                "gate": f"FAIL ({type(e).__name__}: {str(e)[:80]})",
                "conditions": {},
            }, columns=["shape", "dtype", "engine", "time_ms",
                        "trustworthiness", "vs_cuml", "HBM_GB", "gate"])

    del X32, X_np
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
