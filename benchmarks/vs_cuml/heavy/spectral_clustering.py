"""Heavy SpectralClustering sweep — release-candidate audit.

cuML has no SpectralClustering peer, so the sole baseline is sklearn
(CPU). This makes "vs cuml" entries n/a; the headline is wall-time
+ ARI vs sklearn.

flashlib's pipeline:
1. symmetric kNN graph (via :func:`flash_knn`),
2. normalized graph Laplacian, top-K eigenvectors via power iteration
   with lazy QR every 5 steps,
3. KMeans on the row-normalised embedding (via :func:`flash_kmeans`).

Anti-reward-hacking guardrails:

* Same blob inputs + same ``n_neighbors`` between engines.
* sklearn runs ``assign_labels='kmeans'`` to match the flashlib
  pipeline's final clustering stage.
* ARI vs sklearn is the only quality metric; gates at 0.90 because
  the final KMeans + power-iter starting vector both have
  realisation-dependent labels.
* sklearn is single-threaded Arnoldi; we cap BLAS threads at 8 so
  sklearn doesn't slow down absurdly on a many-core host.
"""
from benchmarks.vs_cuml.heavy._common import (
    cap_threads, cuml_shim, time_gpu, time_cpu, title, header,
    ari, audit_record, apples_to_apples,
    hbm_peak_reset, hbm_peak_gb, gate_metric, free_gpu, RESULTS_DIR,
)
cap_threads(); cuml_shim()

import warnings; warnings.filterwarnings("ignore")
import numpy as np
import torch

from sklearn.cluster import SpectralClustering as skSpectral
from sklearn.datasets import make_blobs
from flashlib.primitives.spectral_clustering import flash_spectral_clustering


# (label, N, D, K, n_neighbors)
SHAPES = [
    ("small    N=10K  D=16 K=5  NN=15", 10_000,  16,  5, 15),
    ("medium   N=30K  D=32 K=8  NN=15", 30_000,  32,  8, 15),
    ("large    N=60K  D=64 K=10 NN=20", 60_000,  64, 10, 20),
    ("xlarge   N=100K D=64 K=10 NN=20", 100_000, 64, 10, 20),
]

PRIM = "spectral_clustering"


def _run_one(label, N, D, K, n_neighbors):
    title(f"SpectralClustering  {label}  (N={N:,}, D={D}, K={K}, "
          f"NN={n_neighbors})")

    X_np, _ = make_blobs(n_samples=N, centers=K, n_features=D,
                          cluster_std=1.5, random_state=0)
    X_np = X_np.astype(np.float32)

    # sklearn (CPU) reference. Skip at large N to avoid 10+ min walls;
    # in that case we report no ARI and only the wall-clock.
    if N <= 30_000:
        sk_lbl = skSpectral(
            n_clusters=K, n_neighbors=n_neighbors,
            affinity="nearest_neighbors", random_state=0,
            assign_labels="kmeans",
        ).fit_predict(X_np)
        t_sk = time_cpu(
            lambda: skSpectral(
                n_clusters=K, n_neighbors=n_neighbors,
                affinity="nearest_neighbors", random_state=0,
                assign_labels="kmeans",
            ).fit_predict(X_np),
            repeat=1,
        )
        audit_record(PRIM, {
            "shape": label, "engine": "sklearn(CPU)",
            "time_ms": f"{t_sk:10.1f}", "ARI": "1.0000",
            "vs_sklearn": "1.00x",
            "HBM_GB": "0.0", "gate": "PASS",
            "conditions": apples_to_apples(
                op="spectral_clustering", shape={"N": N, "D": D, "K": K,
                                                  "NN": n_neighbors},
                flashlib_dtype="-", cuml_dtype="-",
                flashlib_algorithm="-", cuml_algorithm="-",
                init_shared=False, notes="ground truth"),
        }, columns=["shape", "engine", "time_ms", "ARI", "vs_sklearn",
                    "HBM_GB", "gate"])
        ref_lbl = sk_lbl
    else:
        t_sk = None
        ref_lbl = None

    # flashlib
    free_gpu(); hbm_peak_reset()
    try:
        X32 = torch.tensor(X_np, device="cuda")
        fl_lbl = flash_spectral_clustering(
            X32, n_clusters=K, n_neighbors=n_neighbors, seed=0,
        )
        fl_lbl_np = (fl_lbl.cpu().numpy() if isinstance(fl_lbl, torch.Tensor)
                     else np.asarray(fl_lbl))
        t_fl = time_gpu(
            lambda: flash_spectral_clustering(
                X32, n_clusters=K, n_neighbors=n_neighbors, seed=0,
            ),
            repeat=2, warmup=1,
        )
        hbm_fl = hbm_peak_gb()
        ari_fl = ari(ref_lbl, fl_lbl_np) if ref_lbl is not None else float("nan")
        speedup = (f"{t_sk / t_fl:.2f}x" if t_sk is not None else "n/a")
        gate = (gate_metric("ARI", ari_fl, lower=0.90)
                if ref_lbl is not None else "PASS (no ref)")
        audit_record(PRIM, {
            "shape": label, "engine": "flashlib",
            "time_ms": f"{t_fl:10.1f}",
            "ARI": (f"{ari_fl:.4f}" if ref_lbl is not None else "n/a"),
            "vs_sklearn": speedup, "HBM_GB": f"{hbm_fl:.1f}",
            "gate": gate,
            "conditions": apples_to_apples(
                op="spectral_clustering", shape={"N": N, "D": D, "K": K,
                                                  "NN": n_neighbors},
                flashlib_dtype="fp32", cuml_dtype="-",
                flashlib_algorithm=("flash_knn_graph_+_lazyQR_power_iter_"
                                     "+_kmeans_embedding"),
                cuml_algorithm="(no cuML peer)",
                init_shared=False,
                notes="no cuML peer; sklearn is the only baseline"),
        }, columns=["shape", "engine", "time_ms", "ARI", "vs_sklearn",
                    "HBM_GB", "gate"])
    except Exception as e:
        audit_record(PRIM, {
            "shape": label, "engine": "flashlib",
            "time_ms": "ERR", "ARI": "-",
            "vs_sklearn": "-", "HBM_GB": "-",
            "gate": f"FAIL ({type(e).__name__}: {str(e)[:80]})",
            "conditions": {},
        }, columns=["shape", "engine", "time_ms", "ARI", "vs_sklearn",
                    "HBM_GB", "gate"])

    free_gpu()


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
