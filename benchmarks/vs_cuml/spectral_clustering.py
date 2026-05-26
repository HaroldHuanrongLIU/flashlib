"""SpectralClustering: ``flash_spectral_clustering`` vs sklearn.

cuML does not ship a SpectralClustering at parity with sklearn's
public API, so the GPU reference here is flashlib itself, with
sklearn (CPU) as the correctness baseline (ARI).

flashlib's pipeline:
  * symmetric KNN graph (via :func:`flashlib.primitives.knn.flash_knn`)
  * normalized graph Laplacian, top-K eigenvectors via power iteration
  * KMeans on the row-normalised embedding.

Correctness signal:
* ARI vs sklearn on small synthetic blobs.
"""
from benchmarks.vs_cuml._common import (
    cap_threads, cuml_shim, time_gpu, time_cpu, title, header,
    ari, fmt_table,
)
cap_threads(); cuml_shim()

import warnings; warnings.filterwarnings("ignore")
import numpy as np
import torch

from sklearn.cluster import SpectralClustering as skSpectral
from sklearn.datasets import make_blobs
from flashlib.primitives.spectral_clustering import flash_spectral_clustering


# (label, N, D, n_clusters, n_neighbors, use_sklearn_cpu)
SHAPES = [
    ("small  N=2K   D=8  K=3 NN=10",  2_000,   8, 3, 10, True),
    ("medium N=10K  D=16 K=5 NN=15", 10_000,  16, 5, 15, False),
    ("large  N=50K  D=32 K=8 NN=20", 50_000,  32, 8, 20, False),
]


def run_one(label, N, D, K, n_neighbors, use_sklearn_cpu: bool):
    title(f"SpectralClustering {label}  (N={N:,}, D={D}, K={K}, "
          f"NN={n_neighbors})")

    X_np, _ = make_blobs(n_samples=N, centers=K, n_features=D,
                          cluster_std=1.5, random_state=0)
    X_np = X_np.astype(np.float32)

    rows = []
    ref_lbl = None
    if use_sklearn_cpu:
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
        rows.append(("fp32", "sklearn (CPU)", f"{t_sk:7.2f}",
                     "1.0000", "n/a"))
        ref_lbl = sk_lbl

    X32 = torch.tensor(X_np, device="cuda")
    fl_lbl = flash_spectral_clustering(
        X32, n_clusters=K, n_neighbors=n_neighbors, seed=0,
    )
    fl_lbl_np = fl_lbl.cpu().numpy() if isinstance(fl_lbl, torch.Tensor) else np.asarray(fl_lbl)
    ari_fl = ari(ref_lbl, fl_lbl_np) if ref_lbl is not None else 1.0
    t_fl = time_gpu(
        lambda: flash_spectral_clustering(
            X32, n_clusters=K, n_neighbors=n_neighbors, seed=0,
        ),
        repeat=3, warmup=1,
    )
    speedup = f"{t_sk / t_fl:.2f}x vs sk" if use_sklearn_cpu else "n/a"
    rows.append(("fp32", "flashlib", f"{t_fl:7.2f}",
                 f"{ari_fl:.4f}", speedup))

    print(fmt_table(rows, ["dtype", "engine", "time(ms)",
                            "ARI vs sk", "speedup"]))


def main():
    header()
    for s in SHAPES:
        run_one(*s)
    print()


if __name__ == "__main__":
    main()
