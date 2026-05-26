"""cuML KMeans CUDA-kernel trace via CUPTI.

Swept axis: K (n_clusters) at fixed N=20M, D=64, max_iter=5 — same axis as
the flashlib kmeans breakdown so the two tables are directly comparable.

cuML's KMeans is in:
  - Python:  cuml/cluster/kmeans.pyx
  - C++:     cpp/src/kmeans/kmeans.cu (RAFT-backed)
              cuvs/cpp/src/cluster/kmeans_balanced.cuh    (assign + update kernels)
"""
from __future__ import annotations

import numpy as np
import torch
from cuml.cluster import KMeans as cuKMeans

from ._common import (
    free_gpu, profile_cuml_call, write_cuml_breakdown_md,
)

N, D, MAX_ITER = 20_000_000, 64, 5

SHAPES = [
    ("K=1K",   1_000),
    ("K=10K",  10_000),
    ("K=100K", 100_000),
]


def main() -> None:
    torch.manual_seed(0)
    print(f"[cuml-profile:kmeans] sweeping K at N={N:,}, D={D}, max_iter={MAX_ITER}")

    # Generate once on GPU then pass via cupy zero-copy (matches the
    # apples-to-apples policy used in benchmarks/vs_cuml/heavy/kmeans.py).
    X_t = torch.randn(N, D, dtype=torch.float32, device="cuda")
    import cupy as cp
    X_cp = cp.from_dlpack(X_t)

    shape_results = []
    for label, K in SHAPES:
        # Pre-pick init centroids on GPU for both runs (so init lottery
        # doesn't move kernel timing).
        rng = np.random.RandomState(0)
        init_idx = rng.choice(N, size=K, replace=False)
        init_centroids_cp = X_cp[init_idx]

        def _call():
            cuKMeans(
                n_clusters=K, init=init_centroids_cp, n_init=1,
                max_iter=MAX_ITER, tol=1e-12,  # cuML rejects tol<=0
                random_state=0, verbose=0,
                output_type="cupy",
            ).fit(X_cp)

        sr = profile_cuml_call(label, _call, warmup=1, repeat=3)
        shape_results.append(sr)
        free_gpu()

    write_cuml_breakdown_md(
        prim="kmeans",
        shape_results=shape_results,
        notes=("cuML KMeans = RAFT Lloyd loop: per-iter (assign + update + "
               "shift). Init centroids forced (init=init_centroids_cp) so "
               "the kernel mix isn't perturbed by cuML's k-means++ default."),
    )


if __name__ == "__main__":
    main()
