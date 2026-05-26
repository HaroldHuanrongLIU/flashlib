"""cuML UMAP CUDA-kernel trace via CUPTI.

Matches flashlib breakdown's umap.py shapes:
  (N=50K, D=64, NN=15, n_epochs=100),
  (N=100K, D=256, NN=15, n_epochs=200),
  (N=200K, D=256, NN=20, n_epochs=500).

cuML UMAP:
  - Python: cuml/manifold/umap.pyx
  - C++:    cpp/src/umap/* — knn_graph (brute or rng) +
              fuzzy_simplicial_set + optimize_layout (SGD).
"""
from __future__ import annotations

import numpy as np
import torch
from cuml.manifold import UMAP as cuUMAP
from sklearn.datasets import make_blobs

from ._common import free_gpu, profile_cuml_call, write_cuml_breakdown_md

SHAPES = [
    ("D=64 N=50K NN=15 ep=100",
     {"N":  50_000, "D":  64, "NN": 15, "n_epochs": 100}),
    ("D=256 N=100K NN=15 ep=200",
     {"N": 100_000, "D": 256, "NN": 15, "n_epochs": 200}),
    ("D=256 N=200K NN=20 ep=500",
     {"N": 200_000, "D": 256, "NN": 20, "n_epochs": 500}),
]


def main() -> None:
    torch.manual_seed(0)
    print("[cuml-profile:umap] sweeping (N, D, n_epochs)")

    shape_results = []
    for label, kw in SHAPES:
        N, D, NN, n_epochs = kw["N"], kw["D"], kw["NN"], kw["n_epochs"]
        X_np, _ = make_blobs(n_samples=N, centers=10, n_features=D,
                              cluster_std=2.0, random_state=0)
        X_np = X_np.astype(np.float32)
        X_t = torch.from_numpy(X_np).cuda()
        import cupy as cp
        X_cp = cp.from_dlpack(X_t)

        def _call():
            cuUMAP(n_neighbors=NN, n_epochs=n_epochs,
                    n_components=2, random_state=0,
                    output_type="cupy").fit_transform(X_cp)

        try:
            sr = profile_cuml_call(label, _call, warmup=1, repeat=1)
            shape_results.append(sr)
        except Exception as e:
            print(f"[cuml-profile:umap] SHAPE {label} FAILED: {e}")
            shape_results.append({
                "label": label, "outer_wall_ms": float("nan"),
                "n_kernels": 0,
                "kernels": [{"kernel": f"FAILED: {type(e).__name__}",
                              "raw_example": "",
                              "launches_per_call": 0,
                              "total_ms_per_call": 0.0,
                              "mean_us_per_launch": 0.0,
                              "pct_of_total": 0.0}],
            })
        del X_t, X_cp
        free_gpu()

    write_cuml_breakdown_md(
        prim="umap",
        shape_results=shape_results,
        notes=("cuML UMAP pipeline: KNN graph build (brute_force at "
               "small N, RNG at large N) + smooth_knn + fuzzy "
               "simplicial set + optimize_layout (SGD). The SGD layout "
               "kernel accumulates n_epochs launches and is the "
               "dominant cost at larger n_epochs."),
    )


if __name__ == "__main__":
    main()
