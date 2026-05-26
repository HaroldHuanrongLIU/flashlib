"""cuML TSNE CUDA-kernel trace via CUPTI.

Matches flashlib breakdown's tsne.py shapes (N ∈ {10K, 15K, 20K}).
cuML defaults to Barnes-Hut (method='barnes_hut'); we pin n_iter=500
to match the flashlib breakdown's SGD count.

cuML TSNE Barnes-Hut path:
  - Python: cuml/manifold/t_sne.pyx
  - C++:    cpp/src/tsne/* — bh_kernels.cuh (bounding box, tree build,
            summarization, repulsive forces) + attractive forces +
            apply_forces. Big sequence of small kernels per iter.
"""
from __future__ import annotations

import numpy as np
import torch
from cuml.manifold import TSNE as cuTSNE
from sklearn.datasets import make_blobs

from ._common import free_gpu, profile_cuml_call, write_cuml_breakdown_md

N_ITER = 500
PERPLEXITY = 30.0
K_BLOBS = 10

SHAPES = [
    ("small N=10K n_iter=500",  {"N": 10_000, "D":  64}),
    ("medium N=15K n_iter=500", {"N": 15_000, "D": 128}),
    ("large N=20K n_iter=500",  {"N": 20_000, "D": 128}),
]


def main() -> None:
    torch.manual_seed(0)
    print(f"[cuml-profile:tsne] sweeping N at n_iter={N_ITER}, "
          f"perplexity={PERPLEXITY}")

    shape_results = []
    for label, kw in SHAPES:
        N, D = kw["N"], kw["D"]
        X_np, _ = make_blobs(n_samples=N, centers=K_BLOBS, n_features=D,
                              cluster_std=2.0, random_state=0)
        X_np = X_np.astype(np.float32)
        X_t = torch.from_numpy(X_np).cuda()
        import cupy as cp
        X_cp = cp.from_dlpack(X_t)

        def _call():
            cuTSNE(n_components=2, perplexity=PERPLEXITY,
                    n_iter=N_ITER, method="barnes_hut",
                    learning_rate=200.0, random_state=0,
                    output_type="cupy").fit_transform(X_cp)

        try:
            sr = profile_cuml_call(label, _call, warmup=1, repeat=1)
            shape_results.append(sr)
        except Exception as e:
            print(f"[cuml-profile:tsne] SHAPE {label} FAILED: {e}")
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
        prim="tsne",
        shape_results=shape_results,
        notes=("cuML TSNE method='barnes_hut' — runs many small kernels "
               "per SGD iter: bounding_box, tree_building, summarization, "
               "compute_repulsive_forces, compute_attractive_forces, "
               "apply_forces. Comparison vs flashlib's EXACT O(N²) path "
               "is **method-mismatched** (BH is O(N log N) approximate, "
               "flashlib exact is O(N²)); the headline metric here is "
               "per-kernel kernel-launch overhead and per-iter wall."),
    )


if __name__ == "__main__":
    main()
