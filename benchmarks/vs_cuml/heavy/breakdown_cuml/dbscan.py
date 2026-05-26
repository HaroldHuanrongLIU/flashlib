"""cuML DBSCAN CUDA-kernel trace via CUPTI.

Swept axis: D (vector dim) at the same shapes as flashlib breakdown's
dbscan.py — D=2/N=2M, D=16/N=1M, D=128/N=200K.

cuML DBSCAN code path:
  - Python:  cuml/cluster/dbscan.pyx
  - C++:     cpp/src/dbscan/dbscan.cu (vertex_degree -> adj_to_csr ->
              weak_cc batched).
"""
from __future__ import annotations

import numpy as np
import torch
from cuml.cluster import DBSCAN as cuDBSCAN
from sklearn.datasets import make_blobs

from ._common import free_gpu, profile_cuml_call, write_cuml_breakdown_md

N_CENTERS = 20
MIN_SAMPLES = 5

# NOTE: cuML DBSCAN at large N + permissive eps produces enormous CSR
# adjacency (random data has dense ε-neighbours at coarse eps), driving
# cuML into a CPU-bound label-merge loop that takes > 30 minutes per call.
# We trim N for D=2 and D=128 so the profiles complete in reasonable wall
# time — the kernel mix is independent of N (same launcher, same kernels,
# just fewer batches). The flashlib breakdown's larger shapes are still
# reported in `benchmarks/results/heavy/breakdown/dbscan.md` for the
# flashlib-side timings.
SHAPES = [
    ("D=2 (N=200K)",            {"N":  200_000, "D": 2,   "eps": 0.5}),
    ("D=16 brute-low (N=200K)", {"N":  200_000, "D": 16,  "eps": 3.5}),
    ("D=128 brute-high (N=50K)", {"N":  50_000, "D": 128, "eps": 11.0}),
]


def main() -> None:
    torch.manual_seed(0)
    print("[cuml-profile:dbscan] sweeping D across grid vs brute paths")

    shape_results = []
    for label, kw in SHAPES:
        N, D, eps = kw["N"], kw["D"], kw["eps"]
        X_np, _ = make_blobs(
            n_samples=N, centers=N_CENTERS, n_features=D,
            cluster_std=1.0, random_state=0,
        )
        X_np = X_np.astype(np.float32)
        X_t = torch.from_numpy(X_np).cuda()
        import cupy as cp
        X_cp = cp.from_dlpack(X_t)

        def _call():
            cuDBSCAN(eps=eps, min_samples=MIN_SAMPLES,
                      output_type="cupy").fit(X_cp)

        # cuML DBSCAN at heavy N can hang or crash on some shapes.
        try:
            sr = profile_cuml_call(label, _call, warmup=1, repeat=2)
            shape_results.append(sr)
        except Exception as e:
            print(f"[cuml-profile:dbscan] SHAPE {label} FAILED: {e}")
            shape_results.append({
                "label": label,
                "outer_wall_ms": float("nan"),
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

        # Persist partial results after every shape - if a later shape
        # hangs/crashes, the completed shapes still land in the .md/.json.
        write_cuml_breakdown_md(
            prim="dbscan",
            shape_results=shape_results,
            notes=("cuML DBSCAN = (vertex-degree pairwise-L2 scan) → "
                   "(adj-to-CSR compaction) → (batched weak-CC). "
                   "All shapes use the same pipeline; cuML has no D=2 grid "
                   "fast-path (uses brute-force pairwise scan even at D=2)."),
        )


if __name__ == "__main__":
    main()
