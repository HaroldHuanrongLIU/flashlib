"""cuML HDBSCAN CUDA-kernel trace via CUPTI.

Swept axis: N at fixed D=16, min_samples=5 — matches flashlib breakdown's
hdbscan.py.

cuML HDBSCAN pipeline:
  - Python:  cuml/cluster/hdbscan.pyx
  - C++:     cpp/src/hdbscan/* (mostly RAFT primitives):
              - core_distances:    raft brute-force kNN
              - mutual_reachability: pairwise + max-broadcast kernel
              - mst (Boruvka):     raft MST on dense mutual-reach
              - SLT label + condense_tree + stability: host-side numba/CPU
"""
from __future__ import annotations

import numpy as np
import torch
from cuml.cluster import HDBSCAN as cuHDBSCAN
from sklearn.datasets import make_blobs

from ._common import free_gpu, profile_cuml_call, write_cuml_breakdown_md

D_FIXED = 16
MIN_SAMPLES = 5
N_CENTERS = 6

SHAPES = [
    ("N=10K", {"N": 10_000, "mcs": 20}),
    ("N=20K", {"N": 20_000, "mcs": 20}),
    ("N=50K", {"N": 50_000, "mcs": 50}),
]


def main() -> None:
    torch.manual_seed(0)
    print(f"[cuml-profile:hdbscan] sweeping N at D={D_FIXED}, "
          f"min_samples={MIN_SAMPLES}")

    shape_results = []
    for label, kw in SHAPES:
        N, mcs = kw["N"], kw["mcs"]
        X_np, _ = make_blobs(
            n_samples=N, centers=N_CENTERS, n_features=D_FIXED,
            cluster_std=1.0, random_state=0,
        )
        X_np = X_np.astype(np.float32)
        X_t = torch.from_numpy(X_np).cuda()
        import cupy as cp
        X_cp = cp.from_dlpack(X_t)

        def _call():
            cuHDBSCAN(
                min_cluster_size=mcs, min_samples=MIN_SAMPLES,
                metric="euclidean", output_type="cupy",
            ).fit(X_cp)

        try:
            sr = profile_cuml_call(label, _call, warmup=1, repeat=2)
            shape_results.append(sr)
        except Exception as e:
            print(f"[cuml-profile:hdbscan] SHAPE {label} FAILED: {e}")
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

    write_cuml_breakdown_md(
        prim="hdbscan",
        shape_results=shape_results,
        notes=("cuML HDBSCAN dense path: core-distances (RAFT brute-kNN) "
               "+ mutual-reachability matrix (pairwise + max-broadcast) "
               "+ Boruvka MST + host-side dendrogram (numba/CPU; not "
               "captured by CUPTI). The GPU kernels timed here are the "
               "MRD matrix construction and the dense MST argmin / "
               "union-find iterations."),
    )


if __name__ == "__main__":
    main()
