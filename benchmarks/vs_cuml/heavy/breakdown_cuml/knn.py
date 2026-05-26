"""cuML NearestNeighbors CUDA-kernel trace via CUPTI.

Swept axis: D (vector dim) at Q=M=100K, K=10, fp32, brute-force exact —
matches benchmarks/vs_cuml/heavy/breakdown/knn.py.

cuML's NearestNeighbors (algorithm='brute') is in:
  - Python:  cuml/neighbors/nearest_neighbors.pyx
  - C++:     cuvs/cpp/src/neighbors/brute_force.cu
              cuvs/cpp/src/neighbors/detail/knn_brute_force.cuh
"""
from __future__ import annotations

import torch
from cuml.neighbors import NearestNeighbors as cuNN

from ._common import free_gpu, profile_cuml_call, write_cuml_breakdown_md

M_FIXED, Q_FIXED, K_FIXED = 100_000, 100_000, 10

SHAPES = [
    ("D=8",   8),
    ("D=128", 128),
    ("D=512", 512),
]


def main() -> None:
    torch.manual_seed(0)
    print(f"[cuml-profile:knn] sweeping D at Q=M={Q_FIXED:,}, K={K_FIXED}, fp32")

    shape_results = []
    for label, D in SHAPES:
        X_t = torch.randn(M_FIXED, D, dtype=torch.float32, device="cuda")
        import cupy as cp
        X_cp = cp.from_dlpack(X_t)

        nn = cuNN(n_neighbors=K_FIXED, algorithm="brute", output_type="cupy")
        nn.fit(X_cp)

        def _call():
            nn.kneighbors(X_cp)

        sr = profile_cuml_call(label, _call, warmup=1, repeat=3)
        shape_results.append(sr)
        del nn, X_cp, X_t
        free_gpu()

    write_cuml_breakdown_md(
        prim="knn",
        shape_results=shape_results,
        notes=("cuML NearestNeighbors algorithm='brute' (self-kNN). "
               "cuML routes to cuVS brute-force which uses tiled CUTLASS "
               "L2 distance + cub::warp/radix select_k for the top-K."),
    )


if __name__ == "__main__":
    main()
