"""cuML PCA CUDA-kernel trace via CUPTI.

Sweep across the same three aspect ratios as flashlib breakdown's pca.py:
tall (N=10M, D=256, K=64), square (N=2M, D=2K, K=128), wide (N=10K,
D=8K, K=32, tol=0.01).

cuML PCA paths:
  - Python:  cuml/decomposition/pca.pyx (algorithm='full' / 'jacobi')
  - C++:     cpp/src/pca/pca.cu   (cov + cusolver eigh)
              cpp/src/tsvd/tsvd.cu (when components < D and Jacobi is used)
The wide (D > N) case routes to algorithm='jacobi' implicitly when
n_components < min(N, D); otherwise it picks 'full' (cusolver dnGesvdj).
"""
from __future__ import annotations

import torch
from cuml.decomposition import PCA as cuPCA

from ._common import free_gpu, profile_cuml_call, write_cuml_breakdown_md

SHAPES = [
    ("tall N=10M D=256 K=64",  {"N": 10_000_000, "D":   256, "K":  64,
                                  "solver": "full"}),
    ("square N=2M D=2K K=128", {"N":  2_000_000, "D": 2_000, "K": 128,
                                  "solver": "full"}),
    ("wide N=10K D=8K K=32",   {"N":     10_000, "D": 8_000, "K":  32,
                                  "solver": "jacobi"}),
]


def main() -> None:
    torch.manual_seed(0)
    print("[cuml-profile:pca] sweeping aspect ratios (tall / square / wide)")

    shape_results = []
    for label, kw in SHAPES:
        N, D, K, solver = kw["N"], kw["D"], kw["K"], kw["solver"]
        X_t = torch.randn(N, D, dtype=torch.float32, device="cuda")
        import cupy as cp
        X_cp = cp.from_dlpack(X_t)

        def _call():
            cuPCA(n_components=K, svd_solver=solver,
                   output_type="cupy").fit_transform(X_cp)

        try:
            sr = profile_cuml_call(label, _call, warmup=1, repeat=2)
            shape_results.append(sr)
        except Exception as e:
            print(f"[cuml-profile:pca] SHAPE {label} FAILED: {e}")
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
        prim="pca",
        shape_results=shape_results,
        notes=("cuML PCA: tall/square use algorithm='full' (cov GEMM + "
               "cusolver gesvdj on D×D); wide uses algorithm='jacobi' "
               "which routes through cusolverDn Jacobi eigendecomp on "
               "the covariance D×D after a TF32 cov GEMM."),
    )


if __name__ == "__main__":
    main()
