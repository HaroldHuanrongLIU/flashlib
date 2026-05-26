"""cuML TruncatedSVD CUDA-kernel trace via CUPTI.

Matches flashlib breakdown's truncated_svd.py shapes.

cuML TruncatedSVD:
  - Python:  cuml/decomposition/tsvd.pyx
  - C++:     cpp/src/tsvd/tsvd.cu  (algorithm='full' -> cusolver dnGesvdj;
              algorithm='jacobi' -> Jacobi iterative SVD on D×D cov).
"""
from __future__ import annotations

import torch
from cuml.decomposition import TruncatedSVD as cuTSVD

from ._common import free_gpu, profile_cuml_call, write_cuml_breakdown_md

SHAPES = [
    ("tall N=10M D=256 K=128",  {"N": 10_000_000, "D":    256, "K": 128}),
    ("wide N=20K D=16K K=64",   {"N":     20_000, "D": 16_000, "K":  64}),
    ("square N=2M D=2K K=128",  {"N":  2_000_000, "D":  2_000, "K": 128}),
]


def main() -> None:
    torch.manual_seed(0)
    print("[cuml-profile:truncated_svd] sweeping (N, D, K)")

    shape_results = []
    for label, kw in SHAPES:
        N, D, K = kw["N"], kw["D"], kw["K"]
        X_t = torch.randn(N, D, dtype=torch.float32, device="cuda")
        import cupy as cp
        X_cp = cp.from_dlpack(X_t)

        def _call():
            cuTSVD(n_components=K, algorithm="jacobi",
                    n_iter=15, tol=1e-3,
                    output_type="cupy").fit_transform(X_cp)

        try:
            sr = profile_cuml_call(label, _call, warmup=1, repeat=2)
            shape_results.append(sr)
        except Exception as e:
            print(f"[cuml-profile:truncated_svd] SHAPE {label} FAILED: {e}")
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
        prim="truncated_svd",
        shape_results=shape_results,
        notes=("cuML TruncatedSVD algorithm='jacobi' — applies cusolver "
               "Jacobi SVD on the dense covariance gram matrix after a "
               "cov GEMM. The Jacobi sweep contributes many small "
               "Givens-rotation kernel launches."),
    )


if __name__ == "__main__":
    main()
