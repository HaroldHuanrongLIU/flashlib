"""cuML LinearRegression CUDA-kernel trace via CUPTI.

Matches flashlib breakdown's linear_regression.py: N=2M, D ∈ {128, 512, 2048}.

cuML LinearRegression:
  - Python: cuml/linear_model/linear_regression.pyx (algorithm='eig')
  - C++:    cpp/src/glm/glm.cu  (preProcessData + eig_solve = XtX + cholesky
             via cusolver dnPotrf + dnPotrs)
"""
from __future__ import annotations

import torch
from cuml.linear_model import LinearRegression as cuLR

from ._common import free_gpu, profile_cuml_call, write_cuml_breakdown_md

N_FIXED = 2_000_000

SHAPES = [
    ("D=128",  128),
    ("D=512",  512),
    ("D=2048", 2_048),
]


def main() -> None:
    torch.manual_seed(0)
    print(f"[cuml-profile:linear_regression] sweeping D at N={N_FIXED:,}")

    shape_results = []
    for label, D in SHAPES:
        X_t = torch.randn(N_FIXED, D, dtype=torch.float32, device="cuda")
        w_true = torch.randn(D, dtype=torch.float32, device="cuda") * 0.1
        y_t = (X_t @ w_true + 0.05 * torch.randn(N_FIXED, device="cuda")).contiguous()
        import cupy as cp
        X_cp = cp.from_dlpack(X_t)
        y_cp = cp.from_dlpack(y_t)

        def _call():
            cuLR(algorithm="eig", fit_intercept=True,
                  output_type="cupy").fit(X_cp, y_cp)

        sr = profile_cuml_call(label, _call, warmup=1, repeat=2)
        shape_results.append(sr)
        del X_t, y_t, X_cp, y_cp
        free_gpu()

    write_cuml_breakdown_md(
        prim="linear_regression",
        shape_results=shape_results,
        notes=("cuML LinearRegression algorithm='eig': closed-form via "
               "X.T@X (cuBLAS GEMM) + cusolver cholesky (dnPotrf) + "
               "triangular solves (dnPotrs). No iterative refinement."),
    )


if __name__ == "__main__":
    main()
