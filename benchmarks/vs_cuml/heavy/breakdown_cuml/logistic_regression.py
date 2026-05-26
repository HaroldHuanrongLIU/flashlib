"""cuML LogisticRegression CUDA-kernel trace via CUPTI.

Matches flashlib breakdown's logistic_regression.py: N=1M, binary,
D ∈ {512, 2048, 4096}, C=1.0.

cuML LogisticRegression:
  - Python: cuml/linear_model/logistic_regression.pyx
  - C++:    cpp/src/glm/qn/* — Quasi-Newton L-BFGS using fused
              sigmoid+loss+grad kernels (qn_loss.cuh) + cuBLAS GEMVs.
"""
from __future__ import annotations

import torch
from cuml.linear_model import LogisticRegression as cuLogReg

from ._common import free_gpu, profile_cuml_call, write_cuml_breakdown_md

N_FIXED = 1_000_000
MAX_ITER = 100
C_REG = 1.0
GTOL = 1e-4

SHAPES = [
    ("D=512",  512),
    ("D=2048", 2_048),
    ("D=4096", 4_096),
]


def _gpu_classification(N, D):
    torch.manual_seed(0)
    y_t = (torch.rand(N, device="cuda") < 0.5).float()
    sign = (2 * y_t - 1).unsqueeze(1)
    X_t = torch.randn(N, D, device="cuda", dtype=torch.float32)
    X_t[:, :D // 2] += 0.30 * sign
    return X_t, y_t


def main() -> None:
    torch.manual_seed(0)
    print(f"[cuml-profile:logistic_regression] sweeping D at N={N_FIXED:,}, "
          f"binary, C={C_REG}, max_iter={MAX_ITER}")

    shape_results = []
    for label, D in SHAPES:
        X_t, y_t = _gpu_classification(N_FIXED, D)
        import cupy as cp
        X_cp = cp.from_dlpack(X_t)
        y_cp = cp.from_dlpack(y_t)

        def _call():
            cuLogReg(C=C_REG, max_iter=MAX_ITER, tol=GTOL,
                      penalty="l2", fit_intercept=True,
                      solver="qn",
                      output_type="cupy").fit(X_cp, y_cp)

        sr = profile_cuml_call(label, _call, warmup=1, repeat=2)
        shape_results.append(sr)
        del X_t, y_t, X_cp, y_cp
        free_gpu()

    write_cuml_breakdown_md(
        prim="logistic_regression",
        shape_results=shape_results,
        notes=("cuML LogisticRegression solver='qn' (L-BFGS): per-iter "
               "forward GEMV + fused sigmoid/log-loss + backward GEMV + "
               "L2 reg + tiny vector ops. Kernel counts scale with the "
               "L-BFGS iter count which is shape-dependent."),
    )


if __name__ == "__main__":
    main()
