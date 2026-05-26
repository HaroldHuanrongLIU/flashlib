"""cuML Ridge CUDA-kernel trace via CUPTI.

Matches flashlib breakdown's ridge.py: N=2M, D=512, T ∈ {1, 16, 64}.
NOTE: cuML Ridge does NOT support multi-target Y natively (only T=1
out of the box). For T>1 we loop T times (same shape, T columns)
so the kernel mix is comparable but the host-side overhead is T×
higher; the per-call time reported below is the average over T fits.

cuML Ridge:
  - Python: cuml/linear_model/ridge.pyx (solver='eig')
  - C++:    cpp/src/glm/ridge.cu (preProcess + XtX + alpha*I + cholesky)
"""
from __future__ import annotations

import torch
from cuml.linear_model import Ridge as cuRidge

from ._common import free_gpu, profile_cuml_call, write_cuml_breakdown_md

N_FIXED, D_FIXED, ALPHA = 2_000_000, 512, 1.0

SHAPES = [
    ("T=1",  1),
    ("T=16", 16),
    ("T=64", 64),
]


def main() -> None:
    torch.manual_seed(0)
    print(f"[cuml-profile:ridge] sweeping T at N={N_FIXED:,}, D={D_FIXED}, "
          f"alpha={ALPHA}")

    shape_results = []
    for label, T in SHAPES:
        X_t = torch.randn(N_FIXED, D_FIXED, dtype=torch.float32, device="cuda")
        w_true = torch.randn(D_FIXED, T, dtype=torch.float32, device="cuda") * 0.1
        Y_t = (X_t @ w_true
                + 0.05 * torch.randn(N_FIXED, T, device="cuda")).contiguous()
        import cupy as cp
        X_cp = cp.from_dlpack(X_t)

        def _call():
            for t in range(T):
                y_cp = cp.from_dlpack(Y_t[:, t].contiguous())
                cuRidge(alpha=ALPHA, solver="eig", fit_intercept=True,
                         output_type="cupy").fit(X_cp, y_cp)

        sr = profile_cuml_call(label, _call, warmup=1, repeat=2)
        shape_results.append(sr)
        del X_t, Y_t, X_cp
        free_gpu()

    write_cuml_breakdown_md(
        prim="ridge",
        shape_results=shape_results,
        notes=("cuML Ridge solver='eig': closed-form via X.T@X (cuBLAS) "
               "+ diag-add(alpha) + cusolver cholesky. cuML's Ridge does "
               "NOT vectorise over T (multi-target); the script loops "
               "T calls per outer call, so kernel **counts** scale with "
               "T but each call's per-kernel ms is T-independent."),
    )


if __name__ == "__main__":
    main()
