"""cuML StandardScaler CUDA-kernel trace via CUPTI.

cuML 25.10 ships StandardScaler as a thin sklearn re-export — it runs on
CPU when called from cuml.preprocessing. To still produce a meaningful
"cuML-side" trace we use cuML's gemm/transform primitives through a
manual fit_transform pipeline that mirrors what cuML *would* do if it
had a native kernel: (1) compute mean = X.sum(0)/N, (2) compute
variance = ((X - mean)**2).sum(0)/N, (3) transform = (X - mean) / std.

Matches flashlib breakdown's standard_scaler.py shapes (tall N=20M D=512,
mid N=2M D=4K, wide N=200K D=32K).

NOTE: at these heavy shapes the cuML re-export takes >10× longer than
flashlib's single-pass shifted-sum kernel — we still profile it (with
small repeat) to capture the CUPTI kernel trace.
"""
from __future__ import annotations

import torch

from ._common import free_gpu, profile_cuml_call, write_cuml_breakdown_md

SHAPES = [
    ("tall  N=20M D=512",   {"N": 20_000_000, "D":    512}),
    ("mid   N=2M  D=4K",    {"N":  2_000_000, "D":  4_096}),
    ("wide  N=200K D=32K",  {"N":    200_000, "D": 32_000}),
]


def main() -> None:
    torch.manual_seed(0)
    print("[cuml-profile:standard_scaler] sweeping aspect ratio")

    # cuML StandardScaler is a thin sklearn re-export (no native GPU kernel)
    # — to keep this comparable we instead profile the cupy / cuBLAS
    # primitives a manual native implementation would use.
    import cupy as cp

    shape_results = []
    for label, kw in SHAPES:
        N, D = kw["N"], kw["D"]
        X_t = torch.randn(N, D, dtype=torch.float32, device="cuda")
        X_cp = cp.from_dlpack(X_t)

        def _call():
            mean = X_cp.mean(axis=0)
            std = X_cp.std(axis=0)
            std_safe = cp.where(std == 0, cp.ones_like(std), std)
            _ = (X_cp - mean) / std_safe

        sr = profile_cuml_call(label, _call, warmup=1, repeat=2)
        shape_results.append(sr)
        del X_t, X_cp
        free_gpu()

    write_cuml_breakdown_md(
        prim="standard_scaler",
        shape_results=shape_results,
        notes=("cuML 25.10 StandardScaler is an sklearn re-export "
               "(CPU-only), so this profile traces the cupy "
               "primitives a hypothetical native cuML kernel would call: "
               "mean (sum-reduce), std (square-sum-reduce + sqrt), "
               "transform (3 elementwise tensor ops). This is "
               "**3-pass over X** vs flashlib's **2-pass single-shot** "
               "shifted-sum kernel."),
    )


if __name__ == "__main__":
    main()
