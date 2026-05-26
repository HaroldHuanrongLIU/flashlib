"""StandardScaler: ``flash_standard_scaler`` vs cuML / sklearn.

flashlib runs a single-pass shifted-sum Triton kernel for ``mean``
and ``std`` (one HBM read of ``X``), then a fused ``(X-mean)*inv_std``
transform kernel.

Correctness signal:
* Element-wise max-abs error in the scaled output vs sklearn.
"""
from benchmarks.vs_cuml._common import (
    cap_threads, cuml_shim, time_gpu, time_cpu, title, header, fmt_table,
)
cap_threads(); cuml_shim()

import warnings; warnings.filterwarnings("ignore")
import numpy as np
import torch

from sklearn.preprocessing import StandardScaler as skScaler
from cuml.preprocessing import StandardScaler as cuScaler
from flashlib.primitives.standard_scaler import (
    flash_standard_scaler_fit_transform,
)


# (label, N, D, use_sklearn_cpu)
SHAPES = [
    ("tall  N=500K  D=128",   500_000,   128, True),
    ("tall  N=2M    D=128", 2_000_000,   128, False),
    ("tall  N=1M    D=512", 1_000_000,   512, False),
    ("wide  N=100K  D=8K",    100_000, 8_000, False),
]


def run_one(label, N, D, use_sklearn_cpu: bool):
    title(f"StandardScaler {label}  (N={N:,}, D={D})")

    rng = np.random.RandomState(0)
    X_np = rng.randn(N, D).astype(np.float32)

    rows = []
    if use_sklearn_cpu:
        sk_out = skScaler().fit_transform(X_np)
        t_sk = time_cpu(lambda: skScaler().fit_transform(X_np), repeat=1)
        rows.append(("fp32", "sklearn (CPU)", f"{t_sk:7.2f}",
                     "0.00e0", "1.00x"))

    # Pre-stage data on GPU so we time the kernel itself rather than the
    # H2D copy that cuml does implicitly when handed a numpy array.
    import cupy as cp
    X_cp = cp.asarray(X_np)
    cu_out = cp.asnumpy(cuScaler().fit_transform(X_cp))
    t_cu = time_gpu(lambda: cuScaler().fit_transform(X_cp),
                    repeat=3, warmup=1)
    err_cu = float(np.max(np.abs(cu_out - sk_out))) if use_sklearn_cpu else 0.0
    rows.append(("fp32", "cuml", f"{t_cu:7.2f}",
                 f"{err_cu:.2e}", "1.00x"))

    ref_out = sk_out if use_sklearn_cpu else cu_out

    X32 = torch.tensor(X_np, device="cuda")
    Y, _ = flash_standard_scaler_fit_transform(X32)
    err = float((Y.cpu().numpy() - ref_out).__abs__().max())
    t_fl = time_gpu(
        lambda: flash_standard_scaler_fit_transform(X32),
        repeat=5, warmup=2,
    )
    rows.append(("fp32", "flashlib", f"{t_fl:7.2f}",
                 f"{err:.2e}", f"{t_cu / t_fl:.2f}x"))

    print(fmt_table(rows, ["dtype", "engine", "time(ms)",
                            "max_abs_err", "vs cuml"]))


def main():
    header()
    for s in SHAPES:
        run_one(*s)
    print()


if __name__ == "__main__":
    main()
