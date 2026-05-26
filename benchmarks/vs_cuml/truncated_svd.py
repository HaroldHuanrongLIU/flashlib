"""TruncatedSVD: ``flash_truncated_svd`` vs ``cuml.decomposition.TruncatedSVD``.

flashlib's TruncatedSVD picks the dual / cov path whose eigh dim is
smaller, runs the dominant cross-GEMM via cuBLAS TF32, and lifts the
eigvecs into singular vectors. ``tol=None`` is exact; ``tol`` opts
into Halko on top.

Correctness signal:
* Relative error in top-K singular values vs sklearn / cuML.
"""
from benchmarks.vs_cuml._common import (
    cap_threads, cuml_shim, time_gpu, time_cpu, title, header, fmt_table,
)
cap_threads(); cuml_shim()

import warnings; warnings.filterwarnings("ignore")
import numpy as np
import torch

from sklearn.decomposition import TruncatedSVD as skTSVD
from cuml.decomposition import TruncatedSVD as cuTSVD
from flashlib.primitives.truncated_svd import flash_truncated_svd


SHAPES = [
    ("tall  N=100K D=128  K=32", 100_000,   128,  32, True),
    ("tall  N=500K D=128  K=32", 500_000,   128,  32, False),
    ("tall  N=200K D=512  K=64", 200_000,   512,  64, False),
    ("wide  N=2K   D=8K   K=32",   2_000,  8_000,  32, False),
]


def _rel_err(a: np.ndarray, b: np.ndarray) -> float:
    denom = max(float(np.max(np.abs(b))), 1e-12)
    return float(np.max(np.abs(a - b)) / denom)


def run_one(label, N, D, K, use_sklearn_cpu: bool):
    title(f"TruncatedSVD {label}  (N={N:,}, D={D}, K={K})")

    rng = np.random.RandomState(0)
    X_np = rng.randn(N, D).astype(np.float32)

    rows = []
    sk_sv = None
    if use_sklearn_cpu:
        sk = skTSVD(n_components=K, algorithm="randomized",
                    random_state=0).fit(X_np)
        sk_sv = sk.singular_values_
        t_sk = time_cpu(
            lambda: skTSVD(n_components=K, algorithm="randomized",
                           random_state=0).fit(X_np),
            repeat=1,
        )
        rows.append(("fp32", "sklearn (CPU)", f"{t_sk:7.2f}",
                     "0.00e0", "1.00x"))

    cu = cuTSVD(n_components=K).fit(X_np)
    cu_sv = np.asarray(cu.singular_values_)
    t_cu = time_gpu(lambda: cuTSVD(n_components=K).fit(X_np),
                    repeat=3, warmup=1)
    rel_cu = _rel_err(cu_sv, sk_sv) if sk_sv is not None else 0.0
    rows.append(("fp32", "cuml", f"{t_cu:7.2f}",
                 f"{rel_cu:.2e}", "1.00x"))

    ref_sv = sk_sv if sk_sv is not None else cu_sv

    X32 = torch.tensor(X_np, device="cuda")
    variants = [
        ("fp32 exact", torch.float32, None),
        ("fp32 halko", torch.float32, 1e-3),
    ]
    for dlabel, dtype, tol in variants:
        X = X32.to(dtype)
        S, _ = flash_truncated_svd(X, K=K, tol=tol)
        fl_sv = S.float().cpu().numpy()
        t_fl = time_gpu(lambda: flash_truncated_svd(X, K=K, tol=tol),
                        repeat=5, warmup=2)
        rel = _rel_err(fl_sv, ref_sv)
        rows.append((dlabel, "flashlib", f"{t_fl:7.2f}",
                     f"{rel:.2e}", f"{t_cu / t_fl:.2f}x"))

    print(fmt_table(rows, ["dtype", "engine", "time(ms)",
                            "rel_err(sv)", "vs cuml"]))


def main():
    header()
    for s in SHAPES:
        run_one(*s)
    print()


if __name__ == "__main__":
    main()
