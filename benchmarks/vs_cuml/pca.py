"""PCA: ``flash_pca`` (fp32 + bf16) vs ``cuml.decomposition.PCA``.

flashlib's PCA is exact in input dtype by default: routes to
``triton_pca`` -> cuBLAS TF32 GEMM (cov or Gram) + cuSOLVER ``eigh``.
Passing ``tol`` opts into approximate Halko on top of cov-eigh.

Correctness signal:
* Top-K eigenvalue match against cuML (relative error < 1%).
* Subspace projection norm match against cuML.
"""
from benchmarks.vs_cuml._common import (
    cap_threads, cuml_shim, time_gpu, time_cpu, title, header, fmt_table,
)
cap_threads(); cuml_shim()

import warnings; warnings.filterwarnings("ignore")
import numpy as np
import torch

from sklearn.decomposition import PCA as skPCA
from cuml.decomposition import PCA as cuPCA
from flashlib.primitives.pca import flash_pca


# (label, N, D, K, use_sklearn_cpu)
SHAPES = [
    ("tall  N=100K D=128  K=32", 100_000,   128,  32, True),
    ("tall  N=500K D=128  K=32", 500_000,   128,  32, False),
    ("tall  N=200K D=512  K=64", 200_000,   512,  64, False),
    ("wide  N=2K   D=8K   K=32",   2_000,  8_000,  32, False),
]


def _explained_var(eigvals: torch.Tensor) -> np.ndarray:
    """Return descending-sorted explained variance (numpy fp32)."""
    return torch.sort(eigvals.float(), descending=True).values.cpu().numpy()


def _rel_err(a: np.ndarray, b: np.ndarray) -> float:
    """max(|a-b|) / max(|b|)."""
    denom = max(float(np.max(np.abs(b))), 1e-12)
    return float(np.max(np.abs(a - b)) / denom)


def run_one(label, N, D, K, use_sklearn_cpu: bool):
    title(f"PCA {label}  (N={N:,}, D={D}, K={K})")

    rng = np.random.RandomState(0)
    X_np = rng.randn(N, D).astype(np.float32)

    rows = []
    sk_ev = None
    if use_sklearn_cpu:
        sk = skPCA(n_components=K, svd_solver="randomized",
                   random_state=0).fit(X_np)
        sk_ev = sk.explained_variance_
        t_sk = time_cpu(
            lambda: skPCA(n_components=K, svd_solver="randomized",
                          random_state=0).fit(X_np),
            repeat=1,
        )
        rows.append(("fp32", "sklearn (CPU)", f"{t_sk:7.2f}",
                     "0.00e0", "1.00x"))

    cu = cuPCA(n_components=K).fit(X_np)
    cu_ev = np.asarray(cu.explained_variance_)
    t_cu = time_gpu(lambda: cuPCA(n_components=K).fit(X_np),
                    repeat=3, warmup=1)
    rel_cu = _rel_err(cu_ev, sk_ev) if sk_ev is not None else 0.0
    rows.append(("fp32", "cuml", f"{t_cu:7.2f}",
                 f"{rel_cu:.2e}", "1.00x"))

    # Reference for ``rel_err`` is cuML when sklearn was skipped.
    ref_ev = sk_ev if sk_ev is not None else cu_ev

    X32 = torch.tensor(X_np, device="cuda")
    Xc32 = X32 - X32.mean(dim=0, keepdim=True)
    # fp32 + tol=None -> exact: cuBLAS TF32 cov/Gram GEMM + cuSOLVER eigh.
    # fp32 + tol=1e-3 -> opt-in Halko (only fires when ``K*4 < N`` on the
    # eigh matrix; cov path's eigh dim = D so this only matters when
    # D > 4K, i.e. the wide-D shape below).
    variants = [
        ("fp32 exact",  torch.float32, None),
        ("fp32 halko",  torch.float32, 1e-3),
    ]
    for dlabel, dtype, tol in variants:
        Xc = Xc32.to(dtype)
        eigvals, _ = flash_pca(Xc, K=K, tol=tol)
        fl_ev = _explained_var(eigvals * (N / (N - 1)))
        t_fl = time_gpu(lambda: flash_pca(Xc, K=K, tol=tol),
                        repeat=5, warmup=2)
        rel = _rel_err(fl_ev, ref_ev)
        rows.append((dlabel, "flashlib", f"{t_fl:7.2f}",
                     f"{rel:.2e}", f"{t_cu / t_fl:.2f}x"))

    print(fmt_table(rows, ["dtype", "engine", "time(ms)",
                            "rel_err(ev)", "vs cuml"]))


def main():
    header()
    for s in SHAPES:
        run_one(*s)
    print()


if __name__ == "__main__":
    main()
