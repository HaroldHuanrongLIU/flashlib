"""KMeans: ``flash_kmeans`` (fp32 + bf16) vs ``cuml.cluster.KMeans``.

Each shape is reported in TWO precisions for flashlib:
  * fp32 input  -- kernel ``tl.dot`` defaults to TF32; argmin is robust
                   to the ~1e-3 distance noise so cluster IDs match.
  * bf16 input  -- native bf16 GEMM; the assignment kernel reads bf16
                   directly without any precision plumbing.

Reference (always fp32 on GPU for cuML):
  * Optionally sklearn ``KMeans`` (CPU) when ``use_sklearn`` is True for
    the row — **large K** makes CPU reference prohibitive, so we skip it
    and report ``ARI vs cuml`` instead.
  * cuml ``KMeans`` (GPU) -- bar to beat.

We use IDENTICAL initial centroids across all engines (same indices on
the same data) so any ARI / inertia gap is implementation quality, not
init lottery. ``vs cuml`` compares flashlib timing against cuml's
timing for the same shape.
"""
from benchmarks.vs_cuml._common import (
    cap_threads, cuml_shim, time_gpu, time_cpu, title,
    ari, header, fmt_table,
)
cap_threads(); cuml_shim()

import warnings; warnings.filterwarnings("ignore")
import numpy as np
import torch

from sklearn.cluster import KMeans as skKMeans
from sklearn.datasets import make_blobs
from cuml.cluster import KMeans as cuKMeans
from flashlib.primitives.kmeans import flash_kmeans


# (label, N, D, K, max_iter, use_sklearn_cpu)
# Large K (512 / 1024): skip sklearn — one CPU fit would dominate wall
# time; ARI vs cuml still verifies flashlib matches the GPU reference.
SHAPES = [
    ("medium K=256",  100_000, 32,   256, 15, True),
    ("large K=512",   200_000, 64,   512, 12, False),
    ("xlarge K=1024", 500_000, 64, 1_024, 10, False),
]

DTYPES = [
    ("fp32", torch.float32),
    ("bf16", torch.bfloat16),
]


def run_one(label, N, D, K, max_iter, use_sklearn_cpu: bool):
    title(
        f"KMeans {label}  (N={N:,}, D={D}, K={K}, max_iter={max_iter}, "
        f"ref={'sklearn+cuml' if use_sklearn_cpu else 'cuml-only'})"
    )

    X_np, _ = make_blobs(n_samples=N, centers=K, n_features=D,
                         cluster_std=2.0, random_state=0)
    X_np = X_np.astype(np.float32)

    rng = np.random.RandomState(0)
    init_idx = rng.choice(N, size=K, replace=False)
    init_np = X_np[init_idx].copy()

    KM_TOL = 1e-6  # cuml rejects tol=0; pick something tiny

    ari_hdr = "ARI vs sk" if use_sklearn_cpu else "ARI vs cuml"
    rows = []

    sk_lbl = None
    sk_inertia = None
    t_sk = None
    if use_sklearn_cpu:
        km_sk = skKMeans(n_clusters=K, n_init=1, max_iter=max_iter,
                         init=init_np, tol=KM_TOL).fit(X_np)
        t_sk = time_cpu(lambda: skKMeans(n_clusters=K, n_init=1,
                                         max_iter=max_iter, init=init_np,
                                         tol=KM_TOL).fit(X_np), repeat=1)
        sk_lbl, sk_inertia = km_sk.labels_, km_sk.inertia_
        rows.append(("fp32", "sklearn (CPU)", f"{t_sk:7.2f}", "1.0000",
                      f"{sk_inertia:.3e}", "1.00x"))

    km_cu = cuKMeans(n_clusters=K, n_init=1, max_iter=max_iter,
                     init=init_np, tol=KM_TOL).fit(X_np)
    cu_lbl = np.asarray(km_cu.labels_); cu_inertia = float(km_cu.inertia_)
    t_cu = time_gpu(lambda: cuKMeans(n_clusters=K, n_init=1,
                                     max_iter=max_iter, init=init_np,
                                     tol=KM_TOL).fit(X_np),
                    repeat=3, warmup=1)
    if use_sklearn_cpu:
        rows.append(("fp32", "cuml", f"{t_cu:7.2f}",
                      f"{ari(sk_lbl, cu_lbl):.4f}",
                      f"{cu_inertia:.3e}", "1.00x"))
    else:
        rows.append(("fp32", "cuml", f"{t_cu:7.2f}", "1.0000",
                      f"{cu_inertia:.3e}", "1.00x"))

    ref_lbl = sk_lbl if use_sklearn_cpu else cu_lbl

    X32 = torch.tensor(X_np, device="cuda")
    for dlabel, dtype in DTYPES:
        X = X32.to(dtype)
        init_th = X[init_idx].clone().unsqueeze(0)
        fl_lbl_t, fl_centers, _ = flash_kmeans(
            X, K, max_iters=max_iter, init_centroids=init_th,
        )
        fl_lbl = fl_lbl_t.squeeze(0).cpu().numpy()
        diff = X32 - fl_centers.squeeze(0).to(torch.float32)[fl_lbl_t.squeeze(0)]
        fl_inertia = float((diff * diff).sum().item())
        t_fl = time_gpu(
            lambda: flash_kmeans(X, K, max_iters=max_iter, init_centroids=init_th),
            repeat=5, warmup=2,
        )
        rows.append((
            dlabel, "flashlib", f"{t_fl:7.2f}",
            f"{ari(ref_lbl, fl_lbl):.4f}",
            f"{fl_inertia:.3e}",
            f"{t_cu / t_fl:.2f}x",
        ))

    print(fmt_table(rows, ["dtype", "engine", "time(ms)", ari_hdr,
                            "inertia", "vs cuml"]))


def main():
    header()
    for s in SHAPES:
        run_one(*s)
    print()


if __name__ == "__main__":
    main()
