"""UMAP: ``flash_umap`` vs ``cuml.manifold.UMAP``.

UMAP, like t-SNE, is SGD-driven so labels are not a faithful
baseline. We use **trustworthiness** (the fraction of low-D
neighbours that were also high-D neighbours) as the correctness
signal.

flashlib's pipeline routes the KNN graph through ``flash_knn`` (so
the ``tol`` lever propagates: bf16 KNN distances), runs fuzzy
simplicial set construction on GPU, and SGD epochs in Triton.
"""
from benchmarks.vs_cuml._common import (
    cap_threads, cuml_shim, time_gpu, time_cpu, title, header, fmt_table,
)
cap_threads(); cuml_shim()

import warnings; warnings.filterwarnings("ignore")
import numpy as np
import torch

from sklearn.datasets import make_blobs
from sklearn.manifold import trustworthiness
from cuml.manifold import UMAP as cuUMAP
from flashlib.primitives.umap import flash_umap


# (label, N, D, K, n_neighbors, n_epochs)
SHAPES = [
    ("small  N=10K   D=32  K=5  NN=15",   10_000,  32,  5, 15, 200),
    ("medium N=50K   D=64  K=10 NN=15",   50_000,  64, 10, 15, 200),
    ("large  N=100K  D=128 K=10 NN=15",  100_000, 128, 10, 15, 200),
]


def run_one(label, N, D, K, n_neighbors, n_epochs):
    title(f"UMAP {label}  (N={N:,}, D={D}, K={K}, "
          f"NN={n_neighbors}, n_epochs={n_epochs})")

    X_np, _ = make_blobs(n_samples=N, centers=K, n_features=D,
                          cluster_std=2.0, random_state=0)
    X_np = X_np.astype(np.float32)

    # Trustworthiness on full N gets expensive for N >> 20K; subsample.
    n_tw = min(N, 5_000)
    rng = np.random.RandomState(0)
    tw_idx = rng.choice(N, size=n_tw, replace=False)

    rows = []
    cu_emb = np.asarray(
        cuUMAP(n_components=2, n_neighbors=n_neighbors,
                n_epochs=n_epochs, random_state=0).fit_transform(X_np)
    )
    cu_tw = trustworthiness(X_np[tw_idx], cu_emb[tw_idx], n_neighbors=12)
    t_cu = time_gpu(
        lambda: cuUMAP(n_components=2, n_neighbors=n_neighbors,
                        n_epochs=n_epochs, random_state=0).fit_transform(X_np),
        repeat=2, warmup=1,
    )
    rows.append(("fp32", "cuml", f"{t_cu:7.2f}",
                 f"{cu_tw:.4f}", "1.00x"))

    X32 = torch.tensor(X_np, device="cuda")
    variants = [
        ("fp32 exact", torch.float32, None),
        ("bf16 tol",   torch.float32, 1e-3),
    ]
    for dlabel, dtype, tol in variants:
        X = X32.to(dtype)
        fl_emb_t = flash_umap(
            X, n_neighbors=n_neighbors, n_components=2,
            n_epochs=n_epochs, tol=tol, seed=0,
        )
        fl_emb = fl_emb_t.float().cpu().numpy()
        fl_tw = trustworthiness(X_np[tw_idx], fl_emb[tw_idx],
                                  n_neighbors=12)
        t_fl = time_gpu(
            lambda: flash_umap(
                X, n_neighbors=n_neighbors, n_components=2,
                n_epochs=n_epochs, tol=tol, seed=0,
            ),
            repeat=2, warmup=1,
        )
        rows.append((dlabel, "flashlib", f"{t_fl:7.2f}",
                     f"{fl_tw:.4f}", f"{t_cu / t_fl:.2f}x"))

    print(fmt_table(rows, ["dtype", "engine", "time(ms)",
                            "trustworthiness", "vs cuml"]))


def main():
    header()
    for s in SHAPES:
        run_one(*s)
    print()


if __name__ == "__main__":
    main()
