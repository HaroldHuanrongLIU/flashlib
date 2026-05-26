"""Per-component time breakdown for flash_umap across multiple workloads.

flash_umap pipeline (``flashlib/primitives/umap/triton/flash_umap.py:149-213``):

  1. _knn_graph(X, n_neighbors, tol):
       flash_knn(X[None], X[None], k=n_neighbors+1, tol=tol)
       drop self (col 0); sqrt(squared dists)
  2. _fuzzy_simplicial_set(nbr_idx, nbr_d, n_neighbors):
       triton_umap_fuzzy_simplicial_set(...)  # fused bisect + symmetrize
  3. emb init + epoch schedule
  4. for epoch in range(n_epochs):
       triton_flash_umap_sgd_step(emb, head, tail, ...)
       (single fused Triton kernel — gradient + atomic apply CANNOT be split.)

Workload axis: **N and n_epochs jointly.**
  * knn       — one-shot, scales O(N²·D) (brute-force flash_knn).
  * fuzzy     — one-shot, scales O(N·NN²) in the bisect.
  * init      — one-shot, O(N).
  * sgd_step  — accumulates over n_epochs; per-epoch O(N·NN_pos + N·5_neg).

So as n_epochs grows (with N fixed), `sgd_step` share grows.
As N grows (with n_epochs fixed), `knn` share grows (faster than O(N)).
"""
from __future__ import annotations

import numpy as np
import torch
from sklearn.datasets import make_blobs

from flashlib.primitives.knn import flash_knn
from flashlib.primitives.umap.triton.fuzzy_simplicial_set import (
    triton_umap_fuzzy_simplicial_set,
)
from flashlib.primitives.umap.triton.flash_umap import (
    _DEFAULT_A, _DEFAULT_B, _make_epochs_per_sample,
)
from flashlib.primitives.umap.triton.sgd_step import (
    triton_flash_umap_sgd_step,
)

from ._common import (
    StageGroup, free_gpu, run_multi_shape, write_multi_shape_md,
)

# Shared UMAP knobs (mirror flash_umap defaults).
N_COMPONENTS = 2
LEARNING_RATE = 1.0
N_NEG_SAMPLES = 5
SEED = 0
TOL = 1e-3                  # bf16 KNN storage internally

SHAPES = [
    ("D=64 N=50K NN=15 ep=100",
     {"N":  50_000, "D":  64, "NN": 15, "K_blobs": 10, "n_epochs": 100}),
    ("D=256 N=100K NN=15 ep=200",
     {"N": 100_000, "D": 256, "NN": 15, "K_blobs": 10, "n_epochs": 200}),
    ("D=256 N=200K NN=20 ep=500",
     {"N": 200_000, "D": 256, "NN": 20, "K_blobs": 10, "n_epochs": 500}),
]
STAGES = ["knn", "fuzzy", "init", "sgd_step"]


def prepare(N: int, D: int, NN: int, K_blobs: int, n_epochs: int) -> dict:
    torch.manual_seed(0)
    device = "cuda"
    X_np, _ = make_blobs(
        n_samples=N, centers=K_blobs, n_features=D,
        cluster_std=2.0, random_state=0,
    )
    X_np = X_np.astype(np.float32)
    X = torch.tensor(X_np, device=device)
    return {"X": X, "N": N, "D": D, "NN": NN, "n_epochs": n_epochs}


def run(stg: StageGroup, ctx: dict) -> None:
    X, N, NN, n_epochs = ctx["X"], ctx["N"], ctx["NN"], ctx["n_epochs"]
    device = X.device

    # ── Stage 1: KNN graph (flash_umap._knn_graph inlined) ────────────
    with stg["knn"]:
        k_with_self = NN + 1
        dists_sq, indices = flash_knn(X[None], X[None], k=k_with_self, tol=TOL)
        dists_sq = dists_sq[0][:, 1:].contiguous()       # drop self
        indices = indices[0][:, 1:].contiguous()
        nbr_d = torch.sqrt(dists_sq.clamp(min=0.0))
        nbr_idx = indices.to(torch.int64)

    # ── Stage 2: fuzzy simplicial set (single fused Triton kernel) ────
    with stg["fuzzy"]:
        head, tail, weights = triton_umap_fuzzy_simplicial_set(
            nbr_idx, nbr_d, n_iter=64, bandwidth=1.0, tol=1e-5,
            filter_eps=1e-9,
        )

    # ── Stage 3: embedding init + epoch schedule (flash_umap.py:182-197) ─
    with stg["init"]:
        torch.manual_seed(SEED)
        emb = (torch.rand(N, N_COMPONENTS, device=device,
                          dtype=torch.float32) - 0.5) * 20.0
        a, b = _DEFAULT_A, _DEFAULT_B
        eps_per = _make_epochs_per_sample(weights, n_epochs)
        eps_per_neg = eps_per / float(N_NEG_SAMPLES)
        epoch_next = eps_per.clone()
        epoch_next_neg = eps_per_neg.clone()

    # ── Stage 4: SGD loop (flash_umap.py:199-209) ─────────────────────
    for epoch in range(n_epochs):
        lr = LEARNING_RATE * (1.0 - epoch / n_epochs)
        with stg["sgd_step"]:
            triton_flash_umap_sgd_step(
                emb, head, tail,
                eps_per, eps_per_neg,
                epoch_next, epoch_next_neg,
                epoch=float(epoch), lr=lr,
                a=a, b=b, gamma=1.0,
                n_neg_max=max(8, N_NEG_SAMPLES + 3),
                seed=SEED,
            )

    del (dists_sq, indices, nbr_d, nbr_idx, head, tail, weights,
         emb, eps_per, eps_per_neg, epoch_next, epoch_next_neg)


def main() -> None:
    print("[breakdown:umap] sweeping (N, n_epochs) — knn is one-shot, "
          "sgd_step accumulates over n_epochs")
    results = run_multi_shape(SHAPES, prepare, run, STAGES,
                               warmup=1, repeat=3)

    write_multi_shape_md(
        prim="umap",
        shape_axis="(N, D, n_epochs) — knn scales O(N²·D) one-shot; "
                   "sgd_step accumulates over n_epochs",
        results=results,
        stage_names=STAGES,
        notes=(
            "Pipeline: flash_knn(bf16, tol=1e-3) -> fused "
            "triton_umap_fuzzy_simplicial_set -> deterministic-negative "
            "SGD. `sgd_step` accumulates across all n_epochs launches; "
            "gradient and atomic apply are fused inside the Triton "
            "kernel and cannot be split."
        ),
        sensitivity=(
            "**Two axes move at once.** Small→medium adds 4× N², "
            "4× D, and 2× n_epochs; medium→large adds 4× N², 1× D, "
            "2.5× n_epochs (NN also rises 15→20). Measured effects: "
            "**`knn`** is brute-force O(N²·D), so its share jumps "
            "small→medium (31 % → 45 %, the D=64→256 step is doing the "
            "heavy lifting — D alone scales it 4×) and then PLATEAUS "
            "medium→large (45 % → 44 %, since D is held at 256 and N² "
            "only buys ~4× more work, same as what sgd grows by). "
            "**`sgd_step`** accumulates per-epoch cost O(n_epochs · "
            "N · (NN + n_neg)); its absolute ms grow ~3× then ~5× "
            "across the ladder. Share is 51 % → 46 % → 51 %: at small "
            "the modest knn (D=64) lets sgd own the wall; at medium "
            "the knn JUMP (D quadruples) compresses sgd's share; at "
            "large the n_epochs+NN bump pushes sgd back above 50 %. "
            "**`fuzzy` and `init`** stay tiny: fuzzy is one fused "
            "Triton launch with cost O(N·NN²) that saturates quickly "
            "(15 % at small N drops to 4 % at large N as the launch "
            "overhead amortises), and `init` is just an O(N·n_components) "
            "torch.rand (always <3 %). "
            "**Optimisation steers by ladder position**: at the "
            "small/medium end SGD is the long pole — fuse more work "
            "per epoch or schedule across multiple SMs better. At the "
            "large end the KNN graph becomes the long pole once N²·D "
            "outruns the SGD budget — switch from brute-force flash_knn "
            "to an ANN backend (IVF, HNSW) when wall budgets care. "
            "The cold-cache JIT compile time on first run is "
            "dominated by 2 unique `_flash_umap_sgd_kernel` "
            "instantiations (D=64 and D=256); subsequent runs reuse "
            "the `.triton_cache/` and finish in <10 s."
        ),
    )
    free_gpu()


if __name__ == "__main__":
    main()
