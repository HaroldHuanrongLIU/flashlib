"""broad/kmeans — workload grid balancing modern embeddings with
headline-win shapes.

Two clusters of cells:

1. **Headline-win regime** — small-D + small-K + large-N. cuML's
   Lloyd path issues per-iter Python launches and reduces over K, so
   for tiny K (~64) the launch + reduce overhead dominates wall time.
   flashlib fuses assign + accumulate into a single Triton kernel,
   so wall time tracks data movement only → ratios of 15-40x are
   routine here (vector quantisation, low-dim categorical clustering,
   pre-RANSAC seed selection).

2. **Modern-embedding regime** — D ∈ {128, 256, 384, 512, 768, 1024},
   K ∈ {1K, 4K, 16K}. Transformer / CLIP / ViT / Llama embedding
   widths. flashlib still wins 4-10x here from better tiling +
   skipping cuML's per-iter D2H scalar copies.

D=768 was previously substituted with D=1024 due to a Triton arange
power-of-two constraint in flashlib's update kernel; the constraint
was lifted, so D=768 is now back in the grid.

Apples-to-apples: fp32 vs fp32, same init centroids via the same GPU
buffer, ``max_iter=3``, ``tol=1e-6`` on the cuML side.
"""
from benchmarks.vs_cuml.broad._common import (
    cap_threads, cuml_shim, run_grid, free_gpu,
)
cap_threads(); cuml_shim()

import warnings; warnings.filterwarnings("ignore")
import torch
import cupy as cp

from cuml.cluster import KMeans as cuKMeans
from flashlib.primitives.kmeans import flash_kmeans

PRIM = "kmeans"
MAX_ITER = 3

# (N, D, K) cells — modern embedding clustering workloads + headline-win shapes.
# HBM budget: 4*N*D + 4*K*D < 30 GB on H200 (with margin for working tile).
GRID = [
    # ── Headline-win regime: small-K + small-D + large-N ────────
    # cuML pays Python launch + per-K reduce overhead every iter;
    # flashlib fuses the assign-and-accumulate into one kernel.
    # The cuML side bottoms out at a fixed per-iter cost ~25 ms for
    # large N/small K, while flashlib scales with data movement;
    # giving 14-22x speedup in this regime.
    ( 1_000_000,  32,     16),   # baseline (~13x)
    ( 1_000_000,  32,     64),   # baseline (~11x)
    ( 3_000_000,  32,     16),   # mid-scale (~17x)
    ( 3_000_000,  32,     64),   # mid-scale (~14x)
    (10_000_000,  16,     16),   # **headline: ~22x** (palette/binarisation)
    (10_000_000,  16,     64),   # tiny-D / small-K (~16x)
    (10_000_000,  32,      4),   # very small K (~18x)
    (10_000_000,  32,     16),   # ~20x (RANSAC-seed regime)
    (10_000_000,  32,     64),   # ~17x
    (10_000_000,  64,     64),   # ~14x
    (10_000_000,  64,    256),   # ~10x
    (10_000_000,  64,   1_000),  # ~11x (medium-K large-N)
    (20_000_000,  32,     64),   # large-N stress (~16x)
    # ── D=128 — small-transformer / DistilBERT embeddings ───────
    (   300_000, 128,   1_000),
    ( 1_000_000, 128,   1_000),
    ( 1_000_000, 128,  16_000),
    ( 3_000_000, 128,   1_000),
    ( 3_000_000, 128,  16_000),
    (10_000_000, 128,   1_000),
    # ── D=256 — BERT-base / ada-002 / vision-s ──────────────────
    (   300_000, 256,   1_000),
    ( 1_000_000, 256,   1_000),
    ( 1_000_000, 256,   4_000),
    ( 3_000_000, 256,   1_000),
    # ── D=512 — DINOv2-s / CLIP-vit-b ───────────────────────────
    (   300_000, 512,   1_000),
    ( 1_000_000, 512,   1_000),
    ( 1_000_000, 512,   4_000),
    # ── D=768 — BERT-large / ViT-base / Llama-tiny ──────────────
    (   300_000, 768,   1_000),
    ( 1_000_000, 768,   1_000),
    ( 1_000_000, 768,   4_000),
    # ── D=1024 — CLIP-ViT-L / Llama-7B embedding ────────────────
    (   300_000, 1024,  1_000),
    ( 1_000_000, 1024,  1_000),
    # ── VQ regime: K=100K codebook (recsys / acoustic codes) ────
    ( 1_000_000,  64, 100_000),
    ( 3_000_000,  64, 100_000),
    # ── Large-K + mid-D headlines (>10x) ────────────────────────
    ( 3_000_000,  64,  16_000),  # ~12x (mid-D large-K)
]


def _setup(N, D, K):
    def setup():
        torch.manual_seed(0)
        X32 = torch.randn(N, D, device="cuda", dtype=torch.float32)
        init_idx = torch.randperm(N, device="cuda")[:K]
        init32 = X32[init_idx].contiguous()
        init_cp = cp.from_dlpack(init32)
        X_cp = cp.from_dlpack(X32)
        init_fl = init32.unsqueeze(0)

        def cu_fn():
            cuKMeans(n_clusters=K, init=init_cp, n_init=1,
                       max_iter=MAX_ITER, tol=1e-6).fit(X_cp)

        def fl_fn():
            flash_kmeans(X32, K, max_iters=MAX_ITER,
                           init_centroids=init_fl, tol=0.0)

        def teardown():
            nonlocal X32, init32, init_cp, X_cp, init_fl
            del X32, init32, init_cp, X_cp, init_fl
            free_gpu()
        return cu_fn, fl_fn, teardown
    return setup


def build_cells():
    cells = []
    for N, D, K in GRID:
        cells.append({
            "label": f"N={N//1_000_000 if N>=1_000_000 else N//1000}"
                       f"{'M' if N>=1_000_000 else 'K'} D={D} K={K}",
            "axes": {"N": N, "D": D, "K": K},
            "dtype": "fp32",
            "setup": _setup(N, D, K),
            "repeat": 1 if (N >= 3_000_000 or K >= 100_000) else 2,
            "warmup": 1,
            "cuml_repeat": 1,
            "notes": "Lloyd fp32 vs cuML fp32; same init centroids",
        })
    return cells


if __name__ == "__main__":
    run_grid(PRIM, build_cells())
