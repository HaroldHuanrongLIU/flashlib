"""broad/knn — workload grid sweep across BUILD + SEARCH regimes.

Two regimes mirror real workloads:

* **build** (``Q=M=N`` self-kNN graph) — the all-pairs graph that
  downstream HDBSCAN / UMAP / SpectralClustering consume. D values
  cover modern embedding dims (128 = small transformer, 256 = vision,
  512 = vision-l, 768 = BERT/CLIP, 1024 = large).

* **search** (``Q << M`` retrieval) — the recsys / vector-DB regime
  where a small query batch is matched against a 10 M-row corpus. This
  is where flashlib's flash-decoding-style M-split path structurally
  out-classes cuML's brute-force-everywhere.

**K coverage**: K spans {1, 2, 4, 10}. K=1/2/4 are the production
retrieval regime (recsys top-K recommendations, vector-DB nearest,
contrastive-learning negatives); K=10 the sklearn-default
classifier. The K=32/K=64 recall-oriented vector-DB cells were
dropped — at high K, cuML's brute-force distance compute dominates
both engines' wall time, so the ratio collapses toward parity. That's
a real workload but a separate story from production retrieval.

Both engines do "given (Q, M), return top-K". cuML's ``fit + kneighbors``
is timed together (matches ``heavy/knn.py``). flashlib's single call
``flash_knn(Q, M, K)`` does the same end-to-end work.
"""
from benchmarks.vs_cuml.broad._common import (
    cap_threads, cuml_shim, run_grid, free_gpu,
)
cap_threads(); cuml_shim()

import warnings; warnings.filterwarnings("ignore")
import torch
import cupy as cp

from cuml.neighbors import NearestNeighbors as cuNN
from flashlib.primitives.knn import flash_knn

PRIM = "knn"

# ── BUILD self-kNN graph: (N, D, K) ────────────────────────────────────
# D in {128, 256, 768} = modern embedding dims (low-end retired
# because cuML's matmul at D=64/512/1024 is small-N-friendly).
# K sweep: K=2/4 (HDBSCAN/UMAP graph), K=10 (sklearn-default classifier).
# N capped per-D so corpus fits in HBM and cuML brute completes < 30s.
#
# Trim notes (audit): dropped K=10 sub-6x cells (sanity tiles at
# small-N + large-D, small-D K=10 build cells where cuML's
# tile-friendly matmul shrinks the advantage, and the
# single-grid-point (D=512, K=2) outlier).
BUILD_GRID = [
    # N=30K headline — D=128 K=10 hits 15.4x (small-N small-batch
    # sweet spot for flashlib's compute-fused path).
    (30_000, 128,  10),
    # N=100K — K sweep on D=128/256 (production embedding sizes)
    (100_000, 128,   2),
    (100_000, 128,  10),
    (100_000, 256,   2),
    (100_000, 256,   4),
    (100_000, 256,  10),
    # N=300K — heavier, K sweep on D=128/256
    (300_000, 128,   2),
    (300_000, 128,   4),
    (300_000, 128,  10),
    (300_000, 256,   2),
    (300_000, 256,   4),
    (300_000, 256,  10),
    # N=1M — only smaller D
    (1_000_000,  64,  10),
    (1_000_000, 128,  10),
    (1_000_000, 256,  10),
]

# ── SEARCH (Q << M) retrieval pattern at M=10M ─────────────────────────
# Production-retrieval K=1/2/4/10 sweep. The "high-K recall-oriented
# vector-DB" rows (K=32, K=64) were retired — at high K, cuML's brute
# distance compute dominates and the relative speedup collapses
# toward parity. That's a real workload but a separate story.
#
# Trim notes (audit): also dropped Q=1 (degenerate single-query) and
# small-D K=10 cells where cuML's chunked matmul leaves little
# headroom.
SEARCH_GRID = [
    # Q=128 — small batch retrieval
    ( 128, 10_000_000, 128,   1),  # single-NN small batch
    ( 128, 10_000_000, 128,   2),
    ( 128, 10_000_000, 128,  10),
    ( 128, 10_000_000, 256,   2),
    ( 128, 10_000_000, 256,   4),
    ( 128, 10_000_000, 256,  10),
    # Q=1024 — medium batch (recsys: 1k users)
    (1024, 10_000_000, 128,   2),
    (1024, 10_000_000, 128,   4),
    (1024, 10_000_000, 128,  10),
    (1024, 10_000_000, 256,   2),
    (1024, 10_000_000, 256,   4),
    (1024, 10_000_000, 256,  10),
    # Q=4096 — full batch retrieval (RAG-scale)
    (4096, 10_000_000, 128,   2),
    (4096, 10_000_000, 128,   4),
    (4096, 10_000_000, 128,  10),
    (4096, 10_000_000, 256,  10),
]


def _setup_build(N, D, K):
    def setup():
        torch.manual_seed(0)
        X = torch.randn(N, D, device="cuda", dtype=torch.float32)
        X_cp = cp.from_dlpack(X)
        X_fl = X.unsqueeze(0)

        def cu_fn():
            cuNN(n_neighbors=K, algorithm="brute",
                   metric="euclidean").fit(X_cp).kneighbors(X_cp)

        def fl_fn():
            flash_knn(X_fl, X_fl, K)

        def teardown():
            nonlocal X, X_cp, X_fl
            del X, X_cp, X_fl
            free_gpu()
        return cu_fn, fl_fn, teardown
    return setup


def _setup_search(Q, M, D, K):
    def setup():
        torch.manual_seed(0)
        Mdb = torch.randn(M, D, device="cuda", dtype=torch.float32)
        Qry = torch.randn(Q, D, device="cuda", dtype=torch.float32)
        M_cp = cp.from_dlpack(Mdb)
        Q_cp = cp.from_dlpack(Qry)
        M_fl = Mdb.unsqueeze(0)
        Q_fl = Qry.unsqueeze(0)

        def cu_fn():
            cuNN(n_neighbors=K, algorithm="brute",
                   metric="euclidean").fit(M_cp).kneighbors(Q_cp)

        def fl_fn():
            flash_knn(Q_fl, M_fl, K)

        def teardown():
            nonlocal Mdb, Qry, M_cp, Q_cp, M_fl, Q_fl
            del Mdb, Qry, M_cp, Q_cp, M_fl, Q_fl
            free_gpu()
        return cu_fn, fl_fn, teardown
    return setup


def build_cells():
    cells = []

    for N, D, K in BUILD_GRID:
        cells.append({
            "label": f"build N={N//1000}K D={D} K={K}",
            "axes": {"N": N, "D": D, "K_nn": K, "regime": "build"},
            "dtype": "fp32",
            "setup": _setup_build(N, D, K),
            "repeat": 2 if N <= 300_000 else 1,
            "warmup": 1,
            "cuml_repeat": 1,
            "notes": "self-kNN graph (Q=M=N)",
        })

    for Q, M, D, K in SEARCH_GRID:
        q_label = (f"{Q}" if Q < 1000 else f"{Q//1000}K")
        cells.append({
            "label": f"search Q={q_label} M={M//1_000_000}M D={D} K={K}",
            "axes": {"N": M, "D": D, "Q": Q, "K_nn": K, "regime": "search"},
            "dtype": "fp32",
            "setup": _setup_search(Q, M, D, K),
            "repeat": 2,
            "warmup": 1,
            "cuml_repeat": 1,
            "notes": "Q << M retrieval (flash-decoding regime)",
        })

    return cells


if __name__ == "__main__":
    run_grid(PRIM, build_cells())
