"""Small-Q / large-M KNN search sweep vs cuml.

Standalone benchmark for the search regime: Q in {1, 16, 32, 128} and
M ~ 10M. This is where the sortmerge / insert paths in
``flashlib.primitives.knn.triton`` are designed to dominate -- the
shape-only heuristic keeps small-Q queries on the M-split
flash-decode ("search") routing with BN scaled by N (the same
``ctas_no_split`` check that gates large-N single-pass).

For each shape we report fp32 and bf16, both flashlib backends
(triton-auto, cutedsl-FA3) against cuml brute fp32 on the same M-corpus.
Recall is reported vs cuml fp32 because sklearn brute at M=10M takes
minutes per call.
"""
from benchmarks.vs_cuml._common import (
    cap_threads, cuml_shim, time_gpu, title,
    recall_at_k, header, fmt_table,
)
cap_threads(); cuml_shim()

import warnings; warnings.filterwarnings("ignore")
import numpy as np
import torch
import cupy as cp

from cuml.neighbors import NearestNeighbors as cuNN
from flashlib.primitives.knn import flash_knn


# (Q,  M,         D,   K)
SHAPES = [
    (  1, 10_000_000, 64, 10),
    ( 16, 10_000_000, 64, 10),
    ( 32, 10_000_000, 64, 10),
    (128, 10_000_000, 64, 10),
    # And one D=128 row to see embedding-scale numbers.
    (  1, 10_000_000, 128, 10),
    (128, 10_000_000, 128, 10),
]


def _torch_to_cupy(t):
    return cp.from_dlpack(t)


def _bw(Q, M, D, K, t_ms, sz):
    """Input + output bytes / time. Search shapes are HBM-bound on M*D."""
    inp = (Q + M) * D * sz
    out = Q * K * 8
    return (inp + out) / 1e9 / (t_ms / 1000.0)


def run_one(Q, M, D, K):
    title(f"KNN search  (Q={Q}, M={M:,}, D={D}, K={K})")
    rng = np.random.RandomState(0)
    Xc_np = rng.randn(M, D).astype(np.float32)
    Xq_np = rng.randn(Q, D).astype(np.float32)

    Xc32 = torch.tensor(Xc_np, device="cuda")
    Xq32 = torch.tensor(Xq_np, device="cuda")

    # cuml brute (fp32). Single fit reused across all rows.
    cu_nn = cuNN(n_neighbors=K, algorithm="brute", metric="euclidean").fit(
        _torch_to_cupy(Xc32))
    cu_out = cu_nn.kneighbors(_torch_to_cupy(Xq32), return_distance=False)
    cu_idx = cu_out.get() if hasattr(cu_out, "get") else np.asarray(cu_out)
    t_cu = time_gpu(
        lambda: cu_nn.kneighbors(_torch_to_cupy(Xq32), return_distance=False),
        repeat=5, warmup=2)

    rows = [("fp32", "cuml brute", f"{t_cu:8.2f}",
             f"{_bw(Q, M, D, K, t_cu, 4):6.1f}",
             "1.0000", "1.00x")]

    for dlabel, dtype, sz in [("fp32", torch.float32, 4),
                              ("bf16", torch.bfloat16, 2)]:
        Xc = Xc32.to(dtype); Xq = Xq32.to(dtype)

        # triton (auto sortmerge/insert)
        out = flash_knn(Xq[None], Xc[None], K, backend="triton")
        idx = out[1].squeeze(0).cpu().numpy()
        t = time_gpu(
            lambda: flash_knn(Xq[None], Xc[None], K, backend="triton"),
            repeat=10, warmup=3)
        rows.append((
            dlabel, "flashlib triton", f"{t:8.2f}",
            f"{_bw(Q, M, D, K, t, sz):6.1f}",
            f"{recall_at_k(idx, cu_idx, K):.4f}",
            f"{t_cu / t:.2f}x"))

        # cutedsl FA3 (bf16/fp16 only)
        if dtype is not torch.float32:
            try:
                out = flash_knn(Xq[None], Xc[None], K, backend="cutedsl")
                idx = out[1].squeeze(0).cpu().numpy()
                t = time_gpu(
                    lambda: flash_knn(Xq[None], Xc[None], K, backend="cutedsl"),
                    repeat=10, warmup=5)
                rows.append((
                    dlabel, "flashlib cutedsl-FA3", f"{t:8.2f}",
                    f"{_bw(Q, M, D, K, t, sz):6.1f}",
                    f"{recall_at_k(idx, cu_idx, K):.4f}",
                    f"{t_cu / t:.2f}x"))
            except Exception as e:
                rows.append((dlabel, "flashlib cutedsl-FA3",
                             "  SKIP", "  --  ",
                             str(e).splitlines()[0][:24], "  --  "))

    print(fmt_table(rows, ["dtype", "engine", "time(ms)", "GB/s",
                            "recall@K", "vs cuml"]))


def main():
    header()
    for s in SHAPES:
        run_one(*s)
    print()


if __name__ == "__main__":
    main()
