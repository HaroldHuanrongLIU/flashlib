"""IVF-Flat: ``flash_ivf_flat`` vs cuVS / cuML IVF-Flat at matched (nlist, nprobe).

The point of IVF-Flat in flashlib is *iso-recall*: at a fixed
``(nlist, nprobe)`` the probed candidate set is the same as any reference
IVF-Flat, so the comparison that matters is **search throughput (QPS) at
equal recall**, not recall traded for speed.

Per row we report, for flashlib and (when installed) cuVS / cuML:

  * build(ms)   -- one-time index construction.
  * search(ms)  -- per-batch query latency (warm, cuda-synced).
  * QPS         -- nq / search_time.
  * recall@k    -- vs exact brute force (``flash_knn`` fp32 ground truth).
  * fine GB/s   -- flashlib only: useful HBM bandwidth of the fused
                   fine-scan (probed-list reads); use this to backfill
                   ``_SUSTAINED_BW_TBS[("ivf_flat_search", dev)]`` in
                   flashlib/info/roofline.py.

cuVS / cuML are optional. Install out-of-band, e.g.::

    pip install --extra-index-url https://pypi.nvidia.com 'cuvs-cu12==25.10.*'

Run::

    python -m benchmarks.vs_cuml.ivf_flat
"""
from __future__ import annotations

import time
import numpy as np
import torch

from benchmarks.vs_cuml._common import time_gpu, title, fmt_table, recall_at_k


# (label,             M,        nq,    D,   nlist, nprobe,  k)
SHAPES = [
    ("1M  D=64",       1_000_000, 10_000, 64,  1024,  16,    10),
    ("1M  D=128",      1_000_000, 10_000, 128, 1024,  32,    10),
    ("online D=64",    1_000_000,    100, 64,  1024,  16,    10),
    ("500K D=96",        500_000,  5_000, 96,   512,  24,    10),
]


def _blobs(M, D, n_centers, seed):
    rng = np.random.RandomState(seed)
    centers = rng.randn(n_centers, D).astype(np.float32) * 4.0
    lab = rng.randint(0, n_centers, size=M)
    return (centers[lab] + rng.randn(M, D).astype(np.float32)).astype(np.float32)


def _brute_ids(Xc_t, Xq_t, k):
    """Exact top-k ids (squared L2) via flash_knn fp32 -- recall ground truth."""
    from flashlib.primitives.knn import flash_knn
    idx = flash_knn(Xq_t.unsqueeze(0), Xc_t.unsqueeze(0), k)[1][0]
    return idx.cpu().numpy()


def _fine_bytes_gb(nq, nprobe, M, nlist, D, sz):
    """Useful HBM read of probed candidate vectors (fused fine-scan)."""
    cand = nq * nprobe * (M / nlist)
    return cand * D * sz / 1e9


def _flashlib_row(label, Xc_t, Xq_t, M, nq, D, nlist, nprobe, k, brute):
    from flashlib import flash_ivf_flat_build, flash_ivf_flat_search

    t_build = time_gpu(
        lambda: flash_ivf_flat_build(Xc_t, nlist, nprobe=nprobe, niter=20, seed=0),
        repeat=1, warmup=0,
    )
    index = flash_ivf_flat_build(Xc_t, nlist, nprobe=nprobe, niter=20, seed=0)

    ids = flash_ivf_flat_search(index, Xq_t, k, nprobe=nprobe)[1].cpu().numpy()
    t_search = time_gpu(
        lambda: flash_ivf_flat_search(index, Xq_t, k, nprobe=nprobe),
        repeat=10, warmup=3,
    )
    qps = nq / (t_search / 1000.0)
    gbps = _fine_bytes_gb(nq, nprobe, M, nlist, D, 4) / (t_search / 1000.0)
    return ("flashlib", f"{t_build:8.1f}", f"{t_search:8.3f}", f"{qps:11,.0f}",
            f"{recall_at_k(ids, brute, k):.4f}", f"{gbps:7.0f}")


def _cuvs_row(label, Xc_np, Xq_np, M, nq, D, nlist, nprobe, k, brute):
    import cupy as cp
    from cuvs.neighbors import ivf_flat

    Xc = cp.asarray(Xc_np); Xq = cp.asarray(Xq_np)
    ip = ivf_flat.IndexParams(n_lists=nlist, metric="sqeuclidean")
    t_build = time_gpu(lambda: ivf_flat.build(ip, Xc), repeat=1, warmup=0)
    index = ivf_flat.build(ip, Xc)
    sp = ivf_flat.SearchParams(n_probes=nprobe)

    _, I = ivf_flat.search(sp, index, Xq, k)
    ids = cp.asarray(I).get()
    t_search = time_gpu(lambda: ivf_flat.search(sp, index, Xq, k),
                        repeat=10, warmup=3)
    qps = nq / (t_search / 1000.0)
    return ("cuvs", f"{t_build:8.1f}", f"{t_search:8.3f}", f"{qps:11,.0f}",
            f"{recall_at_k(ids, brute, k):.4f}", "   -- ")


def run_one(label, M, nq, D, nlist, nprobe, k):
    title(f"IVF-Flat  {label}  (M={M:,}, nq={nq:,}, D={D}, nlist={nlist}, "
          f"nprobe={nprobe}, k={k})")

    Xc_np = _blobs(M, D, max(8, nlist // 8), seed=0)
    Xq_np = _blobs(nq, D, max(8, nlist // 8), seed=1)
    Xc_t = torch.tensor(Xc_np, device="cuda")
    Xq_t = torch.tensor(Xq_np, device="cuda")

    brute = _brute_ids(Xc_t, Xq_t, k)

    rows = []
    try:
        rows.append(_flashlib_row(label, Xc_t, Xq_t, M, nq, D, nlist, nprobe, k, brute))
    except Exception as e:  # pragma: no cover - bench-only
        rows.append(("flashlib", "  ERR", str(e).splitlines()[0][:24], "", "", ""))

    for name, fn in (("cuvs", _cuvs_row),):
        try:
            rows.append(fn(label, Xc_np, Xq_np, M, nq, D, nlist, nprobe, k, brute))
        except Exception as e:  # pragma: no cover - optional dep
            rows.append((name, "  SKIP", str(e).splitlines()[0][:24], "", "", ""))

    print(fmt_table(
        rows,
        ["engine", "build(ms)", "search(ms)", "QPS", "recall@k", "fineGB/s"],
    ))


def main():
    print(f"torch {torch.__version__}   GPU {torch.cuda.get_device_name(0)}")
    for s in SHAPES:
        run_one(*s)
    print()


if __name__ == "__main__":
    main()
