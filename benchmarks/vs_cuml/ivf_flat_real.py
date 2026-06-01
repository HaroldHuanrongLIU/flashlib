"""IVF-Flat on REAL datasets (SIFT1M, GIST1M): flashlib vs cuVS, iso-recall.

Synthetic blobs have clean, balanced clusters that flatter IVF. This
script instead uses the canonical TexMex ANN benchmarks with their
*exact provided* ground truth, and sweeps ``nprobe`` to trace the
**recall@10 vs QPS** curve for both engines -- the honest
apples-to-apples comparison.

  * SIFT1M -- 1M x 128 base, 10k queries (medium dim).
  * GIST1M -- 1M x 960 base,  1k queries (high dim; exercises the
    GEMM kernel's D-split path).

Datasets auto-download to ``~/.cache/flashlib_datasets`` if absent
(SIFT ~168 MB, GIST ~2.7 GB) from::

    ftp://ftp.irisa.fr/local/texmex/corpus/{sift,gist}.tar.gz

Run (all present datasets, or pick by name)::

    python -m benchmarks.vs_cuml.ivf_flat_real
    python -m benchmarks.vs_cuml.ivf_flat_real gist
"""
from __future__ import annotations

import os
import sys
import subprocess
import numpy as np
import torch

from benchmarks.vs_cuml._common import time_gpu, title, fmt_table, recall_at_k


_CACHE = os.path.expanduser("~/.cache/flashlib_datasets")
NPROBES = [8, 16, 32, 64, 128]
K = 10

DATASETS = {
    "sift": dict(
        url="ftp://ftp.irisa.fr/local/texmex/corpus/sift.tar.gz",
        d="sift", base="sift_base.fvecs", query="sift_query.fvecs",
        gt="sift_groundtruth.ivecs", nlist=1024,
    ),
    "gist": dict(
        url="ftp://ftp.irisa.fr/local/texmex/corpus/gist.tar.gz",
        d="gist", base="gist_base.fvecs", query="gist_query.fvecs",
        gt="gist_groundtruth.ivecs", nlist=1024,
    ),
}


def fvecs_read(path: str) -> np.ndarray:
    a = np.fromfile(path, dtype=np.int32)
    d = int(a[0])
    return a.reshape(-1, d + 1)[:, 1:].copy().view(np.float32)


def ivecs_read(path: str) -> np.ndarray:
    a = np.fromfile(path, dtype=np.int32)
    d = int(a[0])
    return a.reshape(-1, d + 1)[:, 1:].copy()


def ensure(spec: dict) -> str:
    root = os.path.join(_CACHE, spec["d"])
    if os.path.exists(os.path.join(root, spec["base"])):
        return root
    os.makedirs(_CACHE, exist_ok=True)
    tgz = os.path.join(_CACHE, f"{spec['d']}.tar.gz")
    if not os.path.exists(tgz):
        print(f"downloading {spec['d']} -> {tgz} ...")
        subprocess.check_call(["curl", "-s", "-o", tgz, spec["url"]])
    print(f"extracting {spec['d']} ...")
    subprocess.check_call(["tar", "xzf", tgz, "-C", _CACHE])
    return root


def _flashlib_rows(Xc_t, Xq_t, gt, nq, nlist):
    from flashlib import flash_ivf_flat_build, flash_ivf_flat_search

    t_build = time_gpu(lambda: flash_ivf_flat_build(Xc_t, nlist, niter=20, seed=0),
                       repeat=1, warmup=0)
    index = flash_ivf_flat_build(Xc_t, nlist, niter=20, seed=0)
    rows = []
    for nprobe in NPROBES:
        ids = flash_ivf_flat_search(index, Xq_t, K, nprobe=nprobe)[1].cpu().numpy()
        t = time_gpu(lambda: flash_ivf_flat_search(index, Xq_t, K, nprobe=nprobe),
                     repeat=10, warmup=3)
        rows.append(("flashlib", str(nprobe), f"{t_build:7.0f}", f"{t:8.3f}",
                     f"{nq / (t / 1000.0):11,.0f}", f"{recall_at_k(ids, gt, K):.4f}"))
    return rows


def _cuvs_rows(Xc_np, Xq_np, gt, nq, nlist):
    import cupy as cp
    from cuvs.neighbors import ivf_flat

    Xc = cp.asarray(Xc_np); Xq = cp.asarray(Xq_np)
    ip = ivf_flat.IndexParams(n_lists=nlist, metric="sqeuclidean")
    t_build = time_gpu(lambda: ivf_flat.build(ip, Xc), repeat=1, warmup=0)
    index = ivf_flat.build(ip, Xc)
    rows = []
    for nprobe in NPROBES:
        sp = ivf_flat.SearchParams(n_probes=nprobe)
        _, I = ivf_flat.search(sp, index, Xq, K)
        ids = cp.asarray(I).get()
        t = time_gpu(lambda: ivf_flat.search(sp, index, Xq, K), repeat=10, warmup=3)
        rows.append(("cuvs", str(nprobe), f"{t_build:7.0f}", f"{t:8.3f}",
                     f"{nq / (t / 1000.0):11,.0f}", f"{recall_at_k(ids, gt, K):.4f}"))
    return rows


def run_one(name: str, spec: dict):
    root = ensure(spec)
    Xc_np = np.ascontiguousarray(fvecs_read(os.path.join(root, spec["base"])))
    Xq_np = np.ascontiguousarray(fvecs_read(os.path.join(root, spec["query"])))
    gt = ivecs_read(os.path.join(root, spec["gt"]))[:, :K]
    M, D = Xc_np.shape
    nq = Xq_np.shape[0]
    nlist = spec["nlist"]

    title(f"{name.upper()}1M  (M={M:,}, nq={nq:,}, D={D}, nlist={nlist}, k={K}; "
          f"recall@{K} vs exact provided ground truth)")

    Xc_t = torch.tensor(Xc_np, device="cuda")
    Xq_t = torch.tensor(Xq_np, device="cuda")

    rows = []
    try:
        rows += _flashlib_rows(Xc_t, Xq_t, gt, nq, nlist)
    except Exception as e:  # pragma: no cover - bench-only
        rows.append(("flashlib", "-", "  ERR", str(e).splitlines()[0][:20], "", ""))
    del Xc_t, Xq_t
    torch.cuda.empty_cache()
    try:
        rows += _cuvs_rows(Xc_np, Xq_np, gt, nq, nlist)
    except Exception as e:  # pragma: no cover - optional dep
        rows.append(("cuvs", "-", " SKIP", str(e).splitlines()[0][:20], "", ""))

    rows.sort(key=lambda r: (int(r[1]) if r[1].isdigit() else 0, r[0]))
    print(fmt_table(rows, ["engine", "nprobe", "build(ms)", "search(ms)", "QPS", "recall@k"]))


def main():
    print(f"torch {torch.__version__}   GPU {torch.cuda.get_device_name(0)}")
    pick = [a for a in sys.argv[1:] if a in DATASETS]
    names = pick or list(DATASETS)
    for name in names:
        spec = DATASETS[name]
        if pick or os.path.exists(os.path.join(_CACHE, spec["d"], spec["base"])):
            run_one(name, spec)
        else:
            print(f"\n[skip {name}: not downloaded; pass '{name}' to auto-fetch]")
    print()


if __name__ == "__main__":
    main()
