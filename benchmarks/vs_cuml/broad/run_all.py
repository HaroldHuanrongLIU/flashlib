"""Parallel dispatcher for the broad benchmark sweep.

Runs each per-primitive script on a free GPU, in parallel. Each
sub-process is given a single ``CUDA_VISIBLE_DEVICES`` so it sees
exactly one GPU. The job queue is FIFO: when a GPU frees up, the next
pending primitive is launched on it.

Usage::

    python -m benchmarks.vs_cuml.broad.run_all              # all prims
    python -m benchmarks.vs_cuml.broad.run_all --only kmeans,knn
    python -m benchmarks.vs_cuml.broad.run_all --gpus 4     # 4 GPUs
"""
from __future__ import annotations

import argparse
import os
import subprocess
import sys
import time
from pathlib import Path

_REPO = Path(__file__).resolve().parents[3]
RESULTS = _REPO / "benchmarks" / "results" / "broad"
RESULTS.mkdir(parents=True, exist_ok=True)
LOGS = RESULTS / "logs"
LOGS.mkdir(parents=True, exist_ok=True)

# Default order — long jobs first so they dominate wall, short jobs
# fill in around them.
#
# Excluded from the "vs cuML" broad sweep (no GPU-native cuML peer in
# cuml 25.10; including them with cupy / sklearn-CPU baselines would
# inflate headline numbers misleadingly):
#   * standard_scaler  — cuml re-exports sklearn.preprocessing.StandardScaler
#   * spectral_clustering — no cuml.cluster.SpectralClustering
# Both modules are preserved as benchmarks/vs_cuml/broad/<name>.py for
# future use if cuML adds peers, but excluded from this dispatcher.
ALL_PRIMS = [
    "tsne",                # tends to be longest per cell
    "random_forest",       # also long
    "umap",                # medium
    "logistic_regression",
    "ridge",
    "linear_regression",
    "truncated_svd",
    "pca",
    "kmeans",
    "knn",
    "hdbscan",
    "dbscan",
    "multinomial_nb",
]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--only", default="", help="comma list of prims")
    ap.add_argument("--gpus", type=int, default=8)
    args = ap.parse_args()

    if args.only:
        prims = [p.strip() for p in args.only.split(",") if p.strip()]
    else:
        prims = list(ALL_PRIMS)

    repo = str(_REPO)
    free_gpus: list[int] = list(range(args.gpus))
    running: dict[int, tuple] = {}   # pid -> (gpu, prim, t0)

    pending = list(prims)

    def launch(prim: str, gpu: int):
        log_path = LOGS / f"{prim}.log"
        env = os.environ.copy()
        env["CUDA_VISIBLE_DEVICES"] = str(gpu)
        cmd = [sys.executable, "-u", "-m",
                f"benchmarks.vs_cuml.broad.{prim}"]
        f = open(log_path, "w")
        f.write(f"## CUDA_VISIBLE_DEVICES={gpu}\n## cmd={' '.join(cmd)}\n\n")
        f.flush()
        p = subprocess.Popen(cmd, cwd=repo, env=env,
                              stdout=f, stderr=subprocess.STDOUT)
        return p, f

    procs: dict[int, tuple] = {}
    t_start = time.time()

    while pending or procs:
        # Launch as many as possible.
        while pending and free_gpus:
            gpu = free_gpus.pop(0)
            prim = pending.pop(0)
            p, f = launch(prim, gpu)
            procs[p.pid] = (p, f, gpu, prim, time.time())
            print(f"[disp] launch {prim} on GPU {gpu} (pid {p.pid})")

        # Wait for any process to finish (short poll loop).
        done_pids = []
        for pid, (p, f, gpu, prim, t0) in procs.items():
            ret = p.poll()
            if ret is not None:
                f.close()
                wall = time.time() - t0
                status = "OK" if ret == 0 else f"EXIT={ret}"
                print(f"[disp] DONE {prim} (GPU {gpu}, {wall:.1f}s, {status})")
                free_gpus.append(gpu)
                done_pids.append(pid)
        for pid in done_pids:
            del procs[pid]

        if not done_pids and procs:
            time.sleep(2.0)

    print(f"[disp] ALL DONE in {time.time()-t_start:.1f}s")


if __name__ == "__main__":
    main()
