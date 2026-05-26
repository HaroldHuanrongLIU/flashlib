"""Fan-out wrapper for ``benchmarks.tune.knn`` -- distributes the shape
grid across multiple GPUs via ``CUDA_VISIBLE_DEVICES``.

Each GPU runs a sequential subprocess over its assigned subset of the
``knn.WORKLOADS`` grid. Per-shape JSONL files are written to the standard
results location, so when all 8 workers finish you can run the derive
script unchanged::

    python -m benchmarks.tune.knn_parallel
    python -m benchmarks.tune.derive.knn

Why a parallel runner: ``cutedsl/build_fa3`` first-call autotune is
multi-minute per shape (compiles ~24 candidates of ~10s each). A 144-
shape grid would take ~10 hours sequentially but ~75 min spread across
8 H200s.

Usage::

    python -m benchmarks.tune.knn_parallel
    python -m benchmarks.tune.knn_parallel --gpus 4
    python -m benchmarks.tune.knn_parallel --rerun --gpus 8
"""
from __future__ import annotations

import argparse
import os
import subprocess
import sys
import time
from pathlib import Path

# Import the workload list directly so we partition the SAME grid the
# child script will iterate over.
from benchmarks.tune.knn import WORKLOADS
from benchmarks.tune._common import shape_key


def partition(items, n):
    """Round-robin split -- keeps lighter / heavier shapes mixed across
    workers so every GPU finishes around the same wall time."""
    return [items[i::n] for i in range(n)]


def main() -> None:
    ap = argparse.ArgumentParser(prog="benchmarks.tune.knn_parallel")
    ap.add_argument("--gpus", type=int, default=8,
                    help="number of CUDA devices to use (default 8)")
    ap.add_argument("--rerun", action="store_true",
                    help="forward to child tuners (overwrite existing JSONL)")
    args = ap.parse_args()

    keys = [shape_key(w) for w in WORKLOADS]
    print(f"[knn] total shapes: {len(keys)}")
    buckets = partition(keys, args.gpus)
    for i, b in enumerate(buckets):
        print(f"  GPU {i}: {len(b)} shapes -- {b[:3]}{'...' if len(b)>3 else ''}")

    procs = []
    log_dir = Path("/tmp") / f"knn_logs_{int(time.time())}"
    log_dir.mkdir(parents=True, exist_ok=True)
    print(f"[knn] worker logs in {log_dir}")

    t0 = time.time()
    for gpu_id, bucket in enumerate(buckets):
        if not bucket:
            continue
        env = os.environ.copy()
        env["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
        cmd = [sys.executable, "-u", "-m", "benchmarks.tune.knn",
               "--size", ",".join(bucket)]
        if args.rerun:
            cmd.append("--rerun")
        log_path = log_dir / f"gpu{gpu_id}.log"
        log_f = log_path.open("w")
        # Stream stdout+stderr to a per-GPU log; never block on the parent.
        p = subprocess.Popen(cmd, env=env, stdout=log_f, stderr=subprocess.STDOUT)
        procs.append((gpu_id, p, log_f, log_path))
        print(f"  -> launched GPU {gpu_id} (pid {p.pid}), log {log_path}")

    failures = []
    for gpu_id, p, log_f, log_path in procs:
        rc = p.wait()
        log_f.close()
        elapsed = time.time() - t0
        status = "OK" if rc == 0 else f"FAIL rc={rc}"
        print(f"[{elapsed/60:5.1f} min] GPU {gpu_id} done [{status}] "
              f"-- log {log_path}")
        if rc != 0:
            failures.append((gpu_id, log_path))

    print(f"\n[knn] all done in {(time.time()-t0)/60:.1f} min")
    if failures:
        print(f"  {len(failures)} GPU worker(s) failed -- inspect logs:")
        for gpu_id, p in failures:
            print(f"    GPU {gpu_id}: {p}")


if __name__ == "__main__":
    main()
