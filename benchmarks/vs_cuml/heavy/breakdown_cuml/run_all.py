"""Dispatch all cuML kernel-trace profilers in parallel across GPUs.

Same shape as ``benchmarks/vs_cuml/heavy/breakdown/run_all.py`` — per-GPU
queue, longest jobs first, each job runs the matching cuML profiler in a
subprocess pinned to one GPU. After all jobs finish, an aggregator pass
produces ``benchmarks/results/heavy/breakdown_cuml/SUMMARY.md`` showing
each primitive's dominant cuML kernel per shape.

Usage:
    python -m benchmarks.vs_cuml.heavy.breakdown_cuml.run_all
    python -m benchmarks.vs_cuml.heavy.breakdown_cuml.run_all --gpus 4
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import threading
import time
from pathlib import Path

RESULTS = Path(__file__).resolve().parents[4] / "benchmarks" / "results" / "heavy" / "breakdown_cuml"
LOG_DIR = RESULTS  # logs live next to the .md files
RESULTS.mkdir(parents=True, exist_ok=True)

# (name, est_seconds). multinomial_nb and random_forest produce *_fit and
# *_predict variants in their own script — represented as a single job.
PRIMITIVES = [
    ("kmeans",              420),  # cuML kmeans K=100K @ N=20M is the long pole
    ("knn",                  60),
    ("dbscan",              180),  # cuML dbscan at N=2M D=2 is slow
    ("hdbscan",             120),
    ("pca",                 180),
    ("truncated_svd",       180),
    ("linear_regression",    60),
    ("ridge",                60),
    ("logistic_regression",  60),
    ("multinomial_nb",       60),
    ("standard_scaler",      90),
    ("random_forest",       300),
    ("tsne",                300),  # BH per-iter has many small kernels; 500 iters
    ("umap",                240),
]


def _run_one(name: str, gpu_id: int) -> dict:
    log = LOG_DIR / f"{name}.log"
    cmd = [sys.executable, "-u", "-m",
            f"benchmarks.vs_cuml.heavy.breakdown_cuml.{name}"]
    t0 = time.time()
    with log.open("w") as f:
        f.write(f"=== breakdown_cuml:{name} on GPU{gpu_id} ===\n")
        f.write(f"=== cmd: {' '.join(cmd)} ===\n")
        f.flush()
        rc = subprocess.run(
            cmd, stdout=f, stderr=subprocess.STDOUT,
            env={**os.environ, "CUDA_VISIBLE_DEVICES": str(gpu_id)},
            cwd=str(Path(__file__).resolve().parents[4]),
        ).returncode
    return {"name": name, "rc": rc, "wall_s": time.time() - t0,
            "log": str(log), "gpu": gpu_id}


def _gpu_worker(gpu_id: int, jobs: list, results: list,
                 lock: threading.Lock, t0: float):
    for name, _ in jobs:
        with lock:
            print(f"[breakdown_cuml] [START ] {name:22s} GPU{gpu_id}  "
                  f"(+{time.time()-t0:6.1f}s)")
        r = _run_one(name, gpu_id)
        results.append(r)
        with lock:
            status = "OK    " if r["rc"] == 0 else f"RC={r['rc']:<3}"
            print(f"[breakdown_cuml] [DONE {status}] {name:22s} GPU{gpu_id} "
                  f"wall={r['wall_s']:6.1f}s  (+{time.time()-t0:6.1f}s)")


def _build_per_gpu_queues(todo, n_gpus):
    queues: list[list] = [[] for _ in range(n_gpus)]
    loads = [0.0] * n_gpus
    for name, est in sorted(todo, key=lambda x: -x[1]):
        i = min(range(n_gpus), key=lambda j: loads[j])
        queues[i].append((name, est))
        loads[i] += est
    return queues, loads


def _aggregate():
    """Read every <prim>.json (or *_fit.json / *_predict.json for the split
    primitives), pick the top kernel per shape, produce a SUMMARY table:
        | primitive | shape | outer wall | top cuML kernel | kernel pct |
    """
    md = ["# heavy/breakdown_cuml — SUMMARY (top cuML kernel per shape)", "",
          f"Generated: {time.strftime('%Y-%m-%d %H:%M:%S')}", "",
          "Per-primitive full per-kernel tables live in "
          "`benchmarks/results/heavy/breakdown_cuml/<prim>.md`. The summary "
          "below picks the single dominant CUDA kernel per shape — paired "
          "with the corresponding flashlib stage from "
          "`benchmarks/results/heavy/breakdown/SUMMARY.md`, this is the "
          "one-row 'why flashlib beats cuML at this shape' answer.", "",
          "| primitive | shape | cuML outer wall | top cuML kernel | top kernel pct |",
          "| --- | --- | --- | --- | --- |"]

    # Build the list of expected JSON files. Primitives that produce two
    # files (multinomial_nb, random_forest) report each separately.
    expected = []
    for prim, _ in PRIMITIVES:
        if prim == "multinomial_nb":
            expected.append(("multinomial_nb (fit)", "multinomial_nb_fit"))
            expected.append(("multinomial_nb (predict)", "multinomial_nb_predict"))
        elif prim == "random_forest":
            expected.append(("random_forest (fit)", "random_forest_fit"))
            expected.append(("random_forest (predict)", "random_forest_predict"))
        else:
            expected.append((prim, prim))

    for disp, file_stem in expected:
        p = RESULTS / f"{file_stem}.json"
        if not p.exists():
            md.append(f"| {disp} | — | MISSING `{file_stem}.json` | — | — |")
            continue
        try:
            data = json.loads(p.read_text())
        except Exception as e:
            md.append(f"| {disp} | — | PARSE_ERROR ({e}) | — | — |")
            continue
        for i, sr in enumerate(data.get("shapes", [])):
            top = sr["kernels"][0] if sr["kernels"] else None
            shape = sr["label"]
            outer = sr.get("outer_wall_ms")
            outer_str = f"{outer:.2f} ms" if isinstance(outer, (int, float)) \
                                                 and outer == outer else "NaN"
            top_name = f"`{top['kernel']}`" if top else "—"
            top_pct = f"{top['pct_of_total']:.1f}%" if top else "—"
            prim_col = disp if i == 0 else ""
            md.append(f"| {prim_col} | `{shape}` | {outer_str} | "
                      f"{top_name} | {top_pct} |")

    md.append("")
    md.append("## How to read this")
    md.append("")
    md.append("- The **top cuML kernel** here is the single CUDA kernel "
              "with the largest sum of device time within one cuML call. "
              "Look up its full per-shape table at "
              "`benchmarks/results/heavy/breakdown_cuml/<prim>.md` for "
              "the kernels ranked 2-12.")
    md.append("- To compare apples-to-apples vs flashlib, open "
              "`benchmarks/results/heavy/breakdown/<prim>.md` (the "
              "flashlib stage table) and read the matching shape.")
    md.append("- The two tables together answer **why** the flashlib "
              "kernel is faster than the cuML kernel for that shape.")

    (RESULTS / "SUMMARY.md").write_text("\n".join(md) + "\n")
    print(f"[breakdown_cuml] wrote {RESULTS / 'SUMMARY.md'}")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--gpus", type=int, default=8)
    p.add_argument("--only", type=str, default="",
                   help="comma-separated subset of primitive names")
    p.add_argument("--aggregate-only", action="store_true")
    args = p.parse_args()

    if args.aggregate_only:
        _aggregate()
        return

    todo = PRIMITIVES
    if args.only:
        wanted = set(args.only.split(","))
        todo = [p for p in PRIMITIVES if p[0] in wanted]

    queues, loads = _build_per_gpu_queues(todo, args.gpus)
    print(f"[breakdown_cuml] {sum(len(q) for q in queues)} jobs across "
          f"{args.gpus} GPUs (estimated max queue: {max(loads):.0f}s)")

    results: list = []
    lock = threading.Lock()
    t0 = time.time()
    threads = []
    for i, jobs in enumerate(queues):
        if not jobs:
            continue
        t = threading.Thread(target=_gpu_worker,
                              args=(i, jobs, results, lock, t0))
        t.start()
        threads.append(t)
    for t in threads:
        t.join()
    print(f"[breakdown_cuml] all jobs done in {time.time()-t0:.1f}s")

    _aggregate()


if __name__ == "__main__":
    main()
