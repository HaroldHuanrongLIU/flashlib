"""Parallel dispatcher for the heavy stress sweep.

Per-GPU job queues with a single worker per GPU. Each primitive runs
in its own subprocess pinned to one GPU via ``CUDA_VISIBLE_DEVICES``.
Heavy-RAM primitives (KMeans 100M+, KNN search-hugeM, RF 1M-trees=200,
LinearRegression at huge N*D) are placed first in the GPU's queue so
short jobs can co-locate on remaining GPUs.

After all jobs complete, a small post-pass aggregates the per-primitive
``<prim>.json`` outputs into ``benchmarks/results/heavy/SUMMARY.md`` +
``SUMMARY.json``.

Usage:
    python -m benchmarks.vs_cuml.heavy.run_all_parallel
    python -m benchmarks.vs_cuml.heavy.run_all_parallel --gpus 8 --only kmeans,knn

Total wall time at ``--gpus 8`` is ~45-60 min for the full sweep.
"""
from __future__ import annotations

import argparse
import json
import os
import queue
import subprocess
import sys
import threading
import time
from pathlib import Path

PRIMITIVES = [
    # (name, est_seconds_per_run, requires_own_gpu)
    # The booleans + sort order let lighter jobs co-locate while the
    # HBM hogs each take a full GPU.
    ("standard_scaler",      120, False),
    ("multinomial_nb",       300, False),
    ("spectral_clustering",  300, False),
    ("hdbscan",              420, False),
    ("dbscan",               480, False),
    ("tsne",                 540, False),
    ("logistic_regression",  720, False),
    ("ridge",                720, False),
    ("truncated_svd",        720, False),
    ("pca",                  720, False),
    ("umap",                 900, False),
    ("random_forest",       1200, True),
    ("linear_regression",   1200, True),
    ("knn",                 1500, True),
    ("kmeans",              1800, True),
]

REPO = Path(__file__).resolve().parents[3]
RESULTS = REPO / "benchmarks" / "results" / "heavy"
LOGS = RESULTS / "logs"


def _detect_gpus(requested: int | None) -> int:
    try:
        out = subprocess.check_output(
            ["nvidia-smi", "--list-gpus"], text=True, stderr=subprocess.DEVNULL,
        )
        n = len([ln for ln in out.splitlines() if ln.strip()])
    except (FileNotFoundError, subprocess.CalledProcessError):
        n = 1
    if requested is None:
        return max(1, n)
    return min(n, requested)


def _run_one(name: str, gpu_id: int) -> dict:
    LOGS.mkdir(parents=True, exist_ok=True)
    log_path = LOGS / f"{name}.log"
    cmd = [sys.executable, "-u", "-m", f"benchmarks.vs_cuml.heavy.{name}"]
    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
    env["PYTHONUNBUFFERED"] = "1"
    # Avoid Triton autotune cache cross-contamination across primitives:
    # each gets its own TRITON_CACHE_DIR.
    env["TRITON_CACHE_DIR"] = str(REPO / ".triton_cache" / name)
    # Cap CPU thread sprawl from numpy / sklearn / openblas inside the
    # heavy procs.
    for v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS",
              "MKL_NUM_THREADS", "BLIS_NUM_THREADS",
              "NUMEXPR_MAX_THREADS"):
        env.setdefault(v, "8")
    t0 = time.time()
    with open(log_path, "w") as f:
        f.write(f"=== {name} on GPU {gpu_id} ===\n")
        f.write(f"=== cmd: {' '.join(cmd)} ===\n")
        f.write(f"=== start: {time.strftime('%Y-%m-%d %H:%M:%S')} ===\n")
        f.flush()
        rc = subprocess.call(cmd, cwd=str(REPO), env=env,
                              stdout=f, stderr=subprocess.STDOUT)
        f.write(f"=== end: {time.strftime('%Y-%m-%d %H:%M:%S')}, "
                f"rc={rc}, elapsed={time.time()-t0:.1f}s ===\n")
    return {"name": name, "gpu": gpu_id, "rc": rc,
            "wall_s": time.time() - t0, "log": str(log_path)}


def _aggregate():
    """Read all per-prim json + md, produce SUMMARY.md / SUMMARY.json."""
    out = {"primitives": {}, "ts": time.time()}
    for prim, _, _ in PRIMITIVES:
        jpath = RESULTS / f"{prim}.json"
        if not jpath.exists():
            out["primitives"][prim] = {"status": "MISSING", "rows": []}
            continue
        try:
            data = json.loads(jpath.read_text())
        except json.JSONDecodeError:
            out["primitives"][prim] = {"status": "CORRUPT JSON", "rows": []}
            continue
        rows = [d.get("row", {}) for d in data]
        n_fail = sum(1 for r in rows if str(r.get("gate", "")).startswith("FAIL"))
        n_skip = sum(1 for r in rows if str(r.get("gate", "")).startswith("SKIP"))
        n_pass = sum(1 for r in rows if str(r.get("gate", "")) == "PASS"
                     or str(r.get("gate", "")).startswith("PASS"))
        n = len(rows)
        out["primitives"][prim] = {
            "status": "OK", "rows": rows, "n_total": n,
            "n_pass": n_pass, "n_fail": n_fail, "n_skip": n_skip,
        }

    # Markdown summary.
    md = ["# flashlib release-candidate heavy stress — SUMMARY",
          "",
          f"Generated: {time.strftime('%Y-%m-%d %H:%M:%S')}",
          "",
          "| primitive | rows | PASS | FAIL | SKIP | headline (best vs cuML) |",
          "| --- | --- | --- | --- | --- | --- |"]
    for prim, _, _ in PRIMITIVES:
        info = out["primitives"][prim]
        if info["status"] != "OK":
            md.append(f"| {prim} | (no output) | - | - | - | {info['status']} |")
            continue
        # Headline = best flashlib row's vs_cuml number.
        best_vs = ""
        best_x = -1.0
        for r in info["rows"]:
            if str(r.get("engine", "")).startswith("flashlib"):
                vs = str(r.get("vs_cuml", ""))
                if vs.endswith("x"):
                    try:
                        v = float(vs[:-1])
                        if v > best_x:
                            best_x = v
                            best_vs = vs
                    except ValueError:
                        pass
        md.append(f"| {prim} | {info['n_total']} | {info['n_pass']} | "
                  f"{info['n_fail']} | {info['n_skip']} | {best_vs} |")
    md.append("")
    md.append("Detailed per-row tables: see "
              "`benchmarks/results/heavy/<primitive>.md`.")
    md.append("")
    md.append("## Audit reading guide")
    md.append("")
    md.append("Every row carries a `gate`:")
    md.append("- **PASS** — correctness metric crossed its published tier.")
    md.append("- **FAIL** — flag for human review; either the kernel "
              "broke or the gate is too strict for this shape.")
    md.append("- **SKIP** — recorded gap (e.g. multinomial LR not in "
              "scope; bf16 unsafe at this conditioning).")

    (RESULTS / "SUMMARY.md").write_text("\n".join(md) + "\n")
    (RESULTS / "SUMMARY.json").write_text(json.dumps(out, indent=2,
                                                       default=str))


def _build_per_gpu_queues(todo, n_gpus):
    """Pin heavy jobs to dedicated GPUs; LPT-pack the rest.

    Returns a list of per-GPU job lists. The longest expected job goes
    onto the GPU with the smallest total estimated runtime so far —
    a longest-processing-time-first packing that keeps the wall
    balanced.
    """
    dedicated = [t for t in todo if t[2]]
    shared = [t for t in todo if not t[2]]
    dedicated.sort(key=lambda x: -x[1])
    shared.sort(key=lambda x: -x[1])

    queues: list[list[tuple[str, int]]] = [[] for _ in range(n_gpus)]
    loads = [0] * n_gpus

    # Pin dedicated first — each gets its own GPU.
    for i, t in enumerate(dedicated[:n_gpus]):
        queues[i].append((t[0], t[1]))
        loads[i] += t[1]
    # Any dedicated overflow + all shared get LPT-packed.
    for t in dedicated[n_gpus:] + shared:
        gpu = min(range(n_gpus), key=lambda g: loads[g])
        queues[gpu].append((t[0], t[1]))
        loads[gpu] += t[1]

    return queues, loads


def _gpu_worker(gpu_id: int, jobs: list[tuple[str, int]], results: list,
                lock: threading.Lock, t0: float):
    """Run one job at a time on the given GPU."""
    for name, _ in jobs:
        with lock:
            print(f"[runner] [START ] {name:22s} GPU{gpu_id}  "
                  f"(+{time.time()-t0:6.1f}s)")
        r = _run_one(name, gpu_id)
        results.append(r)
        with lock:
            status = "OK    " if r["rc"] == 0 else f"RC={r['rc']:<3}"
            print(f"[runner] [DONE {status}] {name:22s} GPU{gpu_id} "
                  f"wall={r['wall_s']:7.1f}s  (+{time.time()-t0:6.1f}s)  "
                  f"-> {r['log']}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--gpus", type=int, default=None,
                    help="num GPUs to use (default: all visible)")
    ap.add_argument("--only", type=str, default=None,
                    help="comma-separated list of primitive names")
    ap.add_argument("--skip", type=str, default=None,
                    help="comma-separated list of primitive names to skip")
    ap.add_argument("--summary-only", action="store_true",
                    help="skip the run, just aggregate existing outputs")
    args = ap.parse_args()

    if args.summary_only:
        _aggregate()
        print(f"Wrote {RESULTS / 'SUMMARY.md'}")
        return

    n_gpus = _detect_gpus(args.gpus)
    print(f"[runner] detected {n_gpus} GPUs")
    LOGS.mkdir(parents=True, exist_ok=True)

    todo = list(PRIMITIVES)
    if args.only:
        keep = set(s.strip() for s in args.only.split(","))
        todo = [t for t in todo if t[0] in keep]
    if args.skip:
        drop = set(s.strip() for s in args.skip.split(","))
        todo = [t for t in todo if t[0] not in drop]
    if not todo:
        print("[runner] nothing to do")
        return

    print(f"[runner] dispatching {len(todo)} primitives to {n_gpus} GPUs")

    queues, loads = _build_per_gpu_queues(todo, n_gpus)
    print(f"[runner] per-GPU schedule (est wall):")
    for g, q in enumerate(queues):
        names = ", ".join(name for name, _ in q)
        print(f"  GPU{g}  est={loads[g]:5d}s  [{names}]")

    t0 = time.time()
    results: list = []
    print_lock = threading.Lock()
    workers = []
    for g, q in enumerate(queues):
        if not q:
            continue
        t = threading.Thread(target=_gpu_worker,
                              args=(g, q, results, print_lock, t0),
                              daemon=False)
        t.start()
        workers.append(t)
    for t in workers:
        t.join()

    print(f"[runner] all done in {time.time()-t0:.1f}s wall")
    _aggregate()
    print(f"[runner] wrote {RESULTS / 'SUMMARY.md'}")


if __name__ == "__main__":
    main()
