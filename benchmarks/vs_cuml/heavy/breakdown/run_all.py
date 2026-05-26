"""Dispatch all per-component breakdown profilers in parallel across GPUs.

Each profiler is a tiny script (~10 s wall) so the bottleneck is sequential
serialisation of the heaviest one (KMeans at the hi-K row, ~6 s on H200).
Run all 14 in parallel and the whole sweep finishes in ~10 s wall on 8 GPUs.

Usage:
    python -m benchmarks.vs_cuml.heavy.breakdown.run_all
    python -m benchmarks.vs_cuml.heavy.breakdown.run_all --gpus 4

After all jobs complete, a small post-pass aggregates the per-primitive
``<prim>.md`` tables into ``benchmarks/results/heavy/breakdown/SUMMARY.md``
— one row per (primitive, dominant_stage, dominant_pct).
"""
from __future__ import annotations

import argparse
import re
import subprocess
import sys
import threading
import time
from pathlib import Path

BREAKDOWN_DIR = Path(__file__).resolve().parents[4] / "benchmarks" / "results" / "heavy" / "breakdown"
HEAVY_DIR = Path(__file__).resolve().parents[4] / "benchmarks" / "vs_cuml" / "heavy" / "breakdown"

# (name, est_seconds). KMeans is the long pole.
PRIMITIVES = [
    ("kmeans",              30),
    ("knn",                 10),
    ("dbscan",              10),
    ("hdbscan",              5),
    ("spectral_clustering",  5),
    ("pca",                  5),
    ("truncated_svd",       10),
    ("linear_regression",   15),
    ("ridge",               15),
    ("logistic_regression",  5),
    ("multinomial_nb",       5),
    ("standard_scaler",     10),
    ("random_forest",       30),
    ("tsne",                30),
    ("umap",                 5),
]


def _run_one(name: str, gpu_id: int) -> dict:
    log = BREAKDOWN_DIR / f"{name}.log"
    log.parent.mkdir(parents=True, exist_ok=True)
    cmd = [sys.executable, "-u", "-m", f"benchmarks.vs_cuml.heavy.breakdown.{name}"]
    t0 = time.time()
    with log.open("w") as f:
        f.write(f"=== breakdown:{name} on GPU{gpu_id} ===\n")
        f.write(f"=== cmd: {' '.join(cmd)} ===\n")
        f.flush()
        rc = subprocess.run(
            cmd, stdout=f, stderr=subprocess.STDOUT,
            env={**__import__("os").environ, "CUDA_VISIBLE_DEVICES": str(gpu_id)},
            cwd=str(Path(__file__).resolve().parents[4]),
        ).returncode
    return {"name": name, "rc": rc, "wall_s": time.time() - t0,
            "log": str(log), "gpu": gpu_id}


def _gpu_worker(gpu_id: int, jobs: list, results: list,
                lock: threading.Lock, t0: float):
    for name, _ in jobs:
        with lock:
            print(f"[breakdown] [START ] {name:22s} GPU{gpu_id}  (+{time.time()-t0:6.1f}s)")
        r = _run_one(name, gpu_id)
        results.append(r)
        with lock:
            status = "OK    " if r["rc"] == 0 else f"RC={r['rc']:<3}"
            print(f"[breakdown] [DONE {status}] {name:22s} GPU{gpu_id} "
                  f"wall={r['wall_s']:6.1f}s  (+{time.time()-t0:6.1f}s)")


def _build_per_gpu_queues(todo, n_gpus):
    queues: list[list] = [[] for _ in range(n_gpus)]
    loads = [0.0] * n_gpus
    for name, est in sorted(todo, key=lambda x: -x[1]):
        i = min(range(n_gpus), key=lambda j: loads[j])
        queues[i].append((name, est))
        loads[i] += est
    return queues, loads


def _parse_multi_shape_md(path: Path) -> dict | None:
    """Parse a multi-shape breakdown .md and return:
        {
          "shape_labels": [s1, s2, s3],
          "outer_walls":  [w1, w2, w3],
          "stages": {stage_name: [(ms, pct), (ms, pct), ...]},
        }
    Returns None if the file isn't a multi-shape table.
    """
    if not path.exists():
        return None
    text = path.read_text()
    lines = text.split("\n")

    # Find the table header line: starts with "| component |"
    hdr_idx = None
    for i, ln in enumerate(lines):
        if ln.strip().startswith("| component |"):
            hdr_idx = i
            break
    if hdr_idx is None:
        return None
    header_cells = [c.strip() for c in lines[hdr_idx].strip("|").split("|")]
    shape_labels = [c for c in header_cells[1:]]
    sep_idx = hdr_idx + 1
    if sep_idx >= len(lines) or not lines[sep_idx].strip().startswith("|"):
        return None

    stages: dict[str, list[tuple[float, float]]] = {}
    outer_walls = [0.0] * len(shape_labels)

    cell_re = re.compile(r"([\d\.]+)\s*ms\s*\(([\d\.]+)%\)")
    outer_re = re.compile(r"\*\*\s*([\d\.]+)\s*ms\s*\*\*")

    i = sep_idx + 1
    while i < len(lines):
        ln = lines[i].strip()
        if not ln.startswith("|"):
            break
        cells = [c.strip() for c in ln.strip("|").split("|")]
        stage_name = cells[0].strip("` *")
        if stage_name == "outer wall" or stage_name.lower() == "outer wall":
            for j, c in enumerate(cells[1:]):
                m = outer_re.search(c)
                if m and j < len(outer_walls):
                    outer_walls[j] = float(m.group(1))
            i += 1
            continue
        vals = []
        for c in cells[1:]:
            m = cell_re.search(c)
            if m:
                vals.append((float(m.group(1)), float(m.group(2))))
            else:
                vals.append((0.0, 0.0))
        stages[stage_name] = vals
        i += 1
    return {"shape_labels": shape_labels, "outer_walls": outer_walls,
            "stages": stages}


def _aggregate():
    """Read all per-prim multi-shape breakdown .md files; produce SUMMARY.md."""
    md = ["# heavy/breakdown — SUMMARY (multi-shape sweeps)", "",
          f"Generated: {time.strftime('%Y-%m-%d %H:%M:%S')}", "",
          "Each primitive's full per-shape × per-component table lives at "
          "`benchmarks/results/heavy/breakdown/<prim>.md`. The summary below "
          "reports, for every shape, the dominant single component and what "
          "% of the wall it consumed — so the **trend of the dominant** "
          "across the workload axis is visible at a glance.", "",
          "| primitive | swept axis | shape | outer wall | dominant component | dom % |",
          "| --- | --- | --- | --- | --- | --- |"]

    sources: list[tuple[str, str, Path]] = []
    for prim, _ in PRIMITIVES:
        if prim == "multinomial_nb":
            sources.append(("multinomial_nb (fit)", "(N,V,C)",
                            BREAKDOWN_DIR / "multinomial_nb_fit.md"))
            sources.append(("multinomial_nb (predict)", "(N,V,C)",
                            BREAKDOWN_DIR / "multinomial_nb_predict.md"))
        elif prim == "random_forest":
            sources.append(("random_forest (fit)", "(trees,depth)",
                            BREAKDOWN_DIR / "random_forest.md"))
            sources.append(("random_forest (predict)", "(trees,depth)",
                            BREAKDOWN_DIR / "random_forest_predict.md"))
        else:
            axis = {
                "kmeans":               "K (n_clusters)",
                "knn":                  "D (dim)",
                "dbscan":               "D (algo path)",
                "hdbscan":              "N",
                "spectral_clustering":  "N",
                "pca":                  "aspect (tall/sq/wide)",
                "truncated_svd":        "(N,D,K)",
                "linear_regression":    "D",
                "ridge":                "T (multi-target)",
                "logistic_regression":  "D",
                "standard_scaler":      "aspect",
                "tsne":                 "N",
                "umap":                 "(N,n_epochs)",
            }.get(prim, "?")
            sources.append((prim, axis, BREAKDOWN_DIR / f"{prim}.md"))

    for disp, axis, path in sources:
        parsed = _parse_multi_shape_md(path)
        if parsed is None:
            md.append(f"| {disp} | {axis} | — | MISSING | — | — |")
            continue
        for j, shape in enumerate(parsed["shape_labels"]):
            outer = parsed["outer_walls"][j]
            # Find the dominant stage (highest pct) for this shape
            best_stage, best_pct = None, -1.0
            for stage, vals in parsed["stages"].items():
                if j >= len(vals):
                    continue
                ms, pct = vals[j]
                if pct > best_pct:
                    best_pct = pct
                    best_stage = stage
            prim_col = disp if j == 0 else ""
            axis_col = axis if j == 0 else ""
            md.append(f"| {prim_col} | {axis_col} | `{shape}` | "
                      f"{outer:.2f} ms | `{best_stage}` | {best_pct:.1f}% |")

    md.append("")
    md.append("## How to read this")
    md.append("")
    md.append("Three regimes for the workload-sensitivity story:")
    md.append("")
    md.append("1. **Stable dominant**: the same component dominates at every shape "
              "(e.g. `kmeans:assign`, `knn:main_knn`, `random_forest:tree_traverse`). "
              "Future kernel work should target that component first.")
    md.append("2. **Crossover**: dominance flips across shapes "
              "(e.g. `pca:cov_or_gram_gemm` at primal shapes vs `eigh_or_halko` at "
              "wide+Halko; `multinomial_nb:onehot_gemm` shrinks as `partial_reduce` "
              "grows). Future optimization must address the regime that matters for "
              "the user's shape.")
    md.append("3. **Balanced**: no single component is >40 % at any shape "
              "(e.g. `spectral_clustering`, `umap`). Each sub-stage in §2 is on "
              "the critical path; aggregate kernel-launch reduction (cudagraphs / "
              "fusion across stages) yields a bigger win than tuning any one kernel.")

    (BREAKDOWN_DIR / "SUMMARY.md").write_text("\n".join(md) + "\n")
    print(f"[breakdown] wrote {BREAKDOWN_DIR / 'SUMMARY.md'}")


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
    print(f"[breakdown] {sum(len(q) for q in queues)} jobs across "
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
    print(f"[breakdown] all jobs done in {time.time()-t0:.1f}s")

    _aggregate()


if __name__ == "__main__":
    main()
