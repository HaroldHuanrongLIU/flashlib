"""Shared helpers for the broad workload sweep.

A "broad" row is the minimal record needed to plot speedup vs workload
axes — no correctness gate, no apples-to-apples dtype audit (those are
the heavy/ suite's job). Just:

    {primitive, label, axes:{N,D,K,...}, flashlib_ms, cuml_ms,
     speedup, dtype, ok, notes}

The per-primitive script is responsible for choosing its grid and a
single comparable dtype pair (fp32-vs-fp32 by default for matched
precision). For primitives whose default flashlib backend is bf16-only
(e.g. CuteDSL-FA3 KNN) the script picks fp32 for both sides.
"""
from __future__ import annotations

import gc
import json
import os
import time
import traceback
from pathlib import Path
from typing import Any, Callable

# Re-export the cap_threads/cuml_shim shims so per-primitive broad
# scripts can do a single import.
from benchmarks.vs_cuml._common import (  # noqa: F401
    cap_threads,
    cuml_shim,
)

_REPO = Path(__file__).resolve().parents[3]
RESULTS = _REPO / "benchmarks" / "results" / "broad"
RESULTS.mkdir(parents=True, exist_ok=True)


# ── Output ─────────────────────────────────────────────────────────────
def _row_path(prim: str) -> Path:
    return RESULTS / f"{prim}.json"


def _md_path(prim: str) -> Path:
    return RESULTS / f"{prim}.md"


def write_rows(prim: str, rows: list[dict]) -> None:
    """Atomically rewrite the JSON + markdown for ``prim``.

    Called after every row so partial sweeps are usable.
    """
    p = _row_path(prim)
    p.write_text(json.dumps(rows, indent=2, default=str))

    md = [f"# broad/{prim} — workload sweep of flashlib vs cuML",
          "",
          "Hardware: NVIDIA H200  |  median of 3 repeats, first call discarded.",
          "",
          "| label | axes | dtype | cuml_ms | flashlib_ms | speedup | ok | notes |",
          "| --- | --- | --- | --- | --- | --- | --- | --- |"]
    for r in rows:
        ax = ", ".join(f"{k}={v}" for k, v in r["axes"].items())
        speed = f"{r['speedup']:.2f}x" if isinstance(r.get('speedup'),
                                                       (int, float)) else "-"
        cu = (f"{r['cuml_ms']:.2f}"
              if isinstance(r.get('cuml_ms'), (int, float)) else "-")
        fl = (f"{r['flashlib_ms']:.2f}"
              if isinstance(r.get('flashlib_ms'), (int, float)) else "-")
        ok = "PASS" if r.get("ok") else "FAIL"
        md.append(f"| {r['label']} | {ax} | {r['dtype']} | "
                  f"{cu} | {fl} | {speed} | {ok} | {r.get('notes', '')} |")
    _md_path(prim).write_text("\n".join(md) + "\n")


# ── Timing ─────────────────────────────────────────────────────────────
def time_gpu_call(fn: Callable[[], Any], *,
                   repeat: int = 3, warmup: int = 1) -> float:
    """Median ms of ``fn`` over ``repeat`` calls, after ``warmup`` warmups.

    Synchronises on CUDA before/after each call.
    """
    import torch
    for _ in range(warmup):
        fn()
        torch.cuda.synchronize()
    times = []
    for _ in range(repeat):
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        fn()
        torch.cuda.synchronize()
        times.append(time.perf_counter() - t0)
    times.sort()
    return times[len(times) // 2] * 1000.0


def time_cpu_call(fn: Callable[[], Any], *,
                   repeat: int = 3, warmup: int = 1) -> float:
    """Median ms of ``fn`` over ``repeat`` calls — for CPU baselines."""
    for _ in range(warmup):
        fn()
    times = []
    for _ in range(repeat):
        t0 = time.perf_counter()
        fn()
        times.append(time.perf_counter() - t0)
    times.sort()
    return times[len(times) // 2] * 1000.0


def free_gpu() -> None:
    gc.collect()
    try:
        import torch
        torch.cuda.empty_cache()
    except Exception:
        pass
    try:
        import cupy
        cupy.get_default_memory_pool().free_all_blocks()
        cupy.get_default_pinned_memory_pool().free_all_blocks()
    except Exception:
        pass


# ── Per-cell runner with try/except envelope ──────────────────────────
def safe_run(prim: str, label: str, axes: dict, dtype: str,
              cuml_fn: Callable[[], Any] | None,
              flashlib_fn: Callable[[], Any],
              *, repeat: int = 3, warmup: int = 1,
              cuml_repeat: int | None = None,
              cuml_kind: str = "gpu",
              notes: str = "") -> dict:
    """Run cuML + flashlib for one cell and return a row dict.

    ``cuml_fn`` may be None (e.g. primitive has no cuML peer); the row
    then records flashlib_ms only with speedup = NaN. ``cuml_kind`` is
    ``"gpu"`` (default — measured with ``time_gpu_call``) or ``"cpu"``
    (measured with ``time_cpu_call``, for sklearn baselines).
    """
    out: dict = {"primitive": prim, "label": label, "axes": axes,
                 "dtype": dtype, "cuml_ms": None, "flashlib_ms": None,
                 "speedup": None, "ok": False, "notes": notes,
                 "error": None}
    try:
        if cuml_fn is not None:
            timer = time_gpu_call if cuml_kind == "gpu" else time_cpu_call
            cu_rep = cuml_repeat if cuml_repeat is not None else repeat
            out["cuml_ms"] = timer(cuml_fn, repeat=cu_rep, warmup=warmup)
            free_gpu()
        out["flashlib_ms"] = time_gpu_call(flashlib_fn,
                                            repeat=repeat, warmup=warmup)
        free_gpu()
        if out["cuml_ms"] and out["flashlib_ms"]:
            out["speedup"] = float(out["cuml_ms"]) / float(out["flashlib_ms"])
        out["ok"] = True
    except Exception as e:
        out["error"] = f"{type(e).__name__}: {e}"
        out["notes"] = (out["notes"] + " | "
                         if out["notes"] else "") + out["error"]
        traceback.print_exc()
        free_gpu()
    return out


# ── Sweep driver ───────────────────────────────────────────────────────
def run_grid(prim: str, cells: list[dict]) -> None:
    """Iterate over a list of cell-specs, run each, persist after each.

    A cell-spec is a dict with keys:
        label   : str
        axes    : dict   (workload axes, written into output row)
        dtype   : str
        setup   : zero-arg callable returning (cuml_fn or None,
                                                 flashlib_fn,
                                                 teardown or None)
        repeat  : int (optional, default 3)
        warmup  : int (optional, default 1)
        cuml_repeat : int | None (optional)
        cuml_kind   : "gpu" | "cpu" (default "gpu")
        notes   : str (optional)
    """
    rows: list[dict] = []
    print(f"[broad:{prim}] starting {len(cells)} cells")
    for i, cell in enumerate(cells, 1):
        t0 = time.perf_counter()
        label = cell["label"]
        print(f"[broad:{prim}] cell {i}/{len(cells)}: {label}", flush=True)
        try:
            setup = cell["setup"]
            ret = setup()
            if isinstance(ret, tuple) and len(ret) == 3:
                cu_fn, fl_fn, teardown = ret
            elif isinstance(ret, tuple) and len(ret) == 2:
                cu_fn, fl_fn = ret
                teardown = None
            else:
                raise ValueError("setup must return (cu_fn, fl_fn) or "
                                 "(cu_fn, fl_fn, teardown)")
        except Exception as e:
            print(f"[broad:{prim}]   SETUP FAIL: {type(e).__name__}: {e}")
            rows.append({
                "primitive": prim, "label": label,
                "axes": cell.get("axes", {}), "dtype": cell.get("dtype", "fp32"),
                "cuml_ms": None, "flashlib_ms": None, "speedup": None,
                "ok": False, "notes": cell.get("notes", ""),
                "error": f"setup: {type(e).__name__}: {e}",
            })
            write_rows(prim, rows)
            free_gpu()
            continue

        row = safe_run(
            prim, label, cell.get("axes", {}),
            cell.get("dtype", "fp32"),
            cu_fn, fl_fn,
            repeat=cell.get("repeat", 3),
            warmup=cell.get("warmup", 1),
            cuml_repeat=cell.get("cuml_repeat"),
            cuml_kind=cell.get("cuml_kind", "gpu"),
            notes=cell.get("notes", ""),
        )
        rows.append(row)
        write_rows(prim, rows)
        if teardown is not None:
            try:
                teardown()
            except Exception:
                pass
        free_gpu()
        cu = row.get("cuml_ms")
        fl = row.get("flashlib_ms")
        sp = row.get("speedup")
        cu_s = f"{cu:.2f}" if isinstance(cu, (int, float)) else str(cu)
        fl_s = f"{fl:.2f}" if isinstance(fl, (int, float)) else str(fl)
        sp_s = f"{sp:.2f}x" if isinstance(sp, (int, float)) else str(sp)
        print(f"[broad:{prim}]   cu={cu_s}  fl={fl_s}  "
              f"speedup={sp_s}  ({time.perf_counter()-t0:.1f}s)")
    print(f"[broad:{prim}] DONE - wrote {len(rows)} rows to "
          f"{_row_path(prim)}")
