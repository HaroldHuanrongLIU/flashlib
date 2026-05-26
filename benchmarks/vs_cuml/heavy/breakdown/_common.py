"""Shared utilities for breakdown profilers."""
from __future__ import annotations

import contextlib
import time
from pathlib import Path
from typing import Callable

import torch

RESULTS = Path(__file__).resolve().parents[4] / "benchmarks" / "results" / "heavy" / "breakdown"
RESULTS.mkdir(parents=True, exist_ok=True)


class Stage:
    """Context manager that records the GPU wall time of a region with
    a pair of ``torch.cuda.Event``s. Multiple calls into the same
    ``Stage`` accumulate (useful for per-iter inner loops).
    """

    def __init__(self, name: str):
        self.name = name
        self._total_ms: float = 0.0
        self._n_calls: int = 0
        self._start: torch.cuda.Event | None = None
        self._end: torch.cuda.Event | None = None

    def __enter__(self) -> "Stage":
        self._start = torch.cuda.Event(enable_timing=True)
        self._end = torch.cuda.Event(enable_timing=True)
        self._start.record()
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        assert self._start is not None and self._end is not None
        self._end.record()
        torch.cuda.synchronize()
        self._total_ms += float(self._start.elapsed_time(self._end))
        self._n_calls += 1
        self._start = None
        self._end = None

    def reset(self) -> None:
        self._total_ms = 0.0
        self._n_calls = 0

    @property
    def ms(self) -> float:
        return self._total_ms

    @property
    def n_calls(self) -> int:
        return self._n_calls


class StageGroup:
    """Holds a set of named Stages so the caller can refer to them by
    name in nested code without threading the object explicitly."""

    def __init__(self, names: list[str]):
        self._stages: dict[str, Stage] = {n: Stage(n) for n in names}

    def __getitem__(self, name: str) -> Stage:
        return self._stages[name]

    def names(self) -> list[str]:
        return list(self._stages.keys())

    def reset_all(self) -> None:
        for s in self._stages.values():
            s.reset()

    def total_ms(self) -> float:
        return sum(s.ms for s in self._stages.values())


def write_breakdown_md(prim: str, shape_label: str, stages: StageGroup,
                        total_outer_ms: float, *,
                        extra_columns: dict[str, str] | None = None,
                        notes: str | None = None) -> None:
    """Render a per-primitive breakdown table to
    ``benchmarks/results/heavy/breakdown/<prim>.md``.
    """
    md = [f"# heavy/breakdown/{prim} — per-component time breakdown",
          "",
          f"Hardware: NVIDIA H200  |  shape: {shape_label}",
          ""]
    if notes:
        md.append(f"_{notes}_")
        md.append("")

    header = ["component", "time_ms", "% of total", "n_calls"]
    if extra_columns:
        header += list(extra_columns.keys())
    md.append("| " + " | ".join(header) + " |")
    md.append("|" + "|".join(["---"] * len(header)) + "|")

    breakdown_total = stages.total_ms()
    for name in stages.names():
        s = stages[name]
        pct = (s.ms / breakdown_total * 100.0) if breakdown_total > 0 else 0.0
        row = [name, f"{s.ms:8.2f}", f"{pct:5.1f}%", str(s.n_calls)]
        if extra_columns:
            row += [extra_columns.get(k, "-") for k in extra_columns.keys()]
        md.append("| " + " | ".join(row) + " |")
    md.append(f"| **sum-of-components** | **{breakdown_total:8.2f}** | **100.0%** | - |")
    md.append(f"| **outer-most wall**   | **{total_outer_ms:8.2f}** | (event-overhead-incl.) | 1 |")

    md.append("")
    md.append("Median of 3 repeats; first call (JIT warmup) discarded.")
    md.append("Per-stage timing uses paired `torch.cuda.Event`s.")

    path = RESULTS / f"{prim}.md"
    path.write_text("\n".join(md) + "\n")
    print(f"[breakdown] wrote {path}")


def median_repeats(fn: Callable[[StageGroup], None], stages: StageGroup, *,
                    warmup: int = 1, repeat: int = 3) -> tuple[float, dict[str, float]]:
    """Run ``fn`` (which takes a StageGroup and populates it in-place) a few
    times, returning the median outer-wall and the per-stage median ms.
    """
    outer_ms_list: list[float] = []
    per_stage_lists: dict[str, list[float]] = {n: [] for n in stages.names()}

    for _ in range(warmup):
        stages.reset_all()
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        fn(stages)
        torch.cuda.synchronize()
        _ = (time.perf_counter() - t0) * 1000.0

    for _ in range(repeat):
        stages.reset_all()
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        fn(stages)
        torch.cuda.synchronize()
        outer_ms_list.append((time.perf_counter() - t0) * 1000.0)
        for n in stages.names():
            per_stage_lists[n].append(stages[n].ms)

    import statistics
    outer_med = statistics.median(outer_ms_list)
    per_med = {n: statistics.median(per_stage_lists[n]) for n in stages.names()}

    # Set the stages to their median values so the caller renders medians,
    # not the last-run values. ``_n_calls`` is the count from the LAST
    # repeat (each repeat ``reset_all()``s); leaving it at zero signals
    # to ``write_multi_shape_md`` that the stage was never entered for
    # this shape (rendered as ``-`` rather than ``0.00 ms``).
    for n in stages.names():
        s = stages[n]
        s._total_ms = per_med[n]

    return outer_med, per_med


def free_gpu() -> None:
    import gc
    gc.collect()
    torch.cuda.empty_cache()


@contextlib.contextmanager
def cuda_event_pair():
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    yield (start, end)
    end.record()
    torch.cuda.synchronize()


# ---------------------------------------------------------------------------
# Multi-shape sweep utilities
# ---------------------------------------------------------------------------

def run_multi_shape(shapes: list[tuple[str, dict]],
                     prepare_fn,
                     run_fn,
                     stage_names: list[str],
                     *,
                     warmup: int = 1,
                     repeat: int = 3) -> list[dict]:
    """Run ``run_fn`` on each shape and return a list of per-shape results.

    Args:
        shapes: list of ``(shape_label, shape_kwargs)``. ``shape_kwargs`` is
            a dict passed to ``prepare_fn`` to build the inputs.
        prepare_fn: ``shape_kwargs -> ctx_dict``. Builds GPU tensors / state
            once per shape. Returns a dict consumed by ``run_fn``.
        run_fn: ``(stages, ctx) -> None``. Wraps each stage in
            ``with stages[name]:`` exactly as the single-shape profilers do.
        stage_names: ordered list of stage names that ``run_fn`` populates.

    Returns:
        list of ``{label, outer_ms, per_stage_ms, n_calls}`` dicts.
    """
    results: list[dict] = []
    for label, kwargs in shapes:
        free_gpu()
        print(f"[multi-shape]   shape: {label}")
        ctx = prepare_fn(**kwargs)
        stages = StageGroup(stage_names)

        def _wrapped(stg):  # noqa: ANN001
            run_fn(stg, ctx)

        outer, per_med = median_repeats(_wrapped, stages,
                                          warmup=warmup, repeat=repeat)
        n_calls = {n: stages[n].n_calls for n in stage_names}
        results.append({
            "label": label,
            "outer_ms": outer,
            "per_stage_ms": per_med,
            "n_calls": n_calls,
            "shape_kwargs": kwargs,
        })
        del ctx
    return results


def write_multi_shape_md(prim: str, shape_axis: str,
                          results: list[dict],
                          stage_names: list[str],
                          *,
                          notes: str | None = None,
                          sensitivity: str | None = None,
                          file_suffix: str = "") -> None:
    """Render a primitive's multi-shape breakdown as a single markdown table.

    Layout (one row per stage, one column per shape):

        | component | <shape1> ms (pct) | <shape2> ms (pct) | ... |

    A final "outer wall" row reports the total wall per shape, and a final
    "Workload sensitivity" paragraph (provided by the caller) summarises
    how the percentages shift with the axis.

    Args:
        prim: primitive name; output goes to
            ``benchmarks/results/heavy/breakdown/<prim>.md``.
        shape_axis: human label for the swept axis (e.g. ``"K (n_clusters)"``).
        results: output of ``run_multi_shape``.
        stage_names: ordered list to render rows for (rows missing a stage
            in a particular shape get ``-``).
        notes: optional paragraph above the table.
        sensitivity: REQUIRED workload-sensitivity paragraph below the table.
    """
    md = [f"# heavy/breakdown/{prim} — per-component time breakdown "
          f"(sweep over {shape_axis})",
          "",
          "Hardware: NVIDIA H200",
          ""]
    if notes:
        md.append(f"_{notes}_")
        md.append("")

    md.append(f"Swept axis: **{shape_axis}**.  Median of 3 repeats per shape; "
              f"first call (JIT warmup) discarded.  Per-stage timing via paired "
              "`torch.cuda.Event`s.")
    md.append("")

    header = ["component"] + [r["label"] for r in results]
    md.append("| " + " | ".join(header) + " |")
    md.append("|" + "|".join(["---"] * len(header)) + "|")

    for stage in stage_names:
        row = [f"`{stage}`"]
        for r in results:
            ms = r["per_stage_ms"].get(stage)
            n = r["n_calls"].get(stage, 0)
            outer = sum(r["per_stage_ms"].values()) or 1.0
            if ms is None or n == 0:
                row.append("-")
            else:
                pct = ms / outer * 100.0
                row.append(f"{ms:.2f} ms ({pct:.1f}%)")
        md.append("| " + " | ".join(row) + " |")

    outer_row = ["**outer wall**"]
    for r in results:
        outer_row.append(f"**{r['outer_ms']:.2f} ms**")
    md.append("| " + " | ".join(outer_row) + " |")

    md.append("")
    if sensitivity:
        md.append("## Workload sensitivity")
        md.append("")
        md.append(sensitivity)
        md.append("")

    path = RESULTS / f"{prim}{file_suffix}.md"
    path.write_text("\n".join(md) + "\n")
    print(f"[breakdown] wrote {path}")
