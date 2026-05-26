"""Shared helpers for ``benchmarks/tune/derive/<op>.py`` scripts.

Reads JSONL files written by :func:`benchmarks.tune._common.run_tuner`
and provides:

* :func:`load_summaries` — yield ``summary`` records (last line of each
  workload JSONL).
* :func:`print_table` — pretty-print a sorted (workload, ms-by-backend)
  pivot table as Markdown.
* :func:`print_rule_suggestions` — print a Python snippet of suggested
  routing rules, sorted by selectivity, plus coverage warnings for
  shapes where no backend wins clearly.

Derive scripts call these with op-specific decisions about how to group
workloads and how to express rules.
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, List, Optional, Tuple

from flashlib import _hw

from .._common import RESULTS_ROOT


@dataclass
class Summary:
    workload: Dict[str, Any]
    best_backend: Optional[str]
    best_ms: Optional[float]
    by_backend: Dict[str, Optional[float]]
    fingerprint: Dict[str, Any]


def load_summaries(op: str, device_tag: Optional[str] = None) -> List[Summary]:
    """Read all ``results/<op>/<device_tag>/*.jsonl`` summary lines."""
    tag = device_tag or _hw.device_tag()
    d = RESULTS_ROOT / op / tag
    if not d.exists():
        return []
    out: List[Summary] = []
    for f in sorted(d.glob("*.jsonl")):
        last_summary: Optional[Dict[str, Any]] = None
        for line in f.read_text().splitlines():
            try:
                rec = json.loads(line)
            except json.JSONDecodeError:
                continue
            if isinstance(rec, dict) and "summary" in rec:
                last_summary = rec["summary"]
        if last_summary:
            out.append(Summary(
                workload=last_summary["workload"],
                best_backend=last_summary["best_backend"],
                best_ms=last_summary["best_ms"],
                by_backend=last_summary["by_backend"],
                fingerprint=last_summary.get("fingerprint", {}),
            ))
    return out


def print_table(summaries: List[Summary], dim_keys: List[str]) -> None:
    """Print a Markdown table: workload dims | per-backend ms | winner | speedup.

    ``dim_keys`` orders the workload dimensions so the table is readable.
    ``speedup`` is best vs second-best (or "—" when only one backend ran).
    """
    if not summaries:
        print("(no results)")
        return
    backends = sorted({k for s in summaries for k in s.by_backend.keys()})
    header = (
        "| " + " | ".join(dim_keys) + " | "
        + " | ".join(f"{b} (ms)" for b in backends)
        + " | winner | speedup |"
    )
    sep = "|" + "---|" * (len(dim_keys) + len(backends) + 2)
    print(header)
    print(sep)
    for s in sorted(summaries,
                    key=lambda s: tuple(s.workload.get(k, 0) for k in dim_keys)):
        cells = [str(s.workload.get(k, "")) for k in dim_keys]
        for b in backends:
            v = s.by_backend.get(b)
            cells.append("—" if v is None else f"{v:.2f}")
        cells.append(s.best_backend or "—")
        # Speedup: best vs second-best.
        ranked = sorted(
            [v for v in s.by_backend.values() if v is not None]
        )
        spd = "—"
        if len(ranked) >= 2 and ranked[0] > 0:
            spd = f"{ranked[1] / ranked[0]:.2f}x"
        cells.append(spd)
        print("| " + " | ".join(cells) + " |")


def print_rule_suggestions(
    summaries: List[Summary],
    *,
    classify: Callable[[Summary], Optional[Tuple[Dict[str, Any], str]]],
    indecisive_threshold: float = 0.05,
) -> None:
    """Print a Python rule snippet inferred from the per-workload winners.

    ``classify`` maps a :class:`Summary` to ``(predicate_dict,
    backend_label)`` where ``predicate_dict`` is the canonical predicate
    (e.g. ``{"N": ">=4096", "D": ">=256"}``) the rule should fire on.
    The derive script writer is responsible for collapsing many shapes
    into a small number of predicates — the derive helper here only
    *aggregates* identical predicates and counts how often each was
    chosen.

    Workloads where best vs second-best is within ``indecisive_threshold``
    (5% by default) are flagged as "no strong preference" — those are
    safe to leave on the default backend.
    """
    bucket: Dict[Tuple[Tuple[str, Any], ...], Dict[str, int]] = {}
    weak: List[Summary] = []
    for s in summaries:
        if s.best_backend is None:
            continue
        ranked = sorted(
            [v for v in s.by_backend.values() if v is not None]
        )
        if len(ranked) >= 2 and ranked[0] > 0:
            margin = (ranked[1] - ranked[0]) / ranked[0]
            if margin < indecisive_threshold:
                weak.append(s)
                continue
        spec = classify(s)
        if spec is None:
            continue
        pred, backend = spec
        key = tuple(sorted(pred.items()))
        bucket.setdefault(key, {})
        bucket[key][backend] = bucket[key].get(backend, 0) + 1

    if not bucket:
        print("# (derive: no rules suggested — sweep is too small or "
              "all backends tied)")
    else:
        print("# === suggested routing rules (paste into route.py) ===")
        for key, votes in sorted(bucket.items(),
                                  key=lambda kv: -sum(kv[1].values())):
            winner = max(votes.items(), key=lambda kv: kv[1])
            cond = " and ".join(f"{k} {op}" for k, op in key)
            n = votes[winner[0]]
            tot = sum(votes.values())
            print(f"if {cond}:  return {winner[0]!r}   # "
                  f"{n}/{tot} workloads")

    if weak:
        print()
        print(f"# === {len(weak)} workload(s) with <5% margin (no preference) ===")
        for s in weak:
            print(f"#   {s.workload}: best={s.best_backend} "
                  f"({s.best_ms:.2f}ms)")


# Convenience: human-readable predicate fragments
def ge(v: Any) -> str: return f">= {v}"
def le(v: Any) -> str: return f"<= {v}"
def gt(v: Any) -> str: return f">  {v}"
def lt(v: Any) -> str: return f"<  {v}"
def eq(v: Any) -> str: return f"== {v}"
