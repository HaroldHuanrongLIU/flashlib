"""Derive GEMM Pareto frontier from sweep results.

GEMM routing is precision-driven, so the right output is a Pareto curve
of (rel_err, time_ms) — each variant matters only if it is on the
frontier. Off-frontier variants are dominated and should be removed
from ``_THROUGHPUT_TF`` (or kept only behind explicit ``backend=``).

Re-tuning workflow on a new GPU::

    python -m benchmarks.tune.gemm
    python -m benchmarks.tune.derive.gemm
    # Then update the throughput numbers in
    # flashlib/linalg/gemm/__init__.py to match the new measurements.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import List, Tuple

from benchmarks.tune._common import RESULTS_ROOT
from flashlib import _hw


def main() -> None:
    tag = _hw.device_tag()
    d = RESULTS_ROOT / "gemm" / tag
    if not d.exists():
        print("no results — run `python -m benchmarks.tune.gemm` first")
        return

    # GEMM only writes one workload (the canonical Pareto shape) by
    # default, but iterate just in case.
    for f in sorted(d.glob("*.jsonl")):
        rows: List[dict] = []
        for line in f.read_text().splitlines():
            try:
                r = json.loads(line)
            except json.JSONDecodeError:
                continue
            if "backend" in r and r.get("status") == "ok":
                rows.append(r)
        if not rows:
            continue
        print(f"# gemm Pareto — {f.stem} on {tag}")
        print()
        print("| variant | runtime_ms | rms_rel_err | Pareto |")
        print("|---|---|---|---|")

        # Pareto front: a row is on the frontier iff no other row is BOTH
        # faster AND tighter.
        def dominated(r):
            for s in rows:
                if s is r:
                    continue
                if (s["time_ms"] < r["time_ms"]
                        and s["rel_err"] < r["rel_err"]):
                    return True
            return False

        pareto = [r for r in rows if not dominated(r)]
        for r in sorted(rows, key=lambda r: r["time_ms"]):
            mark = "Y" if r in pareto else ""
            print(f"| {r['backend']} | {r['time_ms']:.2f} | "
                  f"{r['rel_err']:.2e} | {mark} |")
        print()
        names = sorted({r["backend"] for r in pareto})
        print(f"_Pareto front: {names}_")


if __name__ == "__main__":
    main()
