"""Derive template — copy to ``benchmarks/tune/derive/<op>.py`` and edit.

The contract:

1. Read summaries via :func:`benchmarks.tune.derive._common.load_summaries`.
2. Print a markdown table of per-shape per-backend timings.
3. Print suggested rules via
   :func:`benchmarks.tune.derive._common.print_rule_suggestions` —
   passing a ``classify(summary) -> (predicate, backend)`` function that
   encodes how you want to express the rule (e.g. group by N>=4096).

Run it via::

    python -m benchmarks.tune.derive.<op>
"""
from __future__ import annotations

from benchmarks.tune.derive._common import (
    Summary, ge, le, load_summaries, print_rule_suggestions, print_table,
)


def classify(s: Summary):
    """Map a workload winner to a (predicate, backend) hint.

    Returning ``None`` excludes the workload from rule suggestions
    (useful for shapes you'd rather leave on the default).
    """
    if s.best_backend is None:
        return None
    N = s.workload.get("N")
    D = s.workload.get("D")
    if N is None or D is None:
        return None
    # Example: bucket "big" shapes into one rule.
    if N >= 4096 and D >= 256:
        return ({"N": ge(4096), "D": ge(256)}, s.best_backend)
    return None


def main() -> None:
    sums = load_summaries("YOUR_OP")
    if not sums:
        print("no results — run `python -m benchmarks.tune.YOUR_OP` first")
        return
    print(f"# YOUR_OP — derive ({sums[0].fingerprint.get('device_tag', '?')})")
    print()
    print_table(sums, dim_keys=["N", "D"])
    print()
    print_rule_suggestions(sums, classify=classify)


if __name__ == "__main__":
    main()
