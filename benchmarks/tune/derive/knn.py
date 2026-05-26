"""Derive KNN routing-rule suggestions from sweep results.

Reads ``benchmarks/tune/results/knn/<device_tag>/*.jsonl`` and prints

  1. A markdown pivot table of (B, N, M, D, k) -> per-backend ms.
  2. Suggested rule snippets sorted by selectivity, in a form ready
     to paste into :mod:`flashlib.primitives.knn.route`.

The classifier groups shapes by the routing axes the hand-written
rule cares about. Today's rule (post-2026-05 cleanup):

  * ``N <= 1024``       -> triton/search
  * ``N >= 4096 AND
    D >= 256 AND
    k <= 16``           -> cutedsl/build_fa3 (opt-in only;
                            never auto-routed -- see
                            ``flashlib/primitives/knn/route.py``
                            for why)
  * default             -> triton/build

The derive script doesn't change the auto-routing rule by itself; it
prints suggestions humans can hand-paste after eyeballing the table.
"""
from __future__ import annotations

from benchmarks.tune.derive._common import (
    Summary, ge, le, load_summaries, print_rule_suggestions, print_table,
)


def classify(s: Summary):
    if s.best_backend is None:
        return None
    N = s.workload.get("N", 0)
    D = s.workload.get("D", 0)
    k = s.workload.get("k", 1)

    # Three buckets matching the existing route.py rule structure.
    if N >= 4096 and D >= 256 and k <= 16:
        return ({"N": ge(4096), "D": ge(256), "k": le(16)}, s.best_backend)
    if N <= 1024:
        return ({"N": le(1024)}, s.best_backend)
    return ({"default": ""}, s.best_backend)


def main() -> None:
    sums = load_summaries("knn")
    if not sums:
        print("no results -- run `python -m benchmarks.tune.knn_parallel` first")
        return
    fp = sums[0].fingerprint
    print(f"# knn -- derive on {fp.get('device_tag', '?')} "
          f"(L2={fp.get('l2_bytes', 0) // (1<<20)} MB, "
          f"sm={fp.get('sm_arch', '?')})")
    print(f"# {len(sums)} shapes, "
          f"backends seen: {sorted({k for s in sums for k in s.by_backend})}")
    print()
    print_table(sums, dim_keys=["B", "N", "M", "D", "k"])
    print()
    print_rule_suggestions(sums, classify=classify)


if __name__ == "__main__":
    main()
