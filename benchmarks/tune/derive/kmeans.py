"""Derive K-Means routing-rule suggestions."""
from __future__ import annotations

from benchmarks.tune.derive._common import (
    Summary, ge, le, load_summaries, print_rule_suggestions, print_table,
)


def classify(s: Summary):
    if s.best_backend is None:
        return None
    D = s.workload.get("D", 0)
    if D >= 512:
        return ({"D": ge(512)}, s.best_backend)
    return ({"default": ""}, s.best_backend)


def main() -> None:
    sums = load_summaries("kmeans")
    if not sums:
        print("no results — run `python -m benchmarks.tune.kmeans` first")
        return
    fp = sums[0].fingerprint
    print(f"# kmeans — derive on {fp.get('device_tag', '?')}")
    print()
    print_table(sums, dim_keys=["B", "N", "D", "K"])
    print()
    print_rule_suggestions(sums, classify=classify)


if __name__ == "__main__":
    main()
