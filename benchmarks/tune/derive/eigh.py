"""Derive eigh routing-rule suggestions."""
from __future__ import annotations

from benchmarks.tune.derive._common import (
    Summary, ge, le, load_summaries, print_rule_suggestions, print_table,
)


def classify(s: Summary):
    if s.best_backend is None:
        return None
    N = s.workload.get("N", 0)
    K = s.workload.get("K", 0)
    tol = s.workload.get("tol", 0.0)
    if K and K > 0:
        return ({"K": ge(1), "N": ge(N // 2)}, s.best_backend)
    if N <= 128:
        return ({"N": le(128)}, s.best_backend)
    if tol >= 3e-3 and N >= 5120:
        return ({"tol": ge(3e-3), "N": ge(5120)}, s.best_backend)
    if tol >= 8e-4 and N >= 5120:
        return ({"tol": ge(8e-4), "N": ge(5120)}, s.best_backend)
    return ({"default": ""}, s.best_backend)


def main() -> None:
    sums = load_summaries("eigh")
    if not sums:
        print("no results — run `python -m benchmarks.tune.eigh` first")
        return
    fp = sums[0].fingerprint
    print(f"# eigh — derive on {fp.get('device_tag', '?')}")
    print()
    print_table(sums, dim_keys=["N", "K", "tol"])
    print()
    print_rule_suggestions(sums, classify=classify)


if __name__ == "__main__":
    main()
