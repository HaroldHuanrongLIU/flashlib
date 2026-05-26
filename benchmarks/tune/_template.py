"""Tuner template — copy to ``benchmarks/tune/<op>.py`` and edit.

The contract every tuner must satisfy:

1. Define ``WORKLOADS`` (dict of dim_name -> list of values), or build a
   list of dicts directly. Each row is one shape spec.
2. Define ``BACKENDS`` (list of dicts with at least a ``backend`` key,
   optionally ``variant`` and any extra knobs ``bench_fn`` consumes).
3. Implement ``setup(workload) -> ctx`` returning the input tensors /
   shared state.
4. Implement ``bench(ctx, candidate) -> callable``.
5. (Optional) ``correctness(ctx, candidate) -> rel_err``.
6. Call :func:`benchmarks.tune._common.run_tuner` from ``main()``.

Run it via::

    python -m benchmarks.tune.<op>             # full sweep
    python -m benchmarks.tune.<op> --rerun     # overwrite existing JSONL

Then inspect with ``python -m benchmarks.tune.derive.<op>``.
"""
from __future__ import annotations

import torch

from benchmarks.tune._common import (
    expand_grid, parse_argv, run_tuner,
)


# Dim name -> values to sweep. Keep the cartesian product < ~50 cells so
# the tuner finishes in a few minutes; trim aggressively if needed.
WORKLOADS = expand_grid({
    "N": [1024, 4096],
    "D": [64, 256],
})


# Each candidate dict MUST have a "backend" key. ``variant`` and any
# extra keys are passed through to bench/setup.
BACKENDS = [
    {"backend": "triton",  "variant": None},
    {"backend": "cutedsl", "variant": None},
]


def setup(workload):
    """Allocate inputs once per workload."""
    N, D = workload["N"], workload["D"]
    X = torch.randn(N, D, device="cuda", dtype=torch.float32)
    return {"X": X, "N": N, "D": D}


def bench(ctx, candidate):
    """Return a no-arg callable that runs the candidate on ``ctx``."""
    backend = candidate["backend"]
    if backend == "triton":
        from flashlib.primitives.YOUR_OP import flash_your_op_triton  # type: ignore
        return lambda: flash_your_op_triton(ctx["X"])
    elif backend == "cutedsl":
        from flashlib.primitives.YOUR_OP import flash_your_op_cutedsl  # type: ignore
        return lambda: flash_your_op_cutedsl(ctx["X"])
    raise ValueError(f"unknown backend {backend}")


def main() -> None:
    args = parse_argv("YOUR_OP")
    sizes = set(args.size.split(",")) if args.size else None
    run_tuner(
        op="YOUR_OP",
        workloads=WORKLOADS,
        backends=BACKENDS,
        setup_fn=setup,
        bench_fn=bench,
        warm=3, iters=5, rerun=args.rerun,
        workload_filter=(
            (lambda w: f"N{w['N']}_D{w['D']}" in sizes) if sizes else None
        ),
    )


if __name__ == "__main__":
    main()
