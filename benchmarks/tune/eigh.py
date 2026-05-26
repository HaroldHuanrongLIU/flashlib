"""eigh tuner — sweeps (N, K, tol) × {cusolver, jacobi, halko, qdwh, qdwh_ns}.

Drives the routing thresholds in
[flashlib/linalg/eigh/route.py](../../flashlib/linalg/eigh/route.py):

  * jacobi   wins for N <= 16 (single-CTA Triton cyclic Jacobi;
             opt-in only -- past N ~ 16 cuSOLVER's syevd is faster
             even with its launch fixed-cost)
  * halko    wins for truncated K << N (Pareto over cusolver)
  * qdwh*    win for N >= 5120 with tol >= 8e-4 / 3e-3
  * cusolver default

Usage::

    python -m benchmarks.tune.eigh
"""
from __future__ import annotations

import torch

from benchmarks.tune._common import expand_grid, parse_argv, run_tuner


# Two regimes: full eigh (K=None) and truncated (K << N).
WORKLOADS = (
    expand_grid({
        "N": [32, 64, 128, 1024, 4096, 8192, 16384],
        "K": [0],            # 0 == full eigh
        "tol": [0.0, 8e-4, 3e-3],
    })
    + expand_grid({
        "N": [1024, 4096, 10000],
        "K": [16, 32, 64, 128],
        "tol": [0.0],
    })
)


BACKENDS = [
    {"backend": "cusolver", "variant": None},
    {"backend": "jacobi",   "variant": None},
    {"backend": "halko",    "variant": None},
    {"backend": "qdwh",     "variant": None},
    {"backend": "qdwh_ns",  "variant": None},
]


def setup(workload):
    N = workload["N"]
    A = torch.randn(N, N, device="cuda", dtype=torch.float32)
    A = (A + A.T) * 0.5
    return {"A": A, "N": N, "K": workload["K"] or None, "tol": workload["tol"]}


def bench(ctx, candidate):
    from flashlib.linalg.eigh import eigh
    A = ctx["A"]
    N = ctx["N"]
    K = ctx["K"]
    backend = candidate["backend"]

    # Capability gates: skip cells where the variant cannot run.
    # We honour them by raising — the harness records ``status="error"``.
    if backend == "jacobi" and N > 128:
        raise RuntimeError("jacobi limited to N<=128")
    if backend in ("qdwh", "qdwh_ns") and N < 5120:
        raise RuntimeError("qdwh* limited to N>=5120")
    if backend == "halko" and K is None:
        raise RuntimeError("halko requires K")

    if K is not None and backend != "halko":
        # Compare full-eigh paths against halko on the truncated rows by
        # taking top-K of a full eigh (this measures the fair upper
        # bound for the alternatives).
        return lambda: eigh(A, backend=backend)
    if backend == "halko":
        return lambda: eigh(A, K=K, backend="halko")
    return lambda: eigh(A, backend=backend)


def main() -> None:
    args = parse_argv("eigh")
    sizes = set(args.size.split(",")) if args.size else None

    def keep(w):
        if sizes is None:
            return True
        from benchmarks.tune._common import shape_key
        return shape_key(w) in sizes

    run_tuner(
        op="eigh", workloads=WORKLOADS, backends=BACKENDS,
        setup_fn=setup, bench_fn=bench,
        warm=2, iters=3, rerun=args.rerun,
        workload_filter=keep,
    )


if __name__ == "__main__":
    main()
