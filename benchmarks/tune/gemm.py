"""GEMM tuner — measures throughput AND residual for every variant on
one canonical shape. Drives the ``_THROUGHPUT_TF`` and
``_RESIDUAL_PREFERENCE`` tables in
[flashlib/linalg/gemm/__init__.py](../../flashlib/linalg/gemm/__init__.py).

GEMM routing is driven by tol (precision tier), so the tuner records:

  * ``time_ms``: median wall-clock cost
  * ``rel_err``: RMS rel error vs FP64 reference

A second-pass derive script reads these and prints the Pareto frontier
(each variant only matters if it is fastest at its precision tier).

Usage::

    python -m benchmarks.tune.gemm                # default shape M=N=K=4096
    python -m benchmarks.tune.gemm --rerun
"""
from __future__ import annotations

import torch

from benchmarks.tune._common import expand_grid, parse_argv, run_tuner


# Canonical Pareto shape — matches the ``boundaries_gemm.md`` workload.
WORKLOADS = expand_grid({
    "M": [4096],
    "N": [4096],
    "K": [4096],
})


BACKENDS = [
    {"backend": "fp32"},
    {"backend": "tf32"},
    {"backend": "bf16"},
    {"backend": "fp16"},
    {"backend": "3xtf32"},
    {"backend": "3xbf16"},
    {"backend": "3xfp16"},
    {"backend": "fp16_x9"},
    {"backend": "fp16_x3_kahan"},
    {"backend": "tf32_x6"},
    {"backend": "ozaki2_cute"},
    {"backend": "ozaki2_triton"},
    # ozaki2_int8 needs gemmul8 native; left out of default sweep.
]


def setup(workload):
    M, N, K = workload["M"], workload["N"], workload["K"]
    A = torch.randn(M, K, device="cuda", dtype=torch.float32) / (K ** 0.5)
    B = torch.randn(K, N, device="cuda", dtype=torch.float32) / (K ** 0.5)
    # FP64 reference for residual.
    ref = (A.double() @ B.double()).float()
    return {"A": A, "B": B, "ref": ref}


def bench(ctx, candidate):
    from flashlib.linalg.gemm import gemm
    A, B = ctx["A"], ctx["B"]
    name = candidate["backend"]
    return lambda: gemm(A, B, backend=name)


def correctness(ctx, candidate):
    from flashlib.linalg.gemm import gemm
    A, B, ref = ctx["A"], ctx["B"], ctx["ref"]
    out = gemm(A, B, backend=candidate["backend"]).float()
    return float(((out - ref).pow(2).mean().sqrt() /
                  ref.pow(2).mean().sqrt()).item())


def main() -> None:
    args = parse_argv("gemm")
    sizes = set(args.size.split(",")) if args.size else None

    def keep(w):
        if sizes is None:
            return True
        from benchmarks.tune._common import shape_key
        return shape_key(w) in sizes

    run_tuner(
        op="gemm", workloads=WORKLOADS, backends=BACKENDS,
        setup_fn=setup, bench_fn=bench, correctness_fn=correctness,
        warm=3, iters=5, rerun=args.rerun,
        workload_filter=keep,
    )


if __name__ == "__main__":
    main()
