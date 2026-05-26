"""K-Means tuner — sweeps (N, D, K) × {triton, cutedsl/euclid}.

Drives the boundary documented in
[flashlib/primitives/kmeans/route.py](../../flashlib/primitives/kmeans/route.py):
on H200 the FA3-style CuteDSL assign and the Triton split-D path are
roughly tied, so the default keeps Triton; this tuner exists so a new
GPU (e.g. Blackwell) can quickly tell whether that picture has changed.

Usage::

    python -m benchmarks.tune.kmeans              # full sweep
    python -m benchmarks.tune.kmeans --rerun
"""
from __future__ import annotations

import torch

from benchmarks.tune._common import expand_grid, parse_argv, run_tuner


WORKLOADS = expand_grid({
    "B": [1],
    "N": [16_384, 65_536, 262_144],
    "D": [64, 128, 256, 512],
    "K": [64, 256, 1024, 4096],
})


BACKENDS = [
    {"backend": "triton",  "variant": "lloyd"},
    {"backend": "cutedsl", "variant": "fa3_assign"},
]


def setup(workload):
    B, N, D, K = (workload[k] for k in ("B", "N", "D", "K"))
    # cutedsl path requires fp16/bf16 + B=1; pin those to keep the
    # head-to-head fair (otherwise cutedsl auto-rejects).
    x = torch.randn(B, N, D, device="cuda", dtype=torch.float16)
    init = torch.randn(B, K, D, device="cuda", dtype=torch.float16)
    return {"x": x, "K": K, "init": init}


def bench(ctx, candidate):
    x, K, init = ctx["x"], ctx["K"], ctx["init"]
    if candidate["backend"] == "triton":
        from flashlib.primitives.kmeans import batch_kmeans_Euclid
        return lambda: batch_kmeans_Euclid(
            x, K, max_iters=5, init_centroids=init,
        )
    if candidate["backend"] == "cutedsl":
        from flashlib.primitives.kmeans.cutedsl import cutedsl_kmeans_Euclid
        return lambda: cutedsl_kmeans_Euclid(
            x, K, max_iters=5, init_centroids=init,
        )
    raise ValueError(candidate)


def main() -> None:
    args = parse_argv("kmeans")
    sizes = set(args.size.split(",")) if args.size else None

    def keep(w):
        if sizes is None:
            return True
        from benchmarks.tune._common import shape_key
        return shape_key(w) in sizes

    run_tuner(
        op="kmeans", workloads=WORKLOADS, backends=BACKENDS,
        setup_fn=setup, bench_fn=bench,
        warm=3, iters=5, rerun=args.rerun,
        workload_filter=keep,
    )


if __name__ == "__main__":
    main()
