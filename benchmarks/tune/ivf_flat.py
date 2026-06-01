"""Autotune harness for the IVF-Flat fused fine-scan kernel.

Sweeps the two knobs that drive the fine-scan's occupancy / efficiency:

* ``BM``       -- candidate chunk width (HBM read granularity per step).
* ``n_splits`` -- how finely each inverted list is chopped across the SM
                  waves (matters most in the small-``nq`` / online regime
                  where ``nq * nprobe`` alone underfills the GPU).

The index + coarse probe are built once per workload (in ``setup``); only
the fused fine-scan is timed, so the numbers isolate the new kernel.
Derive sustained bandwidth from the winner to backfill
``flashlib/info/roofline.py``'s ``("ivf_flat_search", dev)`` entry.

Run::

    python -m benchmarks.tune.ivf_flat            # full sweep
    python -m benchmarks.tune.ivf_flat --rerun
"""
from __future__ import annotations

import torch

from benchmarks.tune._common import expand_grid, parse_argv, run_tuner


WORKLOADS = expand_grid({
    "M": [1_000_000],
    "D": [64, 128],
    "nlist": [1024],
    "nprobe": [16],
    "nq": [100, 10_000],   # online + batch regimes
    "k": [10],
})


# Each candidate forces a (BM, n_splits) for the fine-scan kernel.
BACKENDS = [
    {"backend": "triton", "variant": f"BM{bm}_S{ns}", "BM": bm, "n_splits": ns}
    for bm in (64, 128, 256)
    for ns in (1, 4, 16)
]


def setup(workload):
    from flashlib import flash_ivf_flat_build
    from flashlib.primitives.ivf_flat.torch_fallback import _pad_features
    from flashlib.primitives.knn import flash_knn

    M, D = workload["M"], workload["D"]
    nlist, nprobe, nq = workload["nlist"], workload["nprobe"], workload["nq"]

    g = torch.Generator(device="cuda").manual_seed(0)
    centers = torch.randn(max(8, nlist // 8), D, generator=g, device="cuda") * 4.0
    lab = torch.randint(0, centers.shape[0], (M,), generator=g, device="cuda")
    X = (centers[lab] + torch.randn(M, D, generator=g, device="cuda")).float()
    Q = X[torch.randint(0, M, (nq,), generator=g, device="cuda")].clone()

    index = flash_ivf_flat_build(X, nlist, nprobe=nprobe, niter=15, seed=0)
    Qp = _pad_features(Q.to(index.data.dtype), index.Dp).contiguous()
    probed = flash_knn(
        Qp.unsqueeze(0), index.centroids.unsqueeze(0), nprobe,
        return_distances=False,
    )[0].to(torch.int32)
    max_list_len = int(index.list_lengths().max().item())
    return {
        "Qp": Qp, "data": index.data, "probed": probed,
        "offsets": index.list_offsets, "k": workload["k"],
        "max_list_len": max_list_len,
    }


def bench(ctx, candidate):
    from flashlib.primitives.ivf_flat.triton.fine_scan import ivf_fine_scan

    return lambda: ivf_fine_scan(
        ctx["Qp"], ctx["data"], ctx["probed"], ctx["offsets"], ctx["k"],
        max_list_len=ctx["max_list_len"],
        BM=candidate["BM"], n_splits=candidate["n_splits"],
    )


def main() -> None:
    args = parse_argv("ivf_flat")
    sizes = set(args.size.split(",")) if args.size else None
    run_tuner(
        op="ivf_flat",
        workloads=WORKLOADS,
        backends=BACKENDS,
        setup_fn=setup,
        bench_fn=bench,
        warm=3, iters=7, rerun=args.rerun,
        workload_filter=(
            (lambda w: any(s in
                f"M{w['M']}_D{w['D']}_nlist{w['nlist']}_nprobe{w['nprobe']}_nq{w['nq']}_k{w['k']}"
                for s in sizes)) if sizes else None
        ),
    )


if __name__ == "__main__":
    main()
