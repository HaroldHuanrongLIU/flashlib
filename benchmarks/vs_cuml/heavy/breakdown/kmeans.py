"""Per-component time breakdown for flash_kmeans across multiple workloads.

Lloyd-Euclidean stages:
  * init   — initial centroid gather
  * assign — `euclid_assign_triton` (on-chip fused argmin)
  * update — `triton_lloyd_centroid_step_euclid` (sortmerge sums + shift)
  * misc   — ping-pong swap

Workload axis: **K (number of clusters)** at fixed N=20M, D=64, iters=5.
K determines the inner reduction of `assign` (per-row O(K·D) work) vs the
fixed N·D streaming of `update`. The headline shape sits at K=100K.
"""
from __future__ import annotations

import torch

from flashlib.primitives.kmeans.triton.assign import euclid_assign_triton
from flashlib.primitives.kmeans.triton.update import (
    triton_lloyd_centroid_step_euclid,
)

from ._common import (
    StageGroup, free_gpu, run_multi_shape, write_multi_shape_md,
)

N, D = 20_000_000, 64
MAX_ITERS = 5

SHAPES = [
    ("K=1K",   {"K":   1_000}),
    ("K=10K",  {"K":  10_000}),
    ("K=100K", {"K": 100_000}),
]
STAGES = ["init", "assign", "update", "misc"]


def prepare(K: int) -> dict:
    torch.manual_seed(0)
    device = "cuda"
    X = torch.randn(N, D, device=device, dtype=torch.float32)
    init_idx = torch.randint(0, N, (K,), device=device)
    cent_a = torch.gather(X, 0, init_idx.unsqueeze(-1).expand(-1, D))
    cent_a = cent_a.view(1, K, D).contiguous()
    return {
        "X_b": X.unsqueeze(0),
        "init_idx": init_idx,
        "K": K,
        "cent_a": cent_a,
        "cent_b": torch.empty_like(cent_a),
        "sums_buf": torch.zeros((1, K, D), device=device, dtype=torch.float32),
        "cnts_buf": torch.zeros((1, K), device=device, dtype=torch.int32),
        "shift_buf": torch.empty((1, K), device=device, dtype=torch.float32),
    }


def run(stg: StageGroup, ctx: dict) -> None:
    K, D_ = ctx["K"], D
    with stg["init"]:
        ctx["cent_a"].copy_(torch.gather(
            ctx["X_b"].squeeze(0), 0,
            ctx["init_idx"].unsqueeze(-1).expand(-1, D_)
        ).view(1, K, D_))
    cur, nxt = ctx["cent_a"], ctx["cent_b"]
    for _ in range(MAX_ITERS):
        with stg["assign"]:
            cluster_ids = euclid_assign_triton(ctx["X_b"], cur, use_heuristic=True)
        with stg["update"]:
            _, _, _ = triton_lloyd_centroid_step_euclid(
                ctx["X_b"], cluster_ids, cur,
                sums_buf=ctx["sums_buf"], cnts_buf=ctx["cnts_buf"],
                new_buf=nxt, shift_buf=ctx["shift_buf"],
            )
        with stg["misc"]:
            cur, nxt = nxt, cur


def main() -> None:
    print(f"[breakdown:kmeans] sweeping K at N={N:,}, D={D}, iters={MAX_ITERS}")
    results = run_multi_shape(SHAPES, prepare, run, STAGES,
                                warmup=1, repeat=3)

    write_multi_shape_md(
        prim="kmeans",
        shape_axis=f"K (n_clusters) at N={N:,}, D={D}, iters={MAX_ITERS}, fp32",
        results=results,
        stage_names=STAGES,
        notes=("Lloyd loop = (assign + update) × max_iters. "
               "`assign` does per-row O(K·D) work; "
               "`update` does N·D streaming + K·D shift."),
        sensitivity=(
            "As **K grows from 1K → 100K**, the `assign` share rises from "
            "~70 % to ~99.8 % of the wall: each row's argmin scans more "
            "centroids, while `update` (whose N·D streaming is K-independent) "
            "stays roughly constant in absolute ms. At small K, `update` "
            "and `init` contribute meaningfully (the K·D shift kernel is "
            "no longer drowned out); at huge K, `assign` dominates so "
            "thoroughly that further optimization should focus exclusively "
            "on the assign kernel (e.g. the x²-free signed-score reformulation, "
            "split-D for large-D, FA3 warp specialisation for large-K — see §2)."
        ),
    )
    free_gpu()


if __name__ == "__main__":
    main()
