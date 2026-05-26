"""Per-component time breakdown for flash_dbscan across two algorithmic paths.

Workload axis: **D (vector dimension)**, which routes between two distinct
code paths in ``flashlib.primitives.dbscan.triton.dbscan.flash_dbscan``:

  * D=2  -- ``_flash_dbscan_grid``: dense 2D grid index + 3×3 cell scan
            kernel (`_grid_radius_2d_kernel`). No flash_knn call; the
            kernel emits ε-neighbours and per-row degree directly.
  * D=16 -- ``_flash_dbscan_brute`` (the headline shape): flash_knn
            top-K + an eps filter on the returned squared distances.
  * D=128 -- same brute path; the bf16 GEMM tile inside flash_knn now
            dominates more strongly.

To keep the table aligned across both paths the stage list is unified:

  data_prep      : H2D + .contiguous() (make_blobs cached in `prepare()`).
  neighbor_scan  : grid radius kernel (D=2) OR flash_knn brute (D>=3).
                   At D=2 the kernel emits ``deg`` and ``nbr_idx``
                   directly, so the eps_filter stage is skipped.
  eps_filter     : (D>=3 only) knn_dist_sq <= eps^2 -> deg / nbr / core_mask.
                   Renders as "-" for the D=2 grid path.
  cc             : core-core edge construction + ``flash_cc_from_edges``.
  border         : per-row min core-neighbour label.
  compact        : ``torch.unique`` relabel into ``[0, n_clusters)``.

The grid and brute neighbour-scan paths are inlined from
``_flash_dbscan_grid`` and ``_flash_dbscan_brute`` respectively (same
kernel arguments and dispatch logic); the shared cc/border/compact
stages are inlined from the tail of ``flash_dbscan`` -- exactly the
same code path either branch falls through to once ``deg`` /
``nbr_idx`` / ``core_mask`` are in hand.
"""
from __future__ import annotations

import numpy as np
import torch
import triton
from sklearn.datasets import make_blobs

from flashlib.primitives.knn import flash_knn
from flashlib.kernels.flash_mst import flash_cc_from_edges
from flashlib.primitives.dbscan.triton.dbscan import (
    _grid_radius_2d_kernel,
    _build_grid_index,
)

from ._common import (
    StageGroup, free_gpu, run_multi_shape, write_multi_shape_md,
)

N_CENTERS = 20
MAX_NEIGHBORS = 32

SHAPES = [
    ("D=2 grid",        {"N": 2_000_000, "D": 2,
                         "eps": 0.5,  "min_samples": 5}),
    ("D=16 brute-low",  {"N": 1_000_000, "D": 16,
                         "eps": 3.5,  "min_samples": 5}),
    ("D=128 brute-high", {"N": 200_000, "D": 128,
                          "eps": 11.0, "min_samples": 5}),
]
STAGES = ["data_prep", "neighbor_scan", "eps_filter",
          "cc", "border", "compact"]


def prepare(N: int, D: int, eps: float, min_samples: int) -> dict:
    X_np, _ = make_blobs(
        n_samples=N, centers=N_CENTERS, n_features=D,
        cluster_std=1.0, random_state=0,
    )
    X_np = X_np.astype(np.float32)
    K = max(min_samples, MAX_NEIGHBORS)
    K = min(K, N)
    return {
        "X_np": X_np,
        "N": N, "D": D, "eps": eps,
        "min_samples": min_samples, "K": K,
        "eps_sq": float(eps) ** 2,
    }


def run(stg: StageGroup, ctx: dict) -> None:
    device = "cuda"
    N, D = ctx["N"], ctx["D"]
    K = ctx["K"]
    eps = ctx["eps"]
    eps_sq = ctx["eps_sq"]
    min_samples = ctx["min_samples"]
    INT_MAX = 2 ** 31 - 1

    with stg["data_prep"]:
        X = torch.from_numpy(ctx["X_np"]).to(device, non_blocking=False)
        X = X.contiguous()

    if D == 2:
        with stg["neighbor_scan"]:
            (sorted_ptr, cell_start, cell_end, GW, GH,
             inv_eps, x_min, y_min) = _build_grid_index(X, eps)
            deg = torch.zeros(N, dtype=torch.int32, device=device)
            nbr_idx = torch.full((N, K), -1, dtype=torch.int32, device=device)
            BN = 32
            grid_launch = (triton.cdiv(N, BN),)
            _grid_radius_2d_kernel[grid_launch](
                X, sorted_ptr, cell_start, cell_end,
                deg, nbr_idx,
                N=N, K=K, GW=GW, GH=GH,
                INV_EPS=inv_eps, EPS_SQ=eps_sq,
                GRID_X_MIN=x_min, GRID_Y_MIN=y_min,
                BN=BN,
                num_warps=4, num_stages=1,
            )
            core_mask = deg >= min_samples
    else:
        with stg["neighbor_scan"]:
            knn_dist_sq, knn_idx = flash_knn(
                X[None], X[None], k=K, tol=None,
            )
            knn_dist_sq = knn_dist_sq[0]
            knn_idx = knn_idx[0]

        with stg["eps_filter"]:
            valid = knn_dist_sq <= eps_sq
            deg = valid.sum(dim=1).to(torch.int32)
            nbr_idx = torch.where(
                valid, knn_idx.to(torch.int32),
                torch.full_like(knn_idx, -1, dtype=torch.int32),
            )
            core_mask = deg >= min_samples

    with stg["cc"]:
        K_eff = nbr_idx.shape[1]
        nbr_idx_i64 = nbr_idx.to(torch.int64)
        valid_slot = nbr_idx >= 0
        core_per_row = core_mask[:, None].expand(-1, K_eff)
        nbr_idx_safe = torch.where(
            valid_slot, nbr_idx_i64, torch.zeros_like(nbr_idx_i64),
        )
        core_per_col = core_mask[nbr_idx_safe] & valid_slot
        edge_mask = valid_slot & core_per_row & core_per_col
        rows_e = (
            torch.arange(N, device=device, dtype=torch.int32)
            .view(-1, 1).expand(-1, K_eff).contiguous()
        )[edge_mask].contiguous()
        cols_e = nbr_idx[edge_mask].contiguous()
        label_cc = flash_cc_from_edges(rows_e, cols_e, N)
        label = torch.where(
            core_mask, label_cc, torch.full_like(label_cc, -1),
        )

    with stg["border"]:
        nbr_labels = label[nbr_idx_safe]
        nbr_is_core = core_mask[nbr_idx_safe] & valid_slot
        border_cand = torch.where(
            valid_slot & nbr_is_core, nbr_labels,
            torch.full_like(nbr_labels, INT_MAX),
        )
        min_core_label = border_cand.min(dim=1).values
        is_border = (
            (~core_mask) & (min_core_label != INT_MAX)
            & (min_core_label >= 0)
        )
        label = torch.where(is_border, min_core_label, label)

    with stg["compact"]:
        valid_label = label >= 0
        if valid_label.any():
            unique, inv = torch.unique(label[valid_label], return_inverse=True)
            compact = torch.full_like(label, -1)
            compact[valid_label] = inv.to(torch.int32)
            label = compact
        _ = label  # consumer


def main() -> None:
    print("[breakdown:dbscan] sweeping D across grid vs brute paths")
    results = run_multi_shape(SHAPES, prepare, run, STAGES,
                                warmup=1, repeat=3)

    write_multi_shape_md(
        prim="dbscan",
        shape_axis="D (vector dim; grid path at D=2, brute path at D>=3)",
        results=results,
        stage_names=STAGES,
        notes=(
            "data_prep = H2D + .contiguous() (make_blobs cached in prepare). "
            "neighbor_scan = `_grid_radius_2d_kernel` (D=2) OR flash_knn "
            "brute top-K (D>=3). eps_filter = deg/nbr/core_mask from the "
            "K NN distances; the D=2 grid path skips this stage (deg + "
            "nbr_idx come straight out of the kernel) -- rendered as "
            "'-'. cc/border/compact are shared from the tail of "
            "`flash_dbscan`."
        ),
        sensitivity=(
            "**At D=2** the grid radius kernel scans only the 3×3 = 9 "
            "neighbour cells per query (cell-side eps; no flash_knn) and "
            "still dominates the wall (~99%) because dense clusters give "
            "the inner cell-loop ~10K iterations per CTA. The `cc` pass "
            "is the next-largest share at sub-1% -- the bf16 GEMM that "
            "drives the D>=3 path doesn't appear at all (different "
            "kernel sequence). **At D=16** (the headline 8.8x vs cuML "
            "row) the brute path kicks in: `neighbor_scan` is now a "
            "single flash_knn fused bf16 GEMM + on-chip top-K call and "
            "takes ~99% of the wall at N=1M; the post-kNN `cc`/`border`/"
            "`compact` tail is sub-ms even at million-row scale. **At "
            "D=128** the bf16 GEMM tile is still the dominant cost "
            "(~94%) but `data_prep` (a 100 MB H2D of fp32 inputs) finally "
            "becomes visible at ~5%. The optimisation lever therefore "
            "differs by path: at D=2 the grid kernel's per-cell scan is "
            "the bottleneck (and the only one worth attacking on this "
            "shape); at D>=16 any flash_knn improvement (bf16 -> fp8, "
            "FA3 warp-specialisation) moves "
            "the headline number directly. The absolute-ms drop from "
            "D=16 (1450 ms at N=1M) to D=128 (183 ms at N=200K) is "
            "explained by the N² factor in flash_knn pairs: 1M² / 200K² "
            "= 25x more pairs at D=16, partly offset by 8x more bytes "
            "per pair at D=128."
        ),
    )
    free_gpu()


if __name__ == "__main__":
    main()
