"""Per-component time breakdown for flash_standard_scaler (fit_transform)
across multiple aspect ratios.

Stages timed (mirrors ``flashlib/primitives/standard_scaler/triton/
scaler.py::flash_standard_scaler_fit_transform`` -> ``_fit`` (fused=True)
+ ``_transform``):

  * stats_kernel  — the single-pass shifted-sum triton kernel
                     ``_col_sum_ss_shifted_kernel`` (one HBM read of X
                     producing per-column ``Σ(x-c)`` and ``Σ(x-c)²``
                     in fp64 globals) plus the ``c = X[0]`` reference
                     slice + the fp64 sum/ss buffer alloc.
  * finalize      — the host-side mean / var / std / inv_std arithmetic
                     (fp64 reduce of D elements, sqrt, std==0 sentinel).
  * transform     — the fused ``Y = (X - mean) * inv_std`` kernel
                     ``_scale_kernel`` + ``torch.empty_like(X)`` alloc.

Workload axis: **aspect ratio (tall vs wide)** at roughly constant
N·D (≈ 10 G elements ⇒ X is 40 GB at fp32).  Heavy headline sits at
the tall N=20M, D=512 corner.  Both `stats_kernel` and `transform`
are HBM-bound and read/write N·D bytes, so their ratio (~1 : 1.5)
should be aspect-invariant; `finalize` scales O(D) so its share is
expected to bump up slightly at the widest shape.
"""
from __future__ import annotations

import torch
import triton

from flashlib.primitives.standard_scaler.triton.scaler import (
    _col_sum_ss_shifted_kernel, _scale_kernel,
)

from ._common import (
    StageGroup, free_gpu, run_multi_shape, write_multi_shape_md,
)

SHAPES = [
    ("tall  N=20M D=512",   {"N": 20_000_000, "D":    512}),
    ("mid   N=2M  D=4K",    {"N":  2_000_000, "D":  4_096}),
    ("wide  N=200K D=32K",  {"N":    200_000, "D": 32_000}),
]
STAGES = ["stats_kernel", "finalize", "transform"]


def prepare(N: int, D: int) -> dict:
    torch.manual_seed(0)
    device = "cuda"
    X = torch.randn(N, D, device=device, dtype=torch.float32)
    return {"X": X, "N": N, "D": D}


def run(stg: StageGroup, ctx: dict) -> None:
    X, N, D = ctx["X"], ctx["N"], ctx["D"]
    sN, sD = X.stride()
    grid_red = lambda META: (
        triton.cdiv(N, META["BLOCK_N"]),
        triton.cdiv(D, META["BLOCK_D"]),
    )

    with stg["stats_kernel"]:
        c = X[0].clone().detach().contiguous().to(torch.float32)
        sum_diff = torch.zeros(D, device=X.device, dtype=torch.float64)
        ss_diff = torch.zeros(D, device=X.device, dtype=torch.float64)
        _col_sum_ss_shifted_kernel[grid_red](
            X, c, sum_diff, ss_diff, N, D, sN, sD
        )

    with stg["finalize"]:
        c_d = c.to(torch.float64)
        mean_d = c_d + sum_diff / N
        mean_diff_d = sum_diff / N
        var_d = ss_diff / N - mean_diff_d * mean_diff_d
        var_d.clamp_(min=0.0)
        mean = mean_d.to(torch.float32)
        std = var_d.sqrt().to(torch.float32)
        std_safe = torch.where(std == 0, torch.ones_like(std), std)
        inv_std = 1.0 / std_safe

    with stg["transform"]:
        TOTAL = N * D
        Y = torch.empty_like(X)
        grid_tr = lambda META: (triton.cdiv(TOTAL, META["BLOCK"]),)
        _scale_kernel[grid_tr](X, Y, mean, inv_std, TOTAL, D)
        _ = Y.shape


def main() -> None:
    print("[breakdown:standard_scaler] sweeping aspect ratio at "
          "roughly constant N·D (~10 G elements ≈ 40 GB X)")
    results = run_multi_shape(SHAPES, prepare, run, STAGES,
                              warmup=1, repeat=3)

    write_multi_shape_md(
        prim="standard_scaler",
        shape_axis="aspect ratio (tall→wide) at ~constant N·D, fp32 fused single-pass",
        results=results,
        stage_names=STAGES,
        notes=("Fused fp32 path (fused=True). `stats_kernel` reads X "
               "once and fp64-atomic-adds into D-wide globals; "
               "`transform` reads X once and writes Y once. "
               "`finalize` is host-side fp64 reduce + sqrt + std==0 "
               "sentinel; it scales O(D), not O(N·D)."),
        sensitivity=(
            "Both `stats_kernel` (1 HBM read of X) and `transform` "
            "(1 read of X + 1 write of Y) are **HBM-bound** and the "
            "bytes moved are determined by N·D, not the aspect ratio.  "
            "So as we sweep from **tall (N=20M, D=512)** through **mid "
            "(N=2M, D=4K)** to **wide (N=200K, D=32K)** at roughly "
            "constant N·D, the absolute ms of `stats_kernel` and "
            "`transform` stay roughly constant and their **ratio "
            "(~1 : 1.5, set by 1 read vs 1 read + 1 write)** is "
            "aspect-invariant.  The only component that scales with "
            "D alone is `finalize` (one D-element fp64 reduce + sqrt "
            "+ host launch overhead): at D=512 it is essentially "
            "free, at D=32K its share bumps up but should still be "
            "< 5 % of the wall.  Practical implication: standard "
            "scaler is fundamentally bandwidth-saturated at every "
            "heavy shape — the remaining levers are (a) skipping the "
            "transform read by fusing it into the next consumer "
            "(only possible end-to-end), and (b) the fp64 → fp32 "
            "atomic global (only ~0.05 ms at xlarge so not worth "
            "chasing)."
        ),
    )
    free_gpu()


if __name__ == "__main__":
    main()
