"""Per-component time breakdown for flash_ridge across multiple workloads.

Pipeline (mirrors ``flashlib/primitives/ridge/triton/ridge.py::
triton_ridge_regression`` with ``tol=None``, ``alpha=1.0`` — the
fp32-exact path that the heavy benchmark calls), extended to the
multi-target Y shape used by the heavy ridge audit:

  * xtx           — ``Xt = X.T.contiguous()`` + ``XtX = gemm(Xt, X)``
                     (T-independent).
  * add_alpha     — ``XtX.diagonal().add_(alpha)``.
  * cholesky      — ``torch.linalg.cholesky(XtX)`` (SPD because of αI).
  * xty           — ``gemm(Xt, Y)`` — batched when T>1 (Y is (N, T)).
  * tri_solves    — ``cholesky_solve(Xty, L)`` — batched RHS over T.
  * refinement    — the default n_refine=1 alpha-aware refinement step
                     (``r = Y − X@W``; ``Xtr = X.T@r − α·W``; one
                     batched cholesky_solve + add).

Workload axis: **T (multi-target dimension)** at fixed N=2M, D=512,
α=1.0, n_refine=1.  Component asymptotics: xtx is T-independent;
xty is O(N·D·T); tri_solves is O(D²·T); refinement is O(N·D·T)
(two batched GEMMs over T).  Heavy headline sits at T=1.
"""
from __future__ import annotations

import torch

from flashlib.linalg.gemm import gemm as _flash_gemm

from ._common import (
    StageGroup, free_gpu, run_multi_shape, write_multi_shape_md,
)

N, D = 2_000_000, 512
ALPHA = 1.0
N_REFINE = 1

SHAPES = [
    ("T=1",  {"T":  1}),
    ("T=16", {"T": 16}),
    ("T=64", {"T": 64}),
]
STAGES = ["xtx", "add_alpha", "cholesky", "xty", "tri_solves", "refinement"]


def prepare(T: int) -> dict:
    torch.manual_seed(0)
    device = "cuda"
    X = torch.randn(N, D, device=device, dtype=torch.float32)
    w_true = torch.randn(D, T, device=device, dtype=torch.float32) * 0.1
    noise = 0.05 * torch.randn(N, T, device=device, dtype=torch.float32)
    Y = X @ w_true + noise
    if T == 1:
        Y = Y.squeeze(-1)
    return {"X": X, "Y": Y, "T": T}


def run(stg: StageGroup, ctx: dict) -> None:
    X, Y, T = ctx["X"], ctx["Y"], ctx["T"]
    with stg["xtx"]:
        Xt = X.transpose(0, 1).contiguous()
        XtX = _flash_gemm(Xt, X, tol=None)
    with stg["add_alpha"]:
        XtX.diagonal().add_(ALPHA)
    with stg["cholesky"]:
        L = torch.linalg.cholesky(XtX)
    with stg["xty"]:
        if T == 1:
            Xty = _flash_gemm(Xt, Y.unsqueeze(1), tol=None).squeeze(1)
        else:
            Xty = _flash_gemm(Xt, Y, tol=None)
    with stg["tri_solves"]:
        if T == 1:
            W = torch.cholesky_solve(Xty.unsqueeze(1), L).squeeze(1)
        else:
            W = torch.cholesky_solve(Xty, L)
    for _ in range(N_REFINE):
        with stg["refinement"]:
            r = Y - X @ W
            Xtr = X.T @ r
            Xtr -= ALPHA * W
            if T == 1:
                delta = torch.cholesky_solve(Xtr.unsqueeze(1), L).squeeze(1)
            else:
                delta = torch.cholesky_solve(Xtr, L)
            W = W + delta
    _ = W.shape


def main() -> None:
    print(f"[breakdown:ridge] sweeping T at N={N:,}, D={D}, "
          f"alpha={ALPHA}, n_refine={N_REFINE}")
    results = run_multi_shape(SHAPES, prepare, run, STAGES,
                              warmup=1, repeat=3)

    write_multi_shape_md(
        prim="ridge",
        shape_axis=(f"T (multi-target dim) at N={N:,}, D={D}, "
                    f"alpha={ALPHA}, fp32 exact, n_refine={N_REFINE}"),
        results=results,
        stage_names=STAGES,
        notes=("fp32-exact (tol=None) closed-form ridge with batched "
               "RHS over T.  `xtx` is the dominant T-independent "
               "covariance GEMM; `xty` and `tri_solves` switch from "
               "GEMV (T=1) to batched GEMM (T>1), so their absolute "
               "cost scales O(N·D·T) and O(D²·T) respectively.  The "
               "`refinement` stage runs one alpha-aware fp32 polishing "
               "step (residual + Xtr GEMM + −αW + batched "
               "cholesky_solve + add)."),
        sensitivity=(
            "As **T grows from 1 → 64 (64× sweep)**, `xtx`, `add_alpha` "
            "and `cholesky` stay flat (they touch only XtX, which is "
            "T-independent), while `xty`, `tri_solves`, and the per-"
            "target work inside `refinement` all scale roughly linearly "
            "with T (the refinement two GEMMs are O(N·D·T)).  Because "
            "`xtx` is the single biggest cost at every shape, its **% "
            "share shrinks** as T grows: at T=1 the closed-form solve "
            "spends almost all its time inside the covariance GEMM, "
            "but at T=64 the batched `xty` and `refinement` GEMMs each "
            "take a meaningfully larger slice.  The cholesky factor "
            "(O(D³), one-shot per fit) becomes essentially free at "
            "high T because it is reused across all target columns.  "
            "Practical implication: at small T the lever is the "
            "covariance GEMM (same headline as flash_linear_regression); "
            "at large T the levers also include the batched RHS solve "
            "and the refinement GEMM — fusing the two refinement "
            "matmuls + the −αW + the add into a single kernel would "
            "be the biggest multi-target win."
        ),
    )
    free_gpu()


if __name__ == "__main__":
    main()
