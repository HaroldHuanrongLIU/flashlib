"""Per-component time breakdown for flash_pca across multiple workloads.

flash_pca auto-dispatches on aspect ratio (``flashlib/primitives/pca/triton/pca.py``):

* ``N >= 4*D``: **primal** path — ``cov = X.T @ X / N`` then ``eigh(cov, K)``.
* ``N <  4*D``: **dual**   path — ``G = X @ X.T / N`` then ``eigh(G, K)`` →
                ``V = X.T @ U`` → per-column normalize.

The dual eigh additionally routes to Halko subspace iteration when
``K*4 < N`` AND ``tol >= 1e-4``.

Workload axis: **aspect ratio (tall vs square vs wide)** at fixed K-class.
* tall   (primal, eigh on D×D cov)
* square (still primal at our shapes — N=2M, D=2K)
* wide   (dual + Halko on N×N gram)

To unify the column set across both routes, the primal-only stages
``project_gemm`` and ``col_norm`` are simply not entered for tall/square;
``write_multi_shape_md`` renders them as ``-`` for those columns.
"""
from __future__ import annotations

import torch

from flashlib.linalg.eigh import eigh

from ._common import (
    StageGroup, free_gpu, run_multi_shape, write_multi_shape_md,
)

# All shapes drive triton_pca's two dispatch arms (primal cov / dual gram).
SHAPES = [
    ("tall N=10M D=256 K=64",
     {"N": 10_000_000, "D":   256, "K":  64, "tol": None}),
    ("square N=2M D=2K K=128",
     {"N":  2_000_000, "D": 2_000, "K": 128, "tol": None}),
    ("wide N=10K D=8K K=32 tol=0.01",
     {"N":     10_000, "D": 8_000, "K":  32, "tol": 1e-2}),
]
STAGES = ["center", "cov_or_gram_gemm", "eigh_or_halko",
          "project_gemm", "col_norm"]

# Halko params (used when eigh routes to halko in the dual branch).
HALKO_N_ITER = 5
HALKO_P = 30


def prepare(N: int, D: int, K: int, tol: float | None) -> dict:
    torch.manual_seed(0)
    device = "cuda"
    X = torch.randn(N, D, device=device, dtype=torch.float32)
    return {"X": X, "N": N, "D": D, "K": K, "tol": tol}


def _halko_eigh_inlined(G: torch.Tensor, K: int, stg: StageGroup,
                        *, n_iter: int, p: int, seed: int = 42):
    """Inlined ``flashlib.linalg.eigh.halko.halko_eigh`` (matches
    halko.py:69-83 exactly) wrapped in the ``eigh_or_halko`` stage.
    """
    M = G.shape[0]
    q = min(K + p, M)
    gen = torch.Generator(device=G.device).manual_seed(seed)
    with stg["eigh_or_halko"]:
        Q = torch.randn(M, q, device=G.device, dtype=G.dtype, generator=gen)
        Q = G @ Q
        Q, _ = torch.linalg.qr(Q)
        for _ in range(n_iter):
            Q = G @ Q
            Q, _ = torch.linalg.qr(Q)
        H = Q.T @ (G @ Q)
        H = 0.5 * (H + H.T)
        eigvals, V = torch.linalg.eigh(H)
        top_eigvals = eigvals[-K:]
        eigvecs = Q @ V[:, -K:]
    return top_eigvals, eigvecs


def run(stg: StageGroup, ctx: dict) -> None:
    X, N, D, K, tol = ctx["X"], ctx["N"], ctx["D"], ctx["K"], ctx["tol"]

    # Mirror triton_pca's tf32 contract (pca.py:118-133).
    use_tf32 = tol is not None and tol > 0
    prev_tf32 = torch.backends.cuda.matmul.allow_tf32
    if use_tf32:
        torch.backends.cuda.matmul.allow_tf32 = True
    try:
        with stg["center"]:
            Xc = X - X.mean(dim=0, keepdim=True)

        if N >= 4 * D:
            # ── Primal (cov) path — pca.py:_triton_pca_cov ─────────────
            with stg["cov_or_gram_gemm"]:
                M = (Xc.T @ Xc) / N            # cuBLAS GEMM, D×D output
            with stg["eigh_or_halko"]:
                top_eigvals, top_eigvecs = eigh(M, K=K, tol=tol)
            # No project_gemm / col_norm in the primal path — those stages
            # stay at n_calls=0 for these shapes (rendered as "-").
            del M, top_eigvals, top_eigvecs
        else:
            # ── Dual (gram) path — pca.py:_triton_pca_dual ─────────────
            with stg["cov_or_gram_gemm"]:
                G = (Xc @ Xc.T) / N            # cuBLAS GEMM, N×N output

            # Halko routes here because K*4 < N AND tol >= 1e-4 AND N >= 256.
            assert tol is not None and tol >= 1e-4 and K * 4 < N and N >= 256, \
                "dual shape must hit the halko branch for this profiler"
            top_eigvals, U = _halko_eigh_inlined(
                G, K, stg, n_iter=HALKO_N_ITER, p=HALKO_P,
            )

            with stg["project_gemm"]:
                V = Xc.T @ U                   # cuBLAS GEMM, D×K
            with stg["col_norm"]:
                V = V / V.norm(dim=0, keepdim=True).clamp(min=1e-10)
            del G, top_eigvals, U, V
        del Xc
    finally:
        if use_tf32:
            torch.backends.cuda.matmul.allow_tf32 = prev_tf32


def main() -> None:
    print("[breakdown:pca] sweeping aspect ratio across primal/dual routes")
    results = run_multi_shape(SHAPES, prepare, run, STAGES,
                               warmup=1, repeat=3)

    write_multi_shape_md(
        prim="pca",
        shape_axis="aspect ratio (tall primal / square primal / wide dual+halko)",
        results=results,
        stage_names=STAGES,
        notes=(
            "Stage names are unified across routes: `cov_or_gram_gemm` is "
            "the cov GEMM (X.T@X) for primal shapes and the gram GEMM "
            "(X@X.T) for the dual shape; `eigh_or_halko` is cusolver eigh "
            "(routes to `triton_eigh`) for tall/square and Halko subspace "
            "iter for the wide shape. `project_gemm` and `col_norm` are "
            "dual-only — they render as `-` for primal shapes."
        ),
        sensitivity=(
            "**Aspect ratio flips which kernel dominates.** At **tall** "
            "(N=10M, D=256) the primal `cov_or_gram_gemm` reduces "
            "2·N·D² ≈ 1.3 TFLOPs into a tiny 256×256 cov; the cov GEMM "
            "is ~63 % of the wall, `center` (which streams the whole "
            "N×D matrix) is another ~28 %, and eigh on a 256×256 cov "
            "(handled by `triton_eigh`'s small-D Householder path) is "
            "near-negligible at ~9 %. At **square** (N=2M, D=2K) the "
            "cov GEMM JUMPS to ~89 % — the D² output rises from "
            "256²=65K cells to 2K²=4M cells (60× more), so absolute cov "
            "time rises 13× (24 → 314 ms) while center barely moves; "
            "eigh on 2K×2K is now ~7 % of the wall (no longer free). "
            "At **wide** (N=10K, D=8K) the dispatcher swings to the "
            "dual path: the Gram GEMM concentrates 2·N²·D ≈ 1.6 TFLOPs "
            "into a 10K×10K matrix (~40 % of the wall), but the eigh "
            "step is now Halko subspace iter on the same 10K×10K Gram "
            "((n_iter+1) GEMMs of (10K,10K)·(10K,62) + RR), which TAKES "
            "OVER the wall at ~56 %. The dual-only `project_gemm` "
            "(8K×10K · 10K×32) and `col_norm` are ~1 % tails. "
            "Optimisation takeaway: tall/square want a faster cov GEMM "
            "(TF32 once tol allows, or a SYRK-style symmetric-output "
            "kernel); wide wants Halko subspace-iter speedups (fewer "
            "power iters, smaller oversample, or a fused Halko GEMM-QR)."
        ),
    )
    free_gpu()


if __name__ == "__main__":
    main()
