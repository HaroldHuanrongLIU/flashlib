"""Flash-KNN CuteDSL implementation: FA3 fully-fused path (x²-free).

This module exposes a SINGLE public entry point:

    cutedsl_flash_knn(x, c, k, *, autotune=False, ...)

It is FA3-style end-to-end fused: TMA bulk-tensor loads for X / C,
Hopper SM90 WGMMA for the cross-term GEMM, and a per-thread top-K
register heap -- no ``(B, N, M)`` distance matrix is ever materialised
in HBM. The kernel scores ``s = c_sq[m] − 2·<x, c>`` directly; the
``x_sq`` term is constant per row so dropping it preserves argmin-K
while saving one HBM tensor, one register-file pass, and a ``d >= 0``
clamp.

Indices only
------------

The fused kernel writes only ``(B, N, k) int32`` indices -- the true
``||x − c[idx]||²`` distances are recovered (cheaply) by
:func:`flashlib.kernels.distance.triton_knn_gather_sqdist` outside the
fused pass. This matches the contract of the new
:func:`flashlib.primitives.knn.flash_knn` entry point.

Public API
----------
``cutedsl_flash_knn(x, c, k)``
    FA3-style fully-fused KNN (Hopper-only). Pure executor: runs the FA3
    kernel or raises :class:`CuteDSLUnsupported` for any shape outside the
    FA3 sweet spot (B != 1, fp32 input, ``D % 16 != 0``, ``D > 512``,
    ``k > k_max``, or first-call compile failure). It never delegates to
    Triton -- the dispatcher (:func:`flashlib.primitives.knn.flash_knn`)
    owns all backend routing and fallback. Returns ``(B, N, k) int32``
    indices.

``cutedsl_available()``
    Cheap probe for whether the cutlass-dsl Python bindings imported.
"""
from __future__ import annotations

import os
from typing import Optional

import torch

from flashlib.linalg.gemm import storage_dtype_for
from flashlib.primitives.knn.triton._row_norm import _get_or_compute_csq

# =============================================================================
# Maxtree top-K toggle (Blackwell BUILD design ported to Hopper)
# =============================================================================
#
# The ``maxtree`` / ``smem_maxtree`` strategies are the register / smem ports of
# the Blackwell BUILD top-K (unsorted heap + group-min-4 prune + worst-of-K
# recompute). The worst-of-K recompute is K-adaptive (``_worst_row``): a shallow
# balanced max-tree for K<=10, a streaming running-max for K>=11. The streaming
# scan keeps only 2 scalars live instead of all K leaves, so the heap stops
# spilling at high K -- the learning ported from the Blackwell BUILD kernel,
# which hit the same MLIR spill and made the same switch. Measured win regions
# on this H100 (see ``benchmarks/results/micro_knn_maxtree_topk.md``):
#
#   * SEARCH (1-thread-per-row ``smem_maxtree``, WS3): wins at K<=16
#     (1.13-1.36x across M=32k..131k); spills / loses at K>=32.
#   * BUILD (register ``maxtree``, non-WS): wins at LARGE N across a wide K
#     band -- D<=128, N>=50k, K in [5,24] at the BM256/BN64 tile (K=5
#     1.15-1.25x and K in [17,24] 2-4x vs the strategy it replaces; all
#     exact). The K>10 half of the band is new: the streaming worst-of-K
#     stops the heap spilling that capped the old max-tree at K<=10, and the
#     unsorted heap then crushes the sorted ``sortmerge`` that used to own
#     K>16. K=4 loses (one prune group -> ``perthread``); small N loses
#     (fixed merge + epilogue-sort dominates the short mainloop). The
#     ``sortmerge`` bitonic only re-takes the lead at K~28-32, and only at
#     its own BM128/BN128 tile -> K>=25 hands back to ``sortmerge`` there.
#
# The heuristic therefore has THREE modes:
#
#   * "auto" (default): adopt each port ONLY in its measured win region
#     (search K<=16; build D<=128 / N>=50k / K in [5,24]); keep the
#     existing strategies everywhere else. This is the production default
#     -- flash_knn transparently "eats" the wins without regressing any
#     other shape.
#   * "on":  full swap -- ``maxtree`` (build) + ``smem_maxtree`` (search) for
#     every shape. The benchmark's NEW column.
#   * "off": the original strategies everywhere (``smem_perthread`` for
#     search). The benchmark's OLD baseline column.
#
# Drive it via ``FLASHLIB_KNN_MAXTREE`` (``1``/``0``, env, re-read each call)
# or :func:`set_maxtree_enabled` (overrides env; ``None`` reverts to "auto").

_USE_MAXTREE: Optional[bool] = None  # None -> "auto" (env / default).


def set_maxtree_enabled(enabled: Optional[bool]) -> None:
    """Force the maxtree top-K fully on / off (``None`` reverts to "auto")."""
    global _USE_MAXTREE
    _USE_MAXTREE = enabled


def _maxtree_mode() -> str:
    """Return one of ``"on"`` / ``"off"`` / ``"auto"`` (see module note)."""
    if _USE_MAXTREE is True:
        return "on"
    if _USE_MAXTREE is False:
        return "off"
    env = os.environ.get("FLASHLIB_KNN_MAXTREE")
    if env is None:
        return "auto"
    return "off" if env.lower() in ("0", "", "false", "no", "off") else "on"


# =============================================================================
# CuteDSL availability
# =============================================================================

_CUTEDSL_AVAILABLE = False
_CUTE_IMPORT_ERROR: Optional[Exception] = None


def _try_init_cutedsl() -> bool:
    """Lazy import probe; returns True iff cutlass-dsl is importable."""
    global _CUTEDSL_AVAILABLE, _CUTE_IMPORT_ERROR
    if _CUTEDSL_AVAILABLE:
        return True
    if _CUTE_IMPORT_ERROR is not None:
        return False
    try:
        import cutlass  # noqa: F401
        import cutlass.cute as cute  # noqa: F401
        import cutlass.cute.runtime as cute_rt  # noqa: F401

        globals()["cutlass"] = cutlass
        globals()["cute"] = cute
        globals()["cute_rt"] = cute_rt
        _CUTEDSL_AVAILABLE = True
        return True
    except Exception as e:  # noqa: BLE001
        _CUTE_IMPORT_ERROR = e
        return False


def cutedsl_available() -> bool:
    return _try_init_cutedsl()


# =============================================================================
# Compiled-kernel + DLPack handle caches (shared across call sites).
# =============================================================================

_kernel_cache: dict = {}
_dlpack_cache: dict = {}


def _cached_from_dlpack(t: torch.Tensor):
    """Memoise ``cute_rt.from_dlpack`` per ``(ptr, shape, stride, dtype)``."""
    import cutlass.cute.runtime as cute_rt
    key = (t.data_ptr(), tuple(t.shape), tuple(t.stride()), t.dtype)
    val = _dlpack_cache.get(key)
    if val is None:
        val = cute_rt.from_dlpack(t)
        _dlpack_cache[key] = val
    return val


def _trim_dlpack_cache(max_entries: int = 64):
    if len(_dlpack_cache) > max_entries:
        for k in list(_dlpack_cache.keys())[:-max_entries]:
            del _dlpack_cache[k]


# =============================================================================
# Shape-only heuristic (default; one CuteDSL compile per shape, no sweep)
# =============================================================================
#
# Picks the config along the **build vs search** axis. The per-shape
# autotune sweeps on H200 split cleanly:
#
#   * Build shapes (N >= 10K AND 0.5 <= M/N <= 2): non-WS with larger
#     tiles wins. WS3's pipelined topK doesn't pay off when every CTA
#     already has plenty of M-rows -- the pipeline depth costs
#     register / SMEM that hurts WGMMA throughput.
#       - N >= 50K (very-large build): BM=256
#       - else (10K-50K build):        BM=128
#       - K=4    -> ``perthread`` strategy
#       - K >= 16 -> ``sortmerge`` strategy
#       - Narrow D (<=128) wants perthread regardless of K
#         (autotune confirmed it on D=64, K=16).
#       - BN: 128 default; 64 when BM=256 (narrow D) or
#             when sortmerge + BM=256.
#   * Search shapes (small N OR M >> N): WS3 + smem_perthread wins
#     by a wide margin.
#       - K <= 16: BM=64  BN=128
#       - K  > 16: BM=128 BN=64
#
# All configs compile in ~5-8 s (one config) vs ~5-10 min for the
# full sweep.


def _heuristic_fa3_config(N: int, M: int, D: int, K_PAD: int) -> dict:
    maxtree_mode = _maxtree_mode()  # "on" | "off" | "auto"
    is_build = (N >= 10_000) and (0.5 <= M / N <= 2.0)
    if is_build:
        BM = 256 if N >= 50_000 else 128
        # Strategy depends on (D, K) regime. The autotune-derived rule
        # from upstream flash-knn is:
        #   * Narrow D (<= 128) + K <= 16: ``perthread`` wins
        #     (autotune confirmed K=4, K=16 at D=64).
        #   * Narrow D (<= 128) + K >= 32: ``sortmerge`` -- the
        #     per-thread sequential bitonic in CuteDSL's perthread
        #     grows linearly with K_PAD and at K=32 + BM=256 the
        #     bitonic dominates the kernel (12x slower than sortmerge
        #     here on H200). flashlib's kernel `auto` strategy already
        #     flips at K>16, so this matches its empirical breakpoint.
        #   * Wide D (>= 192) + K <= 4: ``perthread`` (autotune).
        #   * Wide D (>= 192) + K >= 16: ``sortmerge`` (autotune
        #     confirmed BM=128 BN=128 wins at D=256 K ∈ {16, 32}).
        if D <= 128 and K_PAD <= 16:
            strat = "perthread"
            BN = 128 if BM == 128 else 64
        elif D > 128 and K_PAD <= 4:
            strat = "perthread"
            BN = 128
        else:
            strat = "sortmerge"
            BN = 128 if BM == 128 else 64
        # maxtree carve-out (per-K winner sweep on H100; see the module note
        # + benchmarks/results/micro_knn_maxtree_topk.md). The unsorted heap
        # + group-min-4 prune beats every other build top-K for mid/high K at
        # large N. The worst-of-K recompute is K-adaptive
        # (``HopperFlashKnnFused._worst_row``): a shallow balanced max-tree
        # for K<=10, a streaming running-max for K>=11 that keeps only 2
        # scalars live so the heap stops spilling at high K (the learning
        # ported from the Blackwell BUILD kernel). Win band: D<=128, N>=50k,
        # K in [5,24] at the BM256/BN64 tile. K=5 1.15-1.25x vs perthread;
        # K in [17,24] beats sortmerge 2-4x (the old K>16 fallback); all
        # exact. K=4 stays ``perthread`` (wins ~1.05-1.08x); from K~28-32 the
        # sortmerge bitonic finally crosses over (at its BM128/BN128 tile) so
        # K>=25 hands back to sortmerge.
        maxtree_build_win = (D <= 128 and N >= 50_000 and 5 <= K_PAD <= 24)
        if maxtree_mode == "on":
            strat = "maxtree"
        elif maxtree_mode == "auto" and maxtree_build_win:
            strat = "maxtree"
        # sortmerge's bitonic network is ~1.6x faster at BM128/BN128 than at
        # the BM256/BN64 perthread/maxtree tile (measured K=20..32, D=64/128,
        # N>=50k: ~68us vs ~110us; matches the upstream D=256 autotune). Give
        # it that tile whenever it is the final pick -- the maxtree="off"
        # baseline, wide-D builds, and the K>24 / K>16-low-N fallbacks.
        if strat == "sortmerge":
            BM, BN = 128, 128
        return dict(
            BM=BM, BN=BN,
            use_ws=False, topk_strategy=strat,
            use_ws3=False, use_ws4=False, dist_stage=1,
        )

    # Search shapes: WS3 1-thread-per-row. smem_maxtree keeps the same WS3
    # pipeline + chunk-min prune, swapping only the inner top-K.
    if K_PAD <= 16:
        # smem_maxtree wins (or ties) here -> adopt it in "auto" and "on".
        search_strat = (
            "smem_perthread" if maxtree_mode == "off" else "smem_maxtree"
        )
        return dict(
            BM=64, BN=128,
            use_ws=True, topk_strategy=search_strat,
            use_ws3=True, use_ws4=False, dist_stage=3,
        )
    # K>16 search: smem_maxtree spills (loses ~0.64x at K=32) -> only when
    # fully forced on; "auto" keeps the sorted smem_perthread.
    search_strat = "smem_maxtree" if maxtree_mode == "on" else "smem_perthread"
    return dict(
        BM=128, BN=64,
        use_ws=True, topk_strategy=search_strat,
        use_ws3=True, use_ws4=False, dist_stage=3,
    )


# =============================================================================
# Per-shape autotune (opt-in -- sweeps the full FA3 grid).
# =============================================================================
#
# All gated strategies use native fp32 compares that work for the signed
# score (``c_sq - 2 * cross`` -- ``x_sq`` is never materialised).

_autotune_cache: dict = {}
_heuristic_cache: dict = {}

# Query-dim padding granularity for the search path. Must be a multiple of
# every search-tile BM the heuristic / autotune can pick (64 for K<=16, 128
# otherwise), so that padding N up to this never leaves a partial query tile
# for the FA3 kernel to read OOB.
_Q_TILE_PAD = 128


class CuteDSLUnsupported(RuntimeError):
    """The FA3 CuteDSL kNN cannot run/compile a shape (capability gate or a
    compile failure). ``cutedsl_flash_knn`` raises this instead of silently
    delegating to Triton -- the *dispatcher* owns all backend routing, so it
    catches this to fall back to Triton (or surfaces it for an explicit
    ``backend="cutedsl"``)."""


# Cache sentinel: a shape we already proved CuteDSL can't take, so we re-raise
# immediately instead of re-attempting the (failed) compile every call.
_CUTEDSL_FAILED = "unsupported"


def _autotune_fa3(x2d, c2d, c_sq_1d, out_i, k_pad: int, *, verbose: bool = False):
    """Compile + bench every fitting (BM, BN, use_ws, strategy) FA3 config."""
    import cuda.bindings.driver as cuda_drv
    import cutlass
    import cutlass.cute as cute_mod
    from cutlass.cute.runtime import from_dlpack as _from_dlpack

    from flashlib.primitives.knn.cutedsl.hopper_impl import HopperFlashKnnFused

    N, D = x2d.shape
    M = c2d.shape[0]
    smem_capacity = torch.cuda.get_device_properties(
        x2d.device
    ).shared_memory_per_block_optin
    bytes_per = 2

    def _fits(bm, bn):
        return (bm * D * bytes_per + bn * D * bytes_per + 1024) <= smem_capacity

    if k_pad <= 3:
        strategies = ("insert", "perthread", "maxtree", "sortmerge")
    else:
        strategies = ("perthread", "maxtree", "sortmerge")

    bms = (64, 128, 256)
    bns = (64, 128, 256)

    def _is_known_hang(bm, bn, use_ws, strat):
        # Pre-existing FA3 hang in the underlying kernel -- BM=256 WS2
        # candidates wedge the GPU during warmup on H200. They never win
        # the autotune anyway (BM=128 WS2 consistently dominates), so skip.
        if bm == 256 and use_ws:
            return True
        return False

    candidates = [
        (bm, bn, ws, strat, False, False, 1)
        for bm in bms
        for bn in bns
        for ws in (False, True)
        for strat in strategies
        if _fits(bm, bn) and not _is_known_hang(bm, bn, ws, strat)
    ]
    ws3_tiles = [(64, 64), (64, 128), (128, 64), (128, 128)]
    for bm, bn in ws3_tiles:
        if _fits(bm, bn):
            for stage in (3, 2):
                for smem_strat in ("smem_perthread", "smem_maxtree"):
                    candidates.append(
                        (bm, bn, True, smem_strat, True, False, stage)
                    )
    ws4_tiles = [(64, 64), (64, 128), (128, 64)]
    for bm, bn in ws4_tiles:
        if _fits(bm, bn) and bm * bn <= 128 * 64:
            for smem_strat in ("smem_perthread", "smem_maxtree"):
                candidates.append(
                    (bm, bn, True, smem_strat, True, True, 2)
                )

    stream = cuda_drv.CUstream(0)
    best = None
    best_t = float("inf")

    for BM, BN, use_ws, strat, use_ws3, use_ws4, dist_stage in candidates:
        try:
            kernel = HopperFlashKnnFused(
                acc_dtype=cutlass.Float32,
                m_block_size=BM, n_block_size=BN,
                k_pad=k_pad, use_ws=use_ws,
                topk_strategy=strat,
                use_ws3=use_ws3, use_ws4=use_ws4, dist_stage=dist_stage,
            )
            compiled = cute_mod.compile(
                kernel,
                _from_dlpack(x2d), _from_dlpack(c2d),
                _from_dlpack(c_sq_1d), _from_dlpack(out_i),
                stream,
            )
        except Exception as exc:
            if verbose:
                ws_tag = (
                    "ws4" if use_ws4 else
                    ("ws3" if use_ws3 else ("ws2" if use_ws else "no "))
                )
                print(
                    f"  fa3 skip BM={BM} BN={BN} {ws_tag} "
                    f"strat={strat}: {str(exc).splitlines()[0][:80]}"
                )
            continue

        x_dl = _cached_from_dlpack(x2d)
        c_dl = _cached_from_dlpack(c2d)
        cs_dl = _cached_from_dlpack(c_sq_1d)
        oi_dl = _cached_from_dlpack(out_i)
        try:
            for _ in range(3):
                compiled(x_dl, c_dl, cs_dl, oi_dl, stream)
            torch.cuda.synchronize()
            s = torch.cuda.Event(enable_timing=True)
            e = torch.cuda.Event(enable_timing=True)
            s.record()
            for _ in range(8):
                compiled(x_dl, c_dl, cs_dl, oi_dl, stream)
            e.record()
            torch.cuda.synchronize()
            t_us = s.elapsed_time(e) / 8 * 1000
        except Exception:
            continue
        if verbose:
            tag = (
                "WS4" if use_ws4 else
                ("WS3" if use_ws3 else ("WS " if use_ws else "non"))
            )
            print(
                f"  fa3 BM={BM:>3} BN={BN:>3} {tag} stg={dist_stage} "
                f"strat={strat:<14}: {t_us:>7.0f} us"
            )
        if t_us < best_t:
            best_t = t_us
            best = (BM, BN, use_ws, strat, use_ws3, use_ws4, dist_stage, compiled)

    if best is None:
        raise RuntimeError(
            f"fa3 autotune found no working config for "
            f"N={N} D={D} M={M} k_pad={k_pad}"
        )
    return (
        best[0], best[1], best[2], best[3], best[4], best[5], best[6],
        best_t, best[7], stream,
    )


# =============================================================================
# Public entry point
# =============================================================================

def cutedsl_flash_knn(
    x: torch.Tensor,
    c: torch.Tensor,
    k: int,
    *,
    tol: Optional[float] = None,
    autotune: bool = False,
    autotune_verbose: bool = False,
    k_max: int = 32,
) -> torch.Tensor:
    """Hopper FA3 fully-fused kNN. Returns ``(B, N, k) int32`` indices.

    Pure executor: it runs the FA3 kernel for this shape or raises
    :class:`CuteDSLUnsupported`. It never delegates to Triton -- the
    dispatcher (:func:`flashlib.primitives.knn.flash_knn_dispatch`) owns all
    Triton<->CuteDSL routing and fallback. Partial / small query tiles are
    padded up to a full tile internally so any Q is valid.

    Args:
        x: (B, N, D) -- bf16 / fp16 native. fp32 input + ``tol`` selecting
            bf16/fp16 storage is cast once; otherwise an fp32 input raises
            (FA3 path requires 16-bit storage).
        c: (B, M, D) -- same dtype as x, D contiguous.
        k: number of neighbours. ``k > k_max`` raises.
        tol: residual tolerance.
            * ``None`` (default) -- preserve input dtype (EXACT path).
            * Otherwise the standard
              :func:`flashlib.linalg.gemm.storage_dtype_for` cast is
              applied to ``x``/``c`` once internally.
        autotune: if False (default), pick a single FA3 config from
            :func:`_heuristic_fa3_config` -- first call pays one
            CuteDSL compile (~3-8 s) instead of ~5-10 min for the full
            sweep. ``True`` runs the brute-force search over FA3 configs and
            caches the fastest one (whether it beats Triton is the
            dispatcher's call, not this function's).
        autotune_verbose: print per-candidate timings during autotune.
        k_max: gate threshold; ``k > k_max`` raises.

    Returns:
        ``(B, N, k) int32`` indices, sorted ascending by squared L2
        (ties broken by index). True distances per neighbour are
        recovered by
        :func:`flashlib.kernels.distance.triton_knn_gather_sqdist`.

    Raises:
        CuteDSLUnsupported: dtype/shape/k outside the FA3 path, CuteDSL
            unavailable, or the kernel failed to compile for this shape.
    """
    assert x.is_cuda and c.is_cuda
    target_dtype = storage_dtype_for(tol)
    if target_dtype is not None and x.dtype != target_dtype:
        x = x.to(target_dtype)
        c = c.to(target_dtype)
    B, N, D = x.shape
    M = c.shape[1]
    assert c.shape == (B, M, D)
    assert 1 <= k <= M

    # The FA3 kernel's query loop processes full BM-row tiles; a query count
    # that is not a multiple of the tile (especially small-Q search, N < BM)
    # would read past the x buffer -> illegal instruction. (Triton's tl.dot
    # has the mirror problem: it cannot tile Q < 16 and asserts on sm_100.)
    # Pad the query dim up to a full tile, run, then slice back. For Q <= BM
    # this stays a single tile, so it adds no extra DB passes (~free); only
    # the search path needs it (build N is large and tile-aligned).
    _is_build = (N >= 10_000) and (0.5 <= M / N <= 2.0)
    if not _is_build and (N % _Q_TILE_PAD != 0):
        N_pad = ((N + _Q_TILE_PAD - 1) // _Q_TILE_PAD) * _Q_TILE_PAD
        x_pad = x.new_zeros((B, N_pad, D))
        x_pad[:, :N] = x
        return cutedsl_flash_knn(
            x_pad, c, k, tol=tol, autotune=autotune,
            autotune_verbose=autotune_verbose, k_max=k_max,
        )[:, :N]

    if (
        not _try_init_cutedsl()
        or B != 1
        or x.dtype not in (torch.float16, torch.bfloat16)
        or c.dtype != x.dtype
        or D % 16 != 0
        or D > 512
        or k > k_max
    ):
        raise CuteDSLUnsupported(
            f"FA3 path cannot run this shape (B={B}, D={D}, k={k}, "
            f"k_max={k_max}, dtype={x.dtype}, cutedsl={_try_init_cutedsl()})"
        )

    c_sq = _get_or_compute_csq(c).view(M).contiguous()

    x2d = x.view(N, D)
    c2d = c.view(M, D)
    if not x2d.is_contiguous(): x2d = x2d.contiguous()
    if not c2d.is_contiguous(): c2d = c2d.contiguous()

    K_PAD = int(k)
    out_i = torch.empty((N, K_PAD), device=x.device, dtype=torch.int32)

    ac_key = (N, M, D, K_PAD, x.dtype)

    if not autotune:
        cached_h = _heuristic_cache.get(ac_key)
        if cached_h == _CUTEDSL_FAILED:
            raise CuteDSLUnsupported(
                f"FA3 heuristic compile previously failed for {ac_key}")
        if cached_h is None:
            import cuda.bindings.driver as cuda_drv
            import cutlass
            import cutlass.cute as cute_mod
            from cutlass.cute.runtime import from_dlpack as _from_dlpack
            from flashlib.primitives.knn.cutedsl.hopper_impl import (
                HopperFlashKnnFused,
            )

            cfg = _heuristic_fa3_config(N, M, D, K_PAD)
            stream = cuda_drv.CUstream(0)
            try:
                kernel = HopperFlashKnnFused(
                    acc_dtype=cutlass.Float32,
                    m_block_size=cfg["BM"], n_block_size=cfg["BN"],
                    k_pad=K_PAD, use_ws=cfg["use_ws"],
                    topk_strategy=cfg["topk_strategy"],
                    use_ws3=cfg["use_ws3"], use_ws4=cfg["use_ws4"],
                    dist_stage=cfg["dist_stage"],
                )
                compiled = cute_mod.compile(
                    kernel,
                    _from_dlpack(x2d), _from_dlpack(c2d),
                    _from_dlpack(c_sq), _from_dlpack(out_i),
                    stream,
                )
            except Exception as exc:
                _heuristic_cache[ac_key] = _CUTEDSL_FAILED
                raise CuteDSLUnsupported(
                    f"FA3 heuristic compile failed for {ac_key}") from exc
            cached_h = (compiled, stream)
            _heuristic_cache[ac_key] = cached_h

        compiled, stream = cached_h
        compiled(
            _cached_from_dlpack(x2d), _cached_from_dlpack(c2d),
            _cached_from_dlpack(c_sq), _cached_from_dlpack(out_i),
            stream,
        )
        return out_i.view(B, N, K_PAD)

    cached = _autotune_cache.get(ac_key)
    if cached == _CUTEDSL_FAILED:
        raise CuteDSLUnsupported(
            f"FA3 autotune previously failed for {ac_key}")
    if cached is None:
        try:
            (
                BM, BN, use_ws, strat, use_ws3, use_ws4, dist_stage,
                _t_us, compiled, stream,
            ) = _autotune_fa3(
                x2d, c2d, c_sq, out_i, K_PAD,
                verbose=autotune_verbose,
            )
        except Exception as exc:
            _autotune_cache[ac_key] = _CUTEDSL_FAILED
            raise CuteDSLUnsupported(
                f"FA3 autotune failed for {ac_key}") from exc
        # Autotune finds the fastest FA3 config and caches it. Whether FA3 is
        # worth it vs Triton for this shape is the dispatcher's routing call
        # (``_cutedsl_autopick``), not this executor's.
        _autotune_cache[ac_key] = (compiled, stream)
        if autotune_verbose:
            tag = (
                "WS4" if use_ws4 else
                ("WS3" if use_ws3 else ("WS" if use_ws else "non-WS"))
            )
            stg = f" stg={dist_stage}" if (use_ws3 or use_ws4) else ""
            print(
                f"  fa3 winner: BM={BM} BN={BN} {tag}{stg} "
                f"strat={strat} ({_t_us:.0f} us)"
            )
    else:
        compiled, stream = cached

    compiled(
        _cached_from_dlpack(x2d), _cached_from_dlpack(c2d),
        _cached_from_dlpack(c_sq), _cached_from_dlpack(out_i),
        stream,
    )
    return out_i.view(B, N, K_PAD)
