"""KNN dispatcher + routing rule.

Public entry point: :func:`flash_knn_dispatch` (also reachable via
:func:`flashlib.primitives.knn.flash_knn`). The hand-tuned routing rule
lives in :func:`_route`; the cost model in :mod:`cost` shares the same
rule via :func:`route_op_name`.

Backends
--------
* ``backend="triton"``   -- Triton kernels (default). One unified
  x²-free dispatcher inside :func:`flashlib.primitives.knn.triton.flash_knn_triton`:

      * iterative-insert top-K kernel with ``BN ∈ {8, 16, 32, 64, 128}``;
        the shape-only heuristic auto-picks the "search" (M-split
        flash-decode) vs "large_n" (single-pass per CTA) routing by
        checking ``ctas_no_split`` after BN is chosen. Pattern-A fast
        paths catch the ``B*N <= 8`` small-Q corners. x²-free score,
        indices-only output; the gather pass appends true squared L2.
      * packed-uint64 sort-merge variant kept for the small-Q +
        medium-K (``B*N <= 8``, ``16 <= K <= 64``) Pattern-A regime.

  Never materialises an N×M cross matrix to HBM and never loads
  ``x_sq``; both are hard contracts.
* ``backend="cutedsl"``  -- CuteDSL fully-fused. The backend axis is
  DSL-only; the router auto-selects the kernel by hardware: Hopper
  (sm_90) runs the FA3 ``hopper_impl``, Blackwell (sm_100) runs
  ``blackwell_impl``. Reachable two ways: the explicit override; and the
  transparent advantage-shape auto-route (large-N build on both archs +
  sm_100 small-Q search Triton can't tile; see :func:`_cutedsl_autopick`).
  The first call per shape pays a CuteDSL compile. Routing is *strict and
  proactive*: the dispatcher picks one backend up front and runs it -- there
  is no cross-backend fallback. The kernel is a *pure executor*
  (:func:`...cutedsl.cutedsl_flash_knn`) that raises ``CuteDSLUnsupported``
  rather than delegating, so an explicit ``backend="cutedsl"`` on a shape it
  can't run surfaces the error (no silent swap to Triton).
* ``backend="torch"``    -- pure-torch reference (CPU OK, slow).

No ``variant`` axis: callers don't pick between build / search /
small-N / large-N kernels -- the shape-only heuristic inside the
Triton dispatcher (and the cost-model gate for CuteDSL FA3) does it.
"""
from __future__ import annotations

from typing import Optional

import torch

from flashlib import _hw
from flashlib.kernels.distance.triton.knn_gather_l2sq import triton_knn_gather_sqdist
from flashlib.primitives.knn.cutedsl import cutedsl_available, cutedsl_flash_knn
from flashlib.primitives.knn.torch_fallback import knn_torch_naive
from flashlib.primitives.knn.triton.dispatch import flash_knn_triton


Backend = str


def _route(
    *,
    B: int,
    N: int,
    D: int,
    k: int,
    backend: Optional[str] = None,
    hw: Optional[_hw.HwProps] = None,
) -> Backend:
    """Pick a backend for KNN given workload + hardware.

    Default rule:
      * CUDA available -> ``"triton"`` (the fused kernel auto-picks
        small-N vs large-N inside the dispatcher).
      * else            -> ``"torch"``.

    This rule returns the *baseline* backend. The CuteDSL advantage shapes
    are layered on top of it in :func:`flash_knn_dispatch` via
    :func:`_cutedsl_autopick` (so the cost model, which only sees ``_route``,
    still treats Triton as the default). CuteDSL is transparently substituted
    only where it clearly wins or where Triton cannot run at all -- the
    large-N build band (both archs) and the sm_100 small-Q search Triton
    cannot compile. Every other shape stays on Triton unless the caller opts
    in with ``backend="cutedsl"``; the first such call pays a one-off CuteDSL
    compile (~5-8 s heuristic, multi-minute autotune).
    """
    if backend is not None:
        if backend not in _VALID_BACKENDS:
            raise ValueError(
                f"backend must be one of {_VALID_BACKENDS} or None, "
                f"got {backend!r}"
            )
        return backend
    hw = hw or _hw.current()
    if not hw.is_cuda:
        return "torch"
    return "triton"


# Backends are DSL-only: the hardware (Hopper vs Blackwell) is NOT a
# backend -- the router picks the matching CuteDSL kernel by hardware
# inside ``_cutedsl_knn``.
_VALID_BACKENDS = ("triton", "cutedsl", "torch")

_OP_NAME = {
    "triton":  "knn_triton",
    "cutedsl": "knn_cutedsl_fa3",
    "torch":   "knn_torch",
}

# Small-Q threshold below which Triton's tl.dot (min M=16) cannot run on
# sm_100 -- the CuteDSL backend (Blackwell kernel) is auto-routed there.
_CUTEDSL_SMALLQ = 16

# Largest k for which the Blackwell CuteDSL *build* is auto-routed. It wins
# decisively for k<=20 on B200 (>=18% vs Triton and below cake at large N,
# exact). Its per-thread top-K is O(k^2) so it crosses over ~k=22-24; above that
# the Triton build is MMA-bound (~flat in k) and already matches/beats cake, so
# high-k builds stay on Triton. (Measured B200 ours-vs-Triton crossover sweep.)
_CUTEDSL_BUILD_KMAX = 20

# Hopper FA3 *build* auto-route band (measured H100 cutedsl-vs-triton, bf16;
# see benchmarks/results/micro_knn_maxtree_topk.md). The fully-fused TMA+WGMMA
# build (with the maxtree register top-K) beats the Triton build 1.4-2.4x
# across D<=128, N>=50k, k in [4, ~22]; Triton catches up at k>=24 (its build is
# MMA-bound, ~flat in k). Cap at k<=20 for a solid margin (k=20 ~1.38x; k=22
# only ~1.08x). Search is NOT auto-routed on Hopper: Triton wins small/mid-Q
# search by up to ~12x (FA3 warp-per-row epilogue is starved until Q~8k), and
# Triton tiles every Q on sm_90 (verified Q=1..64 exact) so there is nothing
# to catch -- no reactive net is needed here.
_HOPPER_BUILD_KMAX = 20
_HOPPER_BUILD_NMIN = 50_000
_HOPPER_BUILD_DMAX = 128


def _cutedsl_autopick(x: torch.Tensor, c: torch.Tensor, k: int,
                      hw: _hw.HwProps) -> bool:
    """Whether to transparently route to the CuteDSL backend instead of Triton.
    The concrete kernel is hardware-routed in :func:`_cutedsl_knn` (Hopper FA3
    / Blackwell). All regimes are bf16, single batch. This is the *single
    source of truth* for the auto path -- routing is strict (no cross-backend
    fallback), so this only returns ``True`` on shapes CuteDSL is known to run
    and win on (or, on sm_100, the small-Q search Triton physically can't tile).

    Blackwell (sm_100), D=128:
      * build (self-kNN, k<=``_CUTEDSL_BUILD_KMAX``): the tcgen05 split-K +
        register top-K build beats Triton 2-3x at large N. High k stays on
        Triton (top-K O(k^2) crosses over; Triton MMA-bound, ~flat in k).
      * search (small-Q, N<``_CUTEDSL_SMALLQ``): Triton's ``tl.dot`` needs M>=16
        so it can't even run; the Blackwell search kernel restores it *and*
        wins.

    Hopper (sm_90), D<=``_HOPPER_BUILD_DMAX``:
      * build (self-kNN, N>=``_HOPPER_BUILD_NMIN``, k<=``_HOPPER_BUILD_KMAX``):
        the FA3 fully-fused build beats the Triton build 1.4-2.4x. Search is
        left on Triton (it wins small/mid-Q by up to ~12x and tiles every Q on
        sm_90, so there is no small-Q gap to route around on Hopper).
    """
    if not hw.is_cuda:
        return False
    B, N, Dd = x.shape
    M = c.shape[1]
    if B != 1 or k > 64:
        return False
    if x.dtype != torch.bfloat16 or c.dtype != torch.bfloat16:
        return False
    is_build = (x.data_ptr() == c.data_ptr() and N == M)

    if hw.is_blackwell:
        if Dd != 128:
            return False
        try:
            from flashlib.primitives.knn.cutedsl.blackwell_impl import (
                blackwell_supported)
        except Exception:  # noqa: BLE001
            return False
        if is_build:
            # CuteDSL build wins for small/mid k; high-k top-K is O(k^2) and
            # loses to Triton (MMA-bound, ~flat in k), which matches cake there.
            if k > _CUTEDSL_BUILD_KMAX:
                return False
            return blackwell_supported(x, c, k)
        # search: only where Triton's MMA-batched path can't run (small Q).
        if N >= _CUTEDSL_SMALLQ:
            return False
        return blackwell_supported(x, c, k)

    if hw.is_hopper:
        # Only the large-N build band is a transparent win on Hopper. Search
        # stays on Triton (it wins small/mid-Q and tiles every Q on sm_90); the
        # very-large-Q FA3 search win is left opt-in to avoid the silent compile
        # tax on the common small-Q shape.
        if not is_build:
            return False
        if Dd % 16 != 0 or Dd > _HOPPER_BUILD_DMAX:
            return False
        if N < _HOPPER_BUILD_NMIN or k > _HOPPER_BUILD_KMAX:
            return False
        return cutedsl_available()

    return False


def _cutedsl_knn(x: torch.Tensor, c: torch.Tensor, k: int,
                 hw: _hw.HwProps, **kwargs) -> torch.Tensor:
    """Run the CuteDSL KNN backend, auto-selecting the kernel by hardware.

    The ``"cutedsl"`` backend is DSL-only; the concrete kernel is chosen
    here from :class:`flashlib._hw.HwProps`:

      * Blackwell (sm_100) -> :mod:`...cutedsl.blackwell_impl`.
      * Hopper    (sm_90)  -> the FA3 :mod:`...cutedsl.hopper_impl`
        (via :func:`cutedsl_flash_knn`).
    """
    if hw.is_blackwell:
        from flashlib.primitives.knn.cutedsl.blackwell_impl import (
            blackwell_flash_knn)
        return blackwell_flash_knn(x, c, k)
    return cutedsl_flash_knn(x, c, k, **kwargs)


def route_op_name(*, B: int, N: int, M: int, D: int, k: int,
                  hw: Optional[_hw.HwProps] = None) -> str:
    """Canonical op_name the runtime dispatcher would pick.

    ``M`` is accepted for signature completeness but unused by the rule.
    """
    del M
    backend = _route(B=B, N=N, D=D, k=k, hw=hw)
    return _OP_NAME[backend]


_KNN_MIN_D = 16  # Triton tl.dot requires K >= 16; sub-16 D inputs are zero-padded.


def _prepare_inputs(x: torch.Tensor, c: torch.Tensor):
    """Pad sub-16 D with zeros (zeros contribute 0 to squared L2).

    Padding never affects results -- the extra zero columns produce a
    zero difference for every (x, c) pair, so both the fused score and
    the gather-recomputed distance stay correct.
    """
    *_, D = x.shape
    if D < _KNN_MIN_D:
        x_pad = torch.zeros((*x.shape[:-1], _KNN_MIN_D),
                            device=x.device, dtype=x.dtype)
        x_pad[..., :D] = x
        c_pad = torch.zeros((*c.shape[:-1], _KNN_MIN_D),
                            device=c.device, dtype=c.dtype)
        c_pad[..., :D] = c
        return x_pad.contiguous(), c_pad.contiguous()
    if not x.is_contiguous():
        x = x.contiguous()
    if not c.is_contiguous():
        c = c.contiguous()
    return x, c


def flash_knn_dispatch(
    x: torch.Tensor,
    c: torch.Tensor,
    k: int,
    *,
    tol: Optional[float] = None,
    backend: Optional[str] = None,
    return_distances: bool = True,
    **kwargs,
) -> tuple[torch.Tensor, torch.Tensor] | torch.Tensor:
    """Smart-dispatch fused brute-force KNN (no HBM cross materialisation).

    Parameters
    ----------
    x : (B, N, D) | (N, D) tensor
        Query points.
    c : (B, M, D) | (M, D) tensor
        Corpus points (same dtype as ``x``).
    k : int
        Number of neighbours per query.
    tol : float, optional
        ``None`` (default) keeps the input dtype (EXACT path). Pass a
        tolerance to opt into low-precision storage via
        :func:`flashlib.linalg.gemm.storage_dtype_for`.
    backend : {"triton", "cutedsl", "torch"}, optional
        Explicit backend override. By default auto-routes to Triton on
        CUDA and Torch on CPU. CuteDSL FA3 is reachable only via this
        override.
    return_distances : bool, default True
        If True, return ``(vals, idxs)`` where ``vals[b, n, k]`` is the
        true ``||x[b, n] - c[b, idxs[b, n, k]]||^2`` (fp32) computed by
        :func:`flashlib.kernels.distance.triton_knn_gather_sqdist`.
        If False, return just ``idxs`` -- saves the gather pass for
        downstream consumers that only need the indices.
    **kwargs
        Backend-specific extras (e.g. ``autotune=True`` for cutedsl).

    Returns
    -------
    (vals, idxs) : (B, N, k) tensors, or just idxs when
    ``return_distances=False``.
    """
    squeeze = (x.dim() == 2)
    if squeeze:
        x = x.unsqueeze(0)
        c = c.unsqueeze(0)
    B, N, D = x.shape

    chosen = _route(B=B, N=N, D=D, k=k, backend=backend)

    if chosen == "torch":
        vals, idxs = knn_torch_naive(x, c, k)
        if not return_distances:
            return idxs[0] if squeeze else idxs
        if squeeze:
            return vals[0], idxs[0]
        return vals, idxs

    from flashlib.linalg.gemm import storage_dtype_for
    target_dtype = storage_dtype_for(tol)
    if target_dtype is not None and x.dtype != target_dtype:
        x = x.to(target_dtype)
        c = c.to(target_dtype)

    x_p, c_p = _prepare_inputs(x, c)

    # Strict, proactive routing: pick exactly one backend and run it. The
    # decision is made once here -- there is no cross-backend fallback. For
    # auto (backend=None), ``_cutedsl_autopick`` is the single source of truth
    # and only upgrades Triton->CuteDSL on the measured advantage shapes (and,
    # on sm_100, the small-Q search Triton can't tile -- handled *proactively*,
    # not caught after the fact). An explicit backend runs or raises.
    hw = _hw.current()
    use_cutedsl = (chosen == "cutedsl")
    if backend is None and chosen == "triton":
        use_cutedsl = _cutedsl_autopick(x_p, c_p, k, hw)
    if use_cutedsl:
        idxs = _cutedsl_knn(x_p, c_p, k, hw, **kwargs)
    else:
        idxs = flash_knn_triton(x_p, c_p, k, **kwargs)

    if not return_distances:
        return idxs[0] if squeeze else idxs

    vals = triton_knn_gather_sqdist(x_p, c_p, idxs)
    if squeeze:
        return vals[0], idxs[0]
    return vals, idxs


# Public canonical entry point. ``flash_knn(x, c, k)`` is what callers
# inside flashlib (DBSCAN / HDBSCAN / UMAP / spectral) use.
flash_knn = flash_knn_dispatch
