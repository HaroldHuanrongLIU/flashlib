"""KNN tuner — sweeps (N, M, D, k) x {triton, cutedsl/fa3} on the
current GPU.

Mirrors :mod:`benchmarks.tune.kmeans` in shape and conventions: per-shape
JSONL output under ``benchmarks/tune/results/knn/<device_tag>/``, one
median timing per (workload, backend) pair, ``--rerun`` / ``--size``
filters, and a paired :mod:`benchmarks.tune.derive.knn` script that
turns the artifacts into routing-rule suggestions.

This is the single tuner for KNN: it covers BOTH the auto-routed
``triton`` path (auto-picks M-split flash-decode vs single-pass per
shape inside :func:`flashlib.primitives.knn.triton.flash_knn_triton`)
and the opt-in ``cutedsl/fa3`` Hopper FA3 path. The first-call FA3
autotune is multi-minute per shape; pair this with
:mod:`benchmarks.tune.knn_parallel` to fan the grid out across all
8 GPUs.

Usage::

    # Sequential single-GPU run (slow because of FA3 autotune):
    CUDA_VISIBLE_DEVICES=0 python -m benchmarks.tune.knn
    CUDA_VISIBLE_DEVICES=0 python -m benchmarks.tune.knn --rerun

    # 8-GPU fan-out (preferred):
    python -m benchmarks.tune.knn_parallel
    python -m benchmarks.tune.derive.knn        # markdown table + rules
"""
from __future__ import annotations

import torch

from benchmarks.tune._common import expand_grid, parse_argv, run_tuner


# ---------------------------------------------------------------------------
# FA3 autotune deadlock workaround
# ---------------------------------------------------------------------------
#
# Symptom (observed on H200, cutlass-dsl 4.5.x): the FA3 autotune
# candidate ``BM=256, use_ws=True`` hangs inside ``cuda.synchronize``
# during the warmup launch and never returns. ``[autotune] warmup`` is
# the last log line on the affected process; the GPU pins at 100% util
# forever and Python-level interrupts can't reach it.
#
# Workaround (tuner-local): raise from ``HopperFlashKnnFused.__init__``
# for that tile combo so the existing ``try/except`` in
# ``_autotune_v1_fa3`` cleanly skips the candidate. Production users
# opting into ``build_fa3`` directly still see the upstream search
# space and could hit the deadlock; the patch lives here in the tuner
# so the sweep terminates reliably.

def _patch_fa3_autotune_skip_bm256():
    from flashlib.primitives.knn.cutedsl.fused_kernel import HopperFlashKnnFused
    orig_init = HopperFlashKnnFused.__init__

    def _patched_init(self, *args, **kwargs):
        if kwargs.get("m_block_size") == 256 and kwargs.get("use_ws", False):
            raise RuntimeError(
                "knn tuner: skipping BM=256 use_ws=True "
                "(known H200 / cutlass-dsl warmup deadlock)"
            )
        return orig_init(self, *args, **kwargs)

    HopperFlashKnnFused.__init__ = _patched_init


_patch_fa3_autotune_skip_bm256()


# ---------------------------------------------------------------------------
# Workload grid
# ---------------------------------------------------------------------------
#
# Axes chosen to drive the routing crossovers across the full KNN
# usage spectrum:
#
#   N  tiny-search (1, 16, 128) -> small-batch (1024) -> graph build
#      (4K, 16K, 64K) -> heavy / sharded build (256K, 512K)
#   M  small corpus (4K) -> in-L2 (16K @ D=128 = 4 MB) -> marginal
#      (65K = 16 MB) -> L2 spill (262K = 64 MB > 60 MB H200 L2) -> big
#      corpus (1M)
#   D  narrow (64) -> standard (128, 256) -> wide-but-FA3-OK (512)
#   k  smallest practical (2) -> NN-graph (4, 8) -> dense neighbours
#      (16, 32; 32 = FA3 k_max)
#
# 9 x 5 x 4 x 5 = 900 shapes. FA3 first-call autotune is ~3-5 min
# per shape, so we apply a backend-side gate: FA3 only runs where it
# actually has a shot (N >= 1024 AND M >= 16K), which leaves 6 x 4 x
# 4 x 5 = 480 FA3 cells. Total wall time on 8 H200s ~= 4-5 hours.
#
# Compared to the previous (144-shape) grid this:
#   - adds N=1/16/128 to cover the tiny-search regime that dominates
#     real RAG/ANN query workloads
#   - adds N=262144/524288 to cover the heavy-graph build / sharded
#     index regime
#   - adds M=4096 (small corpus) and M=1048576 (mega corpus)
#   - adds k=2 and k=4 to expose the FA3 WS4 sweet spot (verified
#     empirically: at N=8192 M=65K k=4 D=256 FA3 WS4 wins 1.68x over
#     triton because the 4-WG architecture (load + 2 GEMM + topk)
#     amortises WGMMA latency)

WORKLOADS = expand_grid({
    "B": [1],
    "N": [1, 16, 128, 1024, 4096, 16384, 65536, 262144, 524288],
    "M": [4096, 16384, 65536, 262144, 1048576],
    "D": [64, 128, 256, 512],
    "k": [2, 4, 8, 16, 32],
})


# ---------------------------------------------------------------------------
# Candidates
# ---------------------------------------------------------------------------
#
# The tuner does NOT call the smart dispatcher's auto-routing -- it
# pins each candidate explicitly so we can compare them head-to-head
# at every shape. Every (workload, backend) pair is one row in the
# JSONL, and the derive script turns the resulting (workload, winner)
# pairs into routing rules.

BACKENDS = [
    {"backend": "triton",  "variant": "auto"},
    {"backend": "cutedsl", "variant": "fa3"},
]


# FA3 makes no sense at the tiny-N / tiny-M end of the grid: its first-
# call autotune is ~4 min per shape, but the kernel itself is dominated
# by autotune compile cost at those sizes (kernel runtime < 100 us).
# Gate the FA3 sweep to the regime where it has a chance of winning;
# the other backends always run.
_FA3_GATE_N_MIN = 1024
_FA3_GATE_M_MIN = 16384


def _fa3_eligible(workload):
    return (workload["N"] >= _FA3_GATE_N_MIN
            and workload["M"] >= _FA3_GATE_M_MIN)


def setup(workload):
    B, N, M, D, k = (workload[k_] for k_ in ("B", "N", "M", "D", "k"))
    # bf16 input matches the production path (FA3 only accepts fp16/bf16
    # and the Triton kernels run bf16 GEMM internally on Hopper).
    x = torch.randn(B, N, D, device="cuda", dtype=torch.bfloat16)
    c = torch.randn(B, M, D, device="cuda", dtype=torch.bfloat16)
    return {"x": x, "c": c, "k": k}


def bench(ctx, candidate):
    backend = candidate["backend"]
    variant = candidate["variant"]
    x, c, k = ctx["x"], ctx["c"], ctx["k"]
    B, N, D = x.shape
    M = c.shape[1]

    if backend == "triton":
        # ``flash_knn`` is the public entry point: runs the fused
        # Triton kernel (auto-picks M-split vs single-pass internally),
        # then gathers true squared L2 distances via
        # :func:`triton_knn_gather_sqdist`. Production users always see
        # this end-to-end cost, so we time it as one unit.
        from flashlib.primitives.knn import flash_knn
        return lambda: flash_knn(x, c, k, backend="triton")
    elif backend == "cutedsl":
        # Skip FA3 outside its sweet spot: ~4 min autotune per shape,
        # but at tiny-N (search regime) the kernel runtime is dwarfed
        # by autotune compile cost. Triton handles those shapes well.
        if not (N >= _FA3_GATE_N_MIN and M >= _FA3_GATE_M_MIN):
            raise NotImplementedError(
                f"FA3 gated off for N={N} M={M} "
                f"(< {_FA3_GATE_N_MIN} or {_FA3_GATE_M_MIN}); "
                f"triton wins this regime"
            )
        # FA3 returns indices only -- pair with the gather pass so
        # the head-to-head against ``triton`` includes the same work.
        from flashlib.primitives.knn import flash_knn
        # First call autotunes (multi-minute). bench_ms calls ``warm``
        # warmups before the timed iterations, so the autotune cost is
        # amortised inside the warmup window and the timed median
        # reflects steady-state latency.
        return lambda: flash_knn(x, c, k, backend="cutedsl", autotune=True)
    raise ValueError(f"unknown candidate {candidate}")


def _jaccard_topk(a, b, k, n_sample=1024):
    """Mean per-row Jaccard intersection of two (N, k) index tensors.

    For very large N the per-row ``set(...)`` Python loop becomes the
    bottleneck (8M set ops at N=512K). Cap at ``n_sample`` random
    rows so correctness stays a sanity check, not a heavyweight pass.
    """
    import torch as _torch
    N = a.shape[0]
    if N > n_sample:
        idx = _torch.randperm(N, device=a.device)[:n_sample]
        a = a[idx]
        b = b[idx]
    inter = sum(
        len(set(a[i].tolist()) & set(b[i].tolist()))
        for i in range(a.shape[0])
    )
    return inter / (a.shape[0] * k)


def correctness(ctx, candidate):
    """Top-K *index* set overlap vs reference. Returns ``1 - overlap``
    so 'lower is better' matches ``rel_err`` semantics in the JSONL.

    Reference selection:
      * If the ``(N, M, D)`` fp32 broadcast intermediate fits in 4 GB,
        use the full torch ``topk`` reference.
      * Otherwise fall back to the Triton ``flash_knn`` path as the
        reference (it has its own parity tests in
        ``tests/test_backend_parity.py``). For the triton candidate
        this trivially returns 0, which is fine -- correctness here
        is a sanity check, not a precision benchmark.

    For very large N we cap the per-row Jaccard at 1024 sampled rows.
    """
    x, c, k = ctx["x"], ctx["c"], ctx["k"]
    B, N, D = x.shape
    M = c.shape[1]

    fn = bench(ctx, candidate)
    _, idx = fn()  # (B, N, k)

    # ``xf[..., None, :] - cf[:, None, :, :]`` broadcasts to
    # ``(B, N, M, D) fp32``, costing ``B * N * M * D * 4`` bytes
    # transiently. Cap that at 4 GB to keep correctness from OOM-ing
    # on heavy shapes.
    if (B * N * M * D * 4) > (4 << 30):
        from flashlib.primitives.knn import flash_knn
        _, ref_idx = flash_knn(x, c, k, backend="triton")
        return 1.0 - _jaccard_topk(idx[0].long(), ref_idx[0].long(), k)

    xf = x.float()
    cf = c.float()
    d2 = (xf[..., None, :] - cf[:, None, :, :]).pow(2).sum(-1)  # (B, N, M)
    _, ref = d2.topk(k, dim=-1, largest=False)
    return 1.0 - _jaccard_topk(idx[0].long(), ref[0].long(), k)


def main() -> None:
    args = parse_argv("knn")
    sizes = set(args.size.split(",")) if args.size else None

    def keep(w):
        if sizes is None:
            return True
        from benchmarks.tune._common import shape_key
        return shape_key(w) in sizes

    run_tuner(
        op="knn",
        workloads=WORKLOADS,
        backends=BACKENDS,
        setup_fn=setup,
        bench_fn=bench,
        correctness_fn=correctness,
        warm=3, iters=5, rerun=args.rerun,
        workload_filter=keep,
    )


if __name__ == "__main__":
    main()
