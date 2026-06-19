"""Advantage-region map: the ``maxtree`` top-K vs the strategy it replaces.

The ``maxtree`` top-K (ported from the Blackwell BUILD kernel,
``blackwell_impl.py``) keeps an **unsorted** per-thread heap with a cached
``(worst_d, worst_pos)`` and beats the sorted bubble-insert paths with:

* a **group-min-4** prune (skip 4 candidates with one compare),
* a balanced **max-tree** that recomputes the evict slot in O(log K).

Two Hopper variants, each the direct analog of an existing strategy:

* ``maxtree``       -- register per-thread (non-WS / WS2); analog of
  ``perthread``.
* ``smem_maxtree``  -- 1-thread-per-row (WS3 / WS4); analog of
  ``smem_perthread`` and the structural twin of the Blackwell kernel.

This script sweeps both regimes and emits the OLD-vs-NEW speedup map so the
router rule can be derived from measurement. Build is measured at the
*router tile* (BM=256/BN=64 for N>=50k, else BM=128/BN=128) plus a
BM=128 cross-tile study at the large sizes.

Methodology notes:
* The kernel never materialises the N*M score matrix; the dense torch
  reference would (64 GiB at 131072^2), so correctness (top-K index-set
  overlap) is checked on a ``REF_ROWS`` subset; timing runs on the full
  shape.
* GPU clocks can't be locked on this box, so each measurement does a
  cooldown sleep + adaptive iters, and the destructive spill cases
  (maxtree at K>=16, large N -> 10s-100s of ms/run) run LAST so they
  can't throttle the clean cells.

Writes ``benchmarks/results/micro_knn_maxtree_topk.md``.

Usage:
  python -u -m benchmarks.micro.bench_knn_maxtree_topk --mode strat
  python -u -m benchmarks.micro.bench_knn_maxtree_topk --mode e2e
"""
from __future__ import annotations

import argparse
import time
from pathlib import Path

import torch


REF_ROWS = 2048
COOLDOWN_S = 0.8


def _router_tile(N: int) -> tuple[int, int]:
    """Mirror the build tile in ``_heuristic_fa3_config``."""
    return (256, 64) if N >= 50_000 else (128, 128)


def _time_us(run):
    """Adaptive-iters CUDA-event timing with a cooldown so back-to-back
    heavy kernels don't throttle the clock into the next measurement."""
    time.sleep(COOLDOWN_S)
    run()
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    run()
    torch.cuda.synchronize()
    probe = time.perf_counter() - t0
    if probe > 0.04:
        warm, iters = 1, 5
    elif probe > 0.008:
        warm, iters = 2, 10
    else:
        warm, iters = 5, 30
    for _ in range(warm):
        run()
    torch.cuda.synchronize()
    s = torch.cuda.Event(enable_timing=True)
    e = torch.cuda.Event(enable_timing=True)
    s.record()
    for _ in range(iters):
        run()
    e.record()
    torch.cuda.synchronize()
    return s.elapsed_time(e) / iters * 1000.0


def _ref_subset(x, c, c_sq, k, rows):
    rows = min(rows, x.shape[0])
    s = c_sq.float().unsqueeze(0) - 2.0 * (x[:rows].float() @ c.float().t())
    return torch.topk(s, k, dim=1, largest=False).indices.int()


def _overlap(a, b):
    n, k = a.shape
    return sum(len(set(a[i].tolist()) & set(b[i].tolist()))
               for i in range(n)) / (n * k)


def _bench_strategy(N, M, D, K, BM, BN, strat, *, use_ws, use_ws3=False,
                    use_ws4=False, dist_stage=1):
    import cutlass
    import cutlass.cute as cute
    from cutlass.cute.runtime import from_dlpack
    import cuda.bindings.driver as cuda
    from flashlib.primitives.knn.cutedsl.hopper_impl import HopperFlashKnnFused

    torch.manual_seed(0)
    x = torch.randn(N, D, device="cuda", dtype=torch.bfloat16)
    c = torch.randn(M, D, device="cuda", dtype=torch.bfloat16)
    c_sq = (c.float() ** 2).sum(1).contiguous()
    out_i = torch.empty((N, K), device="cuda", dtype=torch.int32)
    stream = cuda.CUstream(torch.cuda.current_stream().cuda_stream)
    kern = HopperFlashKnnFused(
        acc_dtype=cutlass.Float32, m_block_size=BM, n_block_size=BN,
        k_pad=K, use_ws=use_ws, topk_strategy=strat,
        use_ws3=use_ws3, use_ws4=use_ws4, dist_stage=dist_stage,
    )
    comp = cute.compile(kern, from_dlpack(x), from_dlpack(c),
                        from_dlpack(c_sq), from_dlpack(out_i), stream)

    def run():
        comp(from_dlpack(x), from_dlpack(c), from_dlpack(c_sq),
             from_dlpack(out_i), stream)

    run()
    torch.cuda.synchronize()
    ov = _overlap(out_i[:REF_ROWS].cpu(), _ref_subset(x, c, c_sq, K, REF_ROWS))
    return _time_us(run), ov


def _pair(N, M, D, K, BM, BN, old, new, role="router", **kw):
    o_us, o_ov = _bench_strategy(N, M, D, K, BM, BN, old, **kw)
    n_us, n_ov = _bench_strategy(N, M, D, K, BM, BN, new, **kw)
    spd = o_us / n_us
    flag = "WIN " if spd > 1.03 else ("tie " if spd >= 0.97 else "lose")
    print(f"  [{flag}] {role:<6} {old:>14}->{new:<12} N={N:>6} D={D:>3} "
          f"K={K:>2} BM{BM}/BN{BN}: old {o_us:9.1f} new {n_us:9.1f}  "
          f"{spd:.2f}x  ov={o_ov:.3f}/{n_ov:.3f}", flush=True)
    return dict(N=N, M=M, D=D, K=K, BM=BM, BN=BN, old=old, new=new, role=role,
                old_us=o_us, new_us=n_us, speedup=spd,
                old_ov=o_ov, new_ov=n_ov)


def sweep_build():
    print("\n=== BUILD / REGISTER (perthread -> maxtree, non-WS) ===",
          flush=True)
    rows = []
    sizes = (16384, 32768, 65536, 131072)
    # Phase 1: win-candidate K (no hard spill), at the router tile.
    for K in (4, 8, 10):
        for NM in sizes:
            BM, BN = _router_tile(NM)
            rows.append(_pair(NM, NM, 64, K, BM, BN,
                              "perthread", "maxtree", use_ws=False))
    # Phase 1b: BM=128 cross-tile study at the large (router=BM256) sizes.
    for K in (8, 10):
        for NM in (65536, 131072):
            rows.append(_pair(NM, NM, 64, K, 128, 128,
                              "perthread", "maxtree", role="alt128",
                              use_ws=False))
    # Phase 1c: wide-D spot at the router tile.
    for NM in (65536, 131072):
        BM, BN = _router_tile(NM)
        rows.append(_pair(NM, NM, 128, 8, BM, BN,
                          "perthread", "maxtree", role="D128", use_ws=False))
    # Phase 2: spill cliff (LAST -- destructive), router tile.
    for K in (16, 32):
        rows.append(_pair(131072, 131072, 64, K, 256, 64,
                          "perthread", "maxtree", role="cliff", use_ws=False))
    return rows


def sweep_search():
    print("\n=== SEARCH / SMEM (smem_perthread -> smem_maxtree, WS3 stg3) ===",
          flush=True)
    rows = []
    for K in (4, 8, 16):
        for M in (32768, 131072):
            BM, BN = (64, 128) if K <= 16 else (128, 64)
            rows.append(_pair(2048, M, 128, K, BM, BN,
                              "smem_perthread", "smem_maxtree",
                              use_ws=True, use_ws3=True, dist_stage=3))
    # K=32 spills (last).
    for M in (32768, 131072):
        rows.append(_pair(2048, M, 128, 32, 128, 64,
                          "smem_perthread", "smem_maxtree", role="cliff",
                          use_ws=True, use_ws3=True, dist_stage=3))
    return rows


def _bench_e2e(N, M, D, K, mode):
    from flashlib.primitives.knn.cutedsl import impl
    torch.manual_seed(0)
    x = torch.randn(1, N, D, device="cuda", dtype=torch.bfloat16)
    c = torch.randn(1, M, D, device="cuda", dtype=torch.bfloat16)
    impl.set_maxtree_enabled({"off": False, "on": True, "auto": None}[mode])
    impl._heuristic_cache.clear()
    cfg = impl._heuristic_fa3_config(N, M, D, K)

    def run():
        return impl.cutedsl_flash_knn(x, c, K)

    out = run()
    torch.cuda.synchronize()
    c_sq = (c.view(M, D).float() ** 2).sum(1).contiguous()
    ov = _overlap(out.view(N, K)[:REF_ROWS].cpu(),
                  _ref_subset(x.view(N, D), c.view(M, D), c_sq, K, REF_ROWS))
    us = _time_us(run)
    impl.set_maxtree_enabled(None)
    return us, ov, cfg["topk_strategy"]


def sweep_e2e():
    print("\n=== END-TO-END cutedsl_flash_knn (off / auto / on) ===",
          flush=True)
    shapes = [
        ("search", 8192, 131072, 256, 8),
        ("search", 8192, 131072, 256, 16),
        ("build", 65536, 65536, 64, 8),
        ("build", 131072, 131072, 64, 8),
        ("build", 131072, 131072, 64, 10),
    ]
    rows = []
    for tag, N, M, D, K in shapes:
        res = {m: _bench_e2e(N, M, D, K, m) for m in ("off", "auto", "on")}
        off_us, auto_us, on_us = (res[m][0] for m in ("off", "auto", "on"))
        print(f"  [{tag}] N={N} M={M} D={D} K={K}: "
              f"off {off_us:8.1f} ({res['off'][2]}) | "
              f"auto {auto_us:8.1f} ({res['auto'][2]}, {off_us/auto_us:.2f}x) "
              f"| on {on_us:8.1f} ({res['on'][2]}, {off_us/on_us:.2f}x)  "
              f"recall={res['auto'][1]:.3f}", flush=True)
        rows.append(dict(tag=tag, N=N, M=M, D=D, K=K, res=res,
                         auto_speedup=off_us / auto_us,
                         on_speedup=off_us / on_us))
    return rows


def _write_md(gpu, sm, build_rows, search_rows, e2e_rows):
    out = (Path(__file__).resolve().parent.parent / "results"
           / "micro_knn_maxtree_topk.md")
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w") as f:
        f.write("# Advantage-region map: `maxtree` top-K on Hopper\n\n")
        f.write(f"GPU **{gpu}**, sm{sm[0]}{sm[1]}, torch {torch.__version__}. "
                f"Speedup = old/new (>1 => `maxtree` wins). Correctness = "
                f"top-K index-set overlap on the first {REF_ROWS} rows vs "
                f"torch exact (set, not order). Adaptive iters + "
                f"{COOLDOWN_S}s cooldown; spill cliffs measured last.\n\n")
        f.write("`maxtree` = unsorted per-thread heap + group-min-4 prune + "
                "balanced max-tree (`blackwell_impl.py`); `smem_maxtree` is "
                "the 1-thread-per-row twin.\n\n")

        if build_rows:
            f.write("## BUILD / register (`perthread` -> `maxtree`, non-WS)\n\n")
            f.write("Measured at the **router tile** (BM256/BN64 for N>=50k, "
                    "else BM128/BN128) unless noted. `alt128` = BM128 "
                    "cross-tile study; `cliff` = spill regime (run last).\n\n")
            f.write("| role | N=M | D | K | tile | old us | new us | speedup "
                    "| ov |\n")
            f.write("|:--|---:|---:|---:|:--|---:|---:|---:|:--:|\n")
            for r in build_rows:
                f.write(f"| {r['role']} | {r['N']} | {r['D']} | {r['K']} | "
                        f"BM{r['BM']}/BN{r['BN']} | {r['old_us']:.1f} | "
                        f"{r['new_us']:.1f} | **{r['speedup']:.2f}x** | "
                        f"{r['old_ov']:.3f}/{r['new_ov']:.3f} |\n")
            f.write("\n")

        if search_rows:
            f.write("## SEARCH / smem (`smem_perthread` -> `smem_maxtree`, "
                    "WS3)\n\n")
            f.write("| role | M (db) | K | tile | old us | new us | speedup "
                    "| ov |\n")
            f.write("|:--|---:|---:|:--|---:|---:|---:|:--:|\n")
            for r in search_rows:
                f.write(f"| {r['role']} | {r['M']} | {r['K']} | "
                        f"BM{r['BM']}/BN{r['BN']} | {r['old_us']:.1f} | "
                        f"{r['new_us']:.1f} | **{r['speedup']:.2f}x** | "
                        f"{r['old_ov']:.3f}/{r['new_ov']:.3f} |\n")
            f.write("\n")

        if e2e_rows:
            f.write("## End-to-end `cutedsl_flash_knn` (off / auto / on)\n\n")
            f.write("| shape | N | M | D | K | off us | auto us | auto x | "
                    "on us | on x | strat off->auto |\n")
            f.write("|:--|---:|---:|---:|---:|---:|---:|---:|---:|---:|:--|\n")
            for r in e2e_rows:
                off_us, off_s = r["res"]["off"][0], r["res"]["off"][2]
                auto_us, auto_s = r["res"]["auto"][0], r["res"]["auto"][2]
                on_us = r["res"]["on"][0]
                f.write(f"| {r['tag']} | {r['N']} | {r['M']} | {r['D']} | "
                        f"{r['K']} | {off_us:.1f} | {auto_us:.1f} | "
                        f"**{r['auto_speedup']:.2f}x** | {on_us:.1f} | "
                        f"{r['on_speedup']:.2f}x | {off_s}->{auto_s} |\n")
            f.write("\n")
        f.write("Source: `benchmarks/micro/bench_knn_maxtree_topk.py`.\n")
    print(f"\nWrote {out}", flush=True)


def main():
    assert torch.cuda.is_available(), "Need CUDA"
    from flashlib.primitives.knn.cutedsl.impl import cutedsl_available
    assert cutedsl_available(), "cutedsl unavailable"
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=("strat", "e2e", "all"), default="all")
    args = ap.parse_args()
    gpu = torch.cuda.get_device_name(0)
    sm = torch.cuda.get_device_capability(0)
    print(f"GPU: {gpu}  sm{sm[0]}{sm[1]}  torch {torch.__version__}", flush=True)
    build_rows = sweep_build() if args.mode in ("strat", "all") else []
    search_rows = sweep_search() if args.mode in ("strat", "all") else []
    e2e_rows = sweep_e2e() if args.mode in ("e2e", "all") else []
    _write_md(gpu, sm, build_rows, search_rows, e2e_rows)


if __name__ == "__main__":
    main()
