"""Advantage-region map: the ``maxtree`` top-K vs the strategy it replaces.

The ``maxtree`` top-K (ported from the Blackwell BUILD kernel,
``blackwell_impl.py``) keeps an **unsorted** per-thread heap with a cached
``(worst_d, worst_pos)`` and beats the sorted bubble-insert paths with:

* a **group-min-4** prune (skip 4 candidates with one compare),
* a K-adaptive **worst-of-K** recompute -- a balanced max-tree at K<=10, a
  streaming running-max at K>=11. The streaming scan keeps only 2 scalars
  live so the heap stops spilling in CuteDSL/MLIR at high K, which recovers
  the build win the plain max-tree gives back (the Blackwell BUILD learning).

Two Hopper variants, each the direct analog of an existing strategy:

* ``maxtree``       -- register per-thread (non-WS / WS2); analog of
  ``perthread``.
* ``smem_maxtree``  -- 1-thread-per-row (WS3 / WS4); analog of
  ``smem_perthread`` and the structural twin of the Blackwell kernel.

This script sweeps both regimes and emits the speedup map (maxtree vs the
strategy the router would otherwise pick) so the routing rule can be derived
from measurement. Build maxtree is timed at BM256/BN64; the low-K baseline
``perthread`` at the same tile, the high-K baseline ``sortmerge`` at its own
BM128/BN128 tile (it is ~1.6x slower at BM256). The measured build win band
is K in [5,24]; below that perthread wins (K=4), above it sortmerge re-takes
the lead at K~28-32.

Methodology notes:
* The kernel never materialises the N*M score matrix; the dense torch
  reference would (64 GiB at 131072^2), so correctness (top-K index-set
  overlap) is checked on a ``REF_ROWS`` subset; timing runs on the full
  shape.
* GPU clocks can't be locked on this box, so each measurement does a
  cooldown sleep + adaptive iters, and the heavy high-K cells (10s-100s of
  ms/run) run LAST so they can't throttle the clean cells.

A third axis, ``--mode routing``, maps the *top-level* router decision
(CuteDSL FA3 vs Triton) that ``_cutedsl_autopick`` encodes: the build band
where the fully-fused FA3 build beats Triton (D<=128, N>=50k, K<=~22, 1.4-2.4x;
Triton retakes ~K=24) and the search crossover (Triton wins small/mid Q,
CuteDSL crosses over ~Q=8192). ``cutedsl_flash_knn`` is a pure executor now,
so it runs the FA3 kernel directly for any shape (no Triton opt-out to bypass).

Writes ``benchmarks/results/micro_knn_maxtree_topk.md``.

Usage:
  python -u -m benchmarks.micro.bench_knn_maxtree_topk --mode strat
  python -u -m benchmarks.micro.bench_knn_maxtree_topk --mode e2e
  python -u -m benchmarks.micro.bench_knn_maxtree_topk --mode routing
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


def _pair(N, M, D, K, BM, BN, old, new, role="router",
          old_bm=None, old_bn=None, **kw):
    """Bench old vs new. ``old`` may use its own tile (``old_bm``/``old_bn``)
    so each strategy is timed at the tile the router would actually give it
    (e.g. maxtree@BM256/BN64 vs sortmerge@BM128/BN128)."""
    obm, obn = old_bm or BM, old_bn or BN
    o_us, o_ov = _bench_strategy(N, M, D, K, obm, obn, old, **kw)
    n_us, n_ov = _bench_strategy(N, M, D, K, BM, BN, new, **kw)
    spd = o_us / n_us
    flag = "WIN " if spd > 1.03 else ("tie " if spd >= 0.97 else "lose")
    otile = f"BM{obm}/BN{obn}" if (old_bm or old_bn) else ""
    print(f"  [{flag}] {role:<6} {old:>14}{otile}->{new:<12} N={N:>6} D={D:>3} "
          f"K={K:>2} BM{BM}/BN{BN}: old {o_us:9.1f} new {n_us:9.1f}  "
          f"{spd:.2f}x  ov={o_ov:.3f}/{n_ov:.3f}", flush=True)
    return dict(N=N, M=M, D=D, K=K, BM=BM, BN=BN, old=old, new=new, role=role,
                old_bm=obm, old_bn=obn,
                old_us=o_us, new_us=n_us, speedup=spd,
                old_ov=o_ov, new_ov=n_ov)


def sweep_build():
    print("\n=== BUILD / REGISTER (vs the strategy the router replaces) ===",
          flush=True)
    rows = []
    sizes = (16384, 32768, 65536, 131072)
    # Phase 1: lower edge + mid K vs perthread, at the router tile. The
    # crossover is K=4 (perthread) -> K=5 (maxtree).
    for K in (4, 5, 6, 8, 10):
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
    # Phase 2: high-K region vs SORTMERGE (the K>16 router pick), tile-fair:
    # maxtree@BM256/BN64 vs sortmerge@BM128/BN128 (sortmerge is ~1.6x slower
    # at the BM256 tile). maxtree wins up to ~K=24; sortmerge re-takes the
    # lead at K~28-32 -- that is the upper band edge. Higher K last (slower).
    for K in (16, 18, 20, 24, 28, 32):
        for NM in (65536, 131072):
            rows.append(_pair(NM, NM, 64, K, 256, 64,
                              "sortmerge", "maxtree", role="highK",
                              old_bm=128, old_bn=128, use_ws=False))
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
        ("build", 131072, 131072, 64, 5),
        ("build", 65536, 65536, 64, 8),
        ("build", 131072, 131072, 64, 8),
        ("build", 131072, 131072, 64, 16),
        ("build", 131072, 131072, 64, 20),
        ("build", 131072, 131072, 64, 24),
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


def _bench_routing(tag, N, M, D, K):
    """Time the *top-level* choice: Triton (the default backend) vs the FA3
    CuteDSL kernel (the pure executor runs it directly). Overlap vs exact is
    checked on the FA3 result over a REF_ROWS subset."""
    from flashlib.primitives.knn import flash_knn
    from flashlib.primitives.knn.cutedsl.impl import cutedsl_flash_knn

    torch.manual_seed(0)
    x = torch.randn(N, D, device="cuda", dtype=torch.bfloat16)
    c = x if tag == "build" else torch.randn(M, D, device="cuda",
                                             dtype=torch.bfloat16)
    c_sq = (c.float() ** 2).sum(1).contiguous()

    def run_tri():
        return flash_knn(x, c, K, backend="triton", return_distances=False)

    def run_cud():
        return cutedsl_flash_knn(x.unsqueeze(0), c.unsqueeze(0), K)

    idx_c = run_cud().view(N, K)
    torch.cuda.synchronize()
    ov = _overlap(idx_c[:REF_ROWS].cpu(), _ref_subset(x, c, c_sq, K, REF_ROWS))
    tri = _time_us(run_tri)
    cud = _time_us(run_cud)
    win = "cutedsl" if cud < tri else "triton"
    print(f"  [{tag:6}] N={N:6d} M={M:6d} D={D:3d} K={K:2d}: "
          f"triton {tri:9.1f} cutedsl {cud:9.1f}  {tri / cud:.2f}x "
          f"-> {win:7s} ov={ov:.3f}", flush=True)
    return dict(tag=tag, N=N, M=M, D=D, K=K, tri_us=tri, cud_us=cud,
                speedup=tri / cud, win=win, ov=ov)


def sweep_routing():
    print("\n=== TOP-LEVEL ROUTING (cutedsl FA3 vs Triton) ===", flush=True)
    rows = []
    # build band: FA3 build wins D<=128 / N>=50k up to K~22, Triton retakes K24.
    for K in (4, 8, 16, 20, 24):
        rows.append(_bench_routing("build", 131072, 131072, 64, K))
    for N, D, K in ((131072, 128, 8), (131072, 128, 20), (65536, 64, 8)):
        rows.append(_bench_routing("build", N, N, D, K))
    # search: Triton wins small/mid Q (FA3 epilogue starved); crosses ~Q=8192.
    for Q in (8, 512, 2048, 8192, 16384):
        rows.append(_bench_routing("search", Q, 131072, 128, 8))
    return rows


def _write_md(gpu, sm, build_rows, search_rows, e2e_rows, routing_rows):
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
                "K-adaptive worst-of-K (balanced max-tree K<=10, streaming "
                "running-max K>=11; ported from `blackwell_impl.py`). "
                "`smem_maxtree` is the 1-thread-per-row twin.\n\n")

        if build_rows:
            f.write("## BUILD / register (`maxtree` vs the strategy it "
                    "replaces, non-WS)\n\n")
            f.write("`maxtree` is timed at BM256/BN64; the baseline at the "
                    "tile the router gives it (`perthread`@BM256/BN64, "
                    "`sortmerge`@BM128/BN128). Win band: K in [5,24]. "
                    "`alt128` = BM128 maxtree cross-tile study; `highK` = "
                    "maxtree vs sortmerge (tile-fair), where sortmerge "
                    "re-takes the lead ~K=28-32.\n\n")
            f.write("| role | N=M | D | K | baseline | maxtree us | base us | "
                    "speedup | ov |\n")
            f.write("|:--|---:|---:|---:|:--|---:|---:|---:|:--:|\n")
            for r in build_rows:
                base = (f"{r['old']}@BM{r['old_bm']}/BN{r['old_bn']}")
                f.write(f"| {r['role']} | {r['N']} | {r['D']} | {r['K']} | "
                        f"{base} | {r['new_us']:.1f} | {r['old_us']:.1f} | "
                        f"**{r['speedup']:.2f}x** | "
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
        if routing_rows:
            f.write("## Top-level routing (CuteDSL FA3 vs Triton)\n\n")
            f.write("The basis for the Hopper branch of "
                    "`knn/impl.py::_cutedsl_autopick`. Speedup = triton/cutedsl "
                    "(>1 => FA3 wins, auto-routed). Build auto-routes for "
                    "D<=128, N>=50k, K<=20 (margin below the ~K=24 crossover); "
                    "search stays on Triton until ~Q=8192 (FA3 epilogue starved "
                    "at small Q) so only the build band is auto-routed.\n\n")
            f.write("| regime | N | M | D | K | triton us | cutedsl us | "
                    "speedup | winner | ov |\n")
            f.write("|:--|---:|---:|---:|---:|---:|---:|---:|:--|:--:|\n")
            for r in routing_rows:
                f.write(f"| {r['tag']} | {r['N']} | {r['M']} | {r['D']} | "
                        f"{r['K']} | {r['tri_us']:.1f} | {r['cud_us']:.1f} | "
                        f"**{r['speedup']:.2f}x** | {r['win']} | {r['ov']:.3f} "
                        f"|\n")
            f.write("\n")
        f.write("Source: `benchmarks/micro/bench_knn_maxtree_topk.py`.\n")
    print(f"\nWrote {out}", flush=True)


def main():
    assert torch.cuda.is_available(), "Need CUDA"
    from flashlib.primitives.knn.cutedsl.impl import cutedsl_available
    assert cutedsl_available(), "cutedsl unavailable"
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=("strat", "e2e", "routing", "all"),
                    default="all")
    args = ap.parse_args()
    gpu = torch.cuda.get_device_name(0)
    sm = torch.cuda.get_device_capability(0)
    print(f"GPU: {gpu}  sm{sm[0]}{sm[1]}  torch {torch.__version__}", flush=True)
    build_rows = sweep_build() if args.mode in ("strat", "all") else []
    search_rows = sweep_search() if args.mode in ("strat", "all") else []
    e2e_rows = sweep_e2e() if args.mode in ("e2e", "all") else []
    routing_rows = sweep_routing() if args.mode in ("routing", "all") else []
    _write_md(gpu, sm, build_rows, search_rows, e2e_rows, routing_rows)


if __name__ == "__main__":
    main()
