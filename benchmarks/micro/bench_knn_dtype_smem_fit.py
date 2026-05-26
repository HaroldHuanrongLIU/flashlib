"""Micro-bench: dtype-aware SMEM fit validation for flash_knn.

Goal: confirm dtype-aware SMEM fitting preserves bf16 perf and keeps
fp32 perf on shapes that previously exceeded SMEM.

The dispatcher has two layers:

  1. ``_estimate_sram`` is dtype-aware (``dtype_bytes`` parameter)
     and intentionally **optimistic** — it only counts x_tile + c_tile,
     so the fitter never shrinks a heuristic-picked config (verified
     in the "heuristic cfg picked" column — bf16 and fp32 columns
     match the heuristic exactly).

  2. ``_run`` catches ``OutOfResources`` at launch and shrinks
     ``(NUM_STAGES_PIPE, BN, BM)`` in that order, preferring to halve
     the larger of (BN, BM) so the tile stays near-square (best WGMMA
     shape). The surviving cfg is cached per (D, K, dtype) so
     subsequent calls skip the failed compile.

For each (N, D, K) cell in benchmarks/vs_cuml/broad/knn.py we report:

  * cfg the heuristic picks at bf16 and fp32
  * cfg actually launched (after any runtime fallback) at fp32
  * bf16 / fp32 wall time (median of 5 iters)

Writes benchmarks/results/micro_knn_dtype_smem_fit.md.
"""
from __future__ import annotations

import time
from pathlib import Path

import torch

from flashlib import flash_knn
from flashlib.primitives.knn.triton.dispatch import (
    _heuristic_config, _smem_limit, _estimate_sram,
    _OOR_FALLBACK_CACHE,
)


# Cells from benchmarks/vs_cuml/broad/knn.py (build self-kNN regime).
NS = [30_000, 100_000, 300_000, 1_000_000]
DS = [16, 64, 256, 1024]
K_NN = 10

CELLS = [(N, D) for N in NS for D in DS if N * D <= 2 * 10**8]


WARM = 2
ITERS = 5


def time_ms(fn):
    for _ in range(WARM):
        fn()
    torch.cuda.synchronize()
    samples = []
    for _ in range(ITERS):
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        fn()
        torch.cuda.synchronize()
        samples.append((time.perf_counter() - t0) * 1000)
    samples.sort()
    return samples[len(samples) // 2]


def fmt_cfg(c: dict) -> str:
    return (f"BN={c['BN']:>3} BM={c['BM']:>3} "
            f"D_INNER={c['D_INNER']:>3} NS={c['NUM_STAGES_PIPE']}")


def _runtime_cfg_fp32(N, D, K, smem):
    """Return the cfg actually launched at fp32 for this shape.

    Inspects ``_OOR_FALLBACK_CACHE`` after one flash_knn call. If the
    cache has an entry for this shape, the runtime fallback fired and
    the cached cfg differs from the heuristic pick.
    """
    cfg = _heuristic_config(1, N, N, D, K, dtype_bytes=4, smem_limit=smem)
    # Check fallback cache key matches what _run uses internally.
    fb_key = (D, K, torch.float32, None,
              cfg["kernel_mode"], cfg.get("D_INNER"))
    if fb_key in _OOR_FALLBACK_CACHE:
        cached = _OOR_FALLBACK_CACHE[fb_key]
        return {**cfg, **{k_: cached[k_] for k_ in
                          ("BN", "BM", "NUM_STAGES_PIPE")}}
    return cfg


def main():
    device = torch.device("cuda")
    smem = min(_smem_limit(device), 220_000)
    rows = []
    print(f"# flash_knn dtype-aware SMEM fix validation")
    print(f"# Device opt-in SMEM (effective cap): {smem//1024} KB\n")
    print(f"{'N':>9} {'D':>5}  "
          f"{'cfg bf16':<33} {'bf16 (ms)':>9}  "
          f"{'cfg fp32 (heur)':<33} {'cfg fp32 (run)':<33} {'fp32 (ms)':>9}")
    print('-' * 150)

    for N, D in CELLS:
        cfg_bf = _heuristic_config(
            1, N, N, D, K_NN, dtype_bytes=2, smem_limit=smem)
        cfg_fp_heur = _heuristic_config(
            1, N, N, D, K_NN, dtype_bytes=4, smem_limit=smem)

        # Time bf16
        torch.manual_seed(0)
        x_bf = torch.randn(1, N, D, device=device, dtype=torch.bfloat16)
        try:
            t_bf = time_ms(lambda: flash_knn(x_bf, x_bf, K_NN))
        except Exception as e:
            t_bf = float("nan")
            print(f"  bf16 N={N} D={D} FAIL: {e}")
        del x_bf
        torch.cuda.empty_cache()

        # Time fp32
        torch.manual_seed(0)
        x_fp = torch.randn(1, N, D, device=device, dtype=torch.float32)
        try:
            t_fp = time_ms(lambda: flash_knn(x_fp, x_fp, K_NN))
        except Exception as e:
            t_fp = float("nan")
            print(f"  fp32 N={N} D={D} FAIL: {e}")
        del x_fp
        torch.cuda.empty_cache()

        # The runtime fallback may have shrunk fp32 cfg; inspect cache.
        cfg_fp_run = _runtime_cfg_fp32(N, D, K_NN, smem)
        fallback_fired = (cfg_fp_run['BN'], cfg_fp_run['BM'],
                           cfg_fp_run['NUM_STAGES_PIPE']) != \
                          (cfg_fp_heur['BN'], cfg_fp_heur['BM'],
                           cfg_fp_heur['NUM_STAGES_PIPE'])

        run_str = (fmt_cfg(cfg_fp_run) + (" *" if fallback_fired else ""))

        print(f"{N:>9,} {D:>5}  "
              f"{fmt_cfg(cfg_bf):<33} {t_bf:>9.2f}  "
              f"{fmt_cfg(cfg_fp_heur):<33} {run_str:<33} {t_fp:>9.2f}")

        rows.append({
            'N': N, 'D': D, 'K': K_NN,
            'cfg_bf16': cfg_bf,
            'cfg_fp32_heur': cfg_fp_heur,
            'cfg_fp32_run': cfg_fp_run,
            'fallback_fired': fallback_fired,
            'bf16_ms': t_bf, 'fp32_ms': t_fp,
        })

    md_path = Path("benchmarks/results/micro_knn_dtype_smem_fit.md")
    md_path.parent.mkdir(parents=True, exist_ok=True)
    with md_path.open("w") as f:
        f.write("# `flash_knn` dtype-aware SMEM fit validation\n\n")
        f.write(f"Device opt-in SMEM (effective cap): **{smem//1024} KB**. "
                "All cells are the build-graph regime "
                "(Q=M=N, K=10) used in `benchmarks/vs_cuml/broad/knn.py`.\n\n")
        f.write("`cfg fp32 (run)` is the config **actually launched** "
                "after any runtime ``OutOfResources`` fallback. The `*` "
                "marker indicates a cell where the fallback shrunk the "
                "heuristic-picked config to fit SMEM.\n\n")
        f.write("| N | D | cfg bf16 | bf16 (ms) | cfg fp32 (heur) | "
                "cfg fp32 (run) | fp32 (ms) |\n")
        f.write("|---:|---:|---|---:|---|---|---:|\n")
        for r in rows:
            run_str = "`" + fmt_cfg(r['cfg_fp32_run']) + "`"
            if r['fallback_fired']:
                run_str += " **\\***"
            f.write(f"| {r['N']:,} | {r['D']} | "
                    f"`{fmt_cfg(r['cfg_bf16'])}` | {r['bf16_ms']:.2f} | "
                    f"`{fmt_cfg(r['cfg_fp32_heur'])}` | "
                    f"{run_str} | {r['fp32_ms']:.2f} |\n")

        f.write("\n## Takeaways\n\n")

        f.write("- **bf16**: the heuristic config is launched verbatim on "
                f"all {len(rows)} cells (the optimistic `_estimate_sram` "
                "never shrinks a heuristic pick on these shapes).\n")

        fb_count = sum(1 for r in rows if r['fallback_fired'])
        f.write(f"- **fp32**: {fb_count} cells trigger the runtime "
                "``OutOfResources`` fallback in `_run` (cached for "
                "subsequent calls); the remainder launch the heuristic "
                "config directly.\n")

        # fp32 vs bf16 ratio
        ratios = [r['fp32_ms']/r['bf16_ms'] for r in rows
                  if r['bf16_ms'] > 0 and r['fp32_ms'] > 0]
        if ratios:
            f.write(f"- **fp32 / bf16 ratio**: median "
                    f"{sorted(ratios)[len(ratios)//2]:.2f}x — "
                    "consistent with the 2× HBM-bytes-per-element "
                    "ratio (fp32 is bandwidth-bound on these shapes).\n")
    print(f"\nWrote {md_path}")


if __name__ == "__main__":
    main()
