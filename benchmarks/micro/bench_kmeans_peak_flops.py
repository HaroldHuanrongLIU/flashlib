"""Find peak TFLOPS sustained by the kmeans assign kernel.

The kmeans assign step computes argmin_k ||x_n - c_k||^2 by issuing a
(N x D) * (D x K) matmul fused with on-chip argmin. The flop count
is 2*N*K*D (mul-adds in the cross GEMM); the on-chip argmin epilogue
adds N*K comparisons, which we ignore in the TFLOPS denominator since
WGMMA peak is reported for matmul flops only.

We sweep (N, K, D) at bf16 and fp16 (the H200 WGMMA peak path) and
report the (shape, backend) that maximises measured TFLOPS.
"""
from __future__ import annotations

import statistics
import time
from pathlib import Path

import torch


WARM = 3
ITERS = 7
PEAK_BF16_TFLOPS = 989.0  # H200 bf16 dense (datasheet)
PEAK_FP16_TFLOPS = 989.0  # H200 fp16 dense (same)


def time_ms(fn, warm=WARM, iters=ITERS):
    for _ in range(warm):
        fn()
    torch.cuda.synchronize()
    samples = []
    for _ in range(iters):
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        fn()
        torch.cuda.synchronize()
        samples.append((time.perf_counter() - t0) * 1000.0)
    samples.sort()
    return samples[len(samples) // 2]


def tflops(N, K, D, ms):
    return (2.0 * N * K * D) / (ms / 1000.0) / 1e12


def main():
    assert torch.cuda.is_available(), "Need CUDA"
    from flashlib.primitives.kmeans import (
        euclid_assign_triton,
        cutedsl_assign_euclid,
    )

    dev = torch.device("cuda")
    gpu = torch.cuda.get_device_name(0)
    print(f"GPU: {gpu}  torch {torch.__version__}\n")

    # Sweep grid: aim for compute-bound (big K, big D) but keep
    # tensors fitting comfortably in 141 GB HBM (x ≤ 2 GB, c ≤ 0.5 GB).
    GRID = []
    for D in (64, 128, 256):
        for K in (1024, 4096, 16384, 65536):
            for N in (65_536, 131_072, 262_144):
                bytes_x = N * D * 2
                bytes_c = K * D * 2
                if bytes_x + bytes_c > 8e9:
                    continue
                GRID.append((N, K, D))

    rows = []
    for dtype, peak in ((torch.bfloat16, PEAK_BF16_TFLOPS),
                        (torch.float16, PEAK_FP16_TFLOPS)):
        for (N, K, D) in GRID:
            torch.manual_seed(0)
            x = torch.randn(1, N, D, device=dev, dtype=dtype).contiguous()
            c = torch.randn(1, K, D, device=dev, dtype=dtype).contiguous()

            try:
                _ = euclid_assign_triton(x, c)
                t_tri = time_ms(lambda: euclid_assign_triton(x, c))
            except Exception as e:
                t_tri = float("nan")

            try:
                _ = cutedsl_assign_euclid(x, c, autotune=False)
                t_cute = time_ms(lambda: cutedsl_assign_euclid(x, c, autotune=False))
            except Exception:
                t_cute = float("nan")

            t_best = min(v for v in (t_tri, t_cute) if v == v)
            backend = "triton" if t_best == t_tri else "cutedsl"
            tf = tflops(N, K, D, t_best)
            pct = 100.0 * tf / peak
            rows.append({
                "dtype": str(dtype).replace("torch.", ""),
                "N": N, "K": K, "D": D,
                "t_tri": t_tri, "t_cute": t_cute,
                "best_ms": t_best, "best_backend": backend,
                "TFLOPS": tf, "pct_peak": pct,
            })
            print(f"  {str(dtype)[6:]:<8}  N={N:>7} K={K:>6} D={D:>4}  "
                  f"tri={t_tri:>7.3f} ms  cute={t_cute:>7.3f} ms  "
                  f"best={t_best:>7.3f} ({backend})  "
                  f"TFLOPS={tf:>6.1f}  %peak={pct:>5.1f}%")

    # Top-3 per dtype
    print("\n== Top 3 by TFLOPS ==")
    rows.sort(key=lambda r: -r["TFLOPS"])
    for r in rows[:10]:
        print(f"  {r['dtype']:<8}  N={r['N']:>7} K={r['K']:>6} D={r['D']:>4}  "
              f"{r['best_ms']:>6.3f} ms ({r['best_backend']})  "
              f"{r['TFLOPS']:>6.1f} TFLOPS  ({r['pct_peak']:.1f}% of 989)")

    # Markdown dump
    out_path = Path(__file__).resolve().parent.parent / "results" / "micro_kmeans_peak_flops.md"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w") as f:
        f.write("# Micro-benchmark: peak TFLOPS sustained by flash-kmeans assign\n\n")
        f.write(f"GPU: **{gpu}**, torch {torch.__version__}. "
                f"Shape grid: N ∈ {{65K, 131K, 262K}}, K ∈ {{1K, 4K, 16K, 64K}}, "
                f"D ∈ {{64, 128, 256}}, dtype ∈ {{bf16, fp16}}. "
                f"H200 bf16/fp16 dense peak = 989 TFLOPS. "
                f"flop count = 2·N·K·D (assign-GEMM); on-chip argmin epilogue not counted.\n\n")
        f.write("## Top 10 configurations by TFLOPS\n\n")
        f.write("| dtype | N | K | D | best (ms) | backend | TFLOPS | % of 989 peak |\n")
        f.write("|---|---:|---:|---:|---:|---|---:|---:|\n")
        for r in rows[:10]:
            f.write(f"| {r['dtype']} | {r['N']:,} | {r['K']:,} | {r['D']} | "
                    f"{r['best_ms']:.3f} | {r['best_backend']} | "
                    f"**{r['TFLOPS']:.1f}** | {r['pct_peak']:.1f}% |\n")

        f.write("\n## Full sweep\n\n")
        f.write("| dtype | N | K | D | triton (ms) | cutedsl (ms) | best | TFLOPS | %peak |\n")
        f.write("|---|---:|---:|---:|---:|---:|---|---:|---:|\n")
        rows_orig = sorted(rows, key=lambda r: (r["dtype"], r["D"], r["K"], r["N"]))
        for r in rows_orig:
            t_tri_str = f"{r['t_tri']:.3f}" if r['t_tri'] == r['t_tri'] else "—"
            t_cute_str = f"{r['t_cute']:.3f}" if r['t_cute'] == r['t_cute'] else "—"
            f.write(f"| {r['dtype']} | {r['N']:,} | {r['K']:,} | {r['D']} | "
                    f"{t_tri_str} | {t_cute_str} | {r['best_backend']} | "
                    f"{r['TFLOPS']:.1f} | {r['pct_peak']:.1f}% |\n")
        f.write("\nSource: `benchmarks/micro/bench_kmeans_peak_flops.py`.\n")
    print(f"\nWrote {out_path}")


if __name__ == "__main__":
    main()
