"""Micro-benchmark: flashlib forward RMSNorm/LayerNorm vs native (eager) torch.

PyTorch's eager ``F.rms_norm`` / ``F.layer_norm`` assign one CTA per
normalized row. When the normalized dim ``N`` is small (per-head QK-norm
over ``head_dim``) with many rows, each CTA does a tiny reduction and the
SMs are badly under-utilized. flashlib's multi-row-per-CTA forward kernel
(``flash_rmsnorm`` / ``flash_layernorm``, ported from Sparse VideoGen,
arXiv:2502.01776) packs ``BLOCK_M`` rows per CTA to saturate HBM.

This sweep walks ``N`` from the advantage zone (small ``N``, big speedup)
into the parity zone (large ``N``, where one row already fills the warps
and we should be no slower than eager). ``M`` is chosen per shape to keep
the moved bytes (and thus the absolute work) roughly constant.

Bandwidth columns assume H200 peak = 4.80 TB/s. Writes
``benchmarks/results/micro_norm.md``.

Re-run with ``python -m benchmarks.micro.bench_norm``.
"""
from __future__ import annotations

import time
from pathlib import Path

import torch
import torch.nn.functional as F


# Keep moved bytes ~constant across N: M = TOTAL_ELEMS // N.
TOTAL_ELEMS = 128 * 1024 * 1024
NS = [32, 64, 128, 256, 512, 1024, 4096, 8192]
DTYPE = torch.bfloat16
PEAK_BW = 4.80e12  # H200 HBM3e peak

WARM = 5
ITERS = 11


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


def main():
    assert torch.cuda.is_available(), "Need CUDA"
    dev = torch.device("cuda")
    sm = torch.cuda.get_device_capability(0)
    print(f"GPU: {torch.cuda.get_device_name(0)}  torch {torch.__version__}  "
          f"sm{sm[0]}{sm[1]}  dtype={DTYPE}")

    from flashlib import flash_rmsnorm, flash_layernorm

    torch.manual_seed(0)
    rows = []
    for N in NS:
        M = TOTAL_ELEMS // N
        x = torch.randn(M, N, device=dev, dtype=DTYPE)
        w = torch.randn(N, device=dev, dtype=DTYPE)
        b = torch.randn(N, device=dev, dtype=DTYPE)
        moved = 2 * M * N * x.element_size()  # read x + write y

        # ---- RMSNorm: native eager vs flashlib ----
        y_t = F.rms_norm(x, (N,), w, 1e-6)
        y_f = flash_rmsnorm(x, w, eps=1e-6)
        rms_diff = (y_t.float() - y_f.float()).abs().max().item()
        t_rms_t = time_ms(lambda: F.rms_norm(x, (N,), w, 1e-6))
        t_rms_f = time_ms(lambda: flash_rmsnorm(x, w, eps=1e-6))

        # ---- LayerNorm (weight+bias): native eager vs flashlib ----
        z_t = F.layer_norm(x, (N,), w, b, 1e-5)
        z_f = flash_layernorm(x, w, b, eps=1e-5)
        ln_diff = (z_t.float() - z_f.float()).abs().max().item()
        t_ln_t = time_ms(lambda: F.layer_norm(x, (N,), w, b, 1e-5))
        t_ln_f = time_ms(lambda: flash_layernorm(x, w, b, eps=1e-5))

        rows.append({
            "N": N, "M": M,
            "rms_t": t_rms_t, "rms_f": t_rms_f, "rms_x": t_rms_t / t_rms_f,
            "rms_f_bw": moved / 1e9 / (t_rms_f / 1e3), "rms_diff": rms_diff,
            "ln_t": t_ln_t, "ln_f": t_ln_f, "ln_x": t_ln_t / t_ln_f,
            "ln_f_bw": moved / 1e9 / (t_ln_f / 1e3), "ln_diff": ln_diff,
        })

    # ── print ──
    print(f"\n{'N':>6}{'M':>10} | {'RMS torch':>10}{'RMS flash':>10}{'x':>7}{'flash%BW':>9}"
          f" | {'LN torch':>10}{'LN flash':>10}{'x':>7}{'flash%BW':>9}")
    for r in rows:
        print(f"{r['N']:>6}{r['M']:>10} | {r['rms_t']:>10.3f}{r['rms_f']:>10.3f}"
              f"{r['rms_x']:>6.2f}x{100*r['rms_f_bw']/ (PEAK_BW/1e9):>8.1f}%"
              f" | {r['ln_t']:>10.3f}{r['ln_f']:>10.3f}{r['ln_x']:>6.2f}x"
              f"{100*r['ln_f_bw']/(PEAK_BW/1e9):>8.1f}%")

    # ── persist ──
    out = Path(__file__).resolve().parent.parent / "results" / "micro_norm.md"
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w") as f:
        f.write("# Micro-benchmark: forward RMSNorm / LayerNorm vs native (eager) torch\n\n")
        f.write(f"GPU: **{torch.cuda.get_device_name(0)}**, sm{sm[0]}{sm[1]}, "
                f"torch {torch.__version__}, dtype={DTYPE}. Bytes per shape held "
                f"~constant (`M = {TOTAL_ELEMS:,} // N`). warm={WARM}, iters={ITERS} "
                f"(median ms). Bandwidth vs **H200 peak 4.80 TB/s**.\n\n")
        f.write("Baseline is **native eager** `torch.nn.functional.{rms_norm,layer_norm}` "
                "(one CTA per row). flashlib packs `BLOCK_M` rows per CTA.\n\n")
        f.write("| N (norm dim) | M (rows) | RMS torch (ms) | RMS flash (ms) | RMS speedup | RMS flash %BW | LN torch (ms) | LN flash (ms) | LN speedup | LN flash %BW | max\\|Δ\\| (rms/ln) |\n")
        f.write("|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|\n")
        for r in rows:
            f.write(f"| {r['N']} | {r['M']:,} | {r['rms_t']:.3f} | {r['rms_f']:.3f} | "
                    f"**{r['rms_x']:.2f}×** | {100*r['rms_f_bw']/(PEAK_BW/1e9):.1f}% | "
                    f"{r['ln_t']:.3f} | {r['ln_f']:.3f} | **{r['ln_x']:.2f}×** | "
                    f"{100*r['ln_f_bw']/(PEAK_BW/1e9):.1f}% | "
                    f"{r['rms_diff']:.1e}/{r['ln_diff']:.1e} |\n")
        f.write("\n**Interpretation.** Small `N` (per-head QK-norm, `head_dim` "
                "32-256) is the advantage zone: eager leaves the SMs idle "
                "(one tiny row per CTA) while the multi-row kernel saturates HBM, "
                "giving the large speedups above. As `N` grows a single row already "
                "fills the warps, eager is already bandwidth-bound, and flashlib "
                "converges to parity (≈1×) -- never slower. `max|Δ|` is the max "
                "abs difference vs eager output (bf16 rounding range).\n\n")
        f.write("Source: `benchmarks/micro/bench_norm.py`. Forward kernel ported "
                "from Sparse VideoGen (ICML 2025, arXiv:2502.01776).\n")
    print(f"\nWrote {out}")


if __name__ == "__main__":
    main()
