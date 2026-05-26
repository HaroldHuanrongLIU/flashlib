"""Micro-benchmark: flash-kmeans assign vs naive materialize-then-argmin.

Quantifies the win from the streaming-fused assign kernel (no N x K
distance matrix in HBM) as a function of K.

Three backends per shape:
    naive    -- ((x[:, None] - c[None])**2).sum(-1).argmin(-1)
                materialises a (N, K) fp32 cross matrix in HBM
                (= 4 * N * K bytes scanned per Lloyd iteration).
    triton   -- flashlib.euclid_assign_triton (FA-style: streams K
                outside, fp32 online argmin, x^2-free signed score).
    cutedsl  -- flashlib.cutedsl_assign_euclid (FA3-style: TMA + WGMMA
                + warp specialization; routes to triton when shape
                is outside its supported regime).

Shape grid: B=1, N=65536, D=128, dtype=fp16, K in {64, 256, 1024, 4096}.

Writes benchmarks/results/micro_assign_kernel.md.
"""
from __future__ import annotations

import statistics
import time
from pathlib import Path

import torch


# ──────────────────────────────────────────────────────────────────────
# Shape grid
# ──────────────────────────────────────────────────────────────────────

B = 1
N = 65_536
D = 128
DTYPE = torch.float16
KS = [64, 256, 1024, 4096]

WARM = 3
ITERS = 7


# ──────────────────────────────────────────────────────────────────────
# Reference baselines
# ──────────────────────────────────────────────────────────────────────

def naive_assign(x: torch.Tensor, c: torch.Tensor) -> torch.Tensor:
    """Materialise N x K cross matrix then argmin.

    Cast to fp32 first because fp16 squared-distance overflows on
    realistic shapes and the runtime gap to materialised-fp16 is in
    the noise. The fp32 path also matches what scikit-learn does.
    """
    # (B, N, K) fp32 — this is the (N*K*4)-byte HBM write we want to avoid.
    xx = x.to(torch.float32)
    cc = c.to(torch.float32)
    d = torch.cdist(xx, cc, p=2)  # cuBLAS-tuned; lower bound for any
                                   # materialise-then-argmin path.
    return d.argmin(dim=-1).to(torch.int32)


def naive_assign_squared(x: torch.Tensor, c: torch.Tensor) -> torch.Tensor:
    """Same as naive_assign but uses the 3-term expansion that fla1's
    docstring describes::

        d^2 = ||x||^2 + ||c||^2 - 2 <x, c>

    matches what a hand-rolled torch impl typically writes, and is
    what cuML / sklearn's pairwise_distances does internally. Slightly
    faster than torch.cdist on Hopper because it skips the sqrt.
    """
    xx = x.to(torch.float32)
    cc = c.to(torch.float32)
    # (B, N, K) fp32 — this is the materialisation we're benchmarking.
    cross = torch.bmm(xx, cc.transpose(1, 2))
    x_sq = (xx * xx).sum(dim=-1, keepdim=True)        # (B, N, 1)
    c_sq = (cc * cc).sum(dim=-1).unsqueeze(1)          # (B, 1, K)
    d = x_sq + c_sq - 2.0 * cross
    return d.argmin(dim=-1).to(torch.int32)


# ──────────────────────────────────────────────────────────────────────
# Timing helper
# ──────────────────────────────────────────────────────────────────────

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


# ──────────────────────────────────────────────────────────────────────
# Run
# ──────────────────────────────────────────────────────────────────────

def main():
    assert torch.cuda.is_available(), "Need CUDA"
    dev = torch.device("cuda")
    print(f"GPU: {torch.cuda.get_device_name(0)}  "
          f"torch {torch.__version__}  sm{torch.cuda.get_device_capability(0)}")

    # Import after CUDA is verified so the lazy attr machinery doesn't
    # fire during 'pytest --collect-only' etc.
    from flashlib.primitives.kmeans import (
        euclid_assign_triton,
        cutedsl_assign_euclid,
    )

    torch.manual_seed(0)
    rows = []

    for K in KS:
        x = torch.randn(B, N, D, device=dev, dtype=DTYPE)
        c = torch.randn(B, K, D, device=dev, dtype=DTYPE)

        # ---- correctness anchor: torch.cdist on fp32 ---------------------
        ref = naive_assign(x, c)

        # ---- naive (materialise via x^2 + c^2 - 2xc) ---------------------
        try:
            out_naive = naive_assign_squared(x, c)
            err_naive = int((out_naive != ref).sum().item())
            t_naive = time_ms(lambda: naive_assign_squared(x, c))
        except Exception as e:
            err_naive = -1
            t_naive = float("nan")
            print(f"  naive  K={K}: {e}")

        # ---- triton ------------------------------------------------------
        x_3d = x.contiguous()
        c_3d = c.contiguous()
        out_tri = euclid_assign_triton(x_3d, c_3d)
        err_tri = int((out_tri != ref).sum().item())
        t_tri = time_ms(lambda: euclid_assign_triton(x_3d, c_3d))

        # ---- cutedsl (autotune=False to skip the multi-second sweep) ----
        try:
            out_cute = cutedsl_assign_euclid(x_3d, c_3d, autotune=False)
            err_cute = int((out_cute != ref).sum().item())
            t_cute = time_ms(
                lambda: cutedsl_assign_euclid(x_3d, c_3d, autotune=False)
            )
        except Exception as e:
            err_cute = -1
            t_cute = float("nan")
            print(f"  cutedsl K={K}: {e}")

        # HBM-byte arithmetic --------------------------------------------
        # Both fused kernels' algorithmic LOWER BOUND for HBM bytes:
        #   read X (N*D*2) + read C (K*D*2) + write labels (N*4)
        #
        # Naive kernel's INCLUSIVE bytes (BLAS + matmul + argmin):
        #   read X (N*D*2) + read C (K*D*2) + write D matrix (N*K*4)
        #   + read D matrix (N*K*4) + write labels (N*4)
        bytes_fused = N * D * 2 + K * D * 2 + N * 4
        bytes_naive = (N * D * 2 + K * D * 2 + 2 * N * K * 4 + N * 4)

        # HBM bandwidth on the H200 is 4.80 TB/s peak.
        peak_bw = 4.80e12
        def gbps(b, ms): return b / 1e9 / (ms / 1000.0)
        def pct(b, ms):  return 100.0 * (b / ms / 1e3) / peak_bw

        rows.append({
            "K": K,
            "naive_ms":   t_naive,
            "triton_ms":  t_tri,
            "cutedsl_ms": t_cute,
            "naive_GBps":  gbps(bytes_naive, t_naive)  if t_naive == t_naive else float("nan"),
            "triton_GBps": gbps(bytes_fused, t_tri),
            "cutedsl_GBps": gbps(bytes_fused, t_cute) if t_cute == t_cute else float("nan"),
            "triton_speedup":  (t_naive / t_tri) if t_naive == t_naive else float("nan"),
            "cutedsl_speedup": (t_naive / t_cute) if (t_cute == t_cute and t_naive == t_naive) else float("nan"),
            "err_naive": err_naive,
            "err_triton": err_tri,
            "err_cutedsl": err_cute,
            "hbm_saved": 1.0 - bytes_fused / bytes_naive,
        })

    # ──────────────────────────────────────────────────────────────────
    # Pretty-print + persist
    # ──────────────────────────────────────────────────────────────────
    print()
    print(f"{'K':>5}  {'naive_ms':>10}  {'tri_ms':>8}  {'cute_ms':>8}  "
          f"{'tri_x':>7}  {'cute_x':>8}  {'HBM_saved':>9}")
    for r in rows:
        print(f"{r['K']:>5}  {r['naive_ms']:>10.3f}  {r['triton_ms']:>8.3f}  "
              f"{r['cutedsl_ms']:>8.3f}  {r['triton_speedup']:>6.2f}x  "
              f"{r['cutedsl_speedup']:>7.2f}x  {100.0*r['hbm_saved']:>8.1f}%")

    out_path = Path(__file__).resolve().parent.parent / "results" / "micro_assign_kernel.md"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w") as f:
        gpu = torch.cuda.get_device_name(0)
        sm = torch.cuda.get_device_capability(0)
        f.write(f"# Micro-benchmark: flash-kmeans assign vs naive materialize-then-argmin\n\n")
        f.write(f"GPU: **{gpu}**, sm{sm[0]}{sm[1]}, torch {torch.__version__}. "
                f"Shape: B={B}, N={N}, D={D}, dtype={DTYPE}. "
                f"warm={WARM}, iters={ITERS} (median ms).\n\n")
        f.write("HBM bytes (lower bound for each kernel):\n\n")
        f.write("- **fused (triton/cutedsl)**: read X (N·D·2) + read C (K·D·2) + write labels (N·4)\n")
        f.write("- **naive (materialise N×K dist)**: above + 2·(N·K·4) for write+read of the cross matrix\n\n")
        f.write("HBM saved column = `1 - bytes_fused / bytes_naive`. "
                "Bandwidth columns assume **H200 peak = 4.80 TB/s**.\n\n")
        f.write("| K | naive (ms) | triton (ms) | cutedsl (ms) | triton vs naive | cutedsl vs naive | HBM saved | naive %BW | triton %BW | cutedsl %BW | argmin errors (n/t/c) |\n")
        f.write("|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|\n")
        for r in rows:
            naive_bytes = N*D*2 + K*D*2 + 2*N*r['K']*4 + N*4
            fused_bytes = N*D*2 + r['K']*D*2 + N*4
            naive_pct = 100.0 * (naive_bytes / r['naive_ms'] / 1e3) / 4.80e12 if r['naive_ms']==r['naive_ms'] else float('nan')
            tri_pct   = 100.0 * (fused_bytes / r['triton_ms'] / 1e3) / 4.80e12
            cute_pct  = 100.0 * (fused_bytes / r['cutedsl_ms'] / 1e3) / 4.80e12 if r['cutedsl_ms']==r['cutedsl_ms'] else float('nan')
            f.write(f"| {r['K']} | {r['naive_ms']:.3f} | {r['triton_ms']:.3f} | "
                    f"{r['cutedsl_ms']:.3f} | **{r['triton_speedup']:.2f}×** | "
                    f"**{r['cutedsl_speedup']:.2f}×** | "
                    f"{100.0*r['hbm_saved']:.1f}% | "
                    f"{naive_pct:.1f}% | {tri_pct:.1f}% | {cute_pct:.1f}% | "
                    f"{r['err_naive']}/{r['err_triton']}/{r['err_cutedsl']} |\n")
        f.write("\n")
        f.write("**Interpretation.** All three kernels solve the same problem; the "
                "`argmin errors` column is the count of rows where the kernel's argmin "
                "disagrees with the cdist reference (ties from rounding noise produce "
                "small disagreements). The naive path materialises the full N×K distance "
                "matrix to HBM (8·N·K bytes round-trip); the FA-style fused kernels keep "
                "the running `(min_d, argmin)` register-resident and write only N int32 "
                "labels. HBM-bytes-saved equals K-dependent `1 − fused/naive`; the "
                "speedup mirrors that ratio at the HBM-bound regime and exceeds it where "
                "the materialised path also pays for the extra launch / kernel-fusion "
                "overhead.\n")
        f.write("\n")
        f.write("Source: `benchmarks/micro/bench_assign_kernel.py`. "
                "Re-run with `python -m benchmarks.micro.bench_assign_kernel`.\n")
    print(f"\nWrote {out_path}")


if __name__ == "__main__":
    main()
