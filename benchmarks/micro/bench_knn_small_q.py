"""Micro-benchmark: flash-knn small-Q bandwidth utilization (flash-decoding regime).

Quantifies the bandwidth a flash-decoding-style M-split KNN kernel can
sustain when Q is small (so the canonical "warp-per-query" map doesn't
saturate the SMs).

For each Q in {1, 4, 16, 64, 256, 1024, 4096} we measure flash-knn at
fixed M=10M and D in {64, 128} (typical embedding-retrieval shape) and
report (a) wall ms, (b) achieved HBM bandwidth, and (c) %peak of the
H200's 4.80 TB/s. The expected shape: flat near-peak BW plateau for the
SM-undersaturated small-Q regime (the flash-decoding split is doing its
job), then a knee around Q ~ 512-1024 where the kernel becomes
compute-bound (cross matrix FLOPs dominate) and BW%peak drops.

Writes benchmarks/results/micro_knn_small_q.md.
"""
from __future__ import annotations

import time
from pathlib import Path

import torch


# ──────────────────────────────────────────────────────────────────────
# Shape grid
# ──────────────────────────────────────────────────────────────────────

M = 10_000_000          # corpus size: typical embedding-retrieval scale
DS = [64, 128]
QS = [1, 4, 16, 64, 256, 1024, 4096]
K = 10
DTYPE = torch.bfloat16  # headline regime for the small-Q sweep

WARM = 2
ITERS = 5
PEAK_BW = 4.80e12       # H200 HBM peak (TB/s)


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

    from flashlib.primitives.knn import flash_knn

    torch.manual_seed(0)
    rows = []

    for D in DS:
        # The corpus is huge -- allocate once per D, reuse across Q.
        print(f"\nAllocating M={M} D={D} bf16 corpus "
              f"({M*D*2/(1<<30):.2f} GiB) ...")
        c = torch.randn(M, D, device=dev, dtype=DTYPE)
        for Q in QS:
            x = torch.randn(Q, D, device=dev, dtype=DTYPE)

            # Warm + time the fused KNN (indices only -- skip the gather)
            def run():
                return flash_knn(x, c, K, return_distances=False)

            try:
                # Sanity: result shape (we don't validate correctness here
                # -- the parity tests cover that; we're only measuring BW).
                idx = run()
                t = time_ms(run)
            except Exception as e:
                print(f"  Q={Q}: failed: {e!r}")
                continue

            # GEMM FLOPs charged on the cross matrix
            flops_tf = (2.0 * Q * M * D) / 1e12 / (t / 1000.0)

            # Algorithmic HBM lower bound:
            #   read X (Q*D*2 bytes bf16) + read C once (M*D*2)
            #   + write indices (Q*K*4)
            # The fused kernel achieves this -- no cross matrix in HBM.
            bytes_lb = Q * D * 2 + M * D * 2 + Q * K * 4
            gbps = bytes_lb / 1e9 / (t / 1000.0)
            pct_bw = 100.0 * (gbps * 1e9) / PEAK_BW

            print(f"  Q={Q:>4} D={D}: {t:>7.3f} ms  "
                  f"{gbps:>7.1f} GB/s  ({pct_bw:>5.1f}% peak)  "
                  f"{flops_tf:>7.1f} TF")

            rows.append({
                "Q": Q,
                "D": D,
                "time_ms": t,
                "gbps": gbps,
                "pct_bw": pct_bw,
                "flops_tf": flops_tf,
            })

        # Free the corpus before allocating the next D's
        del c
        torch.cuda.empty_cache()

    # ──────────────────────────────────────────────────────────────────
    # Persist
    # ──────────────────────────────────────────────────────────────────
    out_path = Path(__file__).resolve().parent.parent / "results" / "micro_knn_small_q.md"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w") as f:
        gpu = torch.cuda.get_device_name(0)
        sm = torch.cuda.get_device_capability(0)
        f.write("# Micro-benchmark: flash-knn small-Q bandwidth utilization\n\n")
        f.write(f"GPU: **{gpu}**, sm{sm[0]}{sm[1]}, torch {torch.__version__}. "
                f"Fixed M={M:_}, K={K}, dtype={DTYPE}. "
                f"warm={WARM}, iters={ITERS} (median ms). "
                f"H200 peak HBM = 4.80 TB/s.\n\n")
        f.write("HBM-bytes lower bound = "
                "`Q·D·2 (read X) + M·D·2 (read C) + Q·K·4 (write idx)`. "
                "The fused kernel writes ONLY this many bytes -- no N×M cross "
                "matrix ever lands in HBM. `%peak` reports `gbps / 4800`. "
                "FLOPs column = `2·Q·M·D / time`.\n\n")
        for D in DS:
            f.write(f"## D={D}\n\n")
            f.write("| Q | time (ms) | GB/s | %peak HBM | TFLOPS (cross matmul) |\n")
            f.write("|---:|---:|---:|---:|---:|\n")
            for r in rows:
                if r["D"] != D:
                    continue
                f.write(f"| {r['Q']} | {r['time_ms']:.3f} | {r['gbps']:.1f} | "
                        f"**{r['pct_bw']:.1f}%** | {r['flops_tf']:.1f} |\n")
            f.write("\n")
        f.write("**Interpretation.** For small Q the kernel is memory-bound: "
                "the corpus must be streamed through HBM once and the kernel sustains "
                "a high fraction of peak BW. flash-knn's dispatcher splits the M axis "
                "(`flashlib/primitives/knn/triton/dispatch.py` `_gen_m_splits`) into "
                "1/4/16-wave partial top-K passes when `ceil(N/16)·B < 132` (the SM-count "
                "gate), exactly the flash-decoding pattern -- multiple CTAs cooperate on "
                "one query's reduction. As Q grows past ~512-1024 the warp-per-query path "
                "naturally saturates the 132 SMs and the kernel transitions to a "
                "compute-bound regime where the cross GEMM dominates and the BW fraction "
                "drops while TFLOPs ramps.\n\n")
        f.write("Source: `benchmarks/micro/bench_knn_small_q.py`. "
                "Re-run with `python -m benchmarks.micro.bench_knn_small_q`.\n")
    print(f"\nWrote {out_path}")


if __name__ == "__main__":
    main()
