"""Micro-benchmark: eigh Halko vs cuSOLVER scaling.

For each (N, K), we compute the top-K eigenpairs of a synthetic SPD
matrix and report (a) cuSOLVER full-eigh wall, (b) Halko subspace
iteration wall, (c) the Halko/cuSOLVER speedup, and (d) top-K
eigenvalue rel-err vs cuSOLVER.

Inputs are generated as ``A = U·diag(λ)·Uᵀ`` with exponentially
decaying ``λ_i = 0.95^i`` — the "PCA-realistic" regime where Halko
shines (flat random Wishart matrices are Halko's worst case).

Writes benchmarks/results/micro_eigh_scaling.md.
"""
from __future__ import annotations

import time
from pathlib import Path

import torch


NS = [1024, 2048, 4096, 8192, 16384]   # 32K skipped by default; cuSOLVER
                                       # is ~6 s at 32K which slows the run
KS = [16, 32, 64, 128]

WARM = 2
ITERS = 3


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


def make_spd(N: int, dev, seed=0):
    """Symmetric PSD with exponentially decaying spectrum λ_i = 0.95^i."""
    g = torch.Generator(device=dev).manual_seed(seed)
    U, _ = torch.linalg.qr(torch.randn(N, N, device=dev, dtype=torch.float32, generator=g))
    lam = 0.95 ** torch.arange(N, device=dev, dtype=torch.float32)
    return (U * lam) @ U.T


def main():
    assert torch.cuda.is_available(), "Need CUDA"
    dev = torch.device("cuda")
    print(f"GPU: {torch.cuda.get_device_name(0)}  torch {torch.__version__}")

    from flashlib.linalg.eigh import eigh_cusolver, eigh_halko

    rows = []
    for N in NS:
        print(f"\n=== N={N} ===")
        A = make_spd(N, dev, seed=N)

        # ── cuSOLVER full eigh (computed once per N; same time across K) ──
        try:
            # Warmup + time
            eigh_cusolver(A)
            t_cu = time_ms(lambda: eigh_cusolver(A))
            evals_ref, _ = eigh_cusolver(A)
            print(f"  cuSOLVER full: {t_cu:>9.2f} ms")
        except Exception as e:
            t_cu = float("nan")
            evals_ref = None
            print(f"  cuSOLVER full FAILED: {e}")
            # Without a reference we can't compute rel_err, skip this N
            continue

        for K in KS:
            if K * 4 >= N:
                # Halko gate is K*4 < N; outside that regime Halko
                # auto-falls-through to cuSOLVER anyway -- record a skip.
                rows.append({
                    "N": N, "K": K,
                    "cusolver_ms": t_cu,
                    "halko_ms": float("nan"),
                    "speedup": float("nan"),
                    "topk_rel_err": float("nan"),
                    "skipped": True,
                })
                print(f"  K={K:>3}: skip (Halko gate K*4 < N not met)")
                continue
            try:
                top_evals, _ = eigh_halko(A, K=K)
                t_hk = time_ms(lambda: eigh_halko(A, K=K))
                # rel-err: top-K from Halko (sorted ascending) vs the
                # top-K of cuSOLVER (also ascending -> last K)
                ref_top = evals_ref[-K:]
                rel = float(
                    ((top_evals - ref_top).abs() / ref_top.abs().clamp(min=1e-30))
                    .mean()
                )
                speedup = t_cu / t_hk
                print(f"  K={K:>3}: Halko {t_hk:>7.2f} ms  "
                      f"({speedup:>5.1f}× over cuSOLVER)  "
                      f"top-K rel-err = {rel:.2e}")
                rows.append({
                    "N": N, "K": K,
                    "cusolver_ms": t_cu, "halko_ms": t_hk,
                    "speedup": speedup, "topk_rel_err": rel,
                    "skipped": False,
                })
            except Exception as e:
                print(f"  K={K:>3}: Halko FAILED: {e}")
                rows.append({
                    "N": N, "K": K,
                    "cusolver_ms": t_cu, "halko_ms": float("nan"),
                    "speedup": float("nan"), "topk_rel_err": float("nan"),
                    "skipped": False,
                })

    # ──────────────────────────────────────────────────────────────────
    # Persist
    # ──────────────────────────────────────────────────────────────────
    out_path = Path(__file__).resolve().parent.parent / "results" / "micro_eigh_scaling.md"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w") as f:
        gpu = torch.cuda.get_device_name(0)
        sm = torch.cuda.get_device_capability(0)
        f.write("# Micro-benchmark: eigh Halko vs cuSOLVER scaling\n\n")
        f.write(f"GPU: **{gpu}**, sm{sm[0]}{sm[1]}, torch {torch.__version__}. "
                f"Synthetic SPD with exponentially decaying spectrum "
                f"`λ_i = 0.95^i`. warm={WARM}, iters={ITERS} (median ms). "
                f"`top-K rel-err` = mean `|λ_halko - λ_cu| / |λ_cu|` over the "
                f"returned K eigenvalues.\n\n")
        f.write("Halko's heuristic gate is `K*4 < N AND N >= 256`. Outside "
                "that regime the dispatcher routes to cuSOLVER and the row "
                "below is annotated as such.\n\n")
        # Table
        f.write("| N | K | cuSOLVER (ms) | Halko (ms) | speedup | top-K rel-err |\n")
        f.write("|---:|---:|---:|---:|---:|---:|\n")
        for r in rows:
            if r["skipped"]:
                f.write(f"| {r['N']} | {r['K']} | {r['cusolver_ms']:.2f} | "
                        f"_skip (K·4 ≥ N)_ | — | — |\n")
            elif r["halko_ms"] != r["halko_ms"]:
                f.write(f"| {r['N']} | {r['K']} | {r['cusolver_ms']:.2f} | "
                        f"FAIL | — | — |\n")
            else:
                f.write(f"| {r['N']} | {r['K']} | {r['cusolver_ms']:.2f} | "
                        f"{r['halko_ms']:.2f} | **{r['speedup']:.1f}×** | "
                        f"{r['topk_rel_err']:.2e} |\n")
        f.write("\n")
        f.write("**Interpretation.** Halko's cost is dominated by `(n_iter+1)` "
                "GEMMs of shape `(N,N)·(N,q)` where `q = K + p` "
                "(`p=30` oversample) → `O(N²·q)` FLOPs, vs cuSOLVER's "
                "`O(N³)`. The theoretical speedup ratio is therefore "
                "`N / ((n_iter+1)·q · efficiency_ratio)`, so for fixed K the "
                "speedup grows linearly with N. At `K*4 ≥ N` the cost "
                "approaches cuSOLVER's and the dispatcher routes to "
                "cuSOLVER instead. Top-K rel-err stays below ~1e-3 on the "
                "decaying spectrum because the random-sketch error mass "
                "concentrates in the small-σ tail.\n\n")
        f.write("Source: `benchmarks/micro/bench_eigh_scaling.py`. "
                "Re-run with `python -m benchmarks.micro.bench_eigh_scaling`.\n")
    print(f"\nWrote {out_path}")


if __name__ == "__main__":
    main()
