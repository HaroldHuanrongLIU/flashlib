"""Multinomial NB ``fit`` scaling probe.

Compares three implementations of the per-class feature-count step:

- ``onehot``  — original ``_nb_count_kernel`` (atomic-free GEMM, fast
  for small C but allocates ``(N/BN, C_PAD, D)`` partial buffer)
- ``sorted``  — new ``_nb_count_sorted_kernel`` (sort y once, ONE
  atomic_add per (class, d_tile) per CTA, O(C*D + N) memory)
- ``scatter`` — torch ``scatter_add_`` reference (pure HBM, no fusion)

We deliberately push C up to the point where the onehot path either
OOMs (``partial_sum`` exceeds free HBM) or stalls its autotune.
"""
from __future__ import annotations

import statistics
import time
from pathlib import Path

import torch


WARM = 2
ITERS = 5


def time_ms(fn):
    for _ in range(WARM):
        fn()
    torch.cuda.synchronize()
    ts = []
    for _ in range(ITERS):
        t0 = time.perf_counter()
        fn()
        torch.cuda.synchronize()
        ts.append((time.perf_counter() - t0) * 1000)
    return sorted(ts)[len(ts) // 2]


def ref_scatter(X, y, C):
    fc = torch.zeros(C, X.shape[1], device=X.device, dtype=torch.float32)
    fc.scatter_add_(0, y[:, None].expand(-1, X.shape[1]), X.to(torch.float32))
    cc = torch.zeros(C, device=X.device, dtype=torch.float32)
    cc.scatter_add_(0, y, torch.ones_like(y, dtype=torch.float32))
    return fc, cc


def partial_bytes(N, D, C):
    from flashlib.primitives.multinomial_nb.triton.nb_core import (
        _select_block_n, _round_up_c_pad,
    )
    bn = _select_block_n(N, D, C)
    return ((N + bn - 1) // bn) * _round_up_c_pad(C) * D * 4


def main():
    from flashlib.primitives.multinomial_nb.triton.nb_core import nb_count_features

    dev = torch.device("cuda")
    gpu = torch.cuda.get_device_name(0)
    free_mem = torch.cuda.mem_get_info()[0]
    print(f"GPU: {gpu}  free HBM: {free_mem/1e9:.1f} GB\n")

    GRID = []
    for N, D in [(1_000_000, 128), (10_000_000, 128), (1_000_000, 1024)]:
        for C in (32, 128, 1_000, 10_000, 100_000):
            GRID.append((N, D, C))

    # Hard upper bound for the onehot partial buffer: above this we don't even
    # try (autotune would have to allocate this for every candidate config).
    ONEHOT_PARTIAL_LIMIT_GB = 2.0

    rows = []
    for (N, D, C) in GRID:
        print(f"  ... {N:,} x {D} x {C}", flush=True)
        pb = partial_bytes(N, D, C)
        torch.cuda.empty_cache()
        free = torch.cuda.mem_get_info()[0]
        x_bytes = N * D * 4
        if pb > ONEHOT_PARTIAL_LIMIT_GB * 1e9:
            onehot_status = f"skip (partial={pb/1e9:.1f}GB > {ONEHOT_PARTIAL_LIMIT_GB:.0f}GB)"
        elif x_bytes + pb > 0.9 * free:
            onehot_status = f"skip (X+partial = {(x_bytes + pb)/1e9:.1f}GB > free)"
        else:
            onehot_status = None

        try:
            torch.manual_seed(0)
            X = torch.rand(N, D, device=dev, dtype=torch.float32)
            y = torch.randint(0, C, (N,), device=dev, dtype=torch.int64)
        except Exception as e:
            print(f"skip {N}/{D}/{C}: X/y alloc OOM ({e})")
            continue

        # sorted (new) — always runs
        print(f"    sorted ...", flush=True)
        try:
            t_sorted = time_ms(lambda: nb_count_features(X, y, C, force_path="sorted"))
        except Exception as e:
            t_sorted = float("nan")

        # onehot (old) — skip if we predict OOM/stall
        if onehot_status is None:
            print(f"    onehot ...", flush=True)
            try:
                t_onehot = time_ms(lambda: nb_count_features(X, y, C, force_path="onehot"))
            except Exception as e:
                t_onehot = float("nan")
                onehot_status = f"failed: {type(e).__name__}"
        else:
            t_onehot = float("nan")
            print(f"    onehot SKIP ({onehot_status})", flush=True)

        # scatter_add reference
        try:
            t_scatter = time_ms(lambda: ref_scatter(X, y, C))
        except Exception:
            t_scatter = float("nan")

        rows.append({
            "N": N, "D": D, "C": C,
            "partial_GB": pb / 1e9,
            "t_onehot": t_onehot, "t_sorted": t_sorted, "t_scatter": t_scatter,
            "onehot_status": onehot_status,
        })
        print(f"  N={N:>10,} D={D:>5} C={C:>7,}  partial={pb/1e9:>6.2f}GB  "
              f"onehot={t_onehot if t_onehot==t_onehot else '   OOM':>8}ms  "
              f"sorted={t_sorted:>7.2f}ms  scatter={t_scatter:>7.2f}ms"
              + (f"  [{onehot_status}]" if onehot_status else ""))
        del X, y
        torch.cuda.empty_cache()

    # ──────────────────────────────────────────────────────────────────
    # Pretty markdown dump
    # ──────────────────────────────────────────────────────────────────
    out_path = Path(__file__).resolve().parent.parent / "results" / "micro_nb_fit_scaling.md"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w") as f:
        f.write("# Multinomial NB fit — sort-then-segment vs original one-hot GEMM\n\n")
        f.write(f"GPU: **{gpu}**, free HBM {free_mem/1e9:.1f} GB. "
                f"Median over {ITERS} iterations, warm-up {WARM}. "
                "X is fp32. ``onehot`` = original ``_nb_count_kernel``; "
                "``sorted`` = new ``_nb_count_sorted_kernel`` (this PR); "
                "``scatter`` = torch ``scatter_add_`` reference.\n\n")
        f.write("The ``partial_GB`` column is the size of the temporary "
                "``(n_blocks, C_PAD, D)`` partial-sum buffer the *onehot* "
                "kernel allocates. Once it crosses ~free-HBM/2 the kernel "
                "OOMs or its autotune stalls (allocating that buffer for "
                "each candidate config).\n\n")
        f.write("| N | D | C | partial buffer | onehot (ms) | **sorted (ms)** | scatter_add (ms) | speedup vs scatter | speedup vs onehot |\n")
        f.write("|---:|---:|---:|---:|---:|---:|---:|---:|---:|\n")
        for r in rows:
            t_on = f"{r['t_onehot']:.2f}" if r['t_onehot'] == r['t_onehot'] else "OOM/stall"
            sp_scatter = (r['t_scatter'] / r['t_sorted']) if r['t_scatter'] == r['t_scatter'] and r['t_sorted'] == r['t_sorted'] else float("nan")
            sp_onehot = (r['t_onehot'] / r['t_sorted']) if r['t_onehot'] == r['t_onehot'] and r['t_sorted'] == r['t_sorted'] else float("nan")
            sp_scatter_s = f"**{sp_scatter:.2f}×**" if sp_scatter == sp_scatter else "—"
            sp_onehot_s = f"**{sp_onehot:.2f}×**" if sp_onehot == sp_onehot else "—"
            f.write(f"| {r['N']:,} | {r['D']} | {r['C']:,} | {r['partial_GB']:.2f} GB | "
                    f"{t_on} | **{r['t_sorted']:.2f}** | {r['t_scatter']:.2f} | "
                    f"{sp_scatter_s} | {sp_onehot_s} |\n")
        f.write("\nSource: `benchmarks/micro/bench_nb_fit_scaling.py`.\n")
    print(f"\nWrote {out_path}")


if __name__ == "__main__":
    main()
