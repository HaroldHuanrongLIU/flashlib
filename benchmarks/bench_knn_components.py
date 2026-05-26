"""Comprehensive KNN component benchmark — every kept variant x every regime.

Variants exercised:
    - knn_torch_naive              (fp32 reference; precision baseline)
    - knn_torch_chunked            (memory-friendly torch)
    - flash_knn_triton_small_n     (Triton M-split warp-per-query path)
    - flash_knn_triton_large_n     (Triton single-pass build path)
    - flash_knn_triton             (auto-pick small-n vs large-n)
    - flash_knn                    (top-level smart dispatcher)
    - cutedsl_flash_knn            (Hopper FA3 fully-fused; opt-in via
                                    BENCH_KNN_FA3=1)

Regimes (each row in the matrix is a representative real-world shape):

    A. small-batch / small-corpus   (online query corner)
    B. small-batch / big-corpus     (CLIP-style retrieval)
    C. medium-batch / medium-corpus (DBSCAN-mid graph build)
    D. big-batch / big-corpus       (kNN graph build for clustering)
    E. wide-D corner                (cuBLAS GEMM-bound)
    F. big-K stress                 (top-K dominated)

Outputs:
    benchmarks/results/knn_components.md     markdown table per regime
    benchmarks/results/knn_components.json   structured (regime / shape / variant / metrics)

Methodology:
    - Inputs are bf16 unless dtype-specific. Seed fixed (42).
    - Each variant: WARM=2, ITERS=5 timed via cuda Events; medians reported.
    - Variants with multi-minute autotune (FA3) are gated behind BENCH_KNN_FA3=1.
    - Variants that OOM, fail-to-compile, or error: reported as "FAIL(<class>)".
    - Correctness: top-K index set overlap vs the smallest reference we can run
      at this shape (torch_naive when the N*M*4 distance matrix fits, else
      flash_knn_triton_large_n in bf16).
"""
from __future__ import annotations

import json
import math
import os
import sys
import time
import traceback
from pathlib import Path
from typing import Any, Callable, Optional

import torch

ROOT = Path(__file__).resolve().parent.parent
RESULTS_DIR = ROOT / "benchmarks" / "results"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

SEED = 42
WARM = 2
ITERS = 5
DEVICE = "cuda"

# Skip torch_naive (which materializes N×M fp32 distance matrix) when the
# distance matrix would exceed this size — at 16 GB we can fit (64K×64K).
TORCH_NAIVE_MAX_BYTES = 16 * 1024**3

ENABLE_FA3 = os.environ.get("BENCH_KNN_FA3", "0") == "1"


# ---------------------------------------------------------------------------
# Workload matrix
# ---------------------------------------------------------------------------

# (regime, label, N, M, D, k)
WORKLOADS: list[tuple[str, str, int, int, int, int]] = [
    # A — small-batch / small-corpus
    ("A.small-batch / small-corpus", "tiny",        8,    1024,  64,  8),
    ("A.small-batch / small-corpus", "narrow-mid",  32,   4096,  128, 8),
    ("A.small-batch / small-corpus", "narrow-wide", 128,  8192,  128, 8),

    # B — small-batch / big-corpus (CLIP-style retrieval)
    ("B.small-batch / big-corpus",   "clip-mini",   32,   65_536, 128, 10),
    ("B.small-batch / big-corpus",   "clip-mid",    256,  65_536, 384, 10),
    ("B.small-batch / big-corpus",   "clip-large",  1024, 100_000, 768, 10),

    # C — medium batch / medium corpus (DBSCAN-mid)
    ("C.medium / medium",            "dbscan-mid-1", 1024, 4096,   64,  8),
    ("C.medium / medium",            "dbscan-mid-2", 1024, 16_384, 128, 8),
    ("C.medium / medium",            "dbscan-mid-3", 1024, 32_768, 256, 16),

    # D — big-batch / big-corpus (graph build)
    ("D.big / big",                  "graph-4k",     4096,  4096,  64,  16),
    ("D.big / big",                  "graph-8k",     8192,  8192,  128, 16),
    ("D.big / big",                  "graph-16k",    16_384, 16_384, 256, 32),
    ("D.big / big",                  "graph-50k",    50_000, 50_000, 64,  16),

    # E — wide-D corner (cuBLAS-bound)
    ("E.wide-D",                     "cuBLAS-512",   256, 16_384,  512,  10),
    ("E.wide-D",                     "cuBLAS-1024",  512, 16_384,  1024, 10),

    # F — big-K stress (top-K dominated)
    ("F.big-K",                      "topK-64",      4096, 16_384, 128, 64),
    ("F.big-K",                      "topK-128",     4096, 16_384, 128, 128),
]


# ---------------------------------------------------------------------------
# Variant registry
# ---------------------------------------------------------------------------

def _make_variants() -> dict[str, Callable]:
    """Return {variant_name: callable(x, c, k)}.

    All callables accept ``(x, c, k)`` of shape ``(B, N, D)``, ``(B, M, D)``
    and return ``(vals, idxs)``. Triton entry points only return indices, so
    we wrap them to compute distances via the gather kernel for parity.
    """
    from flashlib.primitives.knn import flash_knn, knn_torch_naive, knn_torch_chunked
    from flashlib.primitives.knn.triton import (
        flash_knn_triton,
        flash_knn_triton_small_n,
        flash_knn_triton_large_n,
    )
    from flashlib.kernels.distance.triton import triton_knn_gather_sqdist

    def _with_gather(fn):
        def _wrapper(x, c, k):
            idx = fn(x, c, k)
            vals = triton_knn_gather_sqdist(x, c, idx)
            return vals, idx
        return _wrapper

    variants: dict[str, Callable] = {
        "torch_naive":   knn_torch_naive,
        "torch_chunked": knn_torch_chunked,
        "triton_small_n": _with_gather(flash_knn_triton_small_n),
        "triton_large_n": _with_gather(flash_knn_triton_large_n),
        "triton_auto":   _with_gather(flash_knn_triton),
        "flash_knn":     flash_knn,
    }

    if ENABLE_FA3:
        from flashlib.primitives.knn import cutedsl_flash_knn
        variants["cutedsl_fa3"] = _with_gather(cutedsl_flash_knn)

    return variants


# ---------------------------------------------------------------------------
# Bench helpers
# ---------------------------------------------------------------------------

def _bench_ms(fn: Callable, *, warm: int = WARM, iters: int = ITERS) -> float:
    """Median wall-clock ms for ``fn`` (CUDA-synced)."""
    for _ in range(warm):
        fn()
    torch.cuda.synchronize()
    samples: list[float] = []
    for _ in range(iters):
        s = torch.cuda.Event(enable_timing=True)
        e = torch.cuda.Event(enable_timing=True)
        s.record()
        fn()
        e.record()
        torch.cuda.synchronize()
        samples.append(s.elapsed_time(e))
    samples.sort()
    return samples[len(samples) // 2]


def _index_overlap(idx_a: torch.Tensor, idx_b: torch.Tensor) -> float:
    """Mean per-row top-K overlap. Both shaped ``(*, k)``; flatten (*) to 1D."""
    a = idx_a.reshape(-1, idx_a.shape[-1])
    b = idx_b.reshape(-1, idx_b.shape[-1])
    a_sorted = a.sort(dim=-1).values
    b_sorted = b.sort(dim=-1).values
    overlaps = []
    for i in range(a.shape[0]):
        sa = set(a_sorted[i].cpu().tolist())
        sb = set(b_sorted[i].cpu().tolist())
        if not sa:
            continue
        overlaps.append(len(sa & sb) / len(sa))
    return sum(overlaps) / max(len(overlaps), 1)


def _torch_naive_fits(B: int, N: int, M: int) -> bool:
    return (B * N * M * 4) < TORCH_NAIVE_MAX_BYTES


def _make_inputs(B: int, N: int, M: int, D: int, *, dtype=torch.bfloat16):
    torch.manual_seed(SEED)
    x = torch.randn(B, N, D, device=DEVICE, dtype=dtype)
    c = torch.randn(B, M, D, device=DEVICE, dtype=dtype)
    return x, c


def _run_variant(name: str, fn: Callable, x: torch.Tensor, c: torch.Tensor,
                 k: int, *, ref_idx: Optional[torch.Tensor]) -> dict[str, Any]:
    """Run a single variant and return ``{ms, overlap, status, err}``."""
    out: dict[str, Any] = {"variant": name}
    try:
        result = fn(x, c, k)
        if not (isinstance(result, tuple) and len(result) == 2):
            raise RuntimeError(f"unexpected return type from {name}: {type(result)}")
        vals, idxs = result
        if idxs.dim() == 2 and x.dim() == 3:
            idxs = idxs.unsqueeze(0)
            vals = vals.unsqueeze(0)

        ms = _bench_ms(lambda: fn(x, c, k))
        out["ms"] = ms

        if ref_idx is not None and idxs.shape == ref_idx.shape:
            out["overlap"] = _index_overlap(idxs.to(torch.int32),
                                            ref_idx.to(torch.int32))
        else:
            out["overlap"] = None
        out["status"] = "ok"
    except Exception as e:
        out["ms"] = None
        out["overlap"] = None
        out["status"] = f"FAIL({type(e).__name__})"
        out["err"] = repr(e)[:200]
    return out


# ---------------------------------------------------------------------------
# Driver
# ---------------------------------------------------------------------------

def main() -> int:
    print(f"[bench-knn] device: {torch.cuda.get_device_name(0)}")
    print(f"[bench-knn] FA3 enabled: {ENABLE_FA3}")
    print(f"[bench-knn] warm={WARM} iters={ITERS}")
    print()

    variants = _make_variants()

    rows: list[dict[str, Any]] = []
    summary: dict[str, dict[str, Any]] = {}

    for regime, label, N, M, D, k in WORKLOADS:
        shape_str = f"({N},{M},{D},k={k})"
        print(f"[{regime}] {label} {shape_str}")
        x, c = _make_inputs(1, N, M, D, dtype=torch.bfloat16)
        B = 1

        # Reference idx for overlap: torch_naive if it fits, else triton_build.
        ref_idx: Optional[torch.Tensor] = None
        if _torch_naive_fits(B, N, M):
            try:
                from flashlib.primitives.knn import knn_torch_naive
                _, ref_idx = knn_torch_naive(x, c, k)
                print(f"   [ref] torch_naive (fp32) — N*M*4 = {B*N*M*4/1e9:.2f} GB")
            except Exception as e:
                ref_idx = None
                print(f"   [ref] torch_naive FAILED: {e}")

        if ref_idx is None:
            try:
                from flashlib.primitives.knn.triton import flash_knn_triton_large_n
                ref_idx = flash_knn_triton_large_n(x, c, k)
                print("   [ref] flash_knn_triton_large_n (bf16) — torch_naive too large")
            except Exception as e:
                ref_idx = None
                print(f"   [ref] flash_knn_triton_large_n FAILED: {e}")

        per_shape_results: list[dict[str, Any]] = []
        for name, fn in variants.items():
            row = _run_variant(name, fn, x, c, k, ref_idx=ref_idx)
            row.update(regime=regime, label=label, N=N, M=M, D=D, k=k)
            per_shape_results.append(row)
            ms = row.get("ms")
            ov = row.get("overlap")
            stat = row.get("status")
            ms_str = f"{ms:7.3f}ms" if isinstance(ms, float) else "        —"
            ov_str = f"ov={ov:.3f}" if isinstance(ov, float) else "ov=  —  "
            print(f"   {name:<22s} {ms_str}  {ov_str}  [{stat}]")
            rows.append(row)

        # Per-shape summary: best explicit variant (excluding `flash_knn`)
        # vs what the dispatcher actually picks. Also exclude torch_*
        # references (they're slow on purpose).
        explicit = [r for r in per_shape_results
                    if isinstance(r.get("ms"), float)
                    and r["variant"] not in ("flash_knn", "torch_naive", "torch_chunked")]
        winner = min(explicit, key=lambda r: r["ms"]) if explicit else None
        dispatcher_row = next((r for r in per_shape_results if r["variant"] == "flash_knn"), None)
        if winner and dispatcher_row and isinstance(dispatcher_row.get("ms"), float):
            ratio = dispatcher_row["ms"] / winner["ms"]
        else:
            ratio = None
        summary[shape_str] = {
            "regime": regime,
            "label": label,
            "winner": winner["variant"] if winner else None,
            "winner_ms": winner["ms"] if winner else None,
            "dispatch_ms": dispatcher_row["ms"] if dispatcher_row else None,
            "dispatch_ratio": ratio,
        }
        if winner:
            print(f"   ---> best explicit: {winner['variant']} ({winner['ms']:.3f}ms)"
                  + (f"   dispatch ratio: {ratio:.2f}x" if ratio else ""))
        print()

    # ------------------------------------------------------------------
    # Markdown report
    # ------------------------------------------------------------------
    md_path = RESULTS_DIR / "knn_components.md"
    json_path = RESULTS_DIR / "knn_components.json"

    variant_order = list(variants.keys())
    by_regime: dict[str, list[dict]] = {}
    for r in rows:
        by_regime.setdefault(r["regime"], []).append(r)

    n_shapes = len(WORKLOADS)
    n_variants = len(variants)
    n_total = n_shapes * n_variants
    n_ok = sum(1 for r in rows if r.get("status") == "ok")
    n_overlap_ge_95 = sum(
        1 for r in rows
        if isinstance(r.get("overlap"), float) and r["overlap"] >= 0.95
    )
    n_overlap_total = sum(1 for r in rows if isinstance(r.get("overlap"), float))
    dispatch_rows = [v for v in summary.values()
                     if isinstance(v.get("dispatch_ratio"), float)]
    if dispatch_rows:
        avg_ratio = sum(v["dispatch_ratio"] for v in dispatch_rows) / len(dispatch_rows)
        max_ratio = max(v["dispatch_ratio"] for v in dispatch_rows)
        n_within_15x = sum(1 for v in dispatch_rows if v["dispatch_ratio"] <= 1.5)
    else:
        avg_ratio = max_ratio = float("nan")
        n_within_15x = 0

    lines: list[str] = []
    lines.append("# KNN component sweep — every variant × every regime")
    lines.append("")
    lines.append(
        f"Generated: {time.strftime('%Y-%m-%d %H:%M UTC', time.gmtime())} | "
        f"device: {torch.cuda.get_device_name(0)} | "
        f"FA3 enabled: **{ENABLE_FA3}**"
    )
    lines.append("")
    lines.append(
        "All inputs are bf16. Times are CUDA-synced ms (median of "
        f"{ITERS} after {WARM} warm-up iters)."
    )
    lines.append("")
    lines.append(
        "**Reference**: ``torch_naive`` (fp32) when its in-memory distance "
        f"matrix fits in {TORCH_NAIVE_MAX_BYTES/1e9:.0f} GB; otherwise "
        "``flash_knn_triton_large_n`` (bf16). Overlap is mean per-row top-K "
        "index set overlap vs the reference."
    )
    lines.append("")
    lines.append(
        "**Dispatcher accuracy**: per-shape ratio = "
        "`flash_knn ms / fastest-explicit-variant ms`. "
        "Fastest-explicit-variant excludes `flash_knn` itself and the two "
        "torch references. Within `1.5x` is considered a correct pick."
    )
    lines.append("")

    # ----- Headline -----
    lines.append("## Headline")
    lines.append("")
    lines.append(f"- **{n_shapes} shapes × {n_variants} variants = {n_total}** runs; "
                 f"**{n_ok}** completed, **{n_total-n_ok}** failed/errored.")
    lines.append(f"- Top-K index overlap ≥ 0.95: "
                 f"**{n_overlap_ge_95}/{n_overlap_total}** rows.")
    lines.append(f"- Smart dispatcher mean slowdown vs best explicit: "
                 f"**{avg_ratio:.2f}x** (max **{max_ratio:.2f}x**, within "
                 f"1.5x on **{n_within_15x}/{len(dispatch_rows)}** shapes).")
    lines.append("")

    # ----- Per-shape summary -----
    lines.append("### Per-shape summary")
    lines.append("")
    lines.append("| shape (N,M,D,k) | regime | best explicit | dispatch | ratio |")
    lines.append("|---|---|---|---|---:|")
    for k_str, v in summary.items():
        w_name = v.get("winner") or "—"
        w_ms = (f"{v['winner_ms']:.2f}ms"
                if isinstance(v.get("winner_ms"), float) else "—")
        d_ms = (f"{v['dispatch_ms']:.2f}ms"
                if isinstance(v.get("dispatch_ms"), float) else "—")
        ratio_str = (f"{v['dispatch_ratio']:.2f}x"
                     if isinstance(v.get("dispatch_ratio"), float) else "—")
        lines.append(
            f"| {k_str} | {v['regime']} | {w_name} ({w_ms}) | {d_ms} | {ratio_str} |"
        )
    lines.append("")

    # ----- Per-regime detail -----
    for regime in sorted(by_regime.keys()):
        lines.append(f"## {regime}")
        lines.append("")
        lines.append("| label | shape | " + " | ".join(variant_order) +
                     " | winner | ratio |")
        lines.append("|---|---|" + "|".join(["---:"] * len(variant_order)) +
                     "|---|---:|")
        labels: dict[str, dict[str, dict]] = {}
        for r in by_regime[regime]:
            labels.setdefault(r["label"], {})[r["variant"]] = r
        for label in dict.fromkeys(r["label"] for r in by_regime[regime]).keys():
            grouped = labels[label]
            shape = next(iter(grouped.values()))
            shape_str = f"({shape['N']},{shape['M']},{shape['D']},k={shape['k']})"
            cells = []
            for name in variant_order:
                r = grouped.get(name)
                if r is None or r.get("ms") is None:
                    cells.append(r["status"] if r else "—")
                else:
                    ms = r["ms"]
                    ov = r["overlap"]
                    flag = " ⚠" if (isinstance(ov, float) and ov < 0.95
                                    and name != "torch_naive") else ""
                    cells.append(f"{ms:.2f}{flag}")
            sk = f"({shape['N']},{shape['M']},{shape['D']},k={shape['k']})"
            s = summary[sk]
            winner = s["winner"] or "—"
            ratio = s["dispatch_ratio"]
            ratio_str = f"{ratio:.2f}x" if isinstance(ratio, float) else "—"
            lines.append(f"| {label} | {shape_str} | " +
                         " | ".join(cells) + f" | {winner} | {ratio_str} |")
        lines.append("")

    md_path.write_text("\n".join(lines) + "\n")
    json_path.write_text(json.dumps({"rows": rows, "summary": summary}, indent=2,
                                    default=lambda o: None))
    print(f"\nReport: {md_path}")
    print(f"JSON  : {json_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
