"""Shared cuML kernel-trace helpers."""
from __future__ import annotations

import json
import re
import subprocess
import time
from collections import defaultdict
from pathlib import Path
from typing import Callable

import torch
from torch.profiler import ProfilerActivity, profile

RESULTS = Path(__file__).resolve().parents[4] / "benchmarks" / "results" / "heavy" / "breakdown_cuml"
RESULTS.mkdir(parents=True, exist_ok=True)


# ---------------------------------------------------------------------------
# Kernel-name canonicalisation
# ---------------------------------------------------------------------------
# CUPTI returns C++ kernel names, often heavily templated:
#   "void cuvs::neighbors::brute_force::detail::tiled_brute_force_knn_kernel<float, int, 32, 8>(...)"
# We strip templates and namespaces down to the leaf kernel name + a short
# template-arg fingerprint so grouping is meaningful.

_TEMPLATE_RE = re.compile(r"<[^<>]*>")
_ARGS_RE = re.compile(r"\(.*$")

# Hand-picked rewrite rules for the most common cuML / RAFT / cuvs / CUTLASS
# kernels.  We match against both demangled (when c++filt succeeds) and
# mangled (when c++filt can't handle nested CUTLASS namespaces) names.
# Mangled-name substrings to look for, in priority order:
#   "FusedDistanceNN"  -> cutlass fused L2 distance + argmin (KMeans assign)
#   "PairwiseDistance" -> cutlass pairwise L2 (KNN brute force, agglom)
#   "FusedDistance"    -> cutlass FusedDistance epilogue (general)
# These substrings appear verbatim in the mangled C++ name because the
# numeric length prefix is followed by the namespace text.
_REWRITE_RULES: list[tuple[re.Pattern, str]] = [
    # CUTLASS-fused distance-NN (cuML KMeans assign kernel)
    (re.compile(r"FusedDistanceNN"), "cutlass_FusedDistanceNN_kmeans"),
    (re.compile(r"PairwiseDistance"), "cutlass_PairwiseDistance"),
    # CUTLASS generic GEMM (cuML linear-model / SVD / PCA)
    (re.compile(r"cutlass.*Gemm.*Persistent"), "cutlass_persistent_gemm"),
    (re.compile(r"cutlass.*[Gg]emm"), "cutlass_gemm"),
    # cuVS / RAFT distance + KNN
    (re.compile(r"tiled_brute_force_knn"), "cuvs_tiled_brute_force_knn"),
    (re.compile(r"brute_force.*l2"), "cuvs_brute_force_l2"),
    (re.compile(r"select_k.*radix"), "cuvs_radix_select_k"),
    (re.compile(r"select_k.*warp"), "cuvs_warp_select_k"),
    (re.compile(r"select_k"), "cuvs_select_k"),
    # RAFT reduce / scan / sort
    (re.compile(r"sum_rows_by_key.*nkeys.*rowmajor"),
        "raft_sum_rows_by_key_large_K"),
    (re.compile(r"sum_rows_by_key"), "raft_sum_rows_by_key"),
    (re.compile(r"reduce_cols_by_key"), "raft_reduce_cols_by_key"),
    (re.compile(r"coalescedSumThinKernel"), "raft_coalesced_sum_thin"),
    (re.compile(r"mapThenReduceKernel"), "raft_map_then_reduce"),
    # CUB device-wide primitives
    (re.compile(r"DeviceReduceSingleTileKernel"),
        "cub_DeviceReduceSingleTile"),
    (re.compile(r"DeviceReduceKernel"), "cub_DeviceReduce"),
    (re.compile(r"DeviceScanInitKernel"), "cub_DeviceScanInit"),
    (re.compile(r"DeviceScanKernel"), "cub_DeviceScan"),
    (re.compile(r"DeviceRadixSortUpsweepKernel"),
        "cub_DeviceRadixSortUpsweep"),
    (re.compile(r"DeviceRadixSortDownsweepKernel"),
        "cub_DeviceRadixSortDownsweep"),
    (re.compile(r"DeviceRadixSortOnesweepKernel"),
        "cub_DeviceRadixSortOnesweep"),
    (re.compile(r"DeviceRadixSortScanBinsKernel"),
        "cub_DeviceRadixSortScanBins"),
    (re.compile(r"DeviceRadixSortHistogramKernel"),
        "cub_DeviceRadixSortHistogram"),
    (re.compile(r"DeviceSegmentedReduceKernel"),
        "cub_DeviceSegmentedReduce"),
    (re.compile(r"DeviceSegmentedRadixSort"),
        "cub_DeviceSegmentedRadixSort"),
    (re.compile(r"DeviceCompactInitKernel"), "cub_DeviceCompactInit"),
    (re.compile(r"DeviceSelectSweepKernel"), "cub_DeviceSelectSweep"),
    # Memory / launch infrastructure
    (re.compile(r"initBinMutexKernel"), "raft_init_bin_mutex"),
    (re.compile(r"static_kernel"), "anon_static_kernel"),
    (re.compile(r"transform_kernel"), "cub_transform_kernel"),
    # DBSCAN-specific kernels
    (re.compile(r"vertexdeg.*RbcRunQuery|rbc.*query|RbcRunQuery"),
        "cuvs_dbscan_rbc_vertex_degree"),
    (re.compile(r"weak_cc"), "raft_dbscan_weak_cc"),
    (re.compile(r"adj_to_csr"), "raft_dbscan_adj_to_csr"),
    (re.compile(r"merge_labels"), "raft_dbscan_merge_labels"),
    # SVD / eigendecomp
    (re.compile(r"jacobi_eigh"), "raft_jacobi_eigh"),
    (re.compile(r"compute_residual_kernel"), "raft_svd_compute_residual"),
    # MNB / linear-model fused kernels
    (re.compile(r"smem_pack_kernel"), "raft_smem_pack"),
    # KNN brute-force naive
    (re.compile(r"l2_distance.*kernel"), "raft_l2_distance"),
    # UMAP-specific
    (re.compile(r"optimize_layout"), "raft_umap_optimize_layout"),
    (re.compile(r"smooth_knn"), "raft_umap_smooth_knn"),
    # TSNE-specific (Barnes-Hut)
    (re.compile(r"compute_repulsive_forces|RepulsiveKernel"),
        "raft_tsne_compute_repulsive_forces"),
    (re.compile(r"compute_attractive_forces|AttractiveKernel"),
        "raft_tsne_compute_attractive_forces"),
    (re.compile(r"summarization_kernel|Summarization"),
        "raft_tsne_summarization"),
    (re.compile(r"tree_building|BoundingBox"),
        "raft_tsne_bh_tree_building"),
    # cupy / pytorch helpers
    (re.compile(r"cupy_copy"), "cupy_copy"),
    (re.compile(r"^Memcpy"), "Memcpy"),
    (re.compile(r"^Memset"), "Memset"),
]


_DEMANGLE_CACHE: dict[str, str] = {}

def _demangle_cxx(raw: str) -> str:
    """Best-effort C++ demangle via /usr/bin/c++filt; falls back to raw.

    Results are cached because CUPTI can emit tens of thousands of events
    per profile and ``subprocess.run`` overhead would otherwise dominate
    the post-profile aggregation cost (~20 ms per call * 50K events).
    """
    cached = _DEMANGLE_CACHE.get(raw)
    if cached is not None:
        return cached
    if not raw.startswith("_Z"):
        _DEMANGLE_CACHE[raw] = raw
        return raw
    try:
        out = subprocess.run(["c++filt", raw],
                              capture_output=True, text=True, timeout=2)
        v = out.stdout.strip() or raw
    except Exception:
        v = raw
    _DEMANGLE_CACHE[raw] = v
    return v


_CANON_CACHE: dict[str, str] = {}

def canonical_kernel_name(raw: str) -> str:
    """Reduce ``raw`` CUPTI kernel name to a short, human-meaningful label.

    The rewriting heuristic is:
      1. Demangle the C++ name (best effort, via /usr/bin/c++filt).
      2. Apply the hand-curated ``_REWRITE_RULES`` for canonical labels of
         the most common cuML / RAFT / cuvs / CUTLASS / CUB kernels.
      3. If no rule matches, strip templates / arg lists / namespaces and
         return the leaf function name.
    """
    cached = _CANON_CACHE.get(raw)
    if cached is not None:
        return cached
    s = _demangle_cxx(raw)
    for pat, label in _REWRITE_RULES:
        if pat.search(s):
            _CANON_CACHE[raw] = label
            return label
    s = re.sub(r"^\s*void\s+", "", s)
    s = _ARGS_RE.sub("", s)
    for _ in range(8):
        new = _TEMPLATE_RE.sub("", s)
        if new == s:
            break
        s = new
    s = re.sub(r"^[\w:]+::", "", s)
    s = s.strip()
    out = s or raw[:80]
    _CANON_CACHE[raw] = out
    return out


def profile_cuml_call(label: str, run_fn: Callable[[], None],
                       *, warmup: int = 1, repeat: int = 3) -> dict:
    """Profile ``run_fn`` (a cuML call) and return per-kernel aggregated stats.

    Args:
        label: human-readable shape label for the run.
        run_fn: zero-arg callable that performs ONE cuML operation
            (e.g. ``lambda: cuKMeans(n_clusters=K).fit(X)``).
        warmup: number of warmup calls (NOT profiled).
        repeat: number of profiled calls; per-kernel stats are summed across
            repeats then divided by ``repeat``.

    Returns:
        dict with keys:
          - ``label``
          - ``outer_wall_ms``: average wall time per call (host-side)
          - ``kernels``: list of {name, count_per_call, total_ms, mean_us, pct}
    """
    print(f"[cuml-profile]   shape: {label}")
    # Warmup
    for _ in range(warmup):
        run_fn()
        torch.cuda.synchronize()

    # Profile
    torch.cuda.synchronize()
    wall_t0 = time.perf_counter()
    with profile(activities=[ProfilerActivity.CUDA],
                 record_shapes=False) as prof:
        for _ in range(repeat):
            run_fn()
            torch.cuda.synchronize()
    wall_total = (time.perf_counter() - wall_t0) * 1000.0
    avg_wall = wall_total / repeat

    # Aggregate per-kernel time across all events
    by_kernel: dict[str, dict] = defaultdict(
        lambda: {"count": 0, "total_us": 0.0, "raw_examples": set()}
    )
    events = prof.events()
    for ev in events:
        if ev.device_type != torch.autograd.DeviceType.CUDA:
            continue
        # Only count actual CUDA kernels, not memcpy / memset markers
        # (which show up as "Memcpy HtoD", "Memset", etc.).
        name = ev.name
        # Newer PyTorch deprecates ``cuda_time_total`` in favour of
        # ``device_time_total``; fall back chain handles both APIs.
        dur_us = getattr(ev, "device_time_total", None)
        if dur_us is None:
            dur_us = getattr(ev, "cuda_time_total", 0.0)
        # Some non-kernel events have zero device time
        if dur_us <= 0:
            continue
        canon = canonical_kernel_name(name)
        rec = by_kernel[canon]
        rec["count"] += getattr(ev, "count", 1) or 1
        rec["total_us"] += float(dur_us)
        if len(rec["raw_examples"]) < 1:
            rec["raw_examples"].add(name[:200])

    # Normalise (we summed across `repeat` calls of the cuML op AND across
    # all CTAs of each kernel — the .count attribute is launches per repeat)
    total_us = sum(r["total_us"] for r in by_kernel.values())
    rows: list[dict] = []
    for canon, rec in by_kernel.items():
        n_launches_per_call = rec["count"] / max(1, repeat)
        total_ms_per_call = rec["total_us"] / max(1, repeat) / 1000.0
        mean_us = (rec["total_us"] / max(1, rec["count"]))
        pct = (rec["total_us"] / total_us * 100.0) if total_us > 0 else 0.0
        rows.append({
            "kernel": canon,
            "raw_example": next(iter(rec["raw_examples"]), ""),
            "launches_per_call": n_launches_per_call,
            "total_ms_per_call": total_ms_per_call,
            "mean_us_per_launch": mean_us,
            "pct_of_total": pct,
        })
    rows.sort(key=lambda r: -r["total_ms_per_call"])
    return {
        "label": label,
        "outer_wall_ms": avg_wall,
        "n_kernels": len(rows),
        "kernels": rows,
    }


def write_cuml_breakdown_md(prim: str, shape_results: list[dict],
                              *, notes: str | None = None,
                              top_k: int = 12) -> None:
    """Render a per-cuML-primitive multi-shape kernel-trace table.

    For each shape, render the top-``top_k`` kernels by total time.
    """
    md = [f"# heavy/breakdown_cuml/{prim} — cuML CUDA kernel trace",
          "",
          "Hardware: NVIDIA H200  |  cuML 25.10  |  profiler: PyTorch CUPTI",
          ""]
    if notes:
        md.append(f"_{notes}_")
        md.append("")
    md.append("All times reported per-cuML-call (median of 3 repeats; first "
              "call discarded). ``launches_per_call`` is the average number "
              "of CUDA kernel launches per cuML operation; ``mean_us`` is the "
              "per-launch mean device time.")
    md.append("")

    for sr in shape_results:
        md.append(f"## Shape: `{sr['label']}`")
        md.append("")
        md.append(f"Outer wall (host): **{sr['outer_wall_ms']:.2f} ms per cuML "
                  f"call** &nbsp;|&nbsp; total distinct kernels: "
                  f"**{sr['n_kernels']}**")
        md.append("")
        md.append("| # | kernel | launches/call | total ms/call | mean µs | % |")
        md.append("|---|---|---|---|---|---|")
        for i, k in enumerate(sr["kernels"][:top_k], start=1):
            md.append(f"| {i} | `{k['kernel']}` | "
                      f"{k['launches_per_call']:.1f} | "
                      f"{k['total_ms_per_call']:.2f} | "
                      f"{k['mean_us_per_launch']:.1f} | "
                      f"{k['pct_of_total']:.1f}% |")
        if len(sr["kernels"]) > top_k:
            tail_ms = sum(k["total_ms_per_call"]
                           for k in sr["kernels"][top_k:])
            tail_pct = sum(k["pct_of_total"]
                            for k in sr["kernels"][top_k:])
            md.append(f"| ... | _(other {len(sr['kernels'])-top_k} kernels)_ | "
                      f"- | {tail_ms:.2f} | - | {tail_pct:.1f}% |")
        md.append("")

    path_md = RESULTS / f"{prim}.md"
    path_md.write_text("\n".join(md) + "\n")
    print(f"[cuml-profile] wrote {path_md}")

    # JSON dump for downstream tools (drop raw_example to keep it terse)
    out = {
        "primitive": prim,
        "tool": "torch.profiler CUPTI",
        "cuml_version": "25.10.00",
        "shapes": [
            {
                "label": sr["label"],
                "outer_wall_ms": sr["outer_wall_ms"],
                "n_kernels": sr["n_kernels"],
                "kernels": [
                    {k: v for k, v in row.items() if k != "raw_example"}
                    for row in sr["kernels"]
                ],
            }
            for sr in shape_results
        ],
    }
    path_json = RESULTS / f"{prim}.json"
    path_json.write_text(json.dumps(out, indent=2))


def free_gpu() -> None:
    import gc
    gc.collect()
    torch.cuda.empty_cache()
    try:
        import cupy
        cupy.get_default_memory_pool().free_all_blocks()
        cupy.get_default_pinned_memory_pool().free_all_blocks()
    except Exception:
        pass


def torch_tensor_to_cupy(t: "torch.Tensor"):
    """Zero-copy DLPack hand-off; matches benchmarks/vs_cuml/heavy/*.py policy."""
    import cupy as cp
    return cp.from_dlpack(t)
