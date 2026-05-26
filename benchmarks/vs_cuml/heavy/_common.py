"""Shared utilities for ``benchmarks/vs_cuml/heavy/`` — release-candidate audit.

Extends ``benchmarks/vs_cuml/_common.py`` with audit primitives:

* ``audit_record(...)`` — append a JSON row to ``<prim>.json`` AND a
  markdown row to ``<prim>.md`` under ``benchmarks/results/heavy/``.
* ``same_init_check(...)`` — assert identical inits across engines so a
  reported ARI / inertia gap cannot be blamed on the init lottery.
* ``hbm_peak_reset()`` / ``hbm_peak_gb()`` — wrap an op in a peak-memory
  measurement so OOM near-misses are visible.
* ``chunked_inertia(...)`` — fp32 reference inertia for KMeans on huge
  ``N`` (CPU would OOM on the per-token diff tensor).
* ``chunked_recall(...)`` — recall@K on huge query sets without
  materialising the per-query intersection.
* ``apples_to_apples(...)`` — wrap a comparison with a "comparison
  conditions" record stating exactly which dtype/algorithm pair is
  being timed.
"""
from __future__ import annotations

import gc
import json
import os
import time
from pathlib import Path
from typing import Any, Callable, Sequence

# Re-export the base helpers so per-primitive scripts can do a single
# import from ``heavy._common``.
from benchmarks.vs_cuml._common import (  # noqa: F401
    cap_threads, cuml_shim, time_gpu, time_cpu, title, hr,
    ari, recall_at_k, cluster_count, header, fmt_table,
)


# ── Output paths ────────────────────────────────────────────────────────
_REPO = Path(__file__).resolve().parents[3]
RESULTS_DIR = _REPO / "benchmarks" / "results" / "heavy"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)
LOGS_DIR = RESULTS_DIR / "logs"
LOGS_DIR.mkdir(parents=True, exist_ok=True)


# ── HBM peak tracking ────────────────────────────────────────────────────
def hbm_peak_reset() -> None:
    """Reset the per-process peak. Call before timing each row."""
    import torch
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()


def hbm_peak_gb() -> float:
    """Peak GPU memory in GB since the last :func:`hbm_peak_reset`."""
    import torch
    if not torch.cuda.is_available():
        return 0.0
    return torch.cuda.max_memory_allocated() / 1e9


# ── Init-equivalence guardrail ────────────────────────────────────────────
def same_init_check(a: Any, b: Any, *, name: str = "init",
                    atol: float = 0.0) -> None:
    """Assert two inits agree elementwise (within ``atol``).

    Used by KMeans heavy: forces both flashlib and cuML to consume the
    SAME starting centroids, so any ARI / inertia gap is an
    implementation gap, not the init lottery. Pass ``atol=2e-2`` when
    one side has been cast to bf16 (round-trip is lossy).
    """
    import numpy as np
    if hasattr(a, "get"):
        a = a.get()
    if hasattr(b, "get"):
        b = b.get()
    if hasattr(a, "cpu"):
        a = a.cpu().numpy()
    if hasattr(b, "cpu"):
        b = b.cpu().numpy()
    a = np.asarray(a)
    b = np.asarray(b)
    if a.shape != b.shape:
        raise AssertionError(f"{name}: shape mismatch {a.shape} vs {b.shape}")
    if atol == 0.0:
        if not np.array_equal(a, b):
            max_diff = float(np.abs(a.astype(float) - b.astype(float)).max())
            raise AssertionError(
                f"{name}: elementwise mismatch (max |Δ| = {max_diff:.3e})"
            )
    else:
        max_diff = float(np.abs(a.astype(float) - b.astype(float)).max())
        if max_diff > atol:
            raise AssertionError(
                f"{name}: |Δ|={max_diff:.3e} > atol={atol:.3e}"
            )


# ── Apples-to-apples comparison conditions stamp ──────────────────────────
def apples_to_apples(
    *,
    op: str,
    shape: dict,
    flashlib_dtype: str,
    cuml_dtype: str,
    flashlib_algorithm: str,
    cuml_algorithm: str,
    init_shared: bool,
    notes: str = "",
) -> dict:
    """Record exactly how two engines were compared on this row.

    Returned dict is embedded in the per-row JSON output so an auditor
    can verify the comparison conditions later. Used by every heavy
    script before reporting its `vs cuml` ratio.
    """
    return {
        "op": op,
        "shape": shape,
        "flashlib_dtype": flashlib_dtype,
        "cuml_dtype": cuml_dtype,
        "flashlib_algorithm": flashlib_algorithm,
        "cuml_algorithm": cuml_algorithm,
        "init_shared": bool(init_shared),
        "precision_step_down": flashlib_dtype != cuml_dtype,
        "algorithm_step_down": flashlib_algorithm != cuml_algorithm,
        "notes": notes,
    }


# ── Chunked reference metrics (huge N safe) ──────────────────────────────
def chunked_inertia(X, C, ids, *, chunk: int = 1_000_000) -> float:
    """Sum of squared distances to assigned centroid, fp32 reference.

    ``X`` (N, D) any float dtype, ``C`` (K, D), ``ids`` (N,) int.
    Uses fp64 accumulator and a single-batch chunked loop so N up to
    100M fits in HBM with D=128.
    """
    import torch
    Cf = C.to(torch.float32)
    total = torch.zeros((), device=X.device, dtype=torch.float64)
    N = X.shape[0]
    for s in range(0, N, chunk):
        e = min(s + chunk, N)
        diff = X[s:e].to(torch.float32) - Cf[ids[s:e]]
        total += diff.pow(2).sum().to(torch.float64)
    return float(total.item())


def chunked_recall(pred_idx, ref_idx, K: int, *, chunk: int = 64_000) -> float:
    """Recall@K on huge query counts. Both arrays are (Q, K)."""
    import numpy as np
    Q = pred_idx.shape[0]
    hits = 0
    for s in range(0, Q, chunk):
        e = min(s + chunk, Q)
        p_chunk = pred_idx[s:e]
        r_chunk = ref_idx[s:e]
        for p_row, r_row in zip(p_chunk, r_chunk):
            hits += len(set(int(x) for x in p_row) & set(int(x) for x in r_row))
    return float(hits) / float(Q * K)


# ── Audit-row writer (markdown + json side-by-side) ─────────────────────
def audit_record(prim: str, row: dict, columns: Sequence[str]) -> None:
    """Append one row to ``<prim>.json`` AND ``<prim>.md``.

    Atomic: the json file is rewritten end-to-end on every call so a
    crash in the middle of a sweep leaves a complete prefix.
    """
    md_path = RESULTS_DIR / f"{prim}.md"
    json_path = RESULTS_DIR / f"{prim}.json"

    # ── markdown ────────────────────────────────────────────────────────
    new_md = not md_path.exists()
    with open(md_path, "a") as f:
        if new_md:
            f.write(f"# heavy/{prim} — release-candidate audit\n\n")
            import torch
            try:
                gpu = torch.cuda.get_device_name(0)
            except Exception:
                gpu = "n/a"
            f.write(f"Hardware: {gpu}  |  CUDA visible: "
                    f"{os.environ.get('CUDA_VISIBLE_DEVICES', 'all')}\n\n")
            f.write("| " + " | ".join(columns) + " |\n")
            f.write("|" + "|".join(["---"] * len(columns)) + "|\n")
        f.write("| " + " | ".join(str(row.get(c, "")) for c in columns) + " |\n")
        f.flush()

    # ── json (rewrite full file each call so partial sweeps are valid) ──
    existing: list[dict] = []
    if json_path.exists():
        try:
            existing = json.loads(json_path.read_text())
        except json.JSONDecodeError:
            existing = []
    existing.append({
        "row": {k: (v if _is_jsonable(v) else str(v)) for k, v in row.items()},
        "ts": time.time(),
    })
    json_path.write_text(json.dumps(existing, indent=2, default=str))


def _is_jsonable(x: Any) -> bool:
    try:
        json.dumps(x)
        return True
    except TypeError:
        return False


def free_gpu() -> None:
    """Free outstanding GPU memory + cache between heavy rows."""
    import torch
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


# ── Pass/fail gate helpers ──────────────────────────────────────────────
def gate_metric(name: str, value: float, *, lower: float | None = None,
                upper: float | None = None) -> str:
    """Return ``PASS`` / ``FAIL`` for a quality metric with stated bounds."""
    if lower is not None and value < lower:
        return f"FAIL ({name}={value:.4g} < {lower})"
    if upper is not None and value > upper:
        return f"FAIL ({name}={value:.4g} > {upper})"
    return "PASS"


# ── Convenience: format a numeric row dict for the markdown table ───────
def fmt(d: dict) -> dict:
    out = {}
    for k, v in d.items():
        if isinstance(v, float):
            if abs(v) >= 1e6 or (abs(v) > 0 and abs(v) < 1e-3):
                out[k] = f"{v:.3e}"
            else:
                out[k] = f"{v:.4f}" if abs(v) < 1.0 else f"{v:.2f}"
        else:
            out[k] = str(v)
    return out
