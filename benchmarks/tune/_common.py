"""Shared bench / IO helpers for ``benchmarks/tune/<op>.py`` scripts.

Every per-op tuner uses the same machinery:

* :func:`bench_ms` — warm-up + median wall-clock timing (CUDA-synced).
* :func:`expand_grid` — turn a workload-spec dict into a list of
  concrete dict rows.
* :func:`results_dir` — canonical results path
  (``benchmarks/tune/results/<op>/<device_tag>/``).
* :func:`write_jsonl` — append rows to a workload's JSONL file with a
  trailing ``"summary"`` line.
* :func:`run_tuner` — orchestrates the WORKLOADS × BACKENDS sweep.

The tuner contract is documented in :doc:`README.md`.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
import traceback
from dataclasses import asdict, dataclass
from itertools import product
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, List, Optional

import torch

from flashlib import _hw


# ──────────────────────────────────────────────────────────────────────
# Paths
# ──────────────────────────────────────────────────────────────────────

ROOT = Path(__file__).resolve().parent
RESULTS_ROOT = ROOT / "results"


def results_dir(op: str, device_tag: Optional[str] = None) -> Path:
    """Return ``benchmarks/tune/results/<op>/<device_tag>/`` (created)."""
    tag = device_tag or _hw.device_tag()
    p = RESULTS_ROOT / op / tag
    p.mkdir(parents=True, exist_ok=True)
    return p


# ──────────────────────────────────────────────────────────────────────
# Bench
# ──────────────────────────────────────────────────────────────────────

def bench_ms(fn: Callable[[], Any], warm: int = 3, iters: int = 5) -> float:
    """Median wall-clock ms over ``iters`` runs, after ``warm`` warmups.

    Each run is fenced by :func:`torch.cuda.synchronize` so the timing is
    a *true* end-to-end host-visible cost (matches what users feel).
    """
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
# Grid
# ──────────────────────────────────────────────────────────────────────

def expand_grid(spec: Dict[str, Iterable[Any]]) -> List[Dict[str, Any]]:
    """Turn a dict of name -> iterable into a list of cartesian-product dicts.

    >>> expand_grid({"N": [128, 256], "D": [16]})
    [{"N": 128, "D": 16}, {"N": 256, "D": 16}]
    """
    keys = list(spec.keys())
    return [dict(zip(keys, vals)) for vals in product(*[spec[k] for k in keys])]


def shape_key(workload: Dict[str, Any]) -> str:
    """Filesystem-safe stringification of a workload dict.

    ``{"N": 1024, "D": 64, "k": 8}`` -> ``"N1024_D64_k8"``.
    """
    return "_".join(f"{k}{v}" for k, v in workload.items())


# ──────────────────────────────────────────────────────────────────────
# JSONL IO
# ──────────────────────────────────────────────────────────────────────

@dataclass
class TuneRow:
    """One (workload, backend) measurement row."""

    workload: Dict[str, Any]
    backend: str
    variant: Optional[str]
    time_ms: Optional[float]
    rel_err: Optional[float]
    status: str
    error: Optional[str] = None

    def to_json(self) -> Dict[str, Any]:
        return asdict(self)


def write_jsonl(path: Path, rows: List[TuneRow], summary: Dict[str, Any]) -> None:
    """Atomic write: dump rows then a ``summary`` line."""
    tmp = path.with_suffix(path.suffix + ".tmp")
    with tmp.open("w") as f:
        for r in rows:
            f.write(json.dumps(r.to_json()) + "\n")
        f.write(json.dumps({"summary": summary}) + "\n")
    tmp.replace(path)


# ──────────────────────────────────────────────────────────────────────
# Orchestration
# ──────────────────────────────────────────────────────────────────────

def fingerprint(extra: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """Snapshot of the calibration environment for traceability."""
    hw = _hw.current()
    fp = {
        "device_tag": hw.device_tag,
        "device_name": hw.name,
        "sm_arch": hw.sm_arch,
        "sm_count": hw.sm_count,
        "l2_bytes": hw.l2_bytes,
        "smem_per_sm_bytes": hw.smem_per_sm_bytes,
        "total_mem_bytes": hw.total_mem_bytes,
        "torch_version": torch.__version__,
        "torch_cuda": torch.version.cuda,
        "calibrated_at_unix": time.time(),
    }
    if extra:
        fp.update(extra)
    return fp


def run_tuner(
    *,
    op: str,
    workloads: List[Dict[str, Any]],
    backends: List[Dict[str, Any]],
    setup_fn: Callable[[Dict[str, Any]], Dict[str, Any]],
    bench_fn: Callable[[Dict[str, Any], Dict[str, Any]], Callable[[], Any]],
    correctness_fn: Optional[Callable[[Dict[str, Any], Dict[str, Any]], float]] = None,
    warm: int = 3,
    iters: int = 5,
    rerun: bool = False,
    workload_filter: Optional[Callable[[Dict[str, Any]], bool]] = None,
) -> List[Path]:
    """Run a full ``workloads × backends`` sweep and persist JSONL files.

    Parameters
    ----------
    op
        Op name (used to name the results subdir and the script-level CLI
        log line).
    workloads
        List of dicts. Each dict is a complete shape spec
        (e.g. ``{"N": 1024, "M": 4096, "D": 64, "k": 8}``).
    backends
        List of candidate dicts. Each dict MUST contain ``backend`` and
        MAY contain ``variant``; arbitrary extra keys are forwarded to
        :func:`bench_fn` so backend-specific knobs can travel with the
        candidate.
    setup_fn(workload) -> ctx
        Build the input tensors / shared state for ``workload``. Called
        ONCE per workload; the returned ``ctx`` dict is passed to
        ``bench_fn`` and ``correctness_fn``.
    bench_fn(ctx, candidate) -> callable
        Returns a no-arg callable that runs the candidate on the
        prepared inputs. Caller is responsible for any per-call clones
        (we typically do not — the caller picks workloads where mutation
        is OK or the kernel is idempotent).
    correctness_fn(ctx, candidate) -> rel_err
        Optional. Cheap sanity check vs torch reference. ``None``
        leaves ``rel_err=None`` in the JSONL.
    warm, iters
        Per-cell timing parameters.
    rerun
        Overwrite existing per-workload JSONL files.
    workload_filter
        Optional predicate applied to each workload; useful for ``--size``
        flags.

    Returns
    -------
    List of paths that were written.
    """
    out_dir = results_dir(op)
    fp = fingerprint({"op": op, "warm": warm, "iters": iters})

    written: List[Path] = []
    for w in workloads:
        if workload_filter is not None and not workload_filter(w):
            continue
        out_path = out_dir / f"{shape_key(w)}.jsonl"
        if out_path.exists() and not rerun:
            print(f"[{op}] skip (exists): {out_path.name}")
            continue
        try:
            ctx = setup_fn(w)
        except Exception:
            print(f"[{op}] setup failed for {w}:\n{traceback.format_exc()[-400:]}")
            continue

        rows: List[TuneRow] = []
        best_ms: Optional[float] = None
        best_backend: Optional[str] = None

        for cand in backends:
            be = cand["backend"]
            var = cand.get("variant")
            try:
                fn = bench_fn(ctx, cand)
                t = bench_ms(fn, warm=warm, iters=iters)
                err = (correctness_fn(ctx, cand)
                       if correctness_fn is not None else None)
                rows.append(TuneRow(
                    workload=dict(w), backend=be, variant=var,
                    time_ms=t, rel_err=err, status="ok",
                ))
                if best_ms is None or t < best_ms:
                    best_ms, best_backend = t, _label(be, var)
            except Exception:
                rows.append(TuneRow(
                    workload=dict(w), backend=be, variant=var,
                    time_ms=None, rel_err=None, status="error",
                    error=traceback.format_exc()[-400:],
                ))

        summary = {
            "workload": dict(w),
            "best_backend": best_backend,
            "best_ms": best_ms,
            "by_backend": {
                _label(r.backend, r.variant): r.time_ms for r in rows
            },
            "fingerprint": fp,
        }
        write_jsonl(out_path, rows, summary)
        written.append(out_path)
        print(f"[{op}] wrote {out_path.name}: best={best_backend} "
              f"@ {best_ms:.3f}ms" if best_ms is not None
              else f"[{op}] wrote {out_path.name}: ALL FAILED")

    return written


def _label(backend: str, variant: Optional[str]) -> str:
    """Stable string id for a (backend, variant) pair."""
    return backend if not variant else f"{backend}/{variant}"


# ──────────────────────────────────────────────────────────────────────
# CLI helper used by every tuner
# ──────────────────────────────────────────────────────────────────────

def parse_argv(op: str) -> argparse.Namespace:
    """Standard ``--rerun`` / ``--size`` CLI shared across all tuners."""
    p = argparse.ArgumentParser(prog=f"benchmarks.tune.{op}")
    p.add_argument("--rerun", action="store_true",
                   help="overwrite existing per-workload JSONL files")
    p.add_argument("--size", default=None,
                   help="comma-separated list of size labels to run; "
                        "if omitted runs every workload")
    return p.parse_args()
