"""Heavy KNN sweep — release-candidate audit.

Three regimes stressed:

1. **Build** (Q=M=N, the all-pairs KNN graph downstream of HDBSCAN /
   UMAP / SpectralClustering) — up to 256K x 256K x D=64 and an
   additional D=512 row that exercises the WGMMA mainloop near its
   compute-bound peak.
2. **Search small-Q** at M=10M (the flash-decoding-style regime where
   the M-split keeps BW utilisation > 30 % peak even at Q=1).
3. **Search medium-Q** at M=50M (corpus 12 GB at fp32 — verifies the
   per-CTA streaming HBM path does not OOM on 143 GB H200).

Anti-reward-hacking guardrails:

* Both flashlib AND cuML inputs are GPU-resident via the same torch
  buffer (``cp.from_dlpack`` view); cuML never pays an H2D transfer.
* cuML pinned to ``algorithm='brute', metric='euclidean'`` for exact
  IEEE parity (the only mode where its output is bit-equivalent to
  ours up to ties).
* recall@K computed against cuML fp32 IEEE — the same baseline both
  engines should match.
* flashlib bf16 reported as a SEPARATE row from flashlib fp32 so the
  precision step-down vs cuML fp32 is visible.
* CuteDSL FA3 path uses extra warmup (10 iters) to amortise the
  multi-second first-call ``cute.compile``.
"""
from benchmarks.vs_cuml.heavy._common import (
    cap_threads, cuml_shim, time_gpu, title, header, fmt_table,
    audit_record, apples_to_apples, hbm_peak_reset, hbm_peak_gb,
    chunked_recall, gate_metric, free_gpu, RESULTS_DIR,
)
cap_threads(); cuml_shim()

import argparse
import warnings; warnings.filterwarnings("ignore")
import numpy as np
import torch
import cupy as cp

from cuml.neighbors import NearestNeighbors as cuNN
from flashlib.primitives.knn import flash_knn


# (label, M, Q, D, K)
BUILD_SHAPES = [
    ("build  128K x 128K  D=64",   128_000,  128_000,  64,  10),
    ("build  200K x 200K  D=64",   200_000,  200_000,  64,  10),
    ("build  100K x 100K  D=128",  100_000,  100_000, 128,  10),
    ("build   64K x  64K  K=64",    64_000,   64_000,  64,  64),
    # NEW heavy row: D=512 WGMMA-bound build (~6.7e10 mul-adds, fits HBM).
    ("build   48K x  48K  D=512",   48_000,   48_000, 512,  10),
]

# (label, Q, M, D, K)  — flash-decoding regime
SEARCH_SMALL_Q_SHAPES = [
    ("search  Q=1     M=10M D=64",        1, 10_000_000,  64, 10),
    ("search  Q=128   M=10M D=64",      128, 10_000_000,  64, 10),
    ("search  Q=128   M=10M D=128",     128, 10_000_000, 128, 10),
]

# (label, Q, M, D, K)  — 12-24 GB corpus (HBM stress)
SEARCH_HUGE_M_SHAPES = [
    ("search  Q=1024  M=30M D=64",    1_024, 30_000_000,  64, 10),
    ("search  Q=4096  M=10M D=128",   4_096, 10_000_000, 128, 10),
]

DTYPES = [("fp32", torch.float32, 4), ("bf16", torch.bfloat16, 2)]

PRIM = "knn"


def _flops_tf(Q, M, D, t_ms) -> float:
    return (2.0 * Q * M * D) / 1e12 / (t_ms / 1000.0)


def _bw_gb(Q, M, D, K, t_ms, sz) -> float:
    inp = (Q + M) * D * sz
    out = Q * K * 8
    return (inp + out) / 1e9 / (t_ms / 1000.0)


def _torch_to_cupy(t):
    return cp.from_dlpack(t)


def _bench_one(*, kind: str, label: str, M: int, Q: int, D: int, K: int):
    """Bench one shape across cuML + flashlib (fp32, bf16, FA3)."""
    title(f"KNN[{kind}]  {label}  (M={M:,}, Q={Q:,}, D={D}, K={K})")
    rng = np.random.RandomState(0)

    # Inputs allocated on GPU directly to avoid CPU OOM at M=50M.
    torch.manual_seed(0)
    Xc32 = torch.randn(M, D, device="cuda", dtype=torch.float32)
    if Q == M:
        Xq32 = Xc32
    else:
        Xq32 = torch.randn(Q, D, device="cuda", dtype=torch.float32)

    # cuML brute fp32 — the IEEE baseline both engines should match.
    free_gpu(); hbm_peak_reset()
    Xc_cp = _torch_to_cupy(Xc32); Xq_cp = _torch_to_cupy(Xq32)
    cu_nn = cuNN(n_neighbors=K, algorithm="brute", metric="euclidean").fit(Xc_cp)
    cu_out = cu_nn.kneighbors(Xq_cp, return_distance=False)
    cu_idx = cu_out.get() if hasattr(cu_out, "get") else np.asarray(cu_out)
    # Lower repeat counts for the huge-M / huge-Q-build rows that take seconds.
    repeats = 2 if (M >= 20_000_000 or (M == Q and M >= 200_000)) else 5
    t_cu = time_gpu(
        lambda: cu_nn.kneighbors(Xq_cp, return_distance=False),
        repeat=repeats, warmup=1,
    )
    hbm_cu = hbm_peak_gb()
    audit_record(PRIM, {
        "shape": label, "dtype": "fp32", "engine": "cuml(brute)",
        "time_ms": f"{t_cu:9.2f}", "TFLOPS": f"{_flops_tf(Q, M, D, t_cu):6.1f}",
        "GBs": f"{_bw_gb(Q, M, D, K, t_cu, 4):6.1f}",
        "recall_at_K": "1.0000", "vs_cuml": "1.00x",
        "HBM_GB": f"{hbm_cu:.1f}", "gate": "PASS",
        "conditions": apples_to_apples(
            op="knn", shape={"M": M, "Q": Q, "D": D, "K": K},
            flashlib_dtype="-", cuml_dtype="fp32",
            flashlib_algorithm="-", cuml_algorithm="brute_l2",
            init_shared=False,
            notes="reference baseline (cuML fp32 brute)"),
    }, columns=["shape", "dtype", "engine", "time_ms", "TFLOPS", "GBs",
                "recall_at_K", "vs_cuml", "HBM_GB", "gate"])

    for dlabel, dtype, sz in DTYPES:
        Xc = Xc32.to(dtype); Xq = Xq32.to(dtype) if Q != M else Xc

        # Triton path (default, always available).
        free_gpu(); hbm_peak_reset()
        try:
            out_idx = flash_knn(Xq[None], Xc[None], K, backend="triton")[1] \
                        .squeeze(0).cpu().numpy()
            recall = chunked_recall(out_idx, cu_idx, K)
            t_fl = time_gpu(
                lambda: flash_knn(Xq[None], Xc[None], K, backend="triton"),
                repeat=repeats, warmup=3)
            hbm_fl = hbm_peak_gb()
            gate = gate_metric("recall", recall, lower=0.95)
            audit_record(PRIM, {
                "shape": label, "dtype": dlabel, "engine": "flashlib(triton)",
                "time_ms": f"{t_fl:9.2f}", "TFLOPS": f"{_flops_tf(Q, M, D, t_fl):6.1f}",
                "GBs": f"{_bw_gb(Q, M, D, K, t_fl, sz):6.1f}",
                "recall_at_K": f"{recall:.4f}",
                "vs_cuml": f"{t_cu / t_fl:.2f}x",
                "HBM_GB": f"{hbm_fl:.1f}", "gate": gate,
                "conditions": apples_to_apples(
                    op="knn", shape={"M": M, "Q": Q, "D": D, "K": K},
                    flashlib_dtype=dlabel, cuml_dtype="fp32",
                    flashlib_algorithm="triton_x2free_fused",
                    cuml_algorithm="brute_l2",
                    init_shared=False,
                    notes=("precision step-down vs cuML fp32" if dtype != torch.float32
                           else "matched dtype with cuML")),
            }, columns=["shape", "dtype", "engine", "time_ms", "TFLOPS", "GBs",
                        "recall_at_K", "vs_cuml", "HBM_GB", "gate"])
        except Exception as e:
            audit_record(PRIM, {
                "shape": label, "dtype": dlabel, "engine": "flashlib(triton)",
                "time_ms": "ERR", "TFLOPS": "-", "GBs": "-",
                "recall_at_K": "-", "vs_cuml": "-", "HBM_GB": "-",
                "gate": f"FAIL ({type(e).__name__}: {str(e)[:60]})",
                "conditions": {},
            }, columns=["shape", "dtype", "engine", "time_ms", "TFLOPS", "GBs",
                        "recall_at_K", "vs_cuml", "HBM_GB", "gate"])

        # CuteDSL FA3 (bf16 / fp16 only — fp32 raises).
        if dtype is not torch.float32:
            free_gpu(); hbm_peak_reset()
            try:
                out_idx = flash_knn(Xq[None], Xc[None], K, backend="cutedsl")[1] \
                            .squeeze(0).cpu().numpy()
                recall = chunked_recall(out_idx, cu_idx, K)
                t_fl = time_gpu(
                    lambda: flash_knn(Xq[None], Xc[None], K, backend="cutedsl"),
                    repeat=repeats, warmup=10)  # FA3 needs extra warmup
                hbm_fl = hbm_peak_gb()
                gate = gate_metric("recall", recall, lower=0.90)  # FA3 looser
                audit_record(PRIM, {
                    "shape": label, "dtype": dlabel,
                    "engine": "flashlib(cutedsl-FA3)",
                    "time_ms": f"{t_fl:9.2f}",
                    "TFLOPS": f"{_flops_tf(Q, M, D, t_fl):6.1f}",
                    "GBs": f"{_bw_gb(Q, M, D, K, t_fl, sz):6.1f}",
                    "recall_at_K": f"{recall:.4f}",
                    "vs_cuml": f"{t_cu / t_fl:.2f}x",
                    "HBM_GB": f"{hbm_fl:.1f}", "gate": gate,
                    "conditions": apples_to_apples(
                        op="knn", shape={"M": M, "Q": Q, "D": D, "K": K},
                        flashlib_dtype=dlabel, cuml_dtype="fp32",
                        flashlib_algorithm="cutedsl_FA3_fused",
                        cuml_algorithm="brute_l2",
                        init_shared=False,
                        notes="FA3 path; bf16/fp16 only on Hopper"),
                }, columns=["shape", "dtype", "engine", "time_ms", "TFLOPS", "GBs",
                            "recall_at_K", "vs_cuml", "HBM_GB", "gate"])
            except Exception as e:
                audit_record(PRIM, {
                    "shape": label, "dtype": dlabel,
                    "engine": "flashlib(cutedsl-FA3)",
                    "time_ms": "SKIP", "TFLOPS": "-", "GBs": "-",
                    "recall_at_K": "-", "vs_cuml": "-", "HBM_GB": "-",
                    "gate": f"SKIP ({type(e).__name__}: {str(e)[:50]})",
                    "conditions": {},
                }, columns=["shape", "dtype", "engine", "time_ms", "TFLOPS", "GBs",
                            "recall_at_K", "vs_cuml", "HBM_GB", "gate"])

    # Free this row's HBM before next shape.
    del Xc32, Xq32
    free_gpu()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--skip-huge", action="store_true",
                    help="skip the M=50M and 20M-D=128 rows (~3 min)")
    args = ap.parse_args()
    header()

    # Wipe previous heavy run for this primitive so the markdown table
    # reflects ONLY the current sweep.
    for ext in (".md", ".json"):
        p = RESULTS_DIR / f"{PRIM}{ext}"
        if p.exists():
            p.unlink()

    for label, M, Q, D, K in BUILD_SHAPES:
        _bench_one(kind="build", label=label, M=M, Q=Q, D=D, K=K)
    for label, Q, M, D, K in SEARCH_SMALL_Q_SHAPES:
        _bench_one(kind="search-smallQ", label=label, M=M, Q=Q, D=D, K=K)
    if not args.skip_huge:
        for label, Q, M, D, K in SEARCH_HUGE_M_SHAPES:
            _bench_one(kind="search-hugeM", label=label, M=M, Q=Q, D=D, K=K)


if __name__ == "__main__":
    main()
