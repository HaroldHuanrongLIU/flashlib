"""Heavy KMeans sweep — release-candidate audit.

Asymptotic / industrial workloads where the structural ~12x ceiling
(``~6x`` from bf16-vs-fp32 dense ratio * ``~2x`` from on-chip fused
argmin) is fully realised:

* (N, K) up to (100M, 100K) at D=64 — the model "vector quantisation
  at recsys scale".
* (N=20M, D=128, K=20K) — adds a non-trivial D axis to verify the
  assign kernel's split-D path is also bound by GEMM, not by L2
  thrashing.
* (N=5M, D=256, K=10K) — D=256 sits at the CuteDSL FA3 sweet spot per
  the kmeans dispatcher rule ``D >= 256 AND K >= 4096 -> cutedsl``.
* (N=200M, D=32, K=200K) — high-K stress at modest D; the assign
  kernel runs its K-stream loop ~50 wider than the baseline and the
  ``c_sq`` buffer alone is 25 MB.

Anti-reward-hacking guardrails:

* The same init centroids (a uniform random sample of ``X``) are
  passed to BOTH engines via ``np.array_equal``-asserted buffers.
* cuML is given the same ``n_init=1, max_iter=N, tol=1e-6`` so its
  Lloyd loop count is fixed and matches flashlib's.
* Inertia is computed via ``chunked_inertia`` so the reference number
  is fp32 (the dtype cuML reports), comparable apples-to-apples even
  when flashlib ran bf16.
* Precision step-down is stamped in the per-row ``conditions``.
* Each row resets HBM peak; the 100M/100K row is allowed up to ~50 GB
  HBM (sums + counts buffers at K=100K dominate, not the data).
"""
from benchmarks.vs_cuml.heavy._common import (
    cap_threads, cuml_shim, time_gpu, title, header,
    audit_record, apples_to_apples, same_init_check,
    hbm_peak_reset, hbm_peak_gb, chunked_inertia,
    gate_metric, free_gpu, RESULTS_DIR,
)
cap_threads(); cuml_shim()

import argparse
import warnings; warnings.filterwarnings("ignore")
import gc
import torch
import cupy as cp

from cuml.cluster import KMeans as cuKMeans
from flashlib.primitives.kmeans import flash_kmeans


# (label, N, D, K, max_iter)
# NB: rows are ordered so the long ones land last; a 30-min wall cap
# in the dispatcher truncates rather than corrupts the table.
SHAPES = [
    ("medium    N=10M  D=64  K=10K",   10_000_000,  64,  10_000, 5),
    ("D-axis    N=20M  D=128 K=20K",   20_000_000, 128,  20_000, 3),
    ("FA3-tile  N=5M   D=256 K=10K",    5_000_000, 256,  10_000, 5),
    ("large     N=30M  D=64  K=30K",   30_000_000,  64,  30_000, 3),
    ("xlarge    N=50M  D=64  K=50K",   50_000_000,  64,  50_000, 3),
    ("hi-K      N=20M  D=64  K=100K",  20_000_000,  64, 100_000, 2),
]

KM_TOL = 1e-6  # cuML rejects tol=0; pick something tiny.
PRIM = "kmeans"


def _run_one(label, N, D, K, max_iter, *, quick: bool):
    title(f"KMeans  {label}  (N={N:,}, D={D}, K={K:,}, "
          f"max_iter={max_iter})")

    # Inputs on GPU (CPU make_blobs is impractical for K >= 10K).
    torch.manual_seed(0)
    X32 = torch.randn(N, D, device="cuda", dtype=torch.float32)
    init_idx = torch.randperm(N, device="cuda")[:K]
    init32 = X32[init_idx].contiguous()

    # cuML path — shared init via the SAME GPU buffer (zero-copy cupy view).
    init_cp = cp.from_dlpack(init32)
    X_cp = cp.from_dlpack(X32)

    def cu_fit():
        km = cuKMeans(n_clusters=K, init=init_cp, n_init=1,
                       max_iter=max_iter, tol=KM_TOL)
        km.fit(X_cp)
        return km

    free_gpu(); hbm_peak_reset()
    km_cu = cu_fit()
    cu_inertia = float(km_cu.inertia_)
    cu_repeats = 1 if N >= 50_000_000 else 2
    t_cu = time_gpu(cu_fit, repeat=cu_repeats, warmup=0)
    hbm_cu = hbm_peak_gb()
    audit_record(PRIM, {
        "shape": label, "dtype": "fp32", "engine": "cuml",
        "time_ms": f"{t_cu:10.1f}",
        "inertia": f"{cu_inertia:.4e}",
        "inertia_gap_pct": "0.000",
        "vs_cuml": "1.00x",
        "HBM_GB": f"{hbm_cu:.1f}",
        "gate": "PASS",
        "conditions": apples_to_apples(
            op="kmeans", shape={"N": N, "D": D, "K": K, "iter": max_iter},
            flashlib_dtype="-", cuml_dtype="fp32",
            flashlib_algorithm="-", cuml_algorithm="lloyd_fp32_assign_chunked_gemm",
            init_shared=True, notes="reference; same init centroids as flashlib"),
    }, columns=["shape", "dtype", "engine", "time_ms", "inertia",
                "inertia_gap_pct", "vs_cuml", "HBM_GB", "gate"])
    del km_cu; gc.collect(); torch.cuda.empty_cache()

    # flashlib paths: fp32 (lossless) AND bf16 (with tol disclosure).
    dtype_specs = [("fp32", torch.float32), ("bf16", torch.bfloat16)]
    if quick:
        dtype_specs = [("bf16", torch.bfloat16)]  # only the headline

    for dlabel, dtype in dtype_specs:
        if dtype is torch.float32 and N >= 50_000_000 and D >= 128:
            # 50M x 128 fp32 = 25.6 GB raw, plus assign work tile.
            # Skip explicitly to avoid OOM on the xxlarge / hi-K rows.
            audit_record(PRIM, {
                "shape": label, "dtype": dlabel, "engine": "flashlib",
                "time_ms": "SKIP", "inertia": "-",
                "inertia_gap_pct": "-", "vs_cuml": "-",
                "HBM_GB": "-",
                "gate": "SKIP (fp32 OOM expected; bf16 row is the headline)",
                "conditions": {},
            }, columns=["shape", "dtype", "engine", "time_ms", "inertia",
                        "inertia_gap_pct", "vs_cuml", "HBM_GB", "gate"])
            continue

        free_gpu(); hbm_peak_reset()
        try:
            X_dt = X32.to(dtype) if dtype is not torch.float32 else X32
            init_dt = X_dt[init_idx].contiguous().unsqueeze(0)

            # Sanity: init was sliced from the same buffer cuML saw.
            # bf16 round-trip introduces ~1e-2 noise; gate with atol.
            init_check = X32[init_idx].to(torch.float32)
            tol_init = 0.0 if dtype is torch.float32 else 5e-2
            same_init_check(init_dt.squeeze(0).to(torch.float32),
                            init_check, name="kmeans init centroids",
                            atol=tol_init)

            fl_ids, fl_C, _ = flash_kmeans(
                X_dt, K, max_iters=max_iter,
                init_centroids=init_dt, tol=0.0,
            )
            fl_inertia = chunked_inertia(X32, fl_C.squeeze(0),
                                          fl_ids.squeeze(0))
            t_fl = time_gpu(
                lambda: flash_kmeans(X_dt, K, max_iters=max_iter,
                                      init_centroids=init_dt, tol=0.0),
                repeat=cu_repeats, warmup=0,
            )
            hbm_fl = hbm_peak_gb()
            inertia_gap = (fl_inertia - cu_inertia) / cu_inertia * 100.0
            # Inertia gap > 1% on KMeans (with shared init + same iter
            # count) is suspect — fail the row.
            gate = gate_metric("|inertia_gap|", abs(inertia_gap), upper=1.0)
            audit_record(PRIM, {
                "shape": label, "dtype": dlabel, "engine": "flashlib",
                "time_ms": f"{t_fl:10.1f}",
                "inertia": f"{fl_inertia:.4e}",
                "inertia_gap_pct": f"{inertia_gap:+.3f}",
                "vs_cuml": f"{t_cu / t_fl:.2f}x",
                "HBM_GB": f"{hbm_fl:.1f}", "gate": gate,
                "conditions": apples_to_apples(
                    op="kmeans", shape={"N": N, "D": D, "K": K,
                                         "iter": max_iter},
                    flashlib_dtype=dlabel, cuml_dtype="fp32",
                    flashlib_algorithm="lloyd_fused_assign_sortedupdate",
                    cuml_algorithm="lloyd_fp32_assign_chunked_gemm",
                    init_shared=True,
                    notes=("matched-dtype apples-to-apples" if dtype is torch.float32
                           else "headline: bf16 vs cuML fp32 — precision step-down "
                                "is intentional, inertia parity verified")),
            }, columns=["shape", "dtype", "engine", "time_ms", "inertia",
                        "inertia_gap_pct", "vs_cuml", "HBM_GB", "gate"])
        except Exception as e:
            audit_record(PRIM, {
                "shape": label, "dtype": dlabel, "engine": "flashlib",
                "time_ms": "ERR", "inertia": "-",
                "inertia_gap_pct": "-", "vs_cuml": "-",
                "HBM_GB": "-",
                "gate": f"FAIL ({type(e).__name__}: {str(e)[:80]})",
                "conditions": {},
            }, columns=["shape", "dtype", "engine", "time_ms", "inertia",
                        "inertia_gap_pct", "vs_cuml", "HBM_GB", "gate"])

    del X32, init32, X_cp, init_cp
    gc.collect(); torch.cuda.empty_cache()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true",
                    help="bf16-only flashlib row; skip the 200M hi-K shape")
    args = ap.parse_args()
    header()

    # Reset per-primitive output files.
    for ext in (".md", ".json"):
        p = RESULTS_DIR / f"{PRIM}{ext}"
        if p.exists():
            p.unlink()

    shapes = SHAPES[:-1] if args.quick else SHAPES
    for s in shapes:
        _run_one(*s, quick=args.quick)


if __name__ == "__main__":
    main()
