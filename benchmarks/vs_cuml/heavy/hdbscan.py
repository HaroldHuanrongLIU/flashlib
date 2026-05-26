"""Heavy HDBSCAN sweep — release-candidate audit.

HDBSCAN dense-MRD path is O(N^2 D) in MRD matrix construction + sparse
Boruvka MST, so the practical heavy ceiling is N=100K at D=64
(the MRD matrix alone is N^2 fp32 = 40 GB at N=100K).

Anti-reward-hacking guardrails:

* Same blob data + same ``min_cluster_size`` / ``min_samples`` across
  all three engines.
* Reference ARI computed vs sklearn (CPU) when feasible (N <= 30K);
  larger N falls back to cuML self-baseline with the row labelled.
* No precision opt-in is exercised: ``flash_hdbscan`` keeps fp32 on
  the MRD path; we report only the fp32-vs-fp32 row.
"""
from benchmarks.vs_cuml.heavy._common import (
    cap_threads, cuml_shim, time_gpu, time_cpu, title, header,
    ari, cluster_count, audit_record, apples_to_apples,
    hbm_peak_reset, hbm_peak_gb, gate_metric, free_gpu, RESULTS_DIR,
)
cap_threads(); cuml_shim()

import warnings; warnings.filterwarnings("ignore")
import numpy as np
import torch

from sklearn.datasets import make_blobs
from sklearn.cluster import HDBSCAN as skHDBSCAN
from cuml.cluster import HDBSCAN as cuHDBSCAN
from flashlib.primitives.hdbscan import flash_hdbscan


# (label, N, D, mcs, ms, n_centers, use_sklearn_cpu)
SHAPES = [
    ("dense-MRD  N=20K  D=16  mcs=20",  20_000,  16,  20,  5,  6, True),
    ("dense-MRD  N=30K  D=32  mcs=30",  30_000,  32,  30,  5,  8, True),
    ("dense-MRD  N=50K  D=32  mcs=50",  50_000,  32,  50,  5,  8, False),
    ("dense-MRD  N=100K D=16  mcs=80", 100_000,  16,  80,  5, 10, False),
    ("dense-MRD  N=100K D=64  mcs=100", 100_000,  64, 100,  5, 12, False),
]

PRIM = "hdbscan"


def _run_one(label, N, D, mcs, ms, n_centers, use_sklearn_cpu):
    title(f"HDBSCAN  {label}  (N={N:,}, D={D}, min_cluster_size={mcs}, "
          f"min_samples={ms})")

    X_np, _ = make_blobs(n_samples=N, centers=n_centers, n_features=D,
                          cluster_std=1.0, random_state=0)
    X_np = X_np.astype(np.float32)

    if use_sklearn_cpu:
        sk_lbl = skHDBSCAN(min_cluster_size=mcs, min_samples=ms) \
                    .fit_predict(X_np)
        t_sk = time_cpu(
            lambda: skHDBSCAN(min_cluster_size=mcs, min_samples=ms)
                       .fit_predict(X_np),
            repeat=1,
        )
        audit_record(PRIM, {
            "shape": label, "engine": "sklearn(CPU)",
            "time_ms": f"{t_sk:10.1f}", "ARI": "1.0000",
            "num_clusters": str(cluster_count(sk_lbl)),
            "vs_cuml": "n/a", "HBM_GB": "0.0", "gate": "PASS",
            "conditions": apples_to_apples(
                op="hdbscan", shape={"N": N, "D": D, "mcs": mcs, "ms": ms},
                flashlib_dtype="-", cuml_dtype="-",
                flashlib_algorithm="-", cuml_algorithm="-",
                init_shared=False, notes="ground truth"),
        }, columns=["shape", "engine", "time_ms", "ARI", "num_clusters",
                    "vs_cuml", "HBM_GB", "gate"])
        ref_lbl = sk_lbl
    else:
        ref_lbl = None

    # cuML
    free_gpu(); hbm_peak_reset()
    cu_lbl = np.asarray(cuHDBSCAN(min_cluster_size=mcs, min_samples=ms)
                        .fit_predict(X_np))
    t_cu = time_gpu(
        lambda: cuHDBSCAN(min_cluster_size=mcs, min_samples=ms).fit_predict(X_np),
        repeat=2, warmup=1,
    )
    hbm_cu = hbm_peak_gb()
    if ref_lbl is None:
        ref_lbl = cu_lbl
    ari_cu = ari(ref_lbl, cu_lbl)
    audit_record(PRIM, {
        "shape": label, "engine": "cuml",
        "time_ms": f"{t_cu:10.1f}", "ARI": f"{ari_cu:.4f}",
        "num_clusters": str(cluster_count(cu_lbl)),
        "vs_cuml": "1.00x", "HBM_GB": f"{hbm_cu:.1f}",
        "gate": gate_metric("ARI", ari_cu, lower=0.80),
        "conditions": apples_to_apples(
            op="hdbscan", shape={"N": N, "D": D, "mcs": mcs, "ms": ms},
            flashlib_dtype="-", cuml_dtype="fp32",
            flashlib_algorithm="-", cuml_algorithm="raft_hdbscan",
            init_shared=False,
            notes="ARI vs sklearn" if use_sklearn_cpu else "ARI vs cuML self-baseline"),
    }, columns=["shape", "engine", "time_ms", "ARI", "num_clusters",
                "vs_cuml", "HBM_GB", "gate"])

    # flashlib
    free_gpu(); hbm_peak_reset()
    try:
        X32 = torch.tensor(X_np, device="cuda")
        fl_lbl = flash_hdbscan(X32, min_cluster_size=mcs, min_samples=ms)
        fl_lbl = (fl_lbl if isinstance(fl_lbl, np.ndarray)
                  else np.asarray(fl_lbl))
        t_fl = time_gpu(
            lambda: flash_hdbscan(X32, min_cluster_size=mcs, min_samples=ms),
            repeat=2, warmup=1,
        )
        hbm_fl = hbm_peak_gb()
        ari_fl = ari(ref_lbl, fl_lbl)
        audit_record(PRIM, {
            "shape": label, "engine": "flashlib",
            "time_ms": f"{t_fl:10.1f}", "ARI": f"{ari_fl:.4f}",
            "num_clusters": str(cluster_count(fl_lbl)),
            "vs_cuml": f"{t_cu / t_fl:.2f}x",
            "HBM_GB": f"{hbm_fl:.1f}",
            "gate": gate_metric("ARI", ari_fl, lower=0.80),
            "conditions": apples_to_apples(
                op="hdbscan", shape={"N": N, "D": D, "mcs": mcs, "ms": ms},
                flashlib_dtype="fp32", cuml_dtype="fp32",
                flashlib_algorithm="dense_mrd_+_sparse_boruvka_+_sl_+_stability",
                cuml_algorithm="raft_hdbscan",
                init_shared=False,
                notes="matched fp32 end-to-end; algorithm structure differs"),
        }, columns=["shape", "engine", "time_ms", "ARI", "num_clusters",
                    "vs_cuml", "HBM_GB", "gate"])
    except Exception as e:
        audit_record(PRIM, {
            "shape": label, "engine": "flashlib",
            "time_ms": "ERR", "ARI": "-", "num_clusters": "-",
            "vs_cuml": "-", "HBM_GB": "-",
            "gate": f"FAIL ({type(e).__name__}: {str(e)[:80]})",
            "conditions": {},
        }, columns=["shape", "engine", "time_ms", "ARI", "num_clusters",
                    "vs_cuml", "HBM_GB", "gate"])

    free_gpu()


def main():
    header()
    for ext in (".md", ".json"):
        p = RESULTS_DIR / f"{PRIM}{ext}"
        if p.exists():
            p.unlink()
    for s in SHAPES:
        _run_one(*s)


if __name__ == "__main__":
    main()
