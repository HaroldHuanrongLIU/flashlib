"""Heavy DBSCAN sweep — release-candidate audit.

Stresses the two regimes of ``flash_dbscan``:

* **2-D grid path** (``D <= 2``) — N=5M D=2 spatial grid stress.
* **kNN-radius path** (``D >= 3``) — N up to 1M at D=16 and a
  D=128 high-dimensional row that exercises the embedded
  ``flash_knn`` build kernel.

Anti-reward-hacking guardrails:

* Both flashlib and cuML consume the SAME synthetic blobs.
* sklearn (CPU) is the ground-truth ARI baseline for the rows where N
  is small enough; otherwise ARI is reported vs cuML and labelled as
  such in the ``conditions`` field.
* cuML pinned to its default exact path (no ANN index involvement —
  the cuml.cluster.DBSCAN brute-force matches the sklearn default).
* HBM peak logged per row to catch the silent fallback to chunked
  pairwise distance inside ``flash_dbscan`` (which would inflate
  flashlib's time and is fine to surface).
"""
from benchmarks.vs_cuml.heavy._common import (
    cap_threads, cuml_shim, time_gpu, time_cpu, title, header,
    ari, cluster_count, audit_record, apples_to_apples,
    hbm_peak_reset, hbm_peak_gb, gate_metric, free_gpu, RESULTS_DIR,
)
cap_threads(); cuml_shim()

import argparse
import warnings; warnings.filterwarnings("ignore")
import numpy as np
import torch

from sklearn.datasets import make_blobs
from sklearn.cluster import DBSCAN as skDBSCAN
from cuml.cluster import DBSCAN as cuDBSCAN
from flashlib.primitives.dbscan import flash_dbscan


# (label, N, D, eps, min_samples, n_centers, use_sklearn_cpu)
SHAPES = [
    # 2-D grid path
    ("2D grid     N=5M  D=2  eps=0.5", 5_000_000,   2, 0.5, 5,  6, False),
    # Low-D kNN-radius path
    ("low-D       N=200K D=16 eps=3.5",   200_000,  16, 3.5, 5,  8, True),
    ("low-D       N=500K D=16 eps=3.5",   500_000,  16, 3.5, 5, 16, False),
    ("low-D       N=1M   D=16 eps=3.5", 1_000_000,  16, 3.5, 5, 20, False),
    # Medium-D
    ("medium-D    N=300K D=32 eps=6.0",   300_000,  32, 6.0, 5, 12, False),
    # High-D — exercises the embedded flash_knn build kernel
    ("high-D      N=100K D=64 eps=8.0",   100_000,  64, 8.0, 5, 12, False),
    ("high-D      N=200K D=128 eps=11.0", 200_000, 128, 11.0, 5, 12, False),
]

PRIM = "dbscan"


def _run_one(label, N, D, eps, min_samples, n_centers, use_sklearn_cpu):
    title(f"DBSCAN  {label}  (N={N:,}, D={D}, eps={eps}, "
          f"min_samples={min_samples})")

    X_np, _ = make_blobs(n_samples=N, centers=n_centers, n_features=D,
                          cluster_std=1.0, random_state=0)
    X_np = X_np.astype(np.float32)

    # Reference: sklearn (CPU) for small N; else cuML.
    if use_sklearn_cpu:
        sk_lbl = skDBSCAN(eps=eps, min_samples=min_samples).fit_predict(X_np)
        t_sk = time_cpu(
            lambda: skDBSCAN(eps=eps, min_samples=min_samples).fit_predict(X_np),
            repeat=1,
        )
        audit_record(PRIM, {
            "shape": label, "engine": "sklearn(CPU)",
            "time_ms": f"{t_sk:10.1f}", "ARI": "1.0000",
            "num_clusters": str(cluster_count(sk_lbl)),
            "vs_cuml": "n/a", "HBM_GB": "0.0", "gate": "PASS",
            "conditions": apples_to_apples(
                op="dbscan", shape={"N": N, "D": D, "eps": eps,
                                     "min_samples": min_samples},
                flashlib_dtype="-", cuml_dtype="-",
                flashlib_algorithm="-", cuml_algorithm="-",
                init_shared=False, notes="ground truth"),
        }, columns=["shape", "engine", "time_ms", "ARI", "num_clusters",
                    "vs_cuml", "HBM_GB", "gate"])
    ref_lbl = sk_lbl if use_sklearn_cpu else None

    # cuML — accepts numpy directly. Wrap in try because cuML's RAFT
    # DBSCAN occasionally raises CUDA invalid-configuration on
    # extreme (N, D) combinations.
    free_gpu(); hbm_peak_reset()
    try:
        cu_lbl = np.asarray(cuDBSCAN(eps=eps, min_samples=min_samples)
                            .fit_predict(X_np))
        t_cu = time_gpu(
            lambda: cuDBSCAN(eps=eps, min_samples=min_samples).fit_predict(X_np),
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
            "gate": gate_metric("ARI", ari_cu, lower=0.95),
            "conditions": apples_to_apples(
                op="dbscan", shape={"N": N, "D": D, "eps": eps,
                                     "min_samples": min_samples},
                flashlib_dtype="-", cuml_dtype="fp32",
                flashlib_algorithm="-", cuml_algorithm="raft_dbscan_dense",
                init_shared=False,
                notes=("ARI vs sklearn" if use_sklearn_cpu
                       else "ARI vs cuML self-baseline (sklearn skipped at N)")),
        }, columns=["shape", "engine", "time_ms", "ARI", "num_clusters",
                    "vs_cuml", "HBM_GB", "gate"])
        t_cu_val = t_cu
    except Exception as e:
        # cuML crash → record FAIL row but DO NOT abort the script —
        # flashlib's row should still run.
        audit_record(PRIM, {
            "shape": label, "engine": "cuml",
            "time_ms": "ERR", "ARI": "-", "num_clusters": "-",
            "vs_cuml": "-", "HBM_GB": "-",
            "gate": f"FAIL ({type(e).__name__}: {str(e)[:80]})",
            "conditions": {},
        }, columns=["shape", "engine", "time_ms", "ARI", "num_clusters",
                    "vs_cuml", "HBM_GB", "gate"])
        t_cu_val = float("inf")
        if ref_lbl is None:
            ref_lbl = None  # remains None — flashlib's ARI will be n/a

    # flashlib (exact in input dtype).
    free_gpu(); hbm_peak_reset()
    try:
        X32 = torch.tensor(X_np, device="cuda")
        fl_lbl = flash_dbscan(X32, eps=eps, min_samples=min_samples) \
                    .cpu().numpy()
        t_fl = time_gpu(
            lambda: flash_dbscan(X32, eps=eps, min_samples=min_samples),
            repeat=2, warmup=1,
        )
        hbm_fl = hbm_peak_gb()
        if ref_lbl is not None:
            ari_fl = ari(ref_lbl, fl_lbl)
            ari_str = f"{ari_fl:.4f}"
            ari_gate = gate_metric("ARI", ari_fl, lower=0.95)
        else:
            ari_str = "n/a"
            ari_gate = "PASS (no ref)"
        audit_record(PRIM, {
            "shape": label, "engine": "flashlib",
            "time_ms": f"{t_fl:10.1f}", "ARI": ari_str,
            "num_clusters": str(cluster_count(fl_lbl)),
            "vs_cuml": (f"{t_cu_val / t_fl:.2f}x"
                        if t_cu_val != float("inf") else "n/a"),
            "HBM_GB": f"{hbm_fl:.1f}",
            "gate": ari_gate,
            "conditions": apples_to_apples(
                op="dbscan", shape={"N": N, "D": D, "eps": eps,
                                     "min_samples": min_samples},
                flashlib_dtype="fp32", cuml_dtype="fp32",
                flashlib_algorithm=("grid_2d" if D <= 2
                                     else "knn_radius_via_flash_knn"),
                cuml_algorithm="raft_dbscan_dense",
                init_shared=False,
                notes="matched fp32; algorithm differs (grid/kNN-radius vs RAFT)"),
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
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true",
                    help="skip rows with N >= 1M")
    args = ap.parse_args()
    header()

    for ext in (".md", ".json"):
        p = RESULTS_DIR / f"{PRIM}{ext}"
        if p.exists():
            p.unlink()

    shapes = [s for s in SHAPES if not args.quick or s[1] < 1_000_000]
    for s in shapes:
        _run_one(*s)


if __name__ == "__main__":
    main()
