"""cuML RandomForestClassifier CUDA-kernel trace via CUPTI.

FIT + PREDICT passes. Matches flashlib breakdown's random_forest.py
shapes ((trees, depth) ∈ (50,8), (100,12), (200,14)) at N=100K, D=64, C=6.

cuML RandomForestClassifier:
  - Python: cuml/ensemble/randomforestclassifier.pyx
  - C++:    cpp/src/decisiontree/* (level-wise histogram builder
              identical in spirit to flashlib; computes per-feature,
              per-bin class-count histograms then a Gini-best-split
              reduction).
"""
from __future__ import annotations

import numpy as np
import torch
from cuml.ensemble import RandomForestClassifier as cuRF

from ._common import free_gpu, profile_cuml_call, write_cuml_breakdown_md

N_TOTAL, D, C = 100_000, 64, 6
N_BINS = 16
SEED = 0
N_TEST = max(8192, N_TOTAL // 20)
N_TRAIN = N_TOTAL - N_TEST

SHAPES = [
    ("small  trees=50  depth=8",   {"trees":  50, "depth":  8}),
    ("medium trees=100 depth=12",  {"trees": 100, "depth": 12}),
    ("large  trees=200 depth=14",  {"trees": 200, "depth": 14}),
]


def _gpu_classification(N, D, C):
    torch.manual_seed(0)
    y_t = torch.randint(0, C, (N,), device="cuda", dtype=torch.int64)
    centers = torch.randn(C, D, device="cuda") * 1.5
    X_t = torch.randn(N, D, device="cuda") + centers[y_t]
    return X_t.cpu().numpy().astype(np.float32), y_t.cpu().numpy().astype(np.int32)


def _build_data():
    X_np, y_np = _gpu_classification(N_TOTAL, D, C)
    Xtr_np, Xte_np = X_np[:-N_TEST], X_np[-N_TEST:]
    ytr_np = y_np[:-N_TEST]
    Xtr = torch.tensor(Xtr_np, device="cuda")
    ytr = torch.tensor(ytr_np, device="cuda", dtype=torch.int32)
    Xte = torch.tensor(Xte_np, device="cuda")
    return Xtr, ytr, Xte


def main_fit() -> None:
    print(f"[cuml-profile:random_forest.FIT] sweeping (trees, depth) at "
          f"N={N_TRAIN:,}, D={D}, C={C}")
    Xtr, ytr, _ = _build_data()
    import cupy as cp
    Xtr_cp = cp.from_dlpack(Xtr)
    ytr_cp = cp.from_dlpack(ytr)

    shape_results = []
    for label, kw in SHAPES:
        n_est, depth = kw["trees"], kw["depth"]

        def _call():
            cuRF(n_estimators=n_est, max_depth=depth,
                  n_bins=N_BINS, max_features=1.0,  # full-D, matches flashlib
                  random_state=SEED,
                  output_type="cupy").fit(Xtr_cp, ytr_cp)

        try:
            sr = profile_cuml_call(label, _call, warmup=1, repeat=1)
            shape_results.append(sr)
        except Exception as e:
            print(f"[cuml-profile:random_forest.FIT] SHAPE {label} FAILED: {e}")
            shape_results.append({
                "label": label, "outer_wall_ms": float("nan"),
                "n_kernels": 0,
                "kernels": [{"kernel": f"FAILED: {type(e).__name__}",
                              "raw_example": "",
                              "launches_per_call": 0,
                              "total_ms_per_call": 0.0,
                              "mean_us_per_launch": 0.0,
                              "pct_of_total": 0.0}],
            })
        free_gpu()
    write_cuml_breakdown_md(
        prim="random_forest_fit",
        shape_results=shape_results,
        notes=("cuML RandomForestClassifier fit: level-wise BFS tree "
               "builder; per (level, n_active_nodes) it launches a "
               "histogram-counts kernel (the dominant cost), a "
               "best-split kernel, and a partition kernel. There is NO "
               "histogram-subtract trick (`hist_subtract`) — cuML "
               "re-computes the histogram for every active node, which "
               "is one of the headline mechanical differences vs "
               "flashlib's sibling-derive optimisation."),
    )


def main_predict() -> None:
    print(f"[cuml-profile:random_forest.PREDICT] sweeping (trees, depth)")
    Xtr, ytr, Xte = _build_data()
    import cupy as cp
    Xtr_cp = cp.from_dlpack(Xtr)
    ytr_cp = cp.from_dlpack(ytr)
    Xte_cp = cp.from_dlpack(Xte)

    shape_results = []
    for label, kw in SHAPES:
        n_est, depth = kw["trees"], kw["depth"]

        try:
            clf = cuRF(n_estimators=n_est, max_depth=depth,
                        n_bins=N_BINS, max_features=1.0,  # full-D, matches flashlib
                        random_state=SEED,
                        output_type="cupy").fit(Xtr_cp, ytr_cp)
        except Exception as e:
            print(f"[cuml-profile:random_forest.PREDICT] FIT FAIL on {label}: {e}")
            shape_results.append({
                "label": label, "outer_wall_ms": float("nan"),
                "n_kernels": 0,
                "kernels": [{"kernel": f"FAILED: {type(e).__name__}",
                              "raw_example": "",
                              "launches_per_call": 0,
                              "total_ms_per_call": 0.0,
                              "mean_us_per_launch": 0.0,
                              "pct_of_total": 0.0}],
            })
            continue

        def _call():
            clf.predict(Xte_cp)

        sr = profile_cuml_call(label, _call, warmup=1, repeat=2)
        shape_results.append(sr)
        del clf
        free_gpu()

    write_cuml_breakdown_md(
        prim="random_forest_predict",
        shape_results=shape_results,
        notes=("cuML RandomForestClassifier predict: each tree is "
               "evaluated on every row independently via a fused tree-"
               "traversal kernel (no torch-level gather loop), then "
               "majority vote is reduced. The traversal kernel is the "
               "dominant cost; the per-tree launch loop scales with "
               "n_estimators."),
    )


if __name__ == "__main__":
    main_fit()
    main_predict()
