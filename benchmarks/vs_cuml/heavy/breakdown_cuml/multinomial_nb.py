"""cuML MultinomialNB CUDA-kernel trace via CUPTI.

Two passes (FIT and PREDICT), matching flashlib breakdown's
multinomial_nb.py shapes ((N, V, C) = (500K, 1K, 10) / (1M, 2K, 20) /
(2M, 2K, 50)).

cuML MultinomialNB:
  - Python: cuml/naive_bayes/naive_bayes.pyx
  - C++:    cpp/src/naive_bayes/* + RAFT smem_pack + cuBLAS GEMM
            internally builds dense (C, V) feature-count via a sparse
            scatter-add over the one-hot.
"""
from __future__ import annotations

import torch
from cuml.naive_bayes import MultinomialNB as cuMNB

from ._common import free_gpu, profile_cuml_call, write_cuml_breakdown_md

ALPHA = 1.0

SHAPES = [
    ("medium  N=500K V=1K C=10", {"N_total":   500_000, "V": 1_000, "C": 10}),
    ("large   N=1M   V=2K C=20", {"N_total": 1_000_000, "V": 2_000, "C": 20}),
    # cuML 25.10 MultinomialNB has a CUDA_ERROR_ILLEGAL_ADDRESS bug at
    # large (N, V, C); the same bug surfaced in the heavy benchmark at
    # (N=2M, C=50) and at (N=1.5M, V=2K, C=20). For the xlarge slot we
    # sweep C upward (matching the flashlib breakdown's C=50 step) but
    # cap N at 800K to stay inside cuML's working envelope.
    ("xlarge  N=800K V=2K C=50", {"N_total":   800_000, "V": 2_000, "C": 50}),
]


def _gen_counts(N, V, C):
    torch.manual_seed(0)
    y_t = torch.randint(0, C, (N,), device="cuda", dtype=torch.int64)
    base = torch.rand(C, V, device="cuda") * 8.0
    lam = base[y_t]
    X_t = torch.poisson(lam).to(torch.float32)
    return X_t, y_t


def main_fit() -> None:
    print("[cuml-profile:multinomial_nb.FIT] sweeping (N, V, C)")
    shape_results = []
    for label, kw in SHAPES:
        N, V, C = kw["N_total"], kw["V"], kw["C"]
        n_test = max(8192, N // 20)
        X_t, y_t = _gen_counts(N, V, C)
        X_train, y_train = X_t[:-n_test].contiguous(), y_t[:-n_test].contiguous()
        import cupy as cp
        X_cp = cp.from_dlpack(X_train)
        y_cp = cp.from_dlpack(y_train)

        def _call():
            cuMNB(alpha=ALPHA, output_type="cupy").fit(X_cp, y_cp)

        try:
            sr = profile_cuml_call(label, _call, warmup=1, repeat=3)
            shape_results.append(sr)
        except Exception as e:
            print(f"[cuml-profile:multinomial_nb.FIT] SHAPE {label} FAILED: {e}")
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
            # Write partial results immediately so a subsequent CUDA-context
            # corruption doesn't lose the medium/large data we already have.
            write_cuml_breakdown_md(
                prim="multinomial_nb_fit",
                shape_results=shape_results,
                notes=("cuML MultinomialNB fit. The xlarge shape may FAIL with "
                       "CUDA_ERROR_ILLEGAL_ADDRESS — this is a known cuML 25.10 "
                       "bug at large (N, V, C); the medium and large rows still "
                       "give meaningful kernel-mix data."),
            )
            print("[cuml-profile:multinomial_nb.FIT] Wrote partial results; "
                  "aborting (CUDA context likely corrupted).")
            return
        try:
            del X_t, y_t, X_train, y_train, X_cp, y_cp
        except Exception:
            pass
        free_gpu()
    write_cuml_breakdown_md(
        prim="multinomial_nb_fit",
        shape_results=shape_results,
        notes=("cuML MultinomialNB fit: builds (C, V) feature_count by "
               "scatter-add of X rows grouped by y class, then alpha "
               "smoothing + log. The hot kernel is the sparse "
               "scatter-add (atomic per row), not a fused GEMM."),
    )


def main_predict() -> None:
    print("[cuml-profile:multinomial_nb.PREDICT] sweeping (N, V, C)")
    shape_results = []
    for label, kw in SHAPES:
        N, V, C = kw["N_total"], kw["V"], kw["C"]
        n_test = max(8192, N // 20)
        X_t, y_t = _gen_counts(N, V, C)
        X_train, y_train = X_t[:-n_test].contiguous(), y_t[:-n_test].contiguous()
        X_test = X_t[-n_test:].contiguous()
        import cupy as cp
        X_train_cp = cp.from_dlpack(X_train)
        y_train_cp = cp.from_dlpack(y_train)
        X_test_cp = cp.from_dlpack(X_test)

        try:
            clf = cuMNB(alpha=ALPHA, output_type="cupy").fit(X_train_cp, y_train_cp)

            def _call():
                clf.predict(X_test_cp)

            sr = profile_cuml_call(label, _call, warmup=1, repeat=3)
            shape_results.append(sr)
            del clf
        except Exception as e:
            print(f"[cuml-profile:multinomial_nb.PREDICT] SHAPE {label} FAILED: {e}")
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
            write_cuml_breakdown_md(
                prim="multinomial_nb_predict",
                shape_results=shape_results,
                notes=("cuML MultinomialNB predict. The xlarge shape may FAIL "
                       "with CUDA_ERROR_ILLEGAL_ADDRESS — known cuML 25.10 bug. "
                       "Medium and large rows still give meaningful kernel-mix data."),
            )
            print("[cuml-profile:multinomial_nb.PREDICT] Wrote partial results; "
                  "aborting (CUDA context likely corrupted).")
            return
        try:
            del X_t, y_t, X_train, y_train, X_test
            del X_train_cp, y_train_cp, X_test_cp
        except Exception:
            pass
        free_gpu()
    write_cuml_breakdown_md(
        prim="multinomial_nb_predict",
        shape_results=shape_results,
        notes=("cuML MultinomialNB predict: jll = X_test @ feature_log_prob.T "
               "(cuBLAS GEMM) + class_log_prior add + argmax. "
               "The GEMM dominates at every shape."),
    )


if __name__ == "__main__":
    main_fit()
    main_predict()
