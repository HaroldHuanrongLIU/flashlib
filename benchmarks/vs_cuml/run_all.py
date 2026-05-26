"""Run all per-primitive vs-cuML benchmarks in fresh subprocesses.

Subprocesses isolate Triton autotune state and CUDA contexts so one
primitive's caches don't bleed into another's timings.
"""
import subprocess, sys


PRIMS = [
    # Distance / clustering
    "knn", "kmeans", "dbscan", "hdbscan", "spectral_clustering",
    # Linear / decomposition
    "pca", "truncated_svd",
    "linear_regression", "ridge", "logistic_regression",
    "multinomial_nb",
    "standard_scaler",
    # Ensembles
    "random_forest",
    # Manifold
    "tsne", "umap",
]


def main():
    failed = []
    for p in PRIMS:
        print(f"\n{'#' * 78}\n# benchmarks.vs_cuml.{p}\n{'#' * 78}")
        rc = subprocess.call(
            [sys.executable, "-m", f"benchmarks.vs_cuml.{p}"]
        )
        if rc != 0:
            failed.append(p)
    if failed:
        sys.exit(f"FAILED: {failed}")


if __name__ == "__main__":
    main()
