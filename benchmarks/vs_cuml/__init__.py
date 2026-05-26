"""flashlib (exact, ``tol=None``) vs cuML (default, fp32 IEEE) benchmarks.

Each module under this package benches one primitive against cuml on
representative shapes. ARI / recall ground truth comes from scikit-learn.

Run individually::

    python -m benchmarks.vs_cuml.knn
    python -m benchmarks.vs_cuml.kmeans
    python -m benchmarks.vs_cuml.dbscan
    python -m benchmarks.vs_cuml.hdbscan

Or sweep all four::

    python -m benchmarks.vs_cuml.run_all
"""
