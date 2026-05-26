# `benchmarks/vs_cuml`

Side-by-side timing & quality comparison: **flashlib (`tol=None`, IEEE
fp32 exact)** vs **cuML (default exact path)** vs scikit-learn (CPU
ground truth) for every primitive in `flashlib` that has a cuML peer
(plus `spectral_clustering`, where cuML has no public API and sklearn
is the only baseline).

## Install cuML (out-of-band; intentionally NOT in `pyproject.toml`)

cuML is a heavyweight RAPIDS dependency that pulls in its own CUDA
toolchain. We keep it out of the project requirements and install it
separately:

```bash
pip install --extra-index-url https://pypi.nvidia.com 'cuml-cu12==25.10.*'
```

Tested against `cuml-cu12==25.10.0`. The `_common.cuml_shim()` patches a
missing `BaseEstimator._get_default_requests` alias that scikit-learn 1.8
removed (cuml.accel still expects it).

## Run

```bash
# every primitive, each in its own fresh subprocess (isolates Triton
# autotune caches and CUDA contexts across primitives)
python -m benchmarks.vs_cuml.run_all

# or one at a time
python -m benchmarks.vs_cuml.knn
python -m benchmarks.vs_cuml.kmeans
python -m benchmarks.vs_cuml.dbscan
python -m benchmarks.vs_cuml.hdbscan
python -m benchmarks.vs_cuml.pca
python -m benchmarks.vs_cuml.truncated_svd
python -m benchmarks.vs_cuml.linear_regression
python -m benchmarks.vs_cuml.ridge
python -m benchmarks.vs_cuml.logistic_regression
python -m benchmarks.vs_cuml.multinomial_nb
python -m benchmarks.vs_cuml.standard_scaler
python -m benchmarks.vs_cuml.random_forest
python -m benchmarks.vs_cuml.spectral_clustering
python -m benchmarks.vs_cuml.tsne
python -m benchmarks.vs_cuml.umap
```

## What's measured

| Primitive            | Quality metric                  | Notes |
|----------------------|---------------------------------|-------|
| KNN                  | recall@K                        | brute-force; both engines IEEE fp32 (rel-err ~2.3e-7). |
| KMeans               | ARI + inertia                   | identical initial centers across engines. |
| DBSCAN               | ARI                             | low-D / medium-D / high-D / large-N sweep. |
| HDBSCAN              | ARI                             | dense-MRD path. |
| SpectralClustering   | ARI vs sklearn                  | cuML has no public peer; sklearn is CPU baseline. |
| PCA                  | rel-err on top-K eigenvalues    | also reports the Halko (``tol=1e-3``) path. |
| TruncatedSVD         | rel-err on top-K singular vals  | also reports Halko. |
| LinearRegression     | R² on holdout                   | normal equations + Cholesky + cuBLAS TF32. |
| Ridge                | R² on holdout                   | same, with L2 diagonal. |
| LogisticRegression   | accuracy on holdout             | L-BFGS + fused sigmoid/grad Triton kernel. |
| MultinomialNB        | accuracy on holdout             | fit kernel + `linalg.gemm` predict. |
| StandardScaler       | max-abs err vs sklearn          | single-pass shifted-sum Triton kernel. |
| RandomForest         | accuracy on holdout             | quantile-binned, batched level-wise BFS trees. |
| TSNE                 | trustworthiness (k=12)          | t-SNE is SGD; labels are not a valid baseline. cuML forced to `method='exact'` for apples-to-apples timing vs flashlib's exact O(N²); the default `'fft'` is also reported as reference. |
| UMAP                 | trustworthiness (k=12)          | also reports the bf16-KNN (``tol=1e-3``) path. |

For comparable timing each script pre-stages input on the GPU when cuML
implicitly does a host-to-device copy (`StandardScaler`, `MultinomialNB`).

Each script reports `vs cuml` -- the speedup factor of flashlib
relative to cuML (`>1.0` means flashlib is faster). The default
flashlib config is the *exact-in-input-dtype* path (`tol=None`);
some scripts also report a `tol=1e-3` row that opts into a
low-precision storage cast (bf16 GEMM, Halko eigh, bf16 KNN).
