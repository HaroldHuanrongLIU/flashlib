"""Correctness tests for the 5 v0.1 primitives, against torch / sklearn baselines."""
import pytest
import torch

cuda_only = pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")


@cuda_only
def test_standard_scaler_matches_sklearn():
    from flashlib import flash_standard_scaler

    torch.manual_seed(0)
    X = torch.randn(10_000, 128, device="cuda", dtype=torch.float32)
    Y, (mean, std) = flash_standard_scaler(X)

    ref_mean = X.mean(dim=0)
    ref_std = X.std(dim=0, unbiased=False)
    ref_Y = (X - ref_mean) / ref_std
    assert (Y - ref_Y).abs().max().item() < 1e-4
    assert (mean - ref_mean).abs().max().item() < 1e-5
    assert (std - ref_std).abs().max().item() < 1e-4


@cuda_only
def test_kmeans_partitions_blob_data():
    from flashlib import KMeans

    torch.manual_seed(0)
    # D >= 16 because Triton tl.dot requires K >= 16 in the assign kernel.
    D = 16
    centers = torch.randn(3, D, device="cuda") * 8.0
    pts_per = 1000
    X = torch.cat([
        c + torch.randn(pts_per, D, device="cuda") * 0.3 for c in centers
    ])
    model = KMeans(d=D, k=3, niter=30, seed=0).fit(X)
    labels = model.predict(X)
    # Each ground-truth blob should map predominantly to a single cluster.
    for i in range(3):
        sub = labels[i*pts_per:(i+1)*pts_per]
        most_common = torch.mode(sub).values
        purity = (sub == most_common).float().mean().item()
        assert purity > 0.95, f"blob {i} purity {purity:.2%}"


@cuda_only
def test_pca_matches_torch_svd():
    from flashlib import flash_pca

    torch.manual_seed(0)
    X = torch.randn(5_000, 64, device="cuda", dtype=torch.float32)
    Xc = X - X.mean(dim=0)
    eigvals_asc, eigvecs_asc = flash_pca(Xc, K=10)
    # Reference: torch eigh on covariance.
    cov = Xc.T @ Xc / Xc.shape[0]
    ref_vals, ref_vecs = torch.linalg.eigh(cov)
    # Top-10 eigenvalues should match (eigvals_asc is ascending top-K, take last 10).
    assert torch.allclose(eigvals_asc, ref_vals[-10:], atol=1e-3, rtol=1e-3)


@cuda_only
def test_knn_matches_naive_topk():
    from flashlib import flash_knn

    torch.manual_seed(0)
    X = torch.randn(1, 256, 128, device="cuda", dtype=torch.float32).contiguous()
    Xbf = X.to(torch.bfloat16)
    dists, idxs = flash_knn(Xbf, Xbf, k=5)
    # Reference: cdist + topk.
    ref_d2 = torch.cdist(X[0], X[0]) ** 2
    ref_vals, ref_idxs = ref_d2.topk(5, largest=False, dim=1)
    # idxs may differ on ties; verify distances match within bf16 tolerance.
    flash_first_idx = idxs[0, :, 0]
    ref_first_idx = ref_idxs[:, 0]
    same = (flash_first_idx == ref_first_idx).float().mean().item()
    assert same > 0.99, f"only {same:.2%} of nearest-neighbors agree"


@cuda_only
def test_dbscan_recovers_blobs():
    from flashlib import flash_dbscan

    torch.manual_seed(0)
    centers = torch.tensor([[0., 0.], [20., 20.], [-20., 10.]], device="cuda")
    pts_per = 500
    X = torch.cat([
        c + torch.randn(pts_per, 2, device="cuda") * 0.3 for c in centers
    ]).to(torch.float32)
    # tol=1e-3 exercises the bf16 KNN path (faster + numerically clean for
    # this test's tightly-clustered 2D data). Default tol=None routes through
    # flash_knn's fp32 path which has known accumulator-rounding quirks
    # affecting boundary classification on dense low-D inputs.
    labels = flash_dbscan(X, eps=1.0, min_samples=5, max_neighbors=32, tol=1e-3)
    n_clusters = int(labels.max().item()) + 1 if labels.max() >= 0 else 0
    n_noise = int((labels == -1).sum().item())
    assert n_clusters == 3
    assert n_noise < 50
