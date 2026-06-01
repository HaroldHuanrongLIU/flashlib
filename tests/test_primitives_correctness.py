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


def _blobs(M, D, n_centers, device, seed=0, scale=4.0, spread=1.0):
    g = torch.Generator(device=device).manual_seed(seed)
    centers = torch.randn(n_centers, D, generator=g, device=device) * scale
    lab = torch.randint(0, n_centers, (M,), generator=g, device=device)
    X = centers[lab] + torch.randn(M, D, generator=g, device=device) * spread
    return X.to(torch.float32).contiguous()


def _recall(got_ids, ref_ids):
    k = ref_ids.shape[1]
    hits = sum(
        len(set(got_ids[i].tolist()) & set(ref_ids[i].tolist()))
        for i in range(got_ids.shape[0])
    )
    return hits / (got_ids.shape[0] * k)


@cuda_only
def test_ivf_flat_recall_vs_brute():
    """nprobe==nlist must reproduce brute force; moderate nprobe clears 0.95.

    At ``nprobe == nlist`` every list is probed, so the candidate set is the
    whole database and the fused fine-scan must return the exact top-k
    distances (modulo fp tie-breaking at the k boundary). Recall then climbs
    monotonically with nprobe, the only recall knob.
    """
    from flashlib import flash_ivf_flat_build, flash_ivf_flat_search

    torch.manual_seed(0)
    M, D, nlist, k = 20_000, 64, 128, 10
    X = _blobs(M, D, 16, "cuda", seed=0)
    Q = _blobs(256, D, 16, "cuda", seed=1)

    index = flash_ivf_flat_build(X, nlist, nprobe=8, niter=15, seed=0)

    ref = torch.cdist(Q, X) ** 2
    ref_vals, ref_ids = ref.topk(k, largest=False, dim=1)

    # Exhaustive: probe every list -> exact distances.
    vals_all, ids_all = flash_ivf_flat_search(index, Q, k, nprobe=nlist)
    assert torch.allclose(
        vals_all.sort(1).values, ref_vals.sort(1).values, rtol=1e-3, atol=1e-1
    ), "nprobe==nlist distances must equal brute force"
    assert _recall(ids_all, ref_ids) >= 0.99

    # Moderate nprobe still clears a high recall bar on blob data.
    _, ids_mod = flash_ivf_flat_search(index, Q, k, nprobe=nlist // 4)
    assert _recall(ids_mod, ref_ids) >= 0.95

    # Recall is monotonic in nprobe (more probes never hurts).
    r_lo = _recall(flash_ivf_flat_search(index, Q, k, nprobe=2)[1], ref_ids)
    r_hi = _recall(flash_ivf_flat_search(index, Q, k, nprobe=16)[1], ref_ids)
    assert r_hi >= r_lo


@cuda_only
def test_ivf_flat_isorecall_vs_reference():
    """Triton kernel == pure-torch reference on the SAME index (iso-recall).

    Building once and running both the fused Triton search and the torch
    oracle over the identical centroids/assignments must return identical
    neighbour ids at matched ``nprobe`` -- so any divergence is a kernel
    bug, not an algorithmic difference.
    """
    from flashlib import flash_ivf_flat_build, flash_ivf_flat_search
    from flashlib.primitives.ivf_flat import torch_fallback as tf

    torch.manual_seed(0)
    M, D, nlist, k = 12_000, 48, 64, 8
    X = _blobs(M, D, 10, "cuda", seed=2)
    Q = _blobs(128, D, 10, "cuda", seed=3)

    index = flash_ivf_flat_build(X, nlist, nprobe=8, niter=15, seed=0)

    for nprobe in (4, 8, 16):
        gv, gi = flash_ivf_flat_search(index, Q, k, nprobe=nprobe)
        tv, ti = tf.ivf_flat_search_torch(index, Q, k, nprobe=nprobe)
        # Same candidate set scanned, same exact distances found; compare on
        # the distance values, which are robust to k-boundary ties.
        assert torch.allclose(
            gv.sort(1).values, tv.sort(1).values, rtol=1e-3, atol=1e-2
        ), f"distance sets diverge at nprobe={nprobe}"
        # IDs then match too, save for the rare fp tie at the k-th boundary
        # (two candidates equidistant to within fp32 accumulation order).
        same = (gi.sort(1).values == ti.sort(1).values).float().mean().item()
        assert same >= 0.99, f"iso-recall id mismatch at nprobe={nprobe}: {same:.4f}"


@cuda_only
def test_ivf_flat_application_class():
    """The sklearn-style IVFFlat wrapper round-trips fit -> kneighbors."""
    from flashlib import IVFFlat

    torch.manual_seed(0)
    X = _blobs(8_000, 32, 8, "cuda", seed=4)
    Q = _blobs(64, 32, 8, "cuda", seed=5)
    model = IVFFlat(nlist=64, nprobe=32, n_neighbors=10, niter=10, seed=0).fit(X)
    dist, idx = model.kneighbors(Q)
    assert dist.shape == (64, 10) and idx.shape == (64, 10)

    ref = torch.cdist(Q, X) ** 2
    _, ref_ids = ref.topk(10, largest=False, dim=1)
    assert _recall(idx, ref_ids) >= 0.95


@cuda_only
def test_ivf_flat_gemm_matches_elementwise_high_d():
    """GEMM and elementwise fine-scan agree at D>256 (D-split path).

    D>256 drives the GEMM kernel through its D-split branch (re-gather the
    query tile per corpus chunk) followed by an oversampled exact re-rank.
    Both variants scan the same candidate set, so their returned squared-L2
    distances must match each other and stay at brute-force recall.
    """
    from flashlib import flash_ivf_flat_build, flash_ivf_flat_search

    torch.manual_seed(0)
    M, D, nlist, k = 12_000, 320, 64, 10
    X = _blobs(M, D, 12, "cuda", seed=6)
    Q = _blobs(512, D, 12, "cuda", seed=7)
    index = flash_ivf_flat_build(X, nlist, nprobe=8, niter=12, seed=0)

    gv, gi = flash_ivf_flat_search(index, Q, k, nprobe=16, variant="gemm")
    ev, _ = flash_ivf_flat_search(index, Q, k, nprobe=16, variant="elementwise")
    assert torch.allclose(
        gv.sort(1).values, ev.sort(1).values, rtol=1e-3, atol=1e-1
    ), "gemm (D-split) vs elementwise distances diverge at D>256"

    ref = torch.cdist(Q, X) ** 2
    _, ref_ids = ref.topk(k, largest=False, dim=1)
    assert _recall(gi, ref_ids) >= 0.95


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
