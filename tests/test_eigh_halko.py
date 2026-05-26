"""Tests for the eigh dispatcher's Halko route + opt-in routing.

Halko subspace iteration was promoted from primitives/pca to a first-class
linalg.eigh variant. After the 2026-05 "exact-by-default" cleanup it is
NEVER chosen unless the caller passes ``tol`` -- ``eigh(A, K=K)`` with
``tol=None`` (the default) always runs the exact path. This file pins
that contract.
"""
from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")

if not torch.cuda.is_available():
    pytest.skip("CUDA required for eigh_halko", allow_module_level=True)

DEVICE = "cuda"


def _spd_with_decay(N: int, seed: int = 0, decay: float = 0.95) -> torch.Tensor:
    """SPD matrix with controlled exponential spectral decay -- the realistic
    PCA / TruncatedSVD regime. Random Wishart matrices have nearly-flat
    spectra at large N, the worst case for subspace iteration.
    """
    g = torch.Generator(device=DEVICE).manual_seed(seed)
    Q, _ = torch.linalg.qr(torch.randn(N, N, device=DEVICE, dtype=torch.float32, generator=g))
    eigvals = torch.tensor([decay ** i for i in range(N)],
                           device=DEVICE, dtype=torch.float32)
    return (Q * eigvals) @ Q.T


def test_halko_matches_cusolver_top_k():
    """Halko's top-K eigenvalues should match cuSOLVER's top-K closely
    when the spectrum has decay (the regime Halko is designed for)."""
    from flashlib.linalg.eigh import eigh_halko, eigh_cusolver
    torch.manual_seed(0)
    N, K = 1024, 16
    G = _spd_with_decay(N, seed=0)

    eigvals_h, eigvecs_h = eigh_halko(G, K=K, n_iter=8, p=30)
    eigvals_full, _ = eigh_cusolver(G)
    top_k_full = eigvals_full[-K:]

    rel = (eigvals_h - top_k_full).abs() / top_k_full.abs().clamp_min(1e-9)
    assert rel.max().item() < 1e-3, (
        f"Halko top-{K} eigenvalues max rel err {rel.max().item():.2e}"
    )

    # Eigenvector residual: ||G v - lambda v|| / (||lambda v||) should be tight.
    res = (G @ eigvecs_h - eigvecs_h * eigvals_h).norm(dim=0)
    norm = (eigvecs_h * eigvals_h).norm(dim=0).clamp_min(1e-9)
    assert (res / norm).max().item() < 1e-2


def test_eigh_dispatcher_routes_to_halko_only_with_tol():
    """``eigh(A, K=K)`` is exact by default; Halko is only chosen when
    ``tol >= 1e-4`` AND ``K*4 < N`` AND ``N >= 256``.
    """
    from flashlib.linalg.eigh.impl import _route

    # tol=None (default) -> always exact, never Halko, regardless of K.
    assert _route(N=1024, K=16, tol=None) == "cusolver"
    assert _route(N=4096, K=64, tol=None) == "cusolver"

    # tol >= 1e-4 + favourable shape -> Halko.
    assert _route(N=1024, K=16, tol=1e-4) == "halko"
    assert _route(N=4096, K=64, tol=1e-3) == "halko"

    # K too large (K*4 >= N) -> NOT Halko even with loose tol.
    assert _route(N=256, K=128, tol=1e-3) != "halko"
    # N too small -> NOT Halko even with loose tol.
    assert _route(N=128, K=8, tol=1e-3) != "halko"
    # K not provided -> never Halko.
    assert _route(N=4096, K=None, tol=1e-3) in ("cusolver", "qdwh", "qdwh_ns")
    # tol tighter than Halko residual (~1e-4) -> fall through to exact.
    assert _route(N=4096, K=64, tol=1e-7) == "cusolver"


def test_eigh_dispatcher_returns_K_eigenpairs():
    """``eigh(A, K=K)`` returns K eigenpairs regardless of route."""
    from flashlib.linalg.eigh import eigh
    torch.manual_seed(1)
    N, K = 1024, 16
    G = _spd_with_decay(N, seed=1)

    # Default exact path with K -> top-K from cuSOLVER.
    eigvals, eigvecs = eigh(G, K=K)
    assert eigvals.shape == (K,)
    assert eigvecs.shape == (N, K)
    eigvals_full, _ = eigh(G)
    rel = (eigvals - eigvals_full[-K:]).abs() / eigvals_full[-K:].abs().clamp_min(1e-9)
    assert rel.max().item() < 1e-5  # exact -> exact

    # Loose tol -> Halko; still returns K eigenpairs, looser tolerance.
    eigvals_h, eigvecs_h = eigh(G, K=K, tol=1e-4, n_iter=8)
    assert eigvals_h.shape == (K,)
    assert eigvecs_h.shape == (N, K)
    rel_h = (eigvals_h - eigvals_full[-K:]).abs() / eigvals_full[-K:].abs().clamp_min(1e-9)
    assert rel_h.max().item() < 1e-3


def test_eigh_dispatcher_K_too_large_falls_back_to_full():
    """When K is large relative to N, dispatcher must still return K
    eigenpairs by routing through a full path + slice."""
    from flashlib.linalg.eigh import eigh
    torch.manual_seed(2)
    N, K = 256, 128  # K*4 = 512 > N -> no Halko route
    G = _spd_with_decay(N, seed=2)

    eigvals, eigvecs = eigh(G, K=K, tol=1e-3)
    assert eigvals.shape == (K,)
    assert eigvecs.shape == (N, K)


def test_info_estimate_routes_K_to_halko_with_tol():
    """``info.estimate('eigh', shape=(N, N), params={'K': K}, tol=1e-3)``
    routes to ``eigh_halko``; without ``tol`` it stays on cuSOLVER.
    """
    import flashlib.info as info

    est_full = info.estimate("eigh", shape=(4096, 4096), device="H100")
    est_topk_exact = info.estimate("eigh", shape=(4096, 4096),
                                    params={"K": 64}, device="H100")
    est_topk_loose = info.estimate("eigh", shape=(4096, 4096),
                                    params={"K": 64}, tol=1e-3,
                                    device="H100")
    assert est_full.op_name == "eigh_cusolver"
    # tol=None default keeps the exact path even with K.
    assert est_topk_exact.op_name == "eigh_cusolver"
    # tol=1e-3 + favourable shape -> Halko.
    assert est_topk_loose.op_name == "eigh_halko"
    assert est_topk_loose.runtime_ms < est_full.runtime_ms / 5, (
        f"Halko predicted {est_topk_loose.runtime_ms:.2f}ms vs cuSOLVER "
        f"{est_full.runtime_ms:.2f}ms -- expected >5x gap."
    )


def test_eigh_halko_callable():
    """``eigh_halko`` is still exported as a backend-explicit entry
    point for power users; the routing is encapsulated inside ``eigh``.
    """
    from flashlib.linalg.eigh import eigh_halko
    assert callable(eigh_halko)
