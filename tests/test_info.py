"""Informative API tests -- agent-friendly contract:

1. ``import flashlib.info`` doesn't load torch / triton.
2. ``info.estimate()`` returns an :class:`Estimate` with sensible fields.
3. ``info.estimate()`` call is sub-10 ms.
4. Compound primitives expose ``subops``.
5. Derived properties (``achieved_tflops`` / ``achieved_gbs`` /
   ``arithmetic_intensity`` / ``utilization_pct``) work on every op.
6. New surface (``info.compare`` / ``info.summary``) functions.
7. Registry is internally consistent (no dead entries).
"""
import sys
import subprocess
import time

import pytest


# ─── 1. Lazy-import contract ─────────────────────────────────────────────

def test_info_does_not_import_torch():
    """``import flashlib.info`` alone must not pull in torch/triton.

    Agents need to call ``info.estimate()`` in environments without
    GPU access; we also want fast cold imports for CLI tools.
    """
    code = (
        "import sys\n"
        "import flashlib.info as info\n"
        "loaded = [m for m in ('torch', 'triton') if m in sys.modules]\n"
        "assert not loaded, f'unexpected modules loaded: {loaded}'\n"
        "print('OK')\n"
    )
    out = subprocess.check_output([sys.executable, "-c", code], text=True)
    assert "OK" in out


# ─── 2. Estimate basic contract ──────────────────────────────────────────

def test_estimate_returns_dataclass():
    import flashlib.info as info
    est = info.estimate("kmeans", shape=(1_000_000, 128),
                        params={"K": 100, "max_iters": 25}, device="H100")
    assert est.runtime_ms > 0
    assert est.flops > 0
    assert est.bytes_moved > 0
    assert est.bound in ("compute", "memory", "mixed", "latency")
    assert est.confidence in ("calibrated", "measured", "roofline", "heuristic")
    # dispatcher stamps dtype + device for downstream derived props.
    assert est.dtype == "fp32"
    assert est.device == "H100"


def test_estimate_is_fast():
    """Each ``info.estimate()`` call should be sub-10 ms (agent-friendly)."""
    import flashlib.info as info
    info.estimate("kmeans", shape=(1000, 128), params={"K": 10}, device="H100")
    t0 = time.perf_counter()
    for _ in range(100):
        info.estimate("kmeans", shape=(1_000_000, 128),
                       params={"K": 100}, device="H100")
    elapsed = (time.perf_counter() - t0) / 100
    assert elapsed * 1000 < 10, f"estimate too slow: {elapsed*1000:.2f} ms/call"


def test_compound_op_has_subops():
    """PCA / DBSCAN / HDBSCAN / UMAP expose per-sub-op breakdown."""
    import flashlib.info as info
    est_pca = info.estimate("pca", shape=(100_000, 256), params={"K": 50},
                              device="H100")
    assert len(est_pca.subops) >= 2

    est_dbscan = info.estimate("dbscan", shape=(20_000, 32),
                                params={"eps": 0.5, "min_samples": 5},
                                device="H100")
    assert len(est_dbscan.subops) >= 1

    est_hdbscan = info.estimate("hdbscan", shape=(20_000, 16),
                                  params={"min_samples": 5, "k": 32},
                                  device="H100")
    assert len(est_hdbscan.subops) == 4   # knn + mrd + mst + condense
    op_names = [s.op_name for s in est_hdbscan.subops]
    assert "hdbscan.knn" in op_names

    est_umap = info.estimate("umap", shape=(10_000, 64), device="H100")
    assert len(est_umap.subops) >= 2


# ─── 3. Derived performance properties ───────────────────────────────────

def test_achieved_throughput_properties():
    import flashlib.info as info
    est = info.estimate("kmeans", shape=(500_000, 64),
                        params={"K": 64, "max_iters": 25}, device="H200")
    assert est.achieved_tflops >= 0
    assert est.achieved_gbs >= 0
    assert est.arithmetic_intensity > 0
    # utilization_pct must be set when dtype + device are.
    assert est.utilization_pct is not None
    assert 0 <= est.utilization_pct <= 200  # roofline can overshoot for solver


def test_arithmetic_intensity_is_finite_for_zero_bytes():
    """``arithmetic_intensity`` is ``inf`` only when bytes are zero."""
    import flashlib.info as info
    est = info.estimate("kmeans", shape=(1000, 16),
                        params={"K": 4, "max_iters": 1}, device="H100")
    # always-positive bytes -> always-finite intensity
    assert est.arithmetic_intensity != float("inf")


# ─── 4. Compare / summary ────────────────────────────────────────────────

def test_compare_returns_speedup():
    import flashlib.info as info
    out = info.compare("kmeans", shape=(500_000, 64), params={"K": 64})
    assert "flashlib" in out
    assert "references" in out
    assert out["dtype"] == "fp32"
    refs = out["references"]
    assert "cuml" in refs
    assert refs["cuml"]["runtime_ms"] > 0
    assert refs["cuml"]["speedup"] > 0


def test_compare_restricts_references():
    import flashlib.info as info
    out = info.compare("kmeans", shape=(500_000, 64), params={"K": 64},
                         references=["cuml"])
    assert list(out["references"]) == ["cuml"]


def test_summary_one_liner():
    import flashlib.info as info
    line = info.summary("kmeans", shape=(500_000, 64), params={"K": 64})
    assert isinstance(line, str)
    assert "kmeans" in line
    assert "ms" in line


# ─── 5. Registry sanity ─────────────────────────────────────────────────

def test_list_ops_includes_v01_set():
    import flashlib.info as info
    ops = set(info.list_ops())
    for name in ("kmeans", "knn", "pca", "dbscan", "standard_scaler",
                 "cov_gemm", "eigh", "pairwise_l2"):
        assert name in ops, f"missing op {name!r}"


def test_unknown_op_raises():
    import flashlib.info as info
    with pytest.raises(KeyError):
        info.estimate("definitely_not_a_real_op", shape=(10, 10))


def test_no_dead_registry_entries():
    """Every registry entry must resolve to a callable cost function."""
    import flashlib.info as info
    from flashlib.info.dispatch import _load_estimate
    for op in info.list_ops():
        try:
            fn = _load_estimate(op)
        except (ImportError, AttributeError) as e:
            pytest.fail(f"op {op!r} fails to load: {type(e).__name__}: {e}")
        assert callable(fn), f"op {op!r} resolves to non-callable {fn!r}"


def test_variants_pareto_returns_sorted():
    """``info.pareto`` returns a list sorted by ascending runtime."""
    import flashlib.info as info
    front = info.pareto("eigh", shape=(8192, 8192))
    assert front  # at least one
    rts = [v.estimate.runtime_ms for v in front]
    assert rts == sorted(rts), f"pareto not sorted: {rts}"


# ─── 6. Dtype canonicalisation ───────────────────────────────────────────

def test_dtype_canonicalisation():
    import flashlib.info as info
    e_f32 = info.estimate("kmeans", shape=(1000, 16),
                            params={"K": 4}, dtype="float32")
    e_fp32 = info.estimate("kmeans", shape=(1000, 16),
                             params={"K": 4}, dtype="fp32")
    e_f = info.estimate("kmeans", shape=(1000, 16),
                          params={"K": 4}, dtype="f32")
    assert e_f32.runtime_ms == e_fp32.runtime_ms == e_f.runtime_ms
    assert e_f32.dtype == "fp32"
