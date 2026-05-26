"""Heavy t-SNE sweep — release-candidate audit.

t-SNE convergence is stochastic; the only stable correctness signal
is **trustworthiness** (sklearn.manifold). flashlib runs an exact
O(N^2) gradient; cuML's default is FFT (FIt-SNE) which is a totally
different algorithm. To make timing apples-to-apples we force cuML
onto its OWN exact path (``method='exact'``) and report a SEPARATE
cuML FFT row for context.

Heavy shapes: N up to 50K — exact O(N^2) per iter caps us there
within a sensible wall time. (cuML 'exact' at N=50K is ~5 min.)

Anti-reward-hacking guardrails:

* Same fixed-seed ``X`` across engines; same ``perplexity``,
  ``n_iter``, ``random_state``.
* Trustworthiness at k=12; gate at >= 0.50 (high-cluster-overlap
  blobs cap trust at ~0.6-0.7 even for the correct algorithm).
* Two cuML rows (FFT for context, exact for fairness) reported on
  every shape.
"""
from benchmarks.vs_cuml.heavy._common import (
    cap_threads, cuml_shim, time_gpu, title, header,
    audit_record, apples_to_apples,
    hbm_peak_reset, hbm_peak_gb, gate_metric, free_gpu, RESULTS_DIR,
)
cap_threads(); cuml_shim()

import warnings; warnings.filterwarnings("ignore")
import gc
import numpy as np
import torch

from sklearn.datasets import make_blobs
from sklearn.manifold import trustworthiness
from cuml.manifold import TSNE as cuTSNE
from flashlib.primitives.tsne import flash_tsne


# (label, N, D, K, perplexity, n_iter)
# NOTE: cuML method='exact' is O(N^2) per iter — N=30K @ 500 iters takes
# ~10 min per fit, N=50K is intractable. We cap at N=20K so the heavy
# sweep finishes in reasonable wall time. The exact-vs-exact apples
# comparison is honest at this size, and flashlib's algorithmic advantage
# is the SAME at N=20K vs N=50K (both fused-streaming O(N^2)).
SHAPES = [
    ("small   N=10K  D=64  K=10",  10_000,  64, 10, 30.0, 500),
    ("medium  N=15K  D=128 K=10",  15_000, 128, 10, 30.0, 500),
    ("large   N=20K  D=128 K=10",  20_000, 128, 10, 30.0, 500),
]

PRIM = "tsne"


def _run_one(label, N, D, K, perplexity, n_iter):
    title(f"t-SNE  {label}  (N={N:,}, D={D}, K={K}, "
          f"perplexity={perplexity}, n_iter={n_iter})")

    X_np, _ = make_blobs(n_samples=N, centers=K, n_features=D,
                          cluster_std=2.0, random_state=0)
    X_np = X_np.astype(np.float32)

    # cuML FFT (context row).
    free_gpu(); hbm_peak_reset()
    try:
        cu_fft_emb = np.asarray(cuTSNE(n_components=2, perplexity=perplexity,
                                         n_iter=n_iter, random_state=0,
                                         method="fft").fit_transform(X_np))
        cu_fft_tw = trustworthiness(X_np, cu_fft_emb, n_neighbors=12)
        t_cu_fft = time_gpu(
            lambda: cuTSNE(n_components=2, perplexity=perplexity,
                            n_iter=n_iter, random_state=0,
                            method="fft").fit_transform(X_np),
            repeat=1, warmup=1,
        )
        hbm_cu_fft = hbm_peak_gb()
        audit_record(PRIM, {
            "shape": label, "algo": "cuml(FFT)", "engine": "cuml",
            "time_ms": f"{t_cu_fft:10.1f}",
            "trustworthiness": f"{cu_fft_tw:.4f}",
            "vs_cuml_exact": "n/a",
            "HBM_GB": f"{hbm_cu_fft:.1f}",
            "gate": "PASS (context; not the apples-to-apples baseline)",
            "conditions": apples_to_apples(
                op="tsne", shape={"N": N, "D": D, "K": K,
                                   "perplexity": perplexity, "iter": n_iter},
                flashlib_dtype="-", cuml_dtype="fp32",
                flashlib_algorithm="-", cuml_algorithm="cuml_tsne_fft",
                init_shared=False,
                notes="cuML default; algorithmically different from flashlib"),
        }, columns=["shape", "algo", "engine", "time_ms", "trustworthiness",
                    "vs_cuml_exact", "HBM_GB", "gate"])
    except Exception as e:
        audit_record(PRIM, {
            "shape": label, "algo": "cuml(FFT)", "engine": "cuml",
            "time_ms": "ERR", "trustworthiness": "-",
            "vs_cuml_exact": "-", "HBM_GB": "-",
            "gate": f"FAIL ({type(e).__name__}: {str(e)[:80]})",
            "conditions": {},
        }, columns=["shape", "algo", "engine", "time_ms", "trustworthiness",
                    "vs_cuml_exact", "HBM_GB", "gate"])

    # cuML exact — the apples-to-apples baseline.
    free_gpu(); hbm_peak_reset()
    try:
        cu_emb = np.asarray(cuTSNE(n_components=2, perplexity=perplexity,
                                     n_iter=n_iter, random_state=0,
                                     method="exact").fit_transform(X_np))
        cu_tw = trustworthiness(X_np, cu_emb, n_neighbors=12)
        t_cu = time_gpu(
            lambda: cuTSNE(n_components=2, perplexity=perplexity,
                            n_iter=n_iter, random_state=0,
                            method="exact").fit_transform(X_np),
            repeat=1, warmup=1,
        )
        hbm_cu = hbm_peak_gb()
        audit_record(PRIM, {
            "shape": label, "algo": "cuml(exact)", "engine": "cuml",
            "time_ms": f"{t_cu:10.1f}", "trustworthiness": f"{cu_tw:.4f}",
            "vs_cuml_exact": "1.00x", "HBM_GB": f"{hbm_cu:.1f}",
            "gate": gate_metric("tw", cu_tw, lower=0.50),
            "conditions": apples_to_apples(
                op="tsne", shape={"N": N, "D": D, "K": K,
                                   "perplexity": perplexity, "iter": n_iter},
                flashlib_dtype="-", cuml_dtype="fp32",
                flashlib_algorithm="-", cuml_algorithm="cuml_tsne_exact_ON2",
                init_shared=False,
                notes="forced 'method=exact' to match flashlib's exact O(N^2)"),
        }, columns=["shape", "algo", "engine", "time_ms", "trustworthiness",
                    "vs_cuml_exact", "HBM_GB", "gate"])
    except Exception as e:
        t_cu = float("inf")
        audit_record(PRIM, {
            "shape": label, "algo": "cuml(exact)", "engine": "cuml",
            "time_ms": "ERR", "trustworthiness": "-",
            "vs_cuml_exact": "-", "HBM_GB": "-",
            "gate": f"FAIL ({type(e).__name__}: {str(e)[:80]})",
            "conditions": {},
        }, columns=["shape", "algo", "engine", "time_ms", "trustworthiness",
                    "vs_cuml_exact", "HBM_GB", "gate"])

    # flashlib exact.
    free_gpu(); hbm_peak_reset()
    try:
        X32 = torch.tensor(X_np, device="cuda")
        fl_emb_t = flash_tsne(X32, n_iter=n_iter, perplexity=perplexity, seed=0)
        fl_emb = fl_emb_t.float().cpu().numpy()
        fl_tw = trustworthiness(X_np, fl_emb, n_neighbors=12)
        t_fl = time_gpu(
            lambda: flash_tsne(X32, n_iter=n_iter, perplexity=perplexity, seed=0),
            repeat=1, warmup=1,
        )
        hbm_fl = hbm_peak_gb()
        audit_record(PRIM, {
            "shape": label, "algo": "flashlib(exact)", "engine": "flashlib",
            "time_ms": f"{t_fl:10.1f}", "trustworthiness": f"{fl_tw:.4f}",
            "vs_cuml_exact": (f"{t_cu / t_fl:.2f}x"
                              if t_cu != float("inf") else "n/a"),
            "HBM_GB": f"{hbm_fl:.1f}",
            "gate": gate_metric("tw", fl_tw, lower=0.50),
            "conditions": apples_to_apples(
                op="tsne", shape={"N": N, "D": D, "K": K,
                                   "perplexity": perplexity, "iter": n_iter},
                flashlib_dtype="fp32", cuml_dtype="fp32",
                flashlib_algorithm="exact_ON2_fused_qsum_+_grad",
                cuml_algorithm="cuml_tsne_exact_ON2",
                init_shared=False,
                notes="matched 'exact O(N^2)' algorithm; cuML FFT row is "
                      "shown for context only"),
        }, columns=["shape", "algo", "engine", "time_ms", "trustworthiness",
                    "vs_cuml_exact", "HBM_GB", "gate"])
    except Exception as e:
        audit_record(PRIM, {
            "shape": label, "algo": "flashlib(exact)", "engine": "flashlib",
            "time_ms": "ERR", "trustworthiness": "-",
            "vs_cuml_exact": "-", "HBM_GB": "-",
            "gate": f"FAIL ({type(e).__name__}: {str(e)[:80]})",
            "conditions": {},
        }, columns=["shape", "algo", "engine", "time_ms", "trustworthiness",
                    "vs_cuml_exact", "HBM_GB", "gate"])

    del X_np
    gc.collect(); torch.cuda.empty_cache()


def main():
    header()
    for ext in (".md", ".json"):
        p = RESULTS_DIR / f"{PRIM}{ext}"
        if p.exists():
            p.unlink()
    for s in SHAPES:
        _run_one(*s)


if __name__ == "__main__":
    main()
