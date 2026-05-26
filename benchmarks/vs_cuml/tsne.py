"""t-SNE: ``flash_tsne`` vs ``cuml.manifold.TSNE``.

t-SNE convergence is stochastic / SGD-driven and reference labels are
not a faithful baseline (different inits land in different local
minima). Standard correctness metric: **trustworthiness** — the
fraction of low-D neighbours that were also high-D neighbours (close
to 1 means structure preserved).

We use ``sklearn.manifold.trustworthiness`` with k=12. flashlib runs
end-to-end Triton t-SNE; bf16 is not exposed as a knob at the API
layer for this primitive.

**Fairness note**: cuML defaults to ``method='fft'`` (FIt-SNE, ~O(N)
per iter) which gives up trustworthiness on shapes where it does not
sit at its sweet spot (see the N=20K row at default cuML config —
~0.53 trust). flashlib runs an *exact* O(N²) gradient (no
Barnes-Hut, no FFT). To make the timing apples-to-apples we force
cuML onto its own exact path via ``method='exact'``. The default
``'fft'`` numbers are reported in a sidebar for reference.
"""
from benchmarks.vs_cuml._common import (
    cap_threads, cuml_shim, time_gpu, time_cpu, title, header, fmt_table,
)
cap_threads(); cuml_shim()

import warnings; warnings.filterwarnings("ignore")
import numpy as np
import torch

from sklearn.datasets import make_blobs
from sklearn.manifold import trustworthiness
from cuml.manifold import TSNE as cuTSNE
from flashlib.primitives.tsne import flash_tsne


# (label, N, D, K, perplexity, n_iter)
# t-SNE is O(N^2 / iter); we keep N small enough that cuML runs in
# seconds.
SHAPES = [
    ("small  N=5K   D=32  K=5",   5_000,  32, 5, 30.0, 500),
    ("medium N=10K  D=64  K=10", 10_000,  64, 10, 30.0, 500),
    ("large  N=20K  D=128 K=10", 20_000, 128, 10, 30.0, 500),
]


def run_one(label, N, D, K, perplexity, n_iter):
    title(f"t-SNE {label}  (N={N:,}, D={D}, K={K}, "
          f"perplexity={perplexity}, n_iter={n_iter})")

    X_np, _ = make_blobs(n_samples=N, centers=K, n_features=D,
                          cluster_std=2.0, random_state=0)
    X_np = X_np.astype(np.float32)

    rows = []

    # Reference: cuML default (``method='fft'``). Not apples-to-apples
    # against flashlib but is what most users actually run; reported for
    # context.
    cu_fft_emb = np.asarray(cuTSNE(n_components=2, perplexity=perplexity,
                                     n_iter=n_iter, random_state=0,
                                     method="fft").fit_transform(X_np))
    cu_fft_tw = trustworthiness(X_np, cu_fft_emb, n_neighbors=12)
    t_cu_fft = time_gpu(
        lambda: cuTSNE(n_components=2, perplexity=perplexity,
                        n_iter=n_iter, random_state=0,
                        method="fft").fit_transform(X_np),
        repeat=2, warmup=1,
    )
    rows.append(("fp32", "cuml (fft, ref)", f"{t_cu_fft:8.1f}",
                 f"{cu_fft_tw:.4f}", "ref"))

    # Apples-to-apples timing: cuML's own exact O(N²) gradient.
    cu_emb = np.asarray(cuTSNE(n_components=2, perplexity=perplexity,
                                 n_iter=n_iter, random_state=0,
                                 method="exact").fit_transform(X_np))
    cu_tw = trustworthiness(X_np, cu_emb, n_neighbors=12)
    t_cu = time_gpu(
        lambda: cuTSNE(n_components=2, perplexity=perplexity,
                        n_iter=n_iter, random_state=0,
                        method="exact").fit_transform(X_np),
        repeat=2, warmup=1,
    )
    rows.append(("fp32", "cuml (exact)", f"{t_cu:8.1f}",
                 f"{cu_tw:.4f}", "1.00x"))

    X32 = torch.tensor(X_np, device="cuda")
    fl_emb_t = flash_tsne(X32, n_iter=n_iter, perplexity=perplexity, seed=0)
    fl_emb = fl_emb_t.float().cpu().numpy()
    fl_tw = trustworthiness(X_np, fl_emb, n_neighbors=12)
    t_fl = time_gpu(
        lambda: flash_tsne(X32, n_iter=n_iter, perplexity=perplexity, seed=0),
        repeat=2, warmup=1,
    )
    rows.append(("fp32", "flashlib (exact)", f"{t_fl:8.1f}",
                 f"{fl_tw:.4f}", f"{t_cu / t_fl:.2f}x"))

    print(fmt_table(rows, ["dtype", "engine", "time(ms)",
                            "trustworthiness", "vs cuml"]))


def main():
    header()
    for s in SHAPES:
        run_one(*s)
    print()


if __name__ == "__main__":
    main()
