"""Heavy stress + correctness sweep for flashlib release-candidate audit.

Each script under ``benchmarks/vs_cuml/heavy/<prim>.py`` runs the
matching primitive at shapes 5-20x larger than ``benchmarks/vs_cuml/``
with strict anti-reward-hacking guardrails:

* identical init across engines (centroids / random_state),
* both inputs GPU-resident (no H2D timing gimmicks),
* matched algorithm where comparable (``algorithm='brute'``,
  ``method='exact'``, etc.) — algorithmic shortcuts are reported as
  SEPARATE rows with the difference flagged in the row label,
* precision disclosure: if flashlib runs bf16 while cuML runs fp32,
  BOTH dtypes are reported,
* first-call JIT discarded (extra warmup for CuteDSL FA3 paths),
* correctness gate: every row reports a quality metric (rel-err / ARI /
  recall@K / R^2 / accuracy / trustworthiness); rows below the published
  tier are flagged ``FAIL``,
* HBM-peak logged per row so a death-by-OOM is visible.

Use the ``benchmarks/vs_cuml/heavy/run_all_parallel.py`` dispatcher to
fan out across 8 H200 GPUs (~45-60 min wall for the full sweep). Each
script also runs standalone:

    python -m benchmarks.vs_cuml.heavy.kmeans
    python -m benchmarks.vs_cuml.heavy.knn
    ...
"""
