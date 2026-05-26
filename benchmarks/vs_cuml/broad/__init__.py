"""``benchmarks/vs_cuml/broad/`` — workload-grid sweep of flashlib vs cuML.

This module systematically sweeps every flashlib primitive over its main
workload axes (N, D, K, T, depth, etc.) and records the per-cell
flashlib/cuML wall-time ratio. It is COMPLEMENTARY to the heavy/ suite:

* ``heavy/<prim>.py`` ships ~5 hand-picked headline shapes with
  correctness gates.
* ``broad/<prim>.py`` ships ~15-30 grid shapes per primitive WITHOUT
  correctness gates — the goal is wide coverage of the speedup surface
  so the per-primitive plot has enough data points to be informative.

Per-row outputs land in ``benchmarks/results/broad/<prim>.json`` with
the schema documented in ``_common.py:BroadRow``. The plot generator
``broad/plot.py`` consumes those JSON files to render per-primitive
heatmaps + a summary box plot.
"""
