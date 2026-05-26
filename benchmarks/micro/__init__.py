"""Micro-benchmarks isolating individual flashlib optimizations.

These scripts produce small, focused result tables under
``benchmarks/results/micro_*.md``. Unlike ``benchmarks/tune/``, the
scripts here are NOT for deriving routing rules -- they isolate one
optimization technique (no-materialization, flash-decoding-style
small-Q parallelism, precision-throughput Pareto, Halko scaling) and
quantify its win.

Run individually::

    python -m benchmarks.micro.bench_assign_kernel
    python -m benchmarks.micro.bench_knn_small_q
    python -m benchmarks.micro.bench_gemm_pareto
    python -m benchmarks.micro.bench_eigh_scaling
"""
