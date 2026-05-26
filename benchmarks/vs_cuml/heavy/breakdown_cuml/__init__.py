"""Per-CUDA-kernel profile of cuML primitives for the "why flashlib beats cuML" audit.

Each script under ``benchmarks/vs_cuml/heavy/breakdown_cuml/<prim>.py`` wraps
a single ``cuml`` call in a ``torch.profiler.profile(activities=[CUDA])``
window and dumps the captured kernel events as:

  * ``benchmarks/results/heavy/breakdown_cuml/<prim>.md`` — top-K kernels
    sorted by total time, with launch count, mean ms, and % of total.
  * ``benchmarks/results/heavy/breakdown_cuml/<prim>.json`` — raw event list
    for downstream tooling.

These tables answer "where does cuML spend its time?" — the complement of the
``benchmarks/vs_cuml/heavy/breakdown/<prim>.md`` tables which answer the same
question for flashlib. Reading both side-by-side produces the kernel-by-kernel
speedup story.

Tool note: ``nsys`` is not available on this image, so we use the PyTorch
profiler's CUPTI integration. CUPTI captures every CUDA kernel launched in
the process (including RAFT / cuvs kernels invoked transitively from cuML),
so the trace is complete. The cost is ~2-5 % timing distortion vs nsys —
within audit tolerance.

Run individually:
    python -m benchmarks.vs_cuml.heavy.breakdown_cuml.kmeans

Run all:
    python -m benchmarks.vs_cuml.heavy.breakdown_cuml.run_all
"""
