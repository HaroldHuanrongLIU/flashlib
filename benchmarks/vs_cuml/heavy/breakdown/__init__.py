"""Per-component time breakdowns for each multi-kernel flashlib primitive.

Each script under ``benchmarks/vs_cuml/heavy/breakdown/<prim>.py`` re-runs the
primitive at one representative heavy shape and reports — per component /
per kernel-stage — the wall time in milliseconds. Output goes to
``benchmarks/results/heavy/breakdown/<prim>.md`` as a small table
``| component | time_ms | % of total |``.

These scripts are NOT part of the headline vs-cuML audit (which lives in
``benchmarks/vs_cuml/heavy/<prim>.py``). They exist to show, for each
multi-kernel primitive, where the wall actually goes — so the reader can
confirm the kernel they think is dominant actually IS dominant on real
heavy shapes.

Measurement protocol:
  * single representative heavy shape per primitive (the one whose
    speedup is reported in the heavy-sweep summary).
  * warmup = 1 (discarded), repeat = 3 (median).
  * per-stage CUDA events span the matching code region; outer-most CUDA
    event spans the whole call.
  * the breakdown rows + the outer-most "total" row always agree to
    within ~1-2 % (overhead from event launch).

Run individually:
    python -m benchmarks.vs_cuml.heavy.breakdown.kmeans
    python -m benchmarks.vs_cuml.heavy.breakdown.dbscan
    ...

Run all in parallel:
    python -m benchmarks.vs_cuml.heavy.breakdown.run_all
"""
