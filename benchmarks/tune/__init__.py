"""flashlib heuristic tuning harness.

See :doc:`README.md` for the workflow. Each ``benchmarks/tune/<op>.py``
defines a ``WORKLOADS`` grid plus a ``BACKENDS`` candidate dict and is
runnable via ``python -m benchmarks.tune.<op>``. Results land under
``benchmarks/tune/results/<op>/<device_tag>/`` (gitignored).
"""
