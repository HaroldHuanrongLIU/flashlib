# flashlib heuristic tuning

Hand-written cache-aware routing rules live in
`flashlib/<area>/<op>/route.py` (committed). The grids of raw
measurements that those rules were derived from live under
`benchmarks/tune/results/<op>/<device_tag>/` and are **not** committed
(see `.gitignore`).

This directory is the harness that produces those measurements and
prints suggested rule snippets for human review.

## The three-step loop

```
python -m benchmarks.tune.<op>           # 1. measure (writes results/<op>/<device>/*.jsonl)
python -m benchmarks.tune.derive.<op>    # 2. inspect (prints table + suggested rules)
$EDITOR flashlib/.../<op>/route.py       # 3. hand-edit the rule with the suggestion
```

The tuner is idempotent — it skips workloads whose JSONL already
exists. Pass `--rerun` to overwrite, `--size <shape_key>` to limit to a
subset (the shape_key is what the tuner prints and what the JSONL files
are named, e.g. `B1_N1024_M4096_D128_k8`).

## Layout

```
benchmarks/tune/
  README.md            (this file)
  _common.py           bench helper, jsonl writer, fingerprint, grid expander
  _template.py         starter for a new tuner; copy + edit
  knn.py               reference: cache-aware FA3-eligible routing
  kmeans.py            FA3-style assign vs split-D Triton
  eigh.py              cusolver / jacobi / halko / qdwh* tier picker
  gemm.py              precision/throughput Pareto for the multi-precision GEMMs
  derive/
    _common.py         pivot table + rule-suggestion printer
    _template.py       starter for a new derive script
    knn.py
    kmeans.py
    eigh.py
    gemm.py
  results/             gitignored; per-device JSONL grids land here
    .gitkeep
```

Each tuner declares a `WORKLOADS` grid and a `BACKENDS` candidate list,
plus three callables:

| function       | role                                                       |
|----------------|------------------------------------------------------------|
| `setup(w)`     | allocate inputs once per workload                          |
| `bench(ctx,c)` | return a no-arg callable that runs candidate `c` on `ctx`  |
| `correctness`  | optional, returns relative error vs torch reference        |

`benchmarks.tune._common.run_tuner` does the rest: warmup, median timing
with `torch.cuda.synchronize`, JSONL writing with a trailing `summary`
record (best backend, all-backend timings, hw fingerprint).

## Per-device organisation

Results are auto-keyed by `flashlib._hw.device_tag()`:
`H200`, `H100`, `A100`, `B200`, `RTX5090`, `sm120` (fallback). Two
machines tuning the same op never collide.

Hand-written rules can branch on the same fingerprint::

    from flashlib import _hw

    def route(*, N, D, k, hw=None, ...):
        hw = hw or _hw.current()
        if hw.is_blackwell:
            ...   # Blackwell-specific thresholds
        else:
            ...   # Hopper thresholds (current default)

So adding "support for a new GPU" means: run the tuner there, look at
the derive output, add a sm-arch branch in `route.py` if the thresholds
differ enough to matter.

## Cache-aware features

Every `route()` receives a `flashlib._hw.HwProps` with:

* `sm_arch`             Hopper/Blackwell/...
* `l2_bytes`            L2 cache size
* `smem_per_sm_bytes`   per-SM shared memory
* `sm_count`            number of SMs
* `total_mem_bytes`     HBM size
* `device_tag`          stable short string

The KNN rule is a worked example — its cutedsl-build branch refuses to
fire when the corpus would not fit in `0.6 * hw.l2_bytes`, so the same
rule auto-disables on smaller-L2 GPUs without code changes.

## Adding a new candidate backend to an existing op

1. Add the kernel + wrapper under `flashlib/.../<op>/<backend>/<file>.py`.
2. Add the candidate to `BACKENDS` in `benchmarks/tune/<op>.py`.
3. Run `python -m benchmarks.tune.<op> --rerun` — overwrites the JSONL
   so the old rows don't mask the new candidate.
4. `python -m benchmarks.tune.derive.<op>` — read suggested rules.
5. Hand-edit `flashlib/.../<op>/route.py` with the refined rule.

## Re-tuning on a new GPU (e.g. H100 → Blackwell)

1. On the new machine: `python -m benchmarks.tune.<op>`. Output lands
   under `benchmarks/tune/results/<op>/<new_tag>/`.
2. `python -m benchmarks.tune.derive.<op>`.
3. If the suggested rules differ from the existing branch in `route.py`,
   wrap the divergence in `if hw.is_blackwell:` (or
   `hw.device_tag == "B200"` for an exact match) and add the new
   thresholds.
4. Validate with `pytest tests/` — every test that exercises the
   dispatcher should still pass since the rule is gated on hardware.

## Adding a brand-new op

1. Copy `_template.py` → `<new_op>.py` and `derive/_template.py` →
   `derive/<new_op>.py`.
2. Fill in `WORKLOADS`, `BACKENDS`, `setup`, `bench`.
3. Create `flashlib/.../<new_op>/route.py` with a stub returning the
   default backend.
4. Run the tuner + derive once; refine the rule from the suggestion.

## Why the results aren't committed

Each per-device sweep is many MBs of JSONL and rebuilds in minutes.
The decisions distilled from the sweep — the rules in `route.py` — are
what the repo cares about. Committing rule changes (a few lines per op)
gives reviewers something readable; committing the raw measurements
would just generate noisy diffs every time the tuner is rerun. If you
want to share a sweep, attach the JSONL files to a PR comment.
