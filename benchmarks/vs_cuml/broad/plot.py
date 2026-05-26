"""Render visualizations of the broad-sweep speedup data.

Reads every ``benchmarks/results/broad/<prim>.json`` and emits PNGs into
``benchmarks/results/broad/plots/``:

* ``overview_box.png``    — speedup distribution per primitive (box+strip)
* ``overview_bar.png``    — geometric mean speedup per primitive (sorted)
* ``overview_scatter.png``— per-cell flashlib_ms vs cuml_ms across all
* ``<prim>_heatmap.png``  — per-primitive 2D heatmap of speedup
                              (for primitives with >= 2 workload axes)
* ``<prim>_lines.png``    — per-primitive line plot of speedup vs N

The summary table is written to
``benchmarks/results/broad/SUMMARY.md`` (plain markdown).
"""
from __future__ import annotations

import json
import math
from collections import defaultdict
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

_REPO = Path(__file__).resolve().parents[3]
RESULTS = _REPO / "benchmarks" / "results" / "broad"
PLOTS = RESULTS / "plots"
PLOTS.mkdir(parents=True, exist_ok=True)

# Friendly display order for the overview plots.
PRIM_ORDER = [
    "truncated_svd", "pca", "ridge", "multinomial_nb",
    "hdbscan", "tsne", "kmeans", "knn",
    "umap", "logistic_regression", "linear_regression", "dbscan",
    "random_forest", "standard_scaler", "spectral_clustering",
]

# How to label each primitive in plots.
PRIM_LABEL = {
    "truncated_svd": "TruncatedSVD",
    "pca": "PCA",
    "ridge": "Ridge",
    "multinomial_nb": "MultinomialNB",
    "hdbscan": "HDBSCAN",
    "tsne": "t-SNE",
    "kmeans": "KMeans",
    "knn": "KNN",
    "umap": "UMAP",
    "logistic_regression": "LogReg",
    "linear_regression": "LinReg",
    "dbscan": "DBSCAN",
    "random_forest": "RandomForest",
    "standard_scaler": "StandardScaler",
    "spectral_clustering": "Spectral",
}

# Which 2 axes go on the heatmap (rows, cols); third (and beyond) is
# aggregated by median.
HEATMAP_AXES = {
    "kmeans":           ("D", "K"),
    "knn":              ("N", "D"),
    "dbscan":           ("N", "D"),
    "hdbscan":          ("N", "D"),
    "pca":              ("N", "D"),
    "truncated_svd":    ("N", "D"),
    "linear_regression": ("N", "D"),
    "ridge":            ("N", "D"),
    "logistic_regression": ("N", "D"),
    "multinomial_nb":   ("N", "V"),
    "standard_scaler":  ("N", "D"),
    "random_forest":    ("N", "D"),
    "tsne":             ("N", "D"),
    "umap":             ("N", "D"),
    "spectral_clustering": ("N", "D"),
}

# Optional per-primitive "split" axes — if set, generate one heatmap
# per distinct value of the split axis. Used to keep KNN build vs
# search visually separated.
HEATMAP_SPLIT = {
    "knn": ("regime", {
        "build":  ("N", "D"),
        "search": ("Q", "D"),
    }),
}

# Color used by per-primitive overview plots.
PRIM_COLOR = {
    "truncated_svd": "#1f77b4",
    "pca":           "#aec7e8",
    "ridge":         "#ff7f0e",
    "multinomial_nb":"#ffbb78",
    "hdbscan":       "#2ca02c",
    "tsne":          "#98df8a",
    "kmeans":        "#d62728",
    "knn":           "#ff9896",
    "umap":          "#9467bd",
    "logistic_regression": "#c5b0d5",
    "linear_regression": "#8c564b",
    "dbscan":        "#c49c94",
    "random_forest": "#e377c2",
    "standard_scaler": "#7f7f7f",
    "spectral_clustering": "#bcbd22",
}


def load_all() -> dict[str, list[dict]]:
    out: dict[str, list[dict]] = {}
    for f in sorted(RESULTS.glob("*.json")):
        if f.stem in ("SUMMARY",):
            continue
        rows = json.loads(f.read_text())
        # filter to rows with a real speedup number
        rows = [r for r in rows
                if isinstance(r.get("speedup"), (int, float))
                and math.isfinite(r["speedup"])
                and r.get("ok")]
        if rows:
            out[f.stem] = rows
    return out


# ── 1. Overview: speedup distribution per primitive ────────────────────
def plot_overview_box(data: dict[str, list[dict]]) -> None:
    prims = [p for p in PRIM_ORDER if p in data]
    speeds = [[r["speedup"] for r in data[p]] for p in prims]
    if not prims:
        return

    fig, ax = plt.subplots(figsize=(11, 7))
    positions = list(range(1, len(prims) + 1))
    bp = ax.boxplot(speeds, positions=positions, widths=0.6,
                     patch_artist=True, showfliers=False)
    for patch, p in zip(bp["boxes"], prims):
        patch.set_facecolor(PRIM_COLOR.get(p, "#7f7f7f"))
        patch.set_alpha(0.55)
        patch.set_edgecolor("#333")
    for med in bp["medians"]:
        med.set_color("#000")
        med.set_linewidth(1.5)

    rng = np.random.RandomState(0)
    for x, ys, p in zip(positions, speeds, prims):
        xs = x + (rng.rand(len(ys)) - 0.5) * 0.25
        ax.scatter(xs, ys, s=20, color=PRIM_COLOR.get(p, "#7f7f7f"),
                    alpha=0.85, edgecolors="#222", linewidths=0.4,
                    zorder=3)

    ax.set_yscale("log")
    ax.set_xticks(positions)
    ax.set_xticklabels([PRIM_LABEL[p] for p in prims], rotation=35,
                         ha="right")
    ax.axhline(1.0, color="#888", linestyle="--", linewidth=1)
    ax.text(len(prims) + 0.4, 1.0, " parity (1x)", va="center",
            color="#888", fontsize=8)
    ax.set_ylabel("Speedup vs cuML  (median of repeats, log scale)")
    ax.set_title("flashlib vs cuML — broad workload sweep "
                  f"({sum(len(v) for v in data.values())} cells)")
    ax.grid(True, axis="y", linestyle=":", alpha=0.4)
    fig.tight_layout()
    fig.savefig(PLOTS / "overview_box.png", dpi=130)
    plt.close(fig)


# ── 2. Overview: geometric-mean bar chart ──────────────────────────────
def plot_overview_bar(data: dict[str, list[dict]]) -> None:
    prims = [p for p in PRIM_ORDER if p in data]
    rows = []
    for p in prims:
        speeds = [r["speedup"] for r in data[p]]
        if not speeds:
            continue
        gm = math.exp(sum(math.log(s) for s in speeds) / len(speeds))
        rows.append((p, gm, min(speeds), max(speeds), len(speeds)))
    rows.sort(key=lambda r: r[1], reverse=True)

    fig, ax = plt.subplots(figsize=(11, 6))
    xs = np.arange(len(rows))
    gms = [r[1] for r in rows]
    bars = ax.bar(xs, gms,
                    color=[PRIM_COLOR.get(p, "#888") for p, *_ in rows],
                    edgecolor="#222")
    for x, (p, gm, mn, mx, n) in zip(xs, rows):
        ax.text(x, gm * 1.06, f"{gm:.1f}x",
                ha="center", va="bottom", fontsize=9, fontweight="bold")
        ax.text(x, gm * 0.5, f"n={n}\n[{mn:.1f}-{mx:.1f}]x",
                ha="center", va="center", fontsize=7, color="#fff",
                fontweight="bold")
    ax.set_yscale("log")
    ax.set_xticks(xs)
    ax.set_xticklabels([PRIM_LABEL[p] for p, *_ in rows],
                         rotation=35, ha="right")
    ax.axhline(1.0, color="#888", linestyle="--", linewidth=1)
    ax.set_ylabel("Geometric mean speedup vs cuML (log)")
    ax.set_title("flashlib vs cuML — geomean speedup per primitive "
                  "(broad workload sweep)")
    ax.grid(True, axis="y", linestyle=":", alpha=0.4)
    fig.tight_layout()
    fig.savefig(PLOTS / "overview_bar.png", dpi=130)
    plt.close(fig)


# ── 3. Overview scatter: flashlib_ms vs cuml_ms (loglog) ───────────────
def plot_overview_scatter(data: dict[str, list[dict]]) -> None:
    fig, ax = plt.subplots(figsize=(9, 8))
    for p in PRIM_ORDER:
        if p not in data:
            continue
        rows = data[p]
        cu = [r["cuml_ms"] for r in rows]
        fl = [r["flashlib_ms"] for r in rows]
        ax.scatter(cu, fl, s=35, color=PRIM_COLOR.get(p, "#888"),
                    edgecolors="#222", linewidths=0.4, alpha=0.85,
                    label=PRIM_LABEL[p])
    lo = 0.05
    hi = max(max((r["cuml_ms"] for rs in data.values()
                    for r in rs), default=1.0),
              max((r["flashlib_ms"] for rs in data.values()
                    for r in rs), default=1.0))
    grid = np.geomspace(lo, hi * 1.2, 100)
    for k, label in [(1, "1x"), (10, "10x"), (100, "100x")]:
        ax.plot(grid, grid / k, color="#666",
                linestyle="--", alpha=0.5, linewidth=0.8)
        ax.text(hi * 0.7, hi * 0.7 / k, label, color="#666",
                fontsize=8, va="bottom")
    ax.plot(grid, grid, color="#000", linewidth=1.2)
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlim(lo, hi * 1.2)
    ax.set_ylim(lo, hi * 1.2)
    ax.set_xlabel("cuML time (ms, log)")
    ax.set_ylabel("flashlib time (ms, log)")
    ax.set_title("Per-cell wall-time: flashlib vs cuML  (below diagonal = win)")
    ax.legend(loc="upper left", fontsize=8, ncol=2,
                framealpha=0.85)
    ax.grid(True, which="both", linestyle=":", alpha=0.3)
    fig.tight_layout()
    fig.savefig(PLOTS / "overview_scatter.png", dpi=130)
    plt.close(fig)


# ── 4. Per-primitive heatmap (2-axis) ──────────────────────────────────
def plot_per_primitive_heatmap(prim: str, rows: list[dict]) -> None:
    """Render one heatmap, or one per regime if ``HEATMAP_SPLIT`` says so."""
    if prim in HEATMAP_SPLIT:
        split_axis, regime_to_axes = HEATMAP_SPLIT[prim]
        # group rows by the split axis value
        groups: dict = defaultdict(list)
        for r in rows:
            val = r["axes"].get(split_axis, "all")
            groups[val].append(r)
        for regime_val, sub_rows in groups.items():
            ax_pair = regime_to_axes.get(regime_val)
            if ax_pair is None:
                continue
            _plot_one_heatmap(prim, sub_rows, ax_pair,
                                suffix=f"_{regime_val}",
                                title_suffix=f" [{regime_val}]")
        return
    axes_pair = HEATMAP_AXES.get(prim)
    if axes_pair is None:
        return
    _plot_one_heatmap(prim, rows, axes_pair)


def _plot_one_heatmap(prim: str, rows: list[dict], axes_pair,
                       *, suffix: str = "", title_suffix: str = "") -> None:
    ax_a, ax_b = axes_pair
    pts: dict[tuple, list[float]] = defaultdict(list)
    for r in rows:
        a = r["axes"].get(ax_a)
        b = r["axes"].get(ax_b)
        if a is None or b is None:
            continue
        pts[(a, b)].append(r["speedup"])
    if not pts:
        return
    as_ = sorted({k[0] for k in pts})
    bs_ = sorted({k[1] for k in pts})
    if len(as_) < 2 or len(bs_) < 2:
        plot_per_primitive_line(prim, rows)
        return
    Z = np.full((len(as_), len(bs_)), np.nan)
    for (a, b), vs in pts.items():
        Z[as_.index(a), bs_.index(b)] = float(np.median(vs))

    fig, ax = plt.subplots(figsize=(0.6 * len(bs_) + 4,
                                       0.5 * len(as_) + 3))
    finite = Z[~np.isnan(Z)]
    if finite.size == 0:
        plt.close(fig); return
    vmin, vmax = float(finite.min()), float(finite.max())
    im = ax.imshow(Z, aspect="auto", origin="lower",
                     cmap="RdYlGn", vmin=max(0.5, vmin),
                     vmax=vmax)
    ax.set_xticks(range(len(bs_)))
    ax.set_xticklabels([_fmt_ax(ax_b, x) for x in bs_], rotation=30,
                         ha="right")
    ax.set_yticks(range(len(as_)))
    ax.set_yticklabels([_fmt_ax(ax_a, x) for x in as_])
    ax.set_xlabel(ax_b)
    ax.set_ylabel(ax_a)
    ax.set_title(f"{PRIM_LABEL[prim]}{title_suffix} — speedup vs cuML "
                  f"(median over other axes)")
    for i in range(len(as_)):
        for j in range(len(bs_)):
            v = Z[i, j]
            if np.isnan(v):
                continue
            txt = f"{v:.1f}x"
            color = "#000" if 0.8 * vmax > v > 0.6 * vmin + 0.4 * vmax else "#000"
            ax.text(j, i, txt, ha="center", va="center", color=color,
                     fontsize=8, fontweight="bold")
    fig.colorbar(im, ax=ax, label="speedup")
    fig.tight_layout()
    fig.savefig(PLOTS / f"{prim}_heatmap{suffix}.png", dpi=130)
    plt.close(fig)


def plot_per_primitive_line(prim: str, rows: list[dict]) -> None:
    """Fallback if heatmap has only 1 distinct value on one axis.

    Plots speedup vs N, colored by the other axis (if any).
    """
    axes_pair = HEATMAP_AXES.get(prim, ("N", "D"))
    ax_a, ax_b = axes_pair
    fig, ax = plt.subplots(figsize=(8, 5))
    groups: dict = defaultdict(list)
    for r in rows:
        a = r["axes"].get(ax_a)
        b = r["axes"].get(ax_b)
        if a is None:
            continue
        groups[b].append((a, r["speedup"]))
    for b, points in sorted(groups.items(),
                              key=lambda kv: (kv[0] is None, kv[0])):
        points.sort()
        xs = [p[0] for p in points]
        ys = [p[1] for p in points]
        ax.plot(xs, ys, marker="o",
                 label=f"{ax_b}={b}" if b is not None else "")
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel(ax_a)
    ax.set_ylabel("speedup")
    ax.axhline(1.0, color="#888", linestyle="--", linewidth=1)
    ax.set_title(f"{PRIM_LABEL[prim]} — speedup vs {ax_a}")
    ax.legend(loc="best", fontsize=8)
    ax.grid(True, which="both", linestyle=":", alpha=0.4)
    fig.tight_layout()
    fig.savefig(PLOTS / f"{prim}_lines.png", dpi=130)
    plt.close(fig)


def _fmt_ax(name: str, v) -> str:
    if isinstance(v, (int, float)):
        if name == "N" and v >= 1000:
            if v >= 1_000_000:
                return f"{v/1_000_000:g}M"
            return f"{v/1000:g}K"
        if name in ("V",) and v >= 1000:
            return f"{v/1000:g}K"
        return f"{v:g}"
    return str(v)


# ── 5. Summary markdown ────────────────────────────────────────────────
def write_summary(data: dict[str, list[dict]]) -> None:
    n_total = sum(len(v) for v in data.values())
    lines = [
        "# broad sweep — SUMMARY",
        "",
        f"Generated automatically from `benchmarks/results/broad/<prim>.json`.",
        f"Total cells: **{n_total}** across **{len(data)}** primitives.",
        f"Hardware: NVIDIA H200.",
        "",
        "Plots in [`plots/`](plots/):",
        "* [`overview_box.png`](plots/overview_box.png) — per-primitive distribution",
        "* [`overview_bar.png`](plots/overview_bar.png) — geomean speedup per primitive (sorted)",
        "* [`overview_scatter.png`](plots/overview_scatter.png) — per-cell flashlib_ms vs cuml_ms",
        "* per-primitive heatmaps `<prim>_heatmap.png`",
        "* KNN is split into [`knn_heatmap_build.png`](plots/knn_heatmap_build.png) (Q=M=N self-kNN) and [`knn_heatmap_search.png`](plots/knn_heatmap_search.png) (Q << M retrieval)",
        "",
        "## Per-primitive speedup statistics",
        "",
        "| primitive | n cells | min | median | geomean | max | min cuml_ms | max cuml_ms |",
        "| --- | --- | --- | --- | --- | --- | --- | --- |",
    ]
    table_rows = []
    for prim in PRIM_ORDER:
        if prim not in data:
            continue
        rows = data[prim]
        speeds = [r["speedup"] for r in rows]
        cu = [r["cuml_ms"] for r in rows]
        gm = math.exp(sum(math.log(s) for s in speeds) / len(speeds))
        table_rows.append((prim, len(rows), min(speeds),
                            sorted(speeds)[len(speeds) // 2], gm,
                            max(speeds), min(cu), max(cu)))
    table_rows.sort(key=lambda r: r[4], reverse=True)
    for prim, n, mn, md, gm, mx, cu_mn, cu_mx in table_rows:
        lines.append(f"| {PRIM_LABEL[prim]} | {n} | {mn:.2f}x | {md:.2f}x | "
                       f"**{gm:.2f}x** | {mx:.2f}x | {cu_mn:.2f} | {cu_mx:.2f} |")
    lines.append("")
    lines.append("## Headline takeaways")
    lines.append("")
    top3 = table_rows[:3]
    lines.append(f"* Top-3 by geomean speedup: " + ", ".join(
        f"**{PRIM_LABEL[p]}** ({gm:.1f}x)"
        for p, _, _, _, gm, *_ in top3))
    win = sum(1 for rs in data.values() for r in rs if r["speedup"] >= 1.0)
    big = sum(1 for rs in data.values() for r in rs if r["speedup"] >= 5.0)
    huge = sum(1 for rs in data.values() for r in rs if r["speedup"] >= 50.0)
    lines.append(f"* {win}/{n_total} cells ({win/n_total*100:.0f}%) "
                  f"have flashlib >= cuML.")
    lines.append(f"* {big}/{n_total} cells ({big/n_total*100:.0f}%) "
                  f">= 5x speedup.")
    lines.append(f"* {huge}/{n_total} cells ({huge/n_total*100:.0f}%) "
                  f">= 50x speedup.")
    (RESULTS / "SUMMARY.md").write_text("\n".join(lines) + "\n")


def main() -> None:
    data = load_all()
    if not data:
        print("No broad results found.")
        return
    print(f"Loaded {len(data)} primitives, "
          f"{sum(len(v) for v in data.values())} cells total")
    plot_overview_box(data)
    plot_overview_bar(data)
    plot_overview_scatter(data)
    for prim, rows in data.items():
        try:
            plot_per_primitive_heatmap(prim, rows)
        except Exception as e:
            print(f"  [{prim}] heatmap failed: {e}; trying line plot")
            try:
                plot_per_primitive_line(prim, rows)
            except Exception as e2:
                print(f"  [{prim}] line plot also failed: {e2}")
    write_summary(data)
    print(f"Wrote plots to {PLOTS}")
    print(f"Wrote SUMMARY.md to {RESULTS}")


if __name__ == "__main__":
    main()
