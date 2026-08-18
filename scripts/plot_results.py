"""Generate benchmark plots for the HNSW C++ implementation.

Reads the CSV files produced by bench/benchmark.cpp and
bench/run_large_scale.sh (run from the repo root) and renders five charts
to plots/:

  1. dataset_size_latency.png   -- brute force vs HNSW query latency as N grows
  2. ef_search_recall_latency.png -- recall vs query latency as ef_search varies
  3. ef_search_recall_qps.png    -- recall vs throughput as ef_search varies
  4. m_sweep_recall.png          -- recall as M varies
  5. m_sweep_build_time.png      -- build time as M varies
"""
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
PLOTS_DIR = ROOT / "plots"

# Palette -- see dataviz skill references/palette.md.
BLUE = "#2a78d6"      # primary series (HNSW)
ORANGE = "#eb6834"    # secondary series (brute force)
INK_PRIMARY = "#0b0b0b"
INK_SECONDARY = "#52514e"
INK_MUTED = "#898781"
GRID = "#e1e0d9"
SURFACE = "#fcfcfb"

plt.rcParams.update({
    "figure.facecolor": SURFACE,
    "axes.facecolor": SURFACE,
    "savefig.facecolor": SURFACE,
    "axes.edgecolor": INK_MUTED,
    "axes.labelcolor": INK_PRIMARY,
    "text.color": INK_PRIMARY,
    "xtick.color": INK_SECONDARY,
    "ytick.color": INK_SECONDARY,
    "font.size": 11,
    "font.family": "sans-serif",
    "legend.frameon": False,
})


def _style_axes(ax):
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_color(INK_MUTED)
    ax.spines["bottom"].set_color(INK_MUTED)
    ax.grid(True, color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)


def _titled(ax, title, caption=None):
    """Set a bold title with an optional smaller, muted caption line below it."""
    ax.set_title(title, fontsize=13, color=INK_PRIMARY, pad=24, loc="left")
    if caption:
        ax.text(0.0, 1.03, caption, transform=ax.transAxes, fontsize=9.5,
                 color=INK_SECONDARY, ha="left", va="bottom")


def _annotate_points(ax, df, x_col, y_col, label_col):
    for _, row in df.iterrows():
        ax.annotate(
            f"{label_col}={int(row[label_col])}",
            (row[x_col], row[y_col]),
            xytext=(6, 6),
            textcoords="offset points",
            fontsize=9,
            color=INK_SECONDARY,
        )


def _save(fig, name):
    PLOTS_DIR.mkdir(parents=True, exist_ok=True)
    out = PLOTS_DIR / name
    fig.tight_layout()
    fig.savefig(out, dpi=200)
    plt.close(fig)
    print(f"wrote {out.relative_to(ROOT)}")


def plot_dataset_size_latency():
    # bench/benchmark_large_scale.cpp / bench/run_large_scale.sh: N up to 1M,
    # dim=64, k=10, M=16, ef_construction=200, two ef_search rows per N.
    df = pd.read_csv(ROOT / "benchmark_large_scale.csv")

    fig, ax = plt.subplots(figsize=(8, 5))

    bf = df.drop_duplicates("n").sort_values("n")
    ax.plot(bf["n"], bf["bf_avg_query_ms"], marker="o", ms=8, lw=2,
            color=ORANGE, label="Brute force")

    for ef_search, linestyle in zip(sorted(df["ef_search"].unique()), ["-", "--"]):
        sub = df[df["ef_search"] == ef_search].sort_values("n")
        ax.plot(sub["n"], sub["hnsw_avg_query_ms"], marker="o", ms=8, lw=2,
                color=BLUE, linestyle=linestyle,
                label=f"HNSW (ef_search={ef_search})")

    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("Dataset size (n)")
    ax.set_ylabel("Avg query latency (ms)")
    _titled(ax, "Query Latency: Brute Force vs HNSW",
            "HNSW fixed at M=16, ef_construction=200; N up to 1M")
    ax.legend(loc="upper left")
    _style_axes(ax)
    _save(fig, "dataset_size_latency.png")


def plot_ef_search_recall_latency():
    df = pd.read_csv(ROOT / "bench_ef_search_100k.csv")

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(df["avg_query_ms"], df["avg_recall"], marker="o", ms=8, lw=2,
            color=BLUE)
    _annotate_points(ax, df, "avg_query_ms", "avg_recall", "ef_search")
    ax.set_xlabel("Avg query latency (ms)")
    ax.set_ylabel("Recall@10")
    ax.set_title("HNSW Recall vs Query Latency (N=100k)")
    _style_axes(ax)
    _save(fig, "ef_search_recall_latency.png")


def plot_ef_search_recall_qps():
    df = pd.read_csv(ROOT / "bench_ef_search_100k.csv")

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(df["qps"], df["avg_recall"], marker="o", ms=8, lw=2, color=BLUE)
    _annotate_points(ax, df, "qps", "avg_recall", "ef_search")
    ax.set_xlabel("Queries per second (QPS)")
    ax.set_ylabel("Recall@10")
    ax.set_title("HNSW Recall vs Throughput (N=100k)")
    _style_axes(ax)
    _save(fig, "ef_search_recall_qps.png")


def plot_m_recall():
    df = pd.read_csv(ROOT / "bench_m_sweep.csv")

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(df["M"], df["avg_recall"], marker="o", ms=8, lw=2, color=BLUE)
    ax.set_xlabel("M (max neighbors per node)")
    ax.set_ylabel("Recall@10")
    _titled(ax, "M / Construction Setting Sweep — Recall (N=50k)",
            "ef_construction scales with M (= 4×M); ef_search=50 (fixed)")
    ax.set_xticks(df["M"])
    _style_axes(ax)
    _save(fig, "m_sweep_recall.png")


def plot_m_build_time():
    df = pd.read_csv(ROOT / "bench_m_sweep.csv")

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(df["M"], df["build_ms"] / 1000.0, marker="o", ms=8, lw=2,
            color=BLUE)
    ax.set_xlabel("M (max neighbors per node)")
    ax.set_ylabel("Build time (s)")
    _titled(ax, "M / Construction Setting Sweep — Build Time (N=50k)",
            "ef_construction scales with M (= 4×M); ef_search=50 (fixed)")
    ax.set_xticks(df["M"])
    _style_axes(ax)
    _save(fig, "m_sweep_build_time.png")


def main():
    plot_dataset_size_latency()
    plot_ef_search_recall_latency()
    plot_ef_search_recall_qps()
    plot_m_recall()
    plot_m_build_time()


if __name__ == "__main__":
    main()
