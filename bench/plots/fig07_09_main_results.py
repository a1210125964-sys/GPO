#!/usr/bin/env python3
"""Main read-benchmark result figure: 5 systems x the Deep100K dataset (paper Figs. 7-9).

Reads the pagesize_8KB as-run CSVs of the main experiment (cold-cache
single-pass, ef_search=200, buffer tier = baseline 6c5d index size x 50%)
and renders the single-column grouped bar chart used as the paper's main
read-performance result.  Every bar is a measured value; no style
placeholders.  Bars are normalized per dataset to Baseline = 1.

Examples
--------
python3 bench/plots/fig07_09_main_results.py --metric execution_qps
python3 bench/plots/fig07_09_main_results.py --metric latency_p95_ms
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


RESULT_DIR = Path(__file__).resolve().parents[2] / "results" / "main" / "pagesize_8KB"
OUTPUT_DIR = Path(__file__).resolve().parents[2] / "results" / "figures"

DATASETS = ("Deep100K",)  # single-dataset artifact; pass --datasets for more
PLOT_GROUPS = DATASETS + ("Avg",) if len(DATASETS) > 1 else DATASETS
DATASET_TICKS = {
    "Deep100K": "Deep",
    "SIFT100K": "SIFT",
    "Glove100K": "Glove",
    "NYTimes100K": "NYTimes",
    "Avg": "Avg",
}
# Unit-free labels: all bars are normalized to Baseline = 1.
METRIC_LABELS = {
    "execution_qps": "QPS",
    "wall_qps": "Wall QPS",
    "latency_p50_ms": "Median latency",
    "latency_p95_ms": "P95 latency",
    "latency_p99_ms": "P99 latency",
    "buffer_hit_rate": "Buffer hit rate",
}

# (variant dir, commit short, legend label, fill color, bar pattern)
# Legend and bar order, left to right: Baseline, TurboCharging, SEN, ATL, GPO.
# Colors: Baseline light gray / TurboCharging orange / SEN-ATL-GPO a
# light-to-dark blue ramp (SEN lightest, GPO darkest).
# (variant, commit_short, label, fill color, hatch) — commit_short is resolved
# from the result-directory name at load time, never hardcoded.
SERIES = (
    ("baseline", None, "Baseline", "#D3D3D3", "forward"),
    ("TurboCharging", None, "TurboCharging", "#E6A05A", "horizontal"),
    ("SEN", None, "SEN", "#9DC3E6", "backward"),
    ("ATL", None, "ATL", "#5B9BD5", "cross"),
    ("GPO", None, "GPO", "#1F4E79", "dots"),
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Render the main read-benchmark grouped bar figure."
    )
    parser.add_argument("--result-dir", type=Path, default=RESULT_DIR)
    parser.add_argument("--output-dir", type=Path, default=OUTPUT_DIR)
    parser.add_argument("--source-dir", type=Path,
                        default=OUTPUT_DIR.parent / "csv",
                        help="directory for the per-figure source-data CSV")
    parser.add_argument("--metric", default="execution_qps")
    parser.add_argument("--datasets", nargs="+", default=list(DATASETS),
                        help="datasets to include")
    parser.add_argument("--ef-search", type=int, default=200)
    parser.add_argument(
        "--page-size",
        default="3.5039,2.7559",
        help="Figure page size in inches as WIDTH,HEIGHT",
    )
    return parser.parse_args()


def apply_style() -> None:
    # Text sizes are 5% larger than the original pass (7.0/7.5/8.0 family),
    # applied uniformly across the main and sensitivity bar figures.
    mpl.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans", "sans-serif"],
            "font.size": 7.35,
            "axes.labelsize": 7.875,
            "axes.spines.right": True,
            "axes.spines.top": True,
            "axes.linewidth": 0.7,
            "xtick.labelsize": 8.4,
            "ytick.labelsize": 7.35,
            "xtick.major.size": 3.0,
            "ytick.major.size": 3.0,
            "xtick.major.width": 0.7,
            "ytick.major.width": 0.7,
            "xtick.direction": "in",
            "ytick.direction": "in",
            "legend.frameon": False,
            "legend.fontsize": 7.875,
            "pdf.fonttype": 42,
            "svg.fonttype": "none",
        }
    )


def baseline_buffer_50pct(result_dir: Path, dataset: str) -> int:
    """Return the experiment's baseline-defined 50% buffer size in MB."""
    metadata_path = result_dir / "index_sizes" / "baseline_index_sizes.csv"
    metadata = pd.read_csv(metadata_path)
    match = metadata.loc[metadata["dataset"] == dataset, "buffer_50pct_mb"]
    if len(match) != 1:
        raise ValueError(
            f"Expected one 50% buffer entry for {dataset}, found {len(match)}"
        )
    return int(match.iloc[0])


def load_plot_data(result_dir: Path, metric: str, ef_search: int) -> pd.DataFrame:
    """Load one observation per dataset and variant for the requested setting."""
    records: list[dict[str, object]] = []
    for dataset in DATASETS:
        buffer_mb = baseline_buffer_50pct(result_dir, dataset)
        for variant, _unused, label, _, _ in SERIES:
            combo_dirs = sorted((result_dir / variant).glob(f"*_{dataset}"))
            if len(combo_dirs) != 1:
                raise FileNotFoundError(
                    f"Expected exactly one combo dir {variant}/*_{dataset}, "
                    f"found {len(combo_dirs)}")
            commit_short = combo_dirs[0].name.split("_")[0]
            csv_path = (
                combo_dirs[0] / f"read_bench_BufferSize{buffer_mb}MB.csv"
            )
            if not csv_path.is_file():
                raise FileNotFoundError(f"Missing benchmark CSV: {csv_path}")

            frame = pd.read_csv(csv_path)
            if metric not in frame.columns:
                raise ValueError(f"Metric {metric!r} is absent from {csv_path.name}")
            selected = frame.loc[frame["ef_search"] == ef_search]
            if len(selected) != 1:
                raise ValueError(
                    f"Expected one ef_search={ef_search} row in {csv_path.name}, "
                    f"found {len(selected)}"
                )
            row = selected.iloc[0]
            value = float(row[metric])
            if not np.isfinite(value):
                raise ValueError(f"Non-finite {metric} in {csv_path.name}")
            records.append(
                {
                    "dataset": dataset,
                    "variant": variant,
                    "method": label,
                    "commit_short": commit_short,
                    "commit_full": str(row["commit"]),
                    "buffer_fraction": 0.50,
                    "buffer_mb": buffer_mb,
                    "ef_search": ef_search,
                    "metric": metric,
                    "value": value,
                }
            )

    data = pd.DataFrame.from_records(records)
    expected = len(DATASETS) * len(SERIES)
    if len(data) != expected:
        raise AssertionError(
            f"Expected {expected} plotted observations, found {len(data)}"
        )
    return data


def metric_label(metric: str) -> str:
    return METRIC_LABELS.get(metric, metric.replace("_", " ").capitalize())


def normalize_to_baseline(data: pd.DataFrame) -> pd.DataFrame:
    """Divide each value by its dataset's Baseline value (Baseline -> 1)."""
    normalized = data.copy()
    baseline = (
        normalized.loc[normalized["variant"] == "baseline"]
        .set_index("dataset")["value"]
        .rename("baseline_value")
    )
    if len(baseline) != len(DATASETS):
        raise ValueError(
            f"Expected one Baseline observation per dataset, found {len(baseline)}"
        )
    normalized = normalized.join(baseline, on="dataset")
    normalized["value_normalized"] = normalized["value"] / normalized["baseline_value"]
    return normalized


def add_average_group(data: pd.DataFrame) -> pd.DataFrame:
    """Append an 'Avg' group: per-method mean over the plotted datasets.

    The plotted height is the mean of the per-dataset normalized values,
    so the Baseline Avg bar stays exactly 1.
    """
    meta_cols = [
        col
        for col in data.columns
        if col
        not in ("dataset", "variant", "value", "baseline_value", "value_normalized")
    ]
    meta = data.drop_duplicates("variant").set_index("variant")[meta_cols]
    averaged = (
        data.groupby("variant", sort=False)
        .agg(value=("value", "mean"), value_normalized=("value_normalized", "mean"))
        .reset_index()
        .join(meta, on="variant")
    )
    averaged["dataset"] = "Avg"
    averaged["baseline_value"] = float(
        averaged.loc[averaged["variant"] == "baseline", "value"].iloc[0]
    )
    return pd.concat([data, averaged[list(data.columns)]], ignore_index=True)


def diagonal_segments(
    left: float,
    right: float,
    height: float,
    ymax: float,
) -> list[tuple[float, float, float, float]]:
    """Return sparse forward-diagonal segments contained inside a rectangle."""
    rise = ymax * 0.060
    step = ymax * 0.085
    slope = rise / (right - left)
    segments: list[tuple[float, float, float, float]] = []
    intercept = -rise
    while intercept <= height:
        x0, y0 = left, intercept
        x1, y1 = right, intercept + rise
        if y0 < 0:
            x0 = left + (-intercept) / slope
            y0 = 0.0
        if y1 > height:
            x1 = left + (height - intercept) / slope
            y1 = height
        if x0 <= x1 and y1 >= 0:
            segments.append((x0, y0, x1, y1))
        intercept += step
    return segments


def add_pattern(
    ax: plt.Axes,
    bar: mpl.patches.Rectangle,
    pattern: str,
    ymax: float,
) -> None:
    """Draw a local vector pattern without page-wide PDF hatch resources."""
    left = float(bar.get_x())
    right = left + float(bar.get_width())
    height = float(bar.get_height())
    # Dark fills (e.g. the darkest blue) need light strokes to stay readable.
    red, green, blue, _ = mpl.colors.to_rgba(bar.get_facecolor())
    luminance = 0.2126 * red + 0.7152 * green + 0.0722 * blue
    edge = "#FFFFFF" if luminance < 0.40 else "#3A3A3A"

    if pattern in {"forward", "backward", "cross"}:
        directions = {
            "forward": ("forward",),
            "backward": ("backward",),
            "cross": ("forward", "backward"),
        }[pattern]
        for x0, y0, x1, y1 in diagonal_segments(left, right, height, ymax):
            for direction in directions:
                px0, px1 = x0, x1
                if direction == "backward":
                    px0, px1 = left + right - x0, left + right - x1
                ax.plot(
                    [px0, px1],
                    [y0, y1],
                    color=edge,
                    linewidth=0.38,
                    solid_capstyle="butt",
                    zorder=4,
                )
    elif pattern == "dots":
        # Compact staggered dot hatch: two dots on one row, one centred dot
        # on the next, with a small and regular pitch.
        y_step = ymax * 0.028
        ys = np.arange(y_step * 0.90, height - y_step * 0.35, y_step)
        for row_index, ypos in enumerate(ys):
            fractions = (0.29, 0.71) if row_index % 2 == 0 else (0.50,)
            xs = left + (right - left) * np.asarray(fractions)
            ax.scatter(
                xs,
                np.full_like(xs, ypos),
                s=0.75,
                color=edge,
                linewidths=0,
                zorder=4,
            )
    elif pattern == "horizontal":
        y_step = ymax * 0.075
        for ypos in np.arange(y_step * 0.6, height, y_step):
            ax.plot([left, right], [ypos, ypos], color=edge, linewidth=0.38, zorder=4)


def plot_main_bars(data: pd.DataFrame, metric: str, figsize: tuple[float, float]) -> plt.Figure:
    apply_style()
    fig, ax = plt.subplots(figsize=figsize)
    x = np.arange(len(PLOT_GROUPS), dtype=float)
    bar_width = 0.13
    offsets = (np.arange(len(SERIES)) - (len(SERIES) - 1) / 2) * bar_width
    ymax = float(data["value_normalized"].max())

    for offset, (_, _, label, color, pattern) in zip(offsets, SERIES):
        subset = (
            data.loc[data["method"] == label]
            .set_index("dataset")
            .loc[list(PLOT_GROUPS)]
        )
        bars = ax.bar(
            x + offset,
            subset["value_normalized"].to_numpy(),
            width=bar_width,
            color=color,
            edgecolor="#333333",
            linewidth=0.50,
            label=label,
            zorder=3,
        )
        for bar in bars:
            add_pattern(ax, bar, pattern, ymax)

    ax.set_ylim(0, ymax * 1.14 if ymax > 0 else 1.0)
    # Restrict y ticks to the 1/2/5 decimal series (one-decimal labels):
    # matplotlib's extended steps include 0.25/0.4/0.8, which creep in when
    # a metric's range shrinks and double the tick count with two-decimal
    # labels (e.g. buffer hit rate peaks at 1.58 normalized).
    ax.yaxis.set_major_locator(
        mpl.ticker.MaxNLocator(nbins=6, steps=[1, 2, 5, 10])
    )
    ax.set_xticks(x, [DATASET_TICKS[dataset] for dataset in PLOT_GROUPS])
    if len(PLOT_GROUPS) == 1:
        # single-dataset artifact: keep the paper's data-per-inch so the bars
        # are as slim as in the five-group paper figure
        ax.set_xlim(-2.25, 2.25)
    ax.set_ylabel(metric_label(metric))
    ax.set_xlabel("Datasets")
    ax.grid(False)
    ax.tick_params(top=True, right=True, labeltop=False, labelright=False)
    ax.legend(
        loc="lower center",
        bbox_to_anchor=(0.5, 1.02),
        ncol=len(SERIES),
        handlelength=0.8,
        handletextpad=0.35,
        columnspacing=0.60,
        borderaxespad=0.0,
    )
    fig.subplots_adjust(left=0.19, right=0.97, bottom=0.18, top=0.82)
    return fig


def main() -> None:
    global DATASETS, PLOT_GROUPS
    args = parse_args()
    DATASETS = tuple(args.datasets)
    PLOT_GROUPS = DATASETS + ("Avg",) if len(DATASETS) > 1 else DATASETS
    result_dir = args.result_dir.resolve()
    output_dir = args.output_dir.resolve()
    source_dir = args.source_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    source_dir.mkdir(parents=True, exist_ok=True)

    data = add_average_group(
        normalize_to_baseline(load_plot_data(result_dir, args.metric, args.ef_search))
    )
    FIG_BY_METRIC = {"execution_qps": "fig07_result_qps", "buffer_hit_rate": "fig08_result_hitrate", "latency_p95_ms": "fig09_result_latency_p95"}
    stem = output_dir / f"{FIG_BY_METRIC[args.metric]}_ef{args.ef_search}"
    data.to_csv(source_dir / f"{FIG_BY_METRIC[args.metric]}_ef{args.ef_search}_source_data.csv", index=False)

    width, height = (float(part) for part in args.page_size.split(","))
    fig = plot_main_bars(data, args.metric, figsize=(width, height))
    fig.savefig(f"{stem}.png", dpi=600, bbox_inches=None)
    plt.close(fig)
    print(stem)


if __name__ == "__main__":
    main()
