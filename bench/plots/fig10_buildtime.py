#!/usr/bin/env python3
"""Index build time figure: 5 systems x the Deep100K dataset (paper Fig. 10).

Reads build_seconds from the pagesize_8KB index_sizes CSVs of the main
experiment (serial build, maintenance_work_mem=4GB, direct I/O) and renders
the single-column grouped bar chart in the same style as the main result figure. Colors,
patterns, and legend order are shared with it; bars are normalized per
dataset to Baseline = 1.

Example
-------
python3 scripts/fig06_build_time_bar.py
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import fig07_09_main_results as base
from fig07_09_main_results import (
    DATASET_TICKS,
    OUTPUT_DIR,
    RESULT_DIR,
    SERIES,
    add_average_group,
    add_pattern,
    apply_style,
    normalize_to_baseline,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Render the index build time grouped bar figure."
    )
    parser.add_argument("--result-dir", type=Path, default=RESULT_DIR)
    parser.add_argument("--datasets", nargs="+", default=list(base.DATASETS),
                        help="datasets to include")
    parser.add_argument("--output-dir", type=Path, default=OUTPUT_DIR)
    parser.add_argument("--source-dir", type=Path,
                        default=OUTPUT_DIR.parent / "csv",
                        help="directory for the per-figure source-data CSV")
    parser.add_argument(
        "--page-size",
        default="3.5039,2.7559",
        help="Figure page size in inches as WIDTH,HEIGHT",
    )
    return parser.parse_args()


def load_build_data(result_dir: Path) -> pd.DataFrame:
    """Load one build-time observation per dataset and variant."""
    records: list[dict[str, object]] = []
    for variant, _unused, label, _, _ in SERIES:
        csv_path = result_dir / "index_sizes" / f"{variant}_index_sizes.csv"
        if not csv_path.is_file():
            raise FileNotFoundError(f"Missing index-size CSV: {csv_path}")
        frame = pd.read_csv(csv_path)
        for dataset in base.DATASETS:
            selected = frame.loc[frame["dataset"] == dataset]
            if len(selected) != 1:
                raise ValueError(
                    f"Expected one {dataset} row in {csv_path.name}, "
                    f"found {len(selected)}"
                )
            row = selected.iloc[0]
            commit_short = str(row["commit"])[:4]
            if bool(row["reused_existing"]):
                raise ValueError(f"Index for {dataset} was reused, not rebuilt")
            value = float(row["build_seconds"])
            if not pd.notna(value) or value <= 0:
                raise ValueError(f"Invalid build_seconds for {dataset} in {csv_path.name}")
            records.append(
                {
                    "dataset": dataset,
                    "variant": variant,
                    "method": label,
                    "commit_short": commit_short,
                    "commit_full": str(row["commit"]),
                    "metric": "build_seconds",
                    "value": value,
                }
            )

    data = pd.DataFrame.from_records(records)
    expected = len(base.DATASETS) * len(SERIES)
    if len(data) != expected:
        raise AssertionError(
            f"Expected {expected} plotted observations, found {len(data)}"
        )
    return data


def plot_build_bars(
    data: pd.DataFrame, figsize: tuple[float, float]
) -> plt.Figure:
    apply_style()
    fig, ax = plt.subplots(figsize=figsize)
    x = np.arange(len(base.PLOT_GROUPS), dtype=float)
    bar_width = 0.13
    offsets = (np.arange(len(SERIES)) - (len(SERIES) - 1) / 2) * bar_width
    ymax = float(data["value_normalized"].max())

    for offset, (_, _, label, color, pattern) in zip(offsets, SERIES):
        subset = (
            data.loc[data["method"] == label]
            .set_index("dataset")
            .loc[list(base.PLOT_GROUPS)]
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
    ax.set_xticks(x, [DATASET_TICKS[dataset] for dataset in base.PLOT_GROUPS])
    if len(base.PLOT_GROUPS) == 1:
        # single-dataset artifact: keep the paper's data-per-inch so the bars
        # are as slim as in the five-group paper figure
        ax.set_xlim(-2.25, 2.25)
    ax.set_ylabel("Construction time")
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
    args = parse_args()
    base.DATASETS = tuple(args.datasets)
    base.PLOT_GROUPS = (tuple(args.datasets) + ("Avg",)) if len(args.datasets) > 1 else tuple(args.datasets)
    result_dir = args.result_dir.resolve()
    output_dir = args.output_dir.resolve()
    source_dir = args.source_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    source_dir.mkdir(parents=True, exist_ok=True)

    data = add_average_group(normalize_to_baseline(load_build_data(result_dir)))
    stem = output_dir / "fig10_build_time_bar"
    data.to_csv(source_dir / "fig10_build_time_bar_source_data.csv", index=False)

    width, height = (float(part) for part in args.page_size.split(","))
    fig = plot_build_bars(data, figsize=(width, height))
    fig.savefig(f"{stem}.png", dpi=600, bbox_inches=None)
    plt.close(fig)
    print(stem)


if __name__ == "__main__":
    main()
