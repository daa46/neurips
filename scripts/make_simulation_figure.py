#!/usr/bin/env python3
"""Build the four-panel simulation figure used in the HRMS paper.

The script consumes only frozen CSV summaries.  It deliberately contains no
simulation or result-selection logic, so regenerating the figure cannot change
the reported experiment.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.axes import Axes
from matplotlib.figure import Figure
from matplotlib.ticker import MaxNLocator

# Okabe--Ito colors, supplemented by neutral gray.  Line styles and markers
# also differ so that the figure remains interpretable in grayscale.
BLUE = "#0072B2"
ORANGE = "#E69F00"
GREEN = "#009E73"
VERMILLION = "#D55E00"
PURPLE = "#CC79A7"
SKY = "#56B4E9"
GRAY = "#6B6B6B"
INK = "#252525"

METHOD_STYLE = {
    "IID marginal": (GRAY, "o", ":"),
    "Cluster marginal": (SKY, "D", "-."),
    "Paired cluster max-t": (VERMILLION, "s", "--"),
    "Paired cluster jackknife bootstrap-t": (BLUE, "o", "-"),
    "Paired cluster bootstrap-t": (GREEN, "^", "-."),
    "Separate-risk bootstrap-t": (ORANGE, "v", "--"),
    "Mean validation": (VERMILLION, "s", "--"),
    "Point robust regret": (GREEN, "^", "-."),
    "HRMS (rho=0)": (PURPLE, "D", ":"),
    "HRMS": (BLUE, "o", "-"),
}

COVERAGE_LABELS = {
    "IID marginal": "IID marginal",
    "Cluster marginal": "Cluster marginal",
    "Paired cluster max-t": r"Fixed max-$t$",
    "Paired cluster jackknife bootstrap-t": r"HRMS jackknife-$t$",
    "Paired cluster bootstrap-t": r"Raw bootstrap-$t$",
}

WIDTH_LABELS = {
    "Paired cluster max-t": r"Fixed max-$t$",
    "Paired cluster jackknife bootstrap-t": r"Jackknife bootstrap-$t$",
    "Paired cluster bootstrap-t": r"Raw bootstrap-$t$",
    "Separate-risk bootstrap-t": r"Separate-risk bootstrap-$t$",
}

REGRET_LABELS = {
    "Mean validation": "Mean validation",
    "Point robust regret": "Point robust regret",
    "HRMS (rho=0)": r"HRMS ($\rho=0$)",
    "HRMS": "HRMS",
}


def _require_columns(table: pd.DataFrame, required: set[str], source: Path) -> None:
    missing = required.difference(table.columns)
    if missing:
        raise ValueError(f"{source} is missing required columns: {sorted(missing)}")


def _select_complete_methods(
    table: pd.DataFrame,
    methods: list[str],
    *,
    source: Path,
    x_column: str,
) -> pd.DataFrame:
    """Select methods in display order and reject missing or duplicate cells."""

    missing = [method for method in methods if method not in set(table["method"])]
    if missing:
        raise ValueError(f"{source} is missing required methods: {missing}")
    selected = table[table["method"].isin(methods)].copy()
    duplicates = selected.duplicated([x_column, "method"], keep=False)
    if duplicates.any():
        cells = selected.loc[duplicates, [x_column, "method"]].to_dict("records")
        raise ValueError(f"{source} has duplicate method cells: {cells}")
    expected_x = set(selected[x_column].dropna().unique())
    for method in methods:
        observed_x = set(selected.loc[selected["method"] == method, x_column].dropna())
        if observed_x != expected_x:
            raise ValueError(
                f"{source} has an incomplete {x_column} grid for {method!r}: "
                f"expected {sorted(expected_x)}, observed {sorted(observed_x)}"
            )
    return selected


def _panel_label(axis: Axes, label: str, title: str, *, label_x: float = -0.13) -> None:
    axis.text(
        label_x,
        1.14,
        label,
        transform=axis.transAxes,
        ha="left",
        va="top",
        fontsize=10.0,
        fontweight="bold",
        clip_on=False,
    )
    axis.text(
        -0.01,
        1.14,
        title,
        transform=axis.transAxes,
        ha="left",
        va="top",
        fontsize=9.0,
        fontweight="bold",
        clip_on=False,
    )


def _plot_series(
    axis: Axes,
    group: pd.DataFrame,
    *,
    x: str,
    mean: str,
    sem: str,
    method: str,
    label: str,
    interval: str = "errorbar",
) -> None:
    color, marker, linestyle = METHOD_STYLE[method]
    ordered = group.sort_values(x)
    x_values = ordered[x].to_numpy(dtype=float)
    means = ordered[mean].to_numpy(dtype=float)
    sems = ordered[sem].fillna(0.0).to_numpy(dtype=float)
    if interval == "band":
        axis.fill_between(
            x_values,
            means - 1.96 * sems,
            means + 1.96 * sems,
            color=color,
            alpha=0.13,
            linewidth=0,
            zorder=1,
        )
        axis.plot(
            x_values,
            means,
            color=color,
            marker=marker,
            linestyle=linestyle,
            label=label,
            zorder=2,
        )
    else:
        lower = np.minimum(1.96 * sems, means)
        upper = np.minimum(1.96 * sems, 1.0 - means)
        axis.errorbar(
            x_values,
            means,
            yerr=np.vstack([lower, upper]),
            color=color,
            marker=marker,
            linestyle=linestyle,
            capsize=1.8,
            capthick=0.75,
            elinewidth=0.8,
            label=label,
            zorder=2,
        )


def _style_axis(axis: Axes) -> None:
    axis.grid(axis="y", color="#E4E4E4", linewidth=0.55, zorder=0)
    axis.spines["top"].set_visible(False)
    axis.spines["right"].set_visible(False)
    axis.tick_params(length=2.5, width=0.7)


def build_simulation_figure(
    coverage_summary: str | Path,
    decision_summary: str | Path,
) -> Figure:
    """Return the deterministic four-panel simulation figure."""

    coverage_path = Path(coverage_summary)
    decision_path = Path(decision_summary)
    coverage = pd.read_csv(coverage_path)
    decision = pd.read_csv(decision_path)

    _require_columns(
        coverage,
        {
            "icc",
            "method",
            "joint_coverage_mean",
            "joint_coverage_sem",
            "mean_interval_width_mean",
            "mean_interval_width_sem",
        },
        coverage_path,
    )
    _require_columns(
        decision,
        {
            "shift_strength",
            "method",
            "selected_true_robust_regret_mean",
            "selected_true_robust_regret_sem",
            "confidence_set_size_mean",
            "confidence_set_size_sem",
            "confidence_set_covers_mean",
            "confidence_set_covers_sem",
        },
        decision_path,
    )

    coverage_methods = list(COVERAGE_LABELS)
    width_methods = list(WIDTH_LABELS)
    regret_methods = list(REGRET_LABELS)
    coverage_subset = _select_complete_methods(
        coverage, coverage_methods, source=coverage_path, x_column="icc"
    )
    width_subset = _select_complete_methods(
        coverage, width_methods, source=coverage_path, x_column="icc"
    )
    regret_subset = _select_complete_methods(
        decision, regret_methods, source=decision_path, x_column="shift_strength"
    )
    hrms = decision.loc[decision["method"] == "HRMS"].sort_values("shift_strength")
    if hrms.empty:
        raise ValueError(f"{decision_path} is missing required method: HRMS")
    if hrms["shift_strength"].duplicated().any():
        raise ValueError(f"{decision_path} has duplicate HRMS shift-strength cells")

    style = {
        "font.family": "sans-serif",
        "font.sans-serif": ["DejaVu Sans"],
        # The source PDF is scaled to about 83% at NeurIPS text width.  These
        # sizes therefore render at roughly 6--7 pt for ticks/legends and
        # 7--8 pt for labels/titles in the paper rather than only 5--6 pt.
        "font.size": 8.3,
        "axes.labelsize": 8.6,
        "axes.linewidth": 0.7,
        "xtick.labelsize": 7.8,
        "ytick.labelsize": 7.8,
        "legend.fontsize": 7.3,
        "lines.linewidth": 1.35,
        "lines.markersize": 3.5,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
    }
    with plt.rc_context(style):
        figure, axes = plt.subplots(
            2,
            2,
            # A deliberately shallow aspect ratio: at NeurIPS text width the
            # rendered figure occupies roughly 2.4 inches vertically.
            figsize=(6.9, 2.95),
            constrained_layout=False,
            gridspec_kw={"hspace": 0.72, "wspace": 0.31},
        )
        axis_a, axis_b, axis_c, axis_d = axes.flat

        for method in coverage_methods:
            _plot_series(
                axis_a,
                coverage_subset[coverage_subset["method"] == method],
                x="icc",
                mean="joint_coverage_mean",
                sem="joint_coverage_sem",
                method=method,
                label=COVERAGE_LABELS[method],
            )
        axis_a.axhline(0.95, color=INK, linewidth=0.8, linestyle=(0, (2, 2)), zorder=1)
        axis_a.text(
            0.985,
            0.928,
            "95% nominal",
            transform=axis_a.transAxes,
            ha="right",
            va="top",
            color=INK,
            fontsize=7.3,
        )
        axis_a.set(
            xlabel="Latent-Gaussian ICC parameter",
            ylabel="Joint coverage",
            ylim=(-0.03, 1.03),
        )
        axis_a.set_yticks([0.0, 0.25, 0.5, 0.75, 1.0])
        axis_a.legend(
            frameon=True,
            loc="center",
            bbox_to_anchor=(0.54, 0.53),
            ncol=2,
            columnspacing=0.45,
            handlelength=1.55,
            handletextpad=0.3,
            fontsize=7.3,
            facecolor="white",
            edgecolor="none",
            framealpha=0.9,
        )
        _panel_label(axis_a, "a", "Simultaneous interval coverage")
        _style_axis(axis_a)

        for method in width_methods:
            _plot_series(
                axis_b,
                width_subset[width_subset["method"] == method],
                x="icc",
                mean="mean_interval_width_mean",
                sem="mean_interval_width_sem",
                method=method,
                label=WIDTH_LABELS[method],
                interval="band",
            )
        axis_b.set(xlabel="Latent-Gaussian ICC parameter", ylabel="Mean interval width")
        axis_b.set_ylim(bottom=0)
        axis_b.legend(
            frameon=True,
            loc="upper left",
            ncol=2,
            columnspacing=0.55,
            handlelength=1.8,
            handletextpad=0.4,
            facecolor="white",
            edgecolor="none",
            framealpha=0.9,
        )
        _panel_label(axis_b, "b", "Pairing gain and calibration cost")
        _style_axis(axis_b)

        for method in regret_methods:
            _plot_series(
                axis_c,
                regret_subset[regret_subset["method"] == method],
                x="shift_strength",
                mean="selected_true_robust_regret_mean",
                sem="selected_true_robust_regret_sem",
                method=method,
                label=REGRET_LABELS[method],
                interval="band",
            )
        axis_c.set(xlabel="Rank-reversal severity", ylabel="Population robust regret")
        axis_c.set_ylim(bottom=0)
        axis_c.legend(
            frameon=False,
            loc="upper left",
            ncol=2,
            columnspacing=0.55,
            handlelength=1.8,
            handletextpad=0.4,
        )
        _panel_label(axis_c, "c", "Selection under rank reversal")
        _style_axis(axis_c)

        shift = hrms["shift_strength"].to_numpy(dtype=float)
        set_size = hrms["confidence_set_size_mean"].to_numpy(dtype=float)
        set_sem = hrms["confidence_set_size_sem"].fillna(0.0).to_numpy(dtype=float)
        axis_d.fill_between(
            shift,
            set_size - 1.96 * set_sem,
            set_size + 1.96 * set_sem,
            color=BLUE,
            alpha=0.14,
            linewidth=0,
        )
        size_line = axis_d.plot(
            shift,
            set_size,
            color=BLUE,
            marker="o",
            label="Mean set size",
        )[0]
        axis_d.set(xlabel="Rank-reversal severity", ylabel="Confidence-set size")
        axis_d.set_ylim(bottom=0)
        axis_d.yaxis.set_major_locator(MaxNLocator(integer=True, nbins=5))
        _style_axis(axis_d)

        coverage_axis = axis_d.twinx()
        set_coverage = hrms["confidence_set_covers_mean"].to_numpy(dtype=float)
        coverage_sem = hrms["confidence_set_covers_sem"].fillna(0.0).to_numpy(dtype=float)
        lower = np.minimum(1.96 * coverage_sem, set_coverage)
        upper = np.minimum(1.96 * coverage_sem, 1.0 - set_coverage)
        coverage_line = coverage_axis.errorbar(
            shift,
            set_coverage,
            yerr=np.vstack([lower, upper]),
            color=ORANGE,
            marker="s",
            linestyle="--",
            linewidth=1.15,
            markersize=3.2,
            capsize=1.7,
            label="Set covers robust oracle",
        )
        finite_coverage = set_coverage[np.isfinite(set_coverage)]
        coverage_floor = 0.0
        if finite_coverage.size and finite_coverage.min() >= 0.75:
            coverage_floor = 0.7
        coverage_axis.set_ylim(coverage_floor, 1.025)
        coverage_axis.set_ylabel("Oracle coverage", color=ORANGE)
        coverage_axis.tick_params(axis="y", colors=ORANGE, length=2.5, width=0.7)
        coverage_axis.spines["top"].set_visible(False)
        coverage_axis.spines["right"].set_color(ORANGE)
        axis_d.legend(
            [size_line, coverage_line],
            ["Mean set size", "Set covers robust oracle"],
            frameon=False,
            loc="center right",
        )
        _panel_label(axis_d, "d", "Uncertainty-aware output", label_x=-0.17)

        figure.subplots_adjust(left=0.09, right=0.93, bottom=0.13, top=0.91)
    return figure


def save_simulation_figure(
    coverage_summary: str | Path,
    decision_summary: str | Path,
    output_stem: str | Path,
    *,
    png_dpi: int = 240,
) -> tuple[Path, Path]:
    """Build and save vector PDF plus a raster preview."""

    stem = Path(output_stem)
    stem.parent.mkdir(parents=True, exist_ok=True)
    pdf_path = stem.with_suffix(".pdf")
    png_path = stem.with_suffix(".png")
    figure = build_simulation_figure(coverage_summary, decision_summary)
    # The backend reads fonttype at save time, after the plotting rc_context.
    with plt.rc_context({"pdf.fonttype": 42, "ps.fonttype": 42}):
        figure.savefig(
            pdf_path,
            format="pdf",
            bbox_inches="tight",
            pad_inches=0.015,
            metadata={
                "Title": "HRMS simulation evaluation",
                "Creator": "HRMS artifact",
                "Producer": "Matplotlib",
                "CreationDate": None,
                "ModDate": None,
            },
        )
    figure.savefig(
        png_path,
        format="png",
        dpi=png_dpi,
        bbox_inches="tight",
        pad_inches=0.015,
        metadata={"Software": "HRMS artifact"},
    )
    plt.close(figure)
    return pdf_path, png_path


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--coverage-summary",
        type=Path,
        default=Path("results/coverage_headline/coverage_summary.csv"),
        help="Coverage-study summary CSV.",
    )
    parser.add_argument(
        "--decision-summary",
        type=Path,
        default=Path("results/decision_headline/decision_summary.csv"),
        help="Decision-study summary CSV.",
    )
    parser.add_argument(
        "--output-stem",
        type=Path,
        default=Path("paper/figures/simulation"),
        help="Output path without extension; .pdf and .png are written.",
    )
    parser.add_argument("--png-dpi", type=int, default=240)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    outputs = save_simulation_figure(
        args.coverage_summary,
        args.decision_summary,
        args.output_stem,
        png_dpi=args.png_dpi,
    )
    for path in outputs:
        print(path)
    return 0


if __name__ == "__main__":  # pragma: no cover - exercised through CLI
    raise SystemExit(main())
