#!/usr/bin/env python3
"""Render the frozen CivilComments result and root-sensitivity figure.

This script reads only completed result tables.  It performs no fitting,
selection, bootstrap resampling, or test-set optimization.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.axes import Axes
from matplotlib.figure import Figure

BLUE = "#0072B2"
SKY = "#56B4E9"
ORANGE = "#E69F00"
GREEN = "#009E73"
GRAY = "#8A8A8A"
PALE_GRAY = "#D6D6D6"
INK = "#252525"

METHODS = [
    ("point_dro_regret", "Point regret", GREEN),
    ("environment_balanced_mean", "Env.-balanced", SKY),
    ("worst_environment", "Worst env.", ORANGE),
    ("full_hrms", "Full HRMS", BLUE),
]


def _require_columns(table: pd.DataFrame, required: set[str], source: Path) -> None:
    missing = required.difference(table.columns)
    if missing:
        raise ValueError(f"{source} is missing required columns: {sorted(missing)}")


def _read_json(path: Path) -> dict[str, object]:
    with path.open(encoding="utf-8") as stream:
        value = json.load(stream)
    if not isinstance(value, dict):
        raise ValueError(f"{path} must contain a JSON object")
    return value


def _candidate_label(model_id: str) -> str:
    representation, raw_c = model_id.split("__c_", maxsplit=1)
    representation = representation.replace("char_", "char ").replace("word_", "word ")
    representation = representation.replace("_", "--")
    c_value = raw_c.replace("p", ".")
    return rf"{representation}, $C={c_value}$"


def _panel_label(axis: Axes, label: str, title: str) -> None:
    axis.text(
        -0.22,
        1.08,
        label,
        transform=axis.transAxes,
        ha="left",
        va="top",
        fontsize=10.0,
        fontweight="bold",
        clip_on=False,
    )
    axis.text(
        0.0,
        1.08,
        title,
        transform=axis.transAxes,
        ha="left",
        va="top",
        fontsize=9.2,
        fontweight="bold",
        clip_on=False,
    )


def _clean_axis(axis: Axes, *, grid: bool = True) -> None:
    if grid:
        axis.grid(axis="y", color="#E5E5E5", linewidth=0.55, zorder=0)
    axis.spines["top"].set_visible(False)
    axis.spines["right"].set_visible(False)
    axis.tick_params(length=2.4, width=0.65)


def build_civilcomments_figure(
    environment_risks: str | Path,
    method_results: str | Path,
    primary_summary: str | Path,
    iid_summary: str | Path,
) -> Figure:
    """Build the deterministic three-panel CivilComments figure."""

    risks_path = Path(environment_risks)
    methods_path = Path(method_results)
    primary_path = Path(primary_summary)
    iid_path = Path(iid_summary)

    risks = pd.read_csv(risks_path, index_col="environment_id")
    methods = pd.read_csv(methods_path)
    primary = _read_json(primary_path)
    iid = _read_json(iid_path)
    _require_columns(
        methods,
        {"method", "selected_model", "test_robust_regret_rho"},
        methods_path,
    )
    if risks.empty or risks.isna().any().any():
        raise ValueError(f"{risks_path} must be a complete environment-by-candidate table")

    method_rows = methods.set_index("method")
    missing_methods = [key for key, _, _ in METHODS if key not in method_rows.index]
    if missing_methods:
        raise ValueError(f"{methods_path} is missing methods: {missing_methods}")

    primary_block = iid["primary_article_clustered"]
    iid_block = iid["iid_comment_root_ablation"]
    if not isinstance(primary_block, dict) or not isinstance(iid_block, dict):
        raise ValueError(f"{iid_path} does not contain the expected sensitivity blocks")

    ranks = risks.rank(axis=1, method="min", ascending=True).T
    ordered_models = ranks.mean(axis=1).sort_values(kind="stable").index
    ranks = ranks.loc[ordered_models]
    selected_model = str(primary["hrms"]["selected_model"])

    style = {
        "font.family": "sans-serif",
        "font.sans-serif": ["DejaVu Sans"],
        # The source PDF is scaled to about 85% at NeurIPS text width.  Keep
        # publication-facing labels near 7 pt and dense annotations near 6 pt
        # after that scaling.
        "font.size": 8.5,
        "axes.labelsize": 8.8,
        "axes.linewidth": 0.65,
        "xtick.labelsize": 7.6,
        "ytick.labelsize": 7.5,
        "legend.fontsize": 7.5,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
    }
    with plt.rc_context(style):
        figure = plt.figure(figsize=(7.0, 2.82), constrained_layout=False)
        grid = figure.add_gridspec(1, 3, width_ratios=(2.05, 1.0, 1.2), wspace=0.68)
        axis_a = figure.add_subplot(grid[0, 0])
        axis_b = figure.add_subplot(grid[0, 1])
        axis_c = figure.add_subplot(grid[0, 2])

        image = axis_a.imshow(
            ranks.to_numpy(dtype=float),
            aspect="auto",
            cmap="cividis_r",
            vmin=1,
            vmax=len(ordered_models),
            interpolation="nearest",
        )
        for row in range(ranks.shape[0]):
            for column in range(ranks.shape[1]):
                value = int(ranks.iat[row, column])
                text_color = "white" if value >= 7 else INK
                axis_a.text(
                    column,
                    row,
                    str(value),
                    ha="center",
                    va="center",
                    fontsize=7.5,
                    color=text_color,
                )
        environment_labels = [
            label.replace("other_religion", "other rel.") for label in ranks.columns
        ]
        axis_a.set_xticks(np.arange(len(environment_labels)), labels=environment_labels)
        axis_a.tick_params(axis="x", rotation=48, pad=0.8, length=0)
        axis_a.set_yticks(
            np.arange(len(ordered_models)),
            labels=[_candidate_label(model) for model in ordered_models],
        )
        axis_a.tick_params(axis="y", length=0, pad=2)
        for tick, model in zip(axis_a.get_yticklabels(), ordered_models, strict=True):
            if model == selected_model:
                tick.set_color(BLUE)
                tick.set_fontweight("bold")
        axis_a.set_xlabel("Test identity-mention environment", labelpad=1.2)
        for spine in axis_a.spines.values():
            spine.set_visible(False)
        colorbar = figure.colorbar(image, ax=axis_a, fraction=0.028, pad=0.018)
        colorbar.set_ticks([1, 6, 12])
        colorbar.ax.tick_params(length=1.8, labelsize=7.5)
        _panel_label(axis_a, "a", "Test ranks by environment")

        method_values = np.array(
            [float(method_rows.loc[key, "test_robust_regret_rho"]) for key, _, _ in METHODS]
        )
        method_labels = [label for _, label, _ in METHODS]
        method_colors = [color for _, _, color in METHODS]
        y_values = np.arange(len(METHODS))
        bars = axis_b.barh(
            y_values,
            1_000 * method_values,
            color=method_colors,
            edgecolor=[INK, INK, INK, BLUE],
            linewidth=[0.45, 0.45, 0.45, 1.0],
            zorder=2,
        )
        bars[-1].set_hatch("///")
        for bar, value in zip(bars, 1_000 * method_values, strict=True):
            axis_b.text(
                value + 0.035,
                bar.get_y() + bar.get_height() / 2,
                f"{value:.3f}",
                ha="left",
                va="center",
                fontsize=7.5,
            )
        axis_b.set_yticks(y_values, labels=method_labels)
        axis_b.tick_params(axis="y", length=0, pad=2)
        axis_b.invert_yaxis()
        axis_b.set_xlabel(r"Test robust regret ($\times 10^{-3}$)")
        axis_b.set_xlim(0, max(1_000 * method_values) * 1.25)
        _clean_axis(axis_b)
        axis_b.grid(axis="x", color="#E5E5E5", linewidth=0.55, zorder=0)
        axis_b.grid(axis="y", visible=False)
        _panel_label(axis_b, "b", r"Robust regret ($\rho=.25$)")

        sensitivity_labels = [
            "Article roots\n(global)",
            "IID comments*\n(stratified)",
        ]
        certificate_values = 1_000 * np.array(
            [
                float(primary_block["selected_certificate_upper"]),
                float(iid_block["selected_certificate_upper"]),
            ]
        )
        test_values = 1_000 * np.array(
            [
                float(primary_block["test_robust_regret"]),
                float(iid_block["test_robust_regret"]),
            ]
        )
        sensitivity_x = np.arange(2)
        axis_c.bar(
            sensitivity_x,
            certificate_values,
            width=0.62,
            color=PALE_GRAY,
            edgecolor=GRAY,
            linewidth=0.6,
            label="Upper certificate",
            zorder=1,
        )
        axis_c.scatter(
            sensitivity_x,
            test_values,
            marker="D",
            s=22,
            color=BLUE,
            edgecolor="white",
            linewidth=0.45,
            label="Test robust regret",
            zorder=3,
        )
        for x_value, bar_value, point_value in zip(
            sensitivity_x, certificate_values, test_values, strict=True
        ):
            axis_c.text(
                x_value,
                bar_value + 0.25,
                f"{bar_value:.2f}",
                ha="center",
                va="bottom",
                fontsize=7.5,
                color=INK,
            )
            axis_c.text(
                x_value + 0.08,
                point_value,
                f"{point_value:.3f}",
                ha="left",
                va="center",
                fontsize=7.5,
                color=BLUE,
            )
        selected_labels = [
            _candidate_label(str(primary_block["selected_model"])),
            _candidate_label(str(iid_block["selected_model"])),
        ]
        set_sizes = [
            int(primary_block["confidence_set_size"]),
            int(iid_block["confidence_set_size"]),
        ]
        for x_value, label in zip(sensitivity_x, selected_labels, strict=True):
            wrapped_label = label.replace(", ", "\n")
            axis_c.text(
                x_value,
                -0.19,
                wrapped_label,
                transform=axis_c.get_xaxis_transform(),
                ha="center",
                va="top",
                fontsize=7.5,
                clip_on=False,
            )
        axis_c.set_xticks(sensitivity_x, labels=sensitivity_labels)
        axis_c.tick_params(axis="x", length=0, pad=2)
        axis_c.set_ylabel(r"Value ($\times 10^{-3}$)")
        axis_c.set_ylim(0, max(certificate_values) * 1.34)
        axis_c.legend(frameon=False, loc="upper center", bbox_to_anchor=(0.5, 1.01))
        _clean_axis(axis_c)
        _panel_label(axis_c, "c", "Root-unit sensitivity")
        if set_sizes[0] == set_sizes[1]:
            set_note = rf"Sets {set_sizes[0]}/12; *IID misspecified"
        else:
            set_note = (
                rf"Sets L/R: {set_sizes[0]}/12, {set_sizes[1]}/12; "
                "*known misspecification"
            )
        axis_c.text(
            0.5,
            -0.43,
            set_note,
            transform=axis_c.transAxes,
            ha="center",
            va="top",
            fontsize=7.5,
            color=GRAY,
        )

        figure.subplots_adjust(left=0.19, right=0.99, bottom=0.29, top=0.89)
    return figure


def save_civilcomments_figure(
    environment_risks: str | Path,
    method_results: str | Path,
    primary_summary: str | Path,
    iid_summary: str | Path,
    output_stem: str | Path,
    *,
    png_dpi: int = 240,
) -> tuple[Path, Path]:
    """Save the figure as an anonymous vector PDF and a raster preview."""

    stem = Path(output_stem)
    stem.parent.mkdir(parents=True, exist_ok=True)
    pdf_path = stem.with_suffix(".pdf")
    png_path = stem.with_suffix(".png")
    figure = build_civilcomments_figure(
        environment_risks,
        method_results,
        primary_summary,
        iid_summary,
    )
    # The backend reads fonttype at save time, after the plotting rc_context.
    with plt.rc_context({"pdf.fonttype": 42, "ps.fonttype": 42}):
        figure.savefig(
            pdf_path,
            format="pdf",
            bbox_inches="tight",
            pad_inches=0.015,
            metadata={
                "Title": "Derived CivilComments environment-mixture evaluation",
                "Creator": "Anonymous artifact",
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
        metadata={"Software": "Anonymous artifact"},
    )
    plt.close(figure)
    return pdf_path, png_path


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--environment-risks",
        type=Path,
        default=Path("results/civilcomments/test_evaluation/test_environment_risks.csv"),
    )
    parser.add_argument(
        "--method-results",
        type=Path,
        default=Path("results/civilcomments/test_evaluation/method_results.csv"),
    )
    parser.add_argument(
        "--primary-summary",
        type=Path,
        default=Path("results/civilcomments/selection/selection_manifest.json"),
    )
    parser.add_argument(
        "--iid-summary",
        type=Path,
        default=Path("results/civilcomments/ablations/iid_comment_roots/summary.json"),
    )
    parser.add_argument(
        "--output-stem",
        type=Path,
        default=Path("paper/figures/civil_ablation"),
    )
    parser.add_argument("--png-dpi", type=int, default=240)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    outputs = save_civilcomments_figure(
        args.environment_risks,
        args.method_results,
        args.primary_summary,
        args.iid_summary,
        args.output_stem,
        png_dpi=args.png_dpi,
    )
    for path in outputs:
        print(path)
    return 0


if __name__ == "__main__":  # pragma: no cover - exercised through CLI
    raise SystemExit(main())
