#!/usr/bin/env python3
"""Render descriptive figures from generated HRMS result tables."""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

COLORS = ("#2F5D8A", "#D17B27", "#4A8F6C", "#A64B5B")


def _require_columns(table: pd.DataFrame, required: set[str], source: Path) -> None:
    missing = required.difference(table.columns)
    if missing:
        raise ValueError(f"{source} is missing required columns: {sorted(missing)}")


def _save(fig: plt.Figure, stem: Path) -> tuple[Path, Path]:
    stem.parent.mkdir(parents=True, exist_ok=True)
    pdf = stem.with_suffix(".pdf")
    png = stem.with_suffix(".png")
    fig.savefig(
        pdf,
        bbox_inches="tight",
        metadata={"Creator": "HRMS reproduction", "CreationDate": None, "ModDate": None},
    )
    fig.savefig(
        png,
        dpi=200,
        bbox_inches="tight",
        metadata={"Software": "HRMS reproduction"},
    )
    plt.close(fig)
    return pdf, png


def _plot_calibration(summary_path: Path, output_dir: Path) -> tuple[Path, Path]:
    table = pd.read_csv(summary_path)
    required = {
        "icc",
        "n_candidates",
        "cluster_size_imbalance",
        "joint_pairwise_coverage",
        "coordinate_coverage",
        "n_repetitions",
    }
    _require_columns(table, required, summary_path)
    labels = [
        f"ICC {row.icc:g}\nM={row.n_candidates:g}, {row.cluster_size_imbalance}"
        for row in table.itertuples(index=False)
    ]
    x = np.arange(len(table))
    fig, axis = plt.subplots(figsize=(6.8, 3.2), constrained_layout=True)
    axis.plot(
        x,
        table["joint_pairwise_coverage"],
        marker="o",
        color=COLORS[0],
        label="Joint pairwise coverage",
    )
    axis.plot(
        x,
        table["coordinate_coverage"],
        marker="s",
        color=COLORS[1],
        label="Coordinate coverage",
    )
    axis.axhline(0.95, color="0.45", linewidth=1, linestyle="--", label="Nominal 0.95")
    axis.set_xticks(x, labels)
    axis.set_ylim(-0.03, 1.03)
    axis.set_ylabel("Empirical fraction")
    axis.set_title("Calibration-grid smoke test (descriptive only)")
    axis.legend(frameon=False, ncol=3, fontsize=8, loc="lower center")
    axis.grid(axis="y", color="0.9", linewidth=0.7)
    return _save(fig, output_dir / "calibration_smoke")


def _plot_rank_reversal(summary_path: Path, output_dir: Path) -> tuple[Path, Path]:
    table = pd.read_csv(summary_path).sort_values("shift_strength")
    series = (
        ("Mean validation", "mean_selected_true_robust_regret"),
        ("Point DRO", "point_dro_selected_true_robust_regret"),
        ("Worst environment", "worst_environment_selected_true_robust_regret"),
        ("HRMS", "selected_true_robust_regret"),
    )
    _require_columns(
        table,
        {"shift_strength", "n_repetitions", *(column for _, column in series)},
        summary_path,
    )
    fig, axis = plt.subplots(figsize=(5.5, 3.2), constrained_layout=True)
    for color, (label, column) in zip(COLORS, series, strict=True):
        axis.plot(
            table["shift_strength"],
            table[column],
            marker="o",
            color=color,
            label=label,
        )
    axis.set_xlabel("Rank-reversal severity")
    axis.set_ylabel("Selected model's true robust regret")
    axis.set_title("Rank-reversal smoke test (descriptive only)")
    axis.grid(color="0.9", linewidth=0.7)
    axis.legend(frameon=False, fontsize=8)
    return _save(fig, output_dir / "rank_reversal_smoke")


def _plot_coverage(summary_path: Path, output_dir: Path) -> tuple[Path, Path]:
    table = pd.read_csv(summary_path)
    required = {"icc", "method", "joint_coverage_mean", "joint_coverage_sem"}
    _require_columns(table, required, summary_path)
    fig, axis = plt.subplots(figsize=(5.8, 3.4), constrained_layout=True)
    color_map = matplotlib.colormaps["tab10"]
    line_styles = ("-", "--", "-.", ":")
    for index, (method, group) in enumerate(table.groupby("method", sort=False)):
        group = group.sort_values("icc")
        axis.errorbar(
            group["icc"],
            group["joint_coverage_mean"],
            yerr=group["joint_coverage_sem"],
            marker="o",
            capsize=2,
            color=color_map(index % color_map.N),
            linestyle=line_styles[(index // color_map.N) % len(line_styles)],
            label=method,
        )
    axis.axhline(0.95, color="0.45", linewidth=1, linestyle="--", label="Nominal 0.95")
    axis.set_xlabel("Intracluster correlation")
    axis.set_ylabel("Simultaneous coverage")
    axis.set_ylim(-0.03, 1.03)
    axis.set_title("Observed simulation coverage")
    axis.grid(axis="y", color="0.9", linewidth=0.7)
    axis.legend(frameon=False, fontsize=7, ncol=2)
    return _save(fig, output_dir / "coverage_by_icc")


def make_figures(
    *,
    simulations_dir: str | Path,
    output_dir: str | Path,
    coverage_summary: str | Path | None = None,
) -> dict[str, Path]:
    """Render every available declared result table and return output paths."""

    source = Path(simulations_dir)
    destination = Path(output_dir)
    calibration = source / "calibration_summary.csv"
    rank_reversal = source / "rank_reversal_summary.csv"
    missing = [path for path in (calibration, rank_reversal) if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"missing simulation summaries: {missing}")

    outputs: dict[str, Path] = {}
    for name, paths in (
        ("calibration", _plot_calibration(calibration, destination)),
        ("rank_reversal", _plot_rank_reversal(rank_reversal, destination)),
    ):
        outputs[f"{name}_pdf"], outputs[f"{name}_png"] = paths
    if coverage_summary is not None:
        paths = _plot_coverage(Path(coverage_summary), destination)
        outputs["coverage_pdf"], outputs["coverage_png"] = paths
    return outputs


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--simulations-dir",
        type=Path,
        default=Path("results/smoke"),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("results/figures"),
    )
    parser.add_argument("--coverage-summary", type=Path)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    outputs = make_figures(
        simulations_dir=args.simulations_dir,
        output_dir=args.output_dir,
        coverage_summary=args.coverage_summary,
    )
    for name, path in outputs.items():
        print(f"{name}: {path}")
    return 0


if __name__ == "__main__":  # pragma: no cover - exercised through the CLI
    raise SystemExit(main())
