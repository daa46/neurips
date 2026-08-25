#!/usr/bin/env python3
"""Render the HRMS method schematic as deterministic vector artwork."""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Circle, FancyArrowPatch, FancyBboxPatch, Polygon, Rectangle

BLUE, SKY, ORANGE = "#0072B2", "#56B4E9", "#E69F00"
GREEN, RED, PURPLE = "#009E73", "#D55E00", "#CC79A7"
INK, MID, PALE, GRID = "#252525", "#777777", "#F4F6F8", "#D8DDE3"


def _frame(ax, x: float, width: float, number: int, title: str) -> None:
    ax.add_patch(
        FancyBboxPatch(
            (x, 0.10),
            width,
            0.84,
            boxstyle="round,pad=.012,rounding_size=.018",
            facecolor=PALE,
            edgecolor=GRID,
            linewidth=0.7,
        )
    )
    ax.text(x + 0.035, 0.88, str(number), color=BLUE, fontsize=8.2, weight="bold", va="center")
    ax.text(x + 0.15, 0.88, title, color=INK, fontsize=7.2, weight="bold", va="center")


def _arrow(ax, left: float, right: float) -> None:
    ax.add_patch(
        FancyArrowPatch(
            (left, 0.49),
            (right, 0.49),
            arrowstyle="-|>",
            mutation_scale=8,
            linewidth=0.8,
            color=MID,
        )
    )


def _paired(ax, x: float, width: float) -> None:
    for row, color in enumerate((BLUE, ORANGE, GREEN)):
        y = 0.70 - 0.18 * row
        ax.text(x + 0.045, y, rf"$e_{row + 1}$", fontsize=6.3, color=MID, va="center")
        for cluster in range(3):
            cx = x + 0.25 + 0.23 * cluster
            ax.add_patch(Circle((cx, y), 0.04, facecolor="white", edgecolor=color, lw=0.9))
            ax.plot([cx - 0.018, cx + 0.018], [y - 0.012, y + 0.012], color=color, lw=0.7)
            ax.scatter([cx - 0.018, cx + 0.018], [y - 0.012, y + 0.012], s=5, color=[RED, BLUE])
    ax.text(x + width / 2, 0.18, "paired losses by root", ha="center", fontsize=6.1, color=MID)


def _bootstrap(ax, x: float, width: float) -> None:
    for y, color in ((0.28, BLUE), (0.39, ORANGE), (0.50, BLUE), (0.61, GREEN)):
        ax.add_patch(
            FancyBboxPatch(
                (x + 0.18, y),
                0.42,
                0.075,
                boxstyle="round,pad=.006",
                facecolor="white",
                edgecolor=color,
                lw=0.85,
            )
        )
        ax.plot([x + 0.24, x + 0.51], [y + 0.038] * 2, color=color, lw=1)
    ax.add_patch(
        FancyArrowPatch(
            (x + 0.69, 0.31),
            (x + 0.69, 0.69),
            connectionstyle="arc3,rad=-.7",
            arrowstyle="-|>",
            mutation_scale=7,
            color=MID,
            lw=0.8,
        )
    )
    ax.text(x + width / 2, 0.18, "root resamples", ha="center", fontsize=6.1, color=MID)


def _bounds(ax, x: float, width: float) -> None:
    ax.axvline(x + 0.50, ymin=0.28, ymax=0.72, color=MID, lw=0.65, ls=":")
    for left, y, color in (
        (0.15, 0.62, BLUE),
        (0.30, 0.53, ORANGE),
        (0.23, 0.44, GREEN),
        (0.39, 0.35, PURPLE),
    ):
        lo, hi = x + left, x + left + 0.32
        ax.plot([lo, hi], [y, y], color=color, lw=1.25)
        ax.plot([lo, lo], [y - 0.025, y + 0.025], color=color, lw=0.8)
        ax.plot([hi, hi], [y - 0.025, y + 0.025], color=color, lw=0.8)
        ax.scatter([(lo + hi) / 2], [y], s=7, color=color)
    ax.text(x + 0.50, 0.70, r"$0$", ha="center", fontsize=6.0, color=MID)
    ax.text(x + width / 2, 0.18, r"one max-$t$ envelope", ha="center", fontsize=6.1, color=MID)


def _mixture(ax, x: float, width: float) -> None:
    vertices = ((x + 0.16, 0.29), (x + 0.76, 0.29), (x + 0.46, 0.72))
    ax.add_patch(Polygon(vertices, closed=True, facecolor="white", edgecolor=MID, lw=0.8))
    center = (x + 0.46, 0.43)
    ax.add_patch(Circle(center, 0.12, facecolor=SKY, edgecolor=BLUE, alpha=0.25, lw=0.9))
    ax.scatter(
        [center[0], x + 0.53, x + 0.39],
        [center[1], 0.48, 0.39],
        s=[9, 6, 6],
        color=[BLUE, ORANGE, ORANGE],
    )
    ax.text(
        x + width / 2, 0.18, r"$q\in\mathcal{Q}_{\rho}(p)$", ha="center", fontsize=6.4, color=MID
    )


def _decision(ax, x: float, width: float) -> None:
    for index, (name, low, high, color) in enumerate(
        (("A", 0.24, 0.48, BLUE), ("B", 0.31, 0.58, ORANGE), ("C", 0.43, 0.73, MID))
    ):
        y = 0.66 - 0.15 * index
        ax.text(x + 0.08, y, name, fontsize=6.5, color=INK, va="center")
        ax.plot([x + low, x + high], [y, y], color=color, lw=3, solid_capstyle="round")
    ax.add_patch(
        Rectangle((x + 0.04, 0.47), 0.72, 0.27, fill=False, edgecolor=BLUE, lw=0.8, ls="--")
    )
    ax.text(
        x + width / 2,
        0.18,
        r"default + $\mathcal{C}_{1-\alpha}$",
        ha="center",
        fontsize=6.1,
        color=MID,
    )


def build_method_figure():
    figure, ax = plt.subplots(figsize=(6.75, 1.55))
    ax.set(xlim=(0, 5.65), ylim=(0.05, 1.0))
    ax.axis("off")
    starts, widths = (0.02, 1.18, 2.24, 3.38, 4.44), (1.02, 0.92, 1.00, 0.92, 0.98)
    titles = (
        "Paired losses",
        "Root bootstrap",
        "Joint bounds",
        "Unknown mixture",
        "Certified choice",
    )
    for i, (x, width, title, draw) in enumerate(
        zip(
            starts, widths, titles, (_paired, _bootstrap, _bounds, _mixture, _decision), strict=True
        ),
        1,
    ):
        _frame(ax, x, width, i, title)
        draw(ax, x, width)
    for i in range(4):
        _arrow(ax, starts[i] + widths[i] + 0.02, starts[i + 1] - 0.02)
    figure.subplots_adjust(left=0.005, right=0.995, bottom=0.01, top=0.99)
    return figure


def save_method_figure(output_stem: str | Path, *, png_dpi: int = 220) -> tuple[Path, Path]:
    """Write anonymous vector and preview assets and return their paths."""

    stem = Path(output_stem)
    stem.parent.mkdir(parents=True, exist_ok=True)
    pdf, png = stem.with_suffix(".pdf"), stem.with_suffix(".png")
    # Force embedded TrueType outlines. Matplotlib otherwise defaults to
    # Type 3 glyphs, which violate the NeurIPS font requirement once this PDF
    # is included in the manuscript.
    with plt.rc_context({"pdf.fonttype": 42, "ps.fonttype": 42}):
        figure = build_method_figure()
        figure.savefig(
            pdf,
            bbox_inches="tight",
            pad_inches=0.01,
            metadata={"Creator": "HRMS", "CreationDate": None},
        )
    figure.savefig(
        png,
        dpi=png_dpi,
        bbox_inches="tight",
        pad_inches=0.01,
        metadata={"Software": "HRMS"},
    )
    plt.close(figure)
    return pdf, png


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-stem", type=Path, default=Path("paper/figures/method"))
    args = parser.parse_args()
    save_method_figure(args.output_stem)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
