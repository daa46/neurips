#!/usr/bin/env python3
"""Render the frozen public-TE evaluation as a publication figure.

The script is intentionally a pure rendering layer.  It reads only tidy CSV
exports written by ``applications/te_regulation/evaluate_public.py`` and does
not fit, select, resample, or inspect raw predictions.  The only arithmetic is
display-oriented (excess risk, scaling, and descriptive quantiles across the
prespecified target-mixture design).
"""

from __future__ import annotations

import argparse
import math
import re
from dataclasses import dataclass
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.axes import Axes
from matplotlib.figure import Figure
from matplotlib.lines import Line2D
from matplotlib.ticker import MaxNLocator, PercentFormatter

# Okabe--Ito colors, plus neutral grays.  Meaning is also encoded by marker,
# line style, or position so the panels remain interpretable in grayscale.
BLUE = "#0072B2"
SKY = "#56B4E9"
ORANGE = "#E69F00"
GREEN = "#009E73"
VERMILLION = "#D55E00"
PURPLE = "#CC79A7"
GRAY = "#777777"
PALE_GRAY = "#D8D8D8"
GRID = "#E5E5E5"
INK = "#252525"

PRIMARY_ANALYSIS = "root_observation"
PRIMARY_WEIGHTING = "observation"
PRIMARY_SCOPES = frozenset({"nominal", "inside_primary", "primary_boundary"})

ENVIRONMENT_ORDER = (
    "Alu",
    "L1",
    "MIR",
    "L2",
    "ERVL-MaLR",
    "hAT-Charlie",
    "ERV1",
    "ERVL",
)

METHOD_SPECS = (
    ("uniform_mean", "Uniform mean", VERMILLION),
    ("worst_environment", "Worst environment", ORANGE),
    ("point_dro_risk", "Point DRO risk", PURPLE),
    ("point_dro_regret", "Point DRO regret", GREEN),
    ("hrms", "HRMS", BLUE),
)

FAMILY_ORDER = {"LR": 0, "SGD": 1, "cNB": 2, "Other": 99}
VIEW_ORDER = {"128 bp": 0, "256 bp": 1, "512 bp": 2, "TE copy": 3}
FAMILY_COLORS = {"LR": BLUE, "SGD": ORANGE, "cNB": GREEN, "Other": GRAY}
_MODEL_PATTERN = re.compile(
    r"^(?P<family>lr|sgd|cnb)_(?P<view>w(?:128|256|512)|te)_"
    r"k(?P<k>\d+)_(?P<penalty>[ca])(?P<value>[0-9]+(?:p[0-9]+)?|[0-9]+e[0-9]+)$"
)


@dataclass(frozen=True)
class CandidateDisplay:
    """A result-independent display description parsed from a frozen model ID."""

    model_id: str
    family: str
    view: str
    k: int
    penalty: str
    penalty_value: float
    label: str
    sort_key: tuple[object, ...]


def _require_columns(table: pd.DataFrame, required: set[str], source: Path) -> None:
    missing = required.difference(table.columns)
    if missing:
        raise ValueError(f"{source} is missing required columns: {sorted(missing)}")


def _numeric_column(table: pd.DataFrame, column: str, source: Path) -> pd.Series:
    values = pd.to_numeric(table[column], errors="coerce")
    if values.isna().any() or not np.isfinite(values.to_numpy(dtype=float)).all():
        raise ValueError(f"{source} column {column!r} must be finite numeric")
    return values.astype(float)


def _boolean_column(table: pd.DataFrame, column: str, source: Path) -> pd.Series:
    normalized = table[column].astype(str).str.strip().str.lower()
    mapping = {"true": True, "false": False, "1": True, "0": False}
    unknown = sorted(set(normalized).difference(mapping))
    if unknown:
        raise ValueError(
            f"{source} column {column!r} contains invalid booleans: {unknown[:5]}"
        )
    return normalized.map(mapping).astype(bool)


def _format_number(value: float) -> str:
    if math.isclose(value, round(value), abs_tol=1e-12):
        return str(int(round(value)))
    return f"{value:g}"


def _parse_candidate(model_id: str) -> CandidateDisplay:
    match = _MODEL_PATTERN.fullmatch(model_id)
    if match is None:
        label = model_id.replace("_", "\u2009")
        return CandidateDisplay(
            model_id=model_id,
            family="Other",
            view="Other",
            k=999,
            penalty="",
            penalty_value=math.inf,
            label=label,
            sort_key=(FAMILY_ORDER["Other"], model_id),
        )

    family = {"lr": "LR", "sgd": "SGD", "cnb": "cNB"}[match.group("family")]
    view_token = match.group("view")
    view = "TE copy" if view_token == "te" else f"{view_token[1:]} bp"
    k = int(match.group("k"))
    penalty = "C" if match.group("penalty") == "c" else r"$\alpha$"
    raw_value = match.group("value")
    if "e" in raw_value and match.group("penalty") == "a":
        mantissa, exponent = raw_value.split("e", maxsplit=1)
        penalty_value = float(mantissa) * 10.0 ** (-int(exponent))
        value_label = f"{mantissa}e-{int(exponent)}"
    else:
        penalty_value = float(raw_value.replace("p", "."))
        value_label = _format_number(penalty_value)
    label = f"{family} · {view} · k{k} · {penalty}={value_label}"
    return CandidateDisplay(
        model_id=model_id,
        family=family,
        view=view,
        k=k,
        penalty=penalty,
        penalty_value=penalty_value,
        label=label,
        sort_key=(
            FAMILY_ORDER[family],
            VIEW_ORDER[view],
            k,
            penalty_value,
            model_id,
        ),
    )


def _candidate_catalog(
    model_ids: set[str] | list[str] | tuple[str, ...],
) -> tuple[list[CandidateDisplay], dict[str, str]]:
    displays = sorted(
        (_parse_candidate(str(value)) for value in set(model_ids)),
        key=lambda item: item.sort_key,
    )
    codes = {display.model_id: f"C{index:02d}" for index, display in enumerate(displays, 1)}
    return displays, codes


def _environment_order(values: set[str] | list[str] | tuple[str, ...]) -> list[str]:
    observed = {str(value) for value in values}
    declared = [value for value in ENVIRONMENT_ORDER if value in observed]
    return declared + sorted(observed.difference(declared))


def _radius_label(value: object) -> str:
    normalized = str(value).strip().lower()
    if normalized in {"full", "full_simplex", "none"}:
        return "full"
    number = float(normalized)
    if not math.isfinite(number) or number < 0:
        raise ValueError(f"invalid ambiguity radius: {value!r}")
    return f"{number:g}"


def _radius_sort_key(value: str) -> tuple[int, float]:
    return (1, math.inf) if value == "full" else (0, float(value))


def _prepare_heatmap(
    table: pd.DataFrame,
    source: Path,
) -> tuple[pd.DataFrame, list[CandidateDisplay], dict[str, str]]:
    _require_columns(
        table,
        {
            "partition",
            "root_definition",
            "weighting",
            "environment_id",
            "model_id",
            "risk",
        },
        source,
    )
    subset = table.loc[
        table["partition"].astype(str).eq("test")
        & table["root_definition"].astype(str).eq("homology_root")
        & table["weighting"].astype(str).eq(PRIMARY_WEIGHTING)
    ].copy()
    if subset.empty:
        raise ValueError(f"{source} has no primary held-out risk rows")
    subset["environment_id"] = subset["environment_id"].astype(str)
    subset["model_id"] = subset["model_id"].astype(str)
    subset["risk"] = _numeric_column(subset, "risk", source)
    if ((subset["risk"] < 0) | (subset["risk"] > 1)).any():
        raise ValueError(f"{source} Brier risks must lie in [0, 1]")
    duplicated = subset.duplicated(["model_id", "environment_id"], keep=False)
    if duplicated.any():
        cells = subset.loc[duplicated, ["model_id", "environment_id"]].head(5)
        raise ValueError(f"{source} has duplicate held-out risk cells: {cells.to_dict('records')}")

    displays, codes = _candidate_catalog(set(subset["model_id"]))
    model_order = [display.model_id for display in displays]
    environments = _environment_order(set(subset["environment_id"]))
    risk = subset.pivot(index="model_id", columns="environment_id", values="risk").reindex(
        index=model_order, columns=environments
    )
    if risk.isna().any().any():
        missing = risk.isna().stack()
        cells = [tuple(value) for value in missing[missing].index[:5]]
        raise ValueError(f"{source} has an incomplete candidate-by-environment grid: {cells}")
    excess = 1_000.0 * risk.subtract(risk.min(axis=0), axis=1)
    return excess, displays, codes


def _prepare_primary_selection(
    table: pd.DataFrame,
    source: Path,
) -> tuple[str, str]:
    _require_columns(
        table,
        {
            "analysis_id",
            "weighting",
            "method",
            "selection_radius",
            "selected_model",
            "primary_comparison",
        },
        source,
    )
    primary = _boolean_column(table, "primary_comparison", source)
    rows = table.loc[
        primary
        & table["analysis_id"].astype(str).eq(PRIMARY_ANALYSIS)
        & table["weighting"].astype(str).eq(PRIMARY_WEIGHTING)
        & table["method"].astype(str).eq("hrms")
    ]
    if len(rows) != 1:
        raise ValueError(f"{source} must mark exactly one primary observation-weighted HRMS row")
    row = rows.iloc[0]
    return str(row["selected_model"]), _radius_label(row["selection_radius"])


def _prepare_mixture_regrets(
    table: pd.DataFrame,
    source: Path,
) -> dict[str, pd.DataFrame]:
    _require_columns(
        table,
        {
            "mixture_id",
            "scope",
            "method_id",
            "analysis_id",
            "weighting",
            "method",
            "selected_model",
            "test_regret",
            "primary_comparison",
        },
        source,
    )
    primary = _boolean_column(table, "primary_comparison", source)
    method_keys = {key for key, _, _ in METHOD_SPECS}
    subset = table.loc[
        primary
        & table["analysis_id"].astype(str).eq(PRIMARY_ANALYSIS)
        & table["weighting"].astype(str).eq(PRIMARY_WEIGHTING)
        & table["method"].astype(str).isin(method_keys)
        & table["scope"].astype(str).isin(PRIMARY_SCOPES)
    ].copy()
    if subset.empty:
        raise ValueError(f"{source} has no primary plausible-mixture regrets")
    subset["method"] = subset["method"].astype(str)
    subset["mixture_id"] = subset["mixture_id"].astype(str)
    subset["selected_model"] = subset["selected_model"].astype(str)
    subset["test_regret"] = _numeric_column(subset, "test_regret", source)
    if (subset["test_regret"] < -1e-10).any() or (subset["test_regret"] > 1 + 1e-10).any():
        raise ValueError(f"{source} held-out Brier regrets must lie in [0, 1]")
    subset["test_regret"] = subset["test_regret"].clip(lower=0.0, upper=1.0)
    duplicated = subset.duplicated(["method", "mixture_id"], keep=False)
    if duplicated.any():
        cells = subset.loc[duplicated, ["method", "mixture_id"]].head(5)
        raise ValueError(
            f"{source} has duplicate method-by-mixture cells: {cells.to_dict('records')}"
        )

    missing_methods = sorted(method_keys.difference(set(subset["method"])))
    if missing_methods:
        raise ValueError(f"{source} is missing primary methods: {missing_methods}")
    expected_mixtures: set[str] | None = None
    prepared: dict[str, pd.DataFrame] = {}
    for method, _, _ in METHOD_SPECS:
        frame = subset.loc[subset["method"].eq(method)].sort_values(
            ["test_regret", "mixture_id"], kind="stable"
        )
        if frame["method_id"].astype(str).nunique() != 1:
            raise ValueError(f"{source} has multiple primary method IDs for {method!r}")
        if frame["selected_model"].nunique() != 1:
            raise ValueError(f"{source} has inconsistent selected models for {method!r}")
        mixtures = set(frame["mixture_id"])
        if expected_mixtures is None:
            expected_mixtures = mixtures
        elif mixtures != expected_mixtures:
            raise ValueError(f"{source} methods do not share the same target-mixture grid")
        prepared[method] = frame.reset_index(drop=True)
    return prepared


def _prepare_robustness_path(
    table: pd.DataFrame,
    source: Path,
    *,
    expected_models: set[str],
) -> pd.DataFrame:
    _require_columns(
        table,
        {
            "analysis_id",
            "weighting",
            "radius",
            "model",
            "lower",
            "upper",
            "selected",
            "in_confidence_set",
            "confidence_set_size",
        },
        source,
    )
    subset = table.loc[
        table["analysis_id"].astype(str).eq(PRIMARY_ANALYSIS)
        & table["weighting"].astype(str).eq(PRIMARY_WEIGHTING)
    ].copy()
    if subset.empty:
        raise ValueError(f"{source} has no primary observation-weighted robustness path")
    subset["model"] = subset["model"].astype(str)
    subset["radius_label"] = subset["radius"].map(_radius_label)
    subset["lower"] = _numeric_column(subset, "lower", source)
    subset["upper"] = _numeric_column(subset, "upper", source)
    subset["confidence_set_size"] = _numeric_column(subset, "confidence_set_size", source)
    subset["selected_bool"] = _boolean_column(subset, "selected", source)
    subset["set_bool"] = _boolean_column(subset, "in_confidence_set", source)
    if (subset["lower"] < -1e-10).any() or (subset["upper"] < -1e-10).any():
        raise ValueError(f"{source} robust-regret bounds must be non-negative")
    if (subset["lower"] > subset["upper"] + 1e-12).any():
        raise ValueError(f"{source} has lower robust-regret bounds above upper bounds")
    if set(subset["model"]) != expected_models:
        raise ValueError(f"{source} candidate library differs from the held-out risk grid")
    if subset.duplicated(["radius_label", "model"]).any():
        raise ValueError(f"{source} has duplicate radius-by-model cells")

    radius_labels = sorted(set(subset["radius_label"]), key=_radius_sort_key)
    rows: list[dict[str, object]] = []
    for radius in radius_labels:
        frame = subset.loc[subset["radius_label"].eq(radius)]
        if set(frame["model"]) != expected_models:
            raise ValueError(f"{source} has an incomplete candidate grid at rho={radius}")
        selected = frame.loc[frame["selected_bool"]]
        if len(selected) != 1:
            raise ValueError(f"{source} must identify one selected model at rho={radius}")
        selected_model = str(selected.iloc[0]["model"])
        if not bool(selected.iloc[0]["set_bool"]):
            raise ValueError(f"{source} selected model is excluded at rho={radius}")
        reported_sizes = frame["confidence_set_size"].unique()
        observed_size = int(frame["set_bool"].sum())
        if (
            len(reported_sizes) != 1
            or not math.isclose(float(reported_sizes[0]), observed_size, abs_tol=1e-12)
        ):
            raise ValueError(f"{source} has an inconsistent non-exclusion-set size at rho={radius}")
        selected_row = selected.iloc[0]
        rows.append(
            {
                "radius": radius,
                "selected_model": selected_model,
                "selected_upper": float(selected_row["upper"]),
                "set_size": observed_size,
            }
        )
    return pd.DataFrame(rows)


def _prepare_cluster_diagnostics(
    table: pd.DataFrame,
    source: Path,
    *,
    expected_environments: set[str],
) -> pd.DataFrame:
    _require_columns(
        table,
        {
            "analysis_id",
            "weighting",
            "environment_id",
            "independent_root_clusters",
            "effective_root_clusters",
            "max_cluster_leverage",
        },
        source,
    )
    subset = table.loc[
        table["analysis_id"].astype(str).eq(PRIMARY_ANALYSIS)
        & table["weighting"].astype(str).eq(PRIMARY_WEIGHTING)
    ].copy()
    if subset.empty:
        raise ValueError(f"{source} has no primary root diagnostics")
    subset["environment_id"] = subset["environment_id"].astype(str)
    for column in (
        "independent_root_clusters",
        "effective_root_clusters",
        "max_cluster_leverage",
    ):
        subset[column] = _numeric_column(subset, column, source)
    if subset["environment_id"].duplicated().any():
        raise ValueError(f"{source} has duplicate environment diagnostics")
    if set(subset["environment_id"]) != expected_environments:
        raise ValueError(f"{source} environments differ from the held-out risk grid")
    if (subset["independent_root_clusters"] < 1).any():
        raise ValueError(f"{source} independent-root counts must be positive")
    if (
        (subset["effective_root_clusters"] <= 0).any()
        or (
            subset["effective_root_clusters"]
            > subset["independent_root_clusters"] + 1e-9
        ).any()
    ):
        raise ValueError(f"{source} effective-root counts are outside valid bounds")
    if (
        (subset["max_cluster_leverage"] <= 0).any()
        or (subset["max_cluster_leverage"] > 1).any()
    ):
        raise ValueError(f"{source} maximum root leverage must lie in (0, 1]")
    order = _environment_order(set(subset["environment_id"]))
    return subset.set_index("environment_id").loc[order].reset_index()


def _panel_label(
    axis: Axes,
    label: str,
    title: str,
    *,
    label_x: float = -0.08,
    title_x: float = 0.0,
) -> None:
    axis.text(
        label_x,
        1.13,
        label,
        transform=axis.transAxes,
        ha="left",
        va="top",
        fontsize=10.0,
        fontweight="bold",
        clip_on=False,
    )
    axis.text(
        title_x,
        1.13,
        title,
        transform=axis.transAxes,
        ha="left",
        va="top",
        fontsize=9.1,
        fontweight="bold",
        clip_on=False,
    )


def _style_axis(axis: Axes, *, grid_axis: str = "y") -> None:
    axis.grid(axis=grid_axis, color=GRID, linewidth=0.55, zorder=0)
    axis.spines["top"].set_visible(False)
    axis.spines["right"].set_visible(False)
    axis.tick_params(length=2.4, width=0.65)


def _compact_environment_label(value: str) -> str:
    return {
        "ERVL-MaLR": "ERVL–\nMaLR",
        "hAT-Charlie": "hAT–\nCharlie",
    }.get(value, value)


def _plot_heatmap(
    axis: Axes,
    figure: Figure,
    excess: pd.DataFrame,
    displays: list[CandidateDisplay],
    codes: dict[str, str],
    *,
    default_model: str,
) -> None:
    maximum = float(excess.to_numpy().max())
    vmax = maximum if maximum > 0 else 1.0
    image = axis.pcolormesh(
        excess.to_numpy(dtype=float),
        shading="flat",
        cmap="cividis",
        vmin=0.0,
        vmax=vmax,
        rasterized=False,
    )
    axis.set_xlim(0, excess.shape[1])
    axis.set_ylim(excess.shape[0], 0)
    axis.set_xticks(
        np.arange(excess.shape[1]) + 0.5,
        labels=[_compact_environment_label(str(value)) for value in excess.columns],
    )
    axis.tick_params(axis="x", length=0, pad=2.0)
    tick_labels = [f"{codes[item.model_id]}  {item.label}" for item in displays]
    axis.set_yticks(np.arange(excess.shape[0]) + 0.5, labels=tick_labels)
    axis.tick_params(axis="y", length=0, pad=2.7, labelsize=7.0)
    for tick, display in zip(axis.get_yticklabels(), displays, strict=True):
        if display.model_id == default_model:
            tick.set_color(BLUE)
            tick.set_fontweight("bold")
    family_starts: dict[str, int] = {}
    for row, display in enumerate(displays):
        family_starts.setdefault(display.family, row)
    ordered_families = [value for value in FAMILY_ORDER if value in family_starts]
    for family in ordered_families[1:]:
        axis.axhline(family_starts[family], color="white", linewidth=1.2)
    for spine in axis.spines.values():
        spine.set_visible(False)
    colorbar = figure.colorbar(image, ax=axis, fraction=0.018, pad=0.012, aspect=28)
    # Matplotlib rasterizes long colorbar gradients by default even when the
    # heatmap itself is a vector QuadMesh.  Force vector solids so the PDF has
    # no embedded bitmap panel elements.
    if colorbar.solids is not None:
        colorbar.solids.set_rasterized(False)
    colorbar.set_label(r"Excess Brier loss ($\times 10^{-3}$)", labelpad=3)
    colorbar.ax.tick_params(length=2.0, width=0.6, labelsize=7.2)
    axis.set_xlabel("Held-out TE family", labelpad=2)
    _panel_label(
        axis,
        "a",
        "Candidate performance varies across TE families (blue label: default HRMS)",
        label_x=-0.12,
    )


def _deterministic_offsets(count: int, half_width: float = 0.14) -> np.ndarray:
    if count <= 1:
        return np.zeros(count)
    # An irrational angular increment yields a compact, deterministic cloud
    # without implying that the fixed mixture design is a random sample.
    indices = np.arange(count, dtype=float)
    return half_width * np.sin(indices * math.pi * (3.0 - math.sqrt(5.0)))


def _plot_mixture_regret(
    axis: Axes,
    prepared: dict[str, pd.DataFrame],
    codes: dict[str, str],
    *,
    default_radius: str,
) -> None:
    labels: list[str] = []
    maximum = 0.0
    for position, (method, display_label, color) in enumerate(METHOD_SPECS):
        frame = prepared[method]
        values = 1_000.0 * frame["test_regret"].to_numpy(dtype=float)
        maximum = max(maximum, float(values.max(initial=0.0)))
        offsets = _deterministic_offsets(len(values))
        axis.scatter(
            values,
            position + offsets,
            s=4.5,
            color=color,
            alpha=0.12,
            edgecolors="none",
            rasterized=False,
            zorder=1,
        )
        q05, median, q95 = np.quantile(values, [0.05, 0.5, 0.95])
        axis.plot([q05, q95], [position, position], color=color, linewidth=3.0, zorder=2)
        axis.scatter(
            [median],
            [position],
            marker="D",
            s=24,
            color=color,
            edgecolor="white",
            linewidth=0.55,
            zorder=3,
        )
        selected = str(frame["selected_model"].iloc[0])
        if selected not in codes:
            raise ValueError(f"mixture-regret table names unknown candidate {selected!r}")
        labels.append(f"{display_label}  [{codes[selected]}]")
    axis.set_yticks(np.arange(len(labels)), labels=labels)
    axis.tick_params(axis="y", length=0, pad=2.5, labelsize=7.0)
    axis.invert_yaxis()
    axis.set_xlabel(r"Held-out Brier regret ($\times 10^{-3}$)")
    axis.set_xlim(left=min(-0.015 * max(maximum, 1.0), -0.015), right=max(maximum * 1.08, 0.1))
    axis.axvline(0.0, color=INK, linewidth=0.65, zorder=0)
    _style_axis(axis, grid_axis="x")
    radius_text = "full simplex" if default_radius == "full" else rf"$\rho\leq {default_radius}$"
    count = len(next(iter(prepared.values())))
    axis.text(
        0.5,
        -0.33,
        f"{count} fixed {radius_text} mixtures\n◆ median; line 5–95%; faint = individual",
        transform=axis.transAxes,
        ha="center",
        va="top",
        fontsize=6.2,
        color=GRAY,
        clip_on=False,
    )
    _panel_label(axis, "b", "Mixture regret", label_x=0.0, title_x=0.12)


def _plot_robustness_path(
    axis: Axes,
    path: pd.DataFrame,
    displays_by_model: dict[str, CandidateDisplay],
    codes: dict[str, str],
    *,
    default_radius: str,
) -> None:
    x = np.arange(len(path), dtype=float)
    upper = 1_000.0 * path["selected_upper"].to_numpy(dtype=float)
    set_size = path["set_size"].to_numpy(dtype=float)
    radii = path["radius"].astype(str).tolist()
    if default_radius not in radii:
        raise ValueError("the primary HRMS radius is absent from the robustness path")
    default_index = radii.index(default_radius)
    axis.axvspan(default_index - 0.32, default_index + 0.32, color=SKY, alpha=0.14, zorder=0)
    axis.plot(x, upper, color=BLUE, marker="o", linewidth=1.4, markersize=3.8, zorder=3)
    axis.set_xticks(x, labels=["full" if value == "full" else value for value in radii])
    axis.set_xlabel(r"Ambiguity radius $\rho$")
    axis.set_ylabel(r"Upper certificate ($\times 10^{-3}$)", color=BLUE, labelpad=2)
    axis.tick_params(axis="y", colors=BLUE)
    axis.spines["left"].set_color(BLUE)
    axis.set_ylim(bottom=0)
    axis.grid(axis="y", color=GRID, linewidth=0.55, zorder=0)
    axis.spines["top"].set_visible(False)
    axis.tick_params(length=2.4, width=0.65)

    size_axis = axis.twinx()
    size_axis.plot(
        x,
        set_size,
        color=ORANGE,
        marker="s",
        linestyle="--",
        linewidth=1.2,
        markersize=3.5,
        zorder=2,
    )
    size_axis.set_ylabel("Set size", color=ORANGE, labelpad=1)
    size_axis.tick_params(axis="y", colors=ORANGE, length=2.4, width=0.65)
    size_axis.spines["top"].set_visible(False)
    size_axis.spines["right"].set_color(ORANGE)
    size_axis.set_ylim(0, max(set_size.max() * 1.22, 1.2))
    size_axis.yaxis.set_major_locator(MaxNLocator(integer=True, nbins=5))

    axis.text(
        -0.015,
        1.015,
        "selected:",
        transform=axis.transAxes,
        ha="right",
        va="center",
        fontsize=6.5,
        color=GRAY,
        clip_on=False,
    )
    for position, row in enumerate(path.itertuples(index=False)):
        model = str(row.selected_model)
        display = displays_by_model[model]
        is_default = position == default_index
        axis.text(
            position,
            1.015,
            codes[model],
            transform=axis.get_xaxis_transform(),
            ha="center",
            va="center",
            fontsize=6.4,
            fontweight="bold" if is_default else "normal",
            color=FAMILY_COLORS[display.family],
            bbox={
                "boxstyle": "round,pad=0.13",
                "facecolor": "white",
                "edgecolor": BLUE if is_default else PALE_GRAY,
                "linewidth": 0.8 if is_default else 0.45,
            },
            clip_on=False,
        )
    axis.text(
        0.99,
        -0.32,
        "shading: default radius",
        transform=axis.transAxes,
        ha="right",
        va="top",
        fontsize=6.2,
        color=GRAY,
        clip_on=False,
    )
    _panel_label(axis, "c", "Radius sensitivity", label_x=0.0, title_x=0.12)


def _plot_root_diagnostics(axis: Axes, diagnostics: pd.DataFrame) -> None:
    x = np.arange(len(diagnostics), dtype=float)
    raw = diagnostics["independent_root_clusters"].to_numpy(dtype=float)
    effective = diagnostics["effective_root_clusters"].to_numpy(dtype=float)
    leverage = diagnostics["max_cluster_leverage"].to_numpy(dtype=float)
    axis.vlines(x, effective, raw, color=PALE_GRAY, linewidth=1.3, zorder=1)
    axis.scatter(
        x,
        raw,
        marker="o",
        s=19,
        facecolor="white",
        edgecolor=GRAY,
        linewidth=0.75,
        zorder=2,
    )
    axis.scatter(x, effective, marker="o", s=20, color=BLUE, zorder=3)
    axis.set_xticks(
        x,
        labels=[_compact_environment_label(value) for value in diagnostics["environment_id"]],
    )
    axis.tick_params(axis="x", length=0, pad=2.0, labelsize=6.5)
    axis.set_ylabel("Root count", labelpad=2)
    axis.set_ylim(bottom=0)
    _style_axis(axis)

    leverage_axis = axis.twinx()
    leverage_axis.scatter(
        x,
        leverage,
        marker="D",
        s=17,
        color=ORANGE,
        edgecolor="white",
        linewidth=0.45,
        zorder=4,
    )
    leverage_axis.set_ylabel("Max. root weight", color=ORANGE, labelpad=1)
    leverage_axis.tick_params(axis="y", colors=ORANGE, length=2.4, width=0.65)
    leverage_axis.spines["top"].set_visible(False)
    leverage_axis.spines["right"].set_color(ORANGE)
    leverage_axis.set_ylim(0, max(float(leverage.max()) * 1.28, 0.01))
    leverage_axis.yaxis.set_major_formatter(PercentFormatter(xmax=1.0, decimals=0))
    legend_handles = [
        Line2D(
            [],
            [],
            marker="o",
            linestyle="none",
            markerfacecolor="white",
            markeredgecolor=GRAY,
            label="Raw",
        ),
        Line2D([], [], marker="o", linestyle="none", color=BLUE, label="Effective"),
        Line2D([], [], marker="D", linestyle="none", color=ORANGE, label="Max weight"),
    ]
    axis.legend(
        handles=legend_handles,
        frameon=False,
        loc="upper center",
        ncol=3,
        columnspacing=0.6,
        handletextpad=0.25,
        borderaxespad=0.2,
        fontsize=6.4,
    )
    _panel_label(axis, "d", "Independent evidence", label_x=0.0, title_x=0.12)


def build_te_figure(
    ranking_heatmap: str | Path,
    target_mixture_regrets: str | Path,
    robustness_path: str | Path,
    selected_models: str | Path,
    cluster_diagnostics: str | Path | None = None,
) -> Figure:
    """Build the deterministic public-TE multi-panel figure."""

    heatmap_path = Path(ranking_heatmap)
    regrets_path = Path(target_mixture_regrets)
    path_path = Path(robustness_path)
    selected_path = Path(selected_models)
    diagnostics_path = Path(cluster_diagnostics) if cluster_diagnostics is not None else None
    heatmap_table = pd.read_csv(heatmap_path)
    regret_table = pd.read_csv(regrets_path)
    path_table = pd.read_csv(path_path, dtype={"radius": str})
    selected_table = pd.read_csv(selected_path, dtype={"selection_radius": str})

    excess, displays, codes = _prepare_heatmap(heatmap_table, heatmap_path)
    default_model, default_radius = _prepare_primary_selection(selected_table, selected_path)
    if default_model not in set(excess.index):
        raise ValueError(f"{selected_path} primary model is absent from the held-out risk grid")
    regrets = _prepare_mixture_regrets(regret_table, regrets_path)
    path = _prepare_robustness_path(
        path_table,
        path_path,
        expected_models=set(excess.index.astype(str)),
    )
    displays_by_model = {display.model_id: display for display in displays}

    diagnostics = None
    if diagnostics_path is not None:
        diagnostic_table = pd.read_csv(diagnostics_path)
        diagnostics = _prepare_cluster_diagnostics(
            diagnostic_table,
            diagnostics_path,
            expected_environments=set(excess.columns.astype(str)),
        )

    style = {
        "font.family": "sans-serif",
        "font.sans-serif": ["DejaVu Sans"],
        "font.size": 8.0,
        "axes.labelsize": 8.2,
        "axes.linewidth": 0.65,
        "xtick.labelsize": 7.1,
        "ytick.labelsize": 7.1,
        "legend.fontsize": 6.8,
        "lines.linewidth": 1.25,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
    }
    with plt.rc_context(style):
        figure = plt.figure(figsize=(7.25, 5.05), constrained_layout=False)
        outer = figure.add_gridspec(2, 1, height_ratios=(1.72, 1.0), hspace=0.51)
        axis_a = figure.add_subplot(outer[0, 0])
        if diagnostics is None:
            bottom = outer[1, 0].subgridspec(1, 2, width_ratios=(1.0, 1.05), wspace=0.53)
            axis_b = figure.add_subplot(bottom[0, 0])
            axis_c = figure.add_subplot(bottom[0, 1])
            axis_d = None
        else:
            bottom = outer[1, 0].subgridspec(
                1, 3, width_ratios=(1.08, 1.13, 1.0), wspace=0.80
            )
            axis_b = figure.add_subplot(bottom[0, 0])
            axis_c = figure.add_subplot(bottom[0, 1])
            axis_d = figure.add_subplot(bottom[0, 2])

        _plot_heatmap(axis_a, figure, excess, displays, codes, default_model=default_model)
        _plot_mixture_regret(
            axis_b,
            regrets,
            codes,
            default_radius=default_radius,
        )
        _plot_robustness_path(
            axis_c,
            path,
            displays_by_model,
            codes,
            default_radius=default_radius,
        )
        if axis_d is not None and diagnostics is not None:
            _plot_root_diagnostics(axis_d, diagnostics)

        figure.subplots_adjust(left=0.235, right=0.955, bottom=0.145, top=0.94)
    return figure


def save_te_figure(
    ranking_heatmap: str | Path,
    target_mixture_regrets: str | Path,
    robustness_path: str | Path,
    selected_models: str | Path,
    cluster_diagnostics: str | Path | None,
    output_stem: str | Path,
    *,
    png_dpi: int = 240,
) -> tuple[Path, Path]:
    """Save an anonymous vector PDF and a deterministic raster preview."""

    if png_dpi < 72:
        raise ValueError("png_dpi must be at least 72")
    stem = Path(output_stem)
    stem.parent.mkdir(parents=True, exist_ok=True)
    pdf_path = stem.with_suffix(".pdf")
    png_path = stem.with_suffix(".png")
    figure = build_te_figure(
        ranking_heatmap,
        target_mixture_regrets,
        robustness_path,
        selected_models,
        cluster_diagnostics,
    )
    with plt.rc_context({"pdf.fonttype": 42, "ps.fonttype": 42}):
        figure.savefig(
            pdf_path,
            format="pdf",
            bbox_inches="tight",
            pad_inches=0.015,
            metadata={
                "Title": "Public TE-associated cCRE model-selection evaluation",
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
    root = Path("results/te_public/evaluation")
    parser.add_argument(
        "--ranking-heatmap",
        type=Path,
        default=root / "test_evaluation" / "ranking_heatmap.csv",
    )
    parser.add_argument(
        "--target-mixture-regrets",
        type=Path,
        default=root / "test_evaluation" / "target_mixture_regrets.csv.gz",
    )
    parser.add_argument(
        "--robustness-path",
        type=Path,
        default=root / "selection" / "robustness_path.csv",
    )
    parser.add_argument(
        "--selected-models",
        type=Path,
        default=root / "selection" / "selected_models.csv",
    )
    parser.add_argument(
        "--cluster-diagnostics",
        type=Path,
        default=root / "selection" / "cluster_diagnostics.csv",
    )
    parser.add_argument(
        "--without-root-diagnostics",
        action="store_true",
        help="Render only panels a--c even if cluster diagnostics are available.",
    )
    parser.add_argument(
        "--output-stem",
        type=Path,
        default=Path("paper/figures/te_public"),
    )
    parser.add_argument("--png-dpi", type=int, default=240)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    diagnostics = None if args.without_root_diagnostics else args.cluster_diagnostics
    outputs = save_te_figure(
        args.ranking_heatmap,
        args.target_mixture_regrets,
        args.robustness_path,
        args.selected_models,
        diagnostics,
        args.output_stem,
        png_dpi=args.png_dpi,
    )
    for path in outputs:
        print(path)
    return 0


if __name__ == "__main__":  # pragma: no cover - exercised through the CLI
    raise SystemExit(main())
