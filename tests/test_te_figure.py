from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pytest

from scripts.make_te_figure import (
    METHOD_SPECS,
    _prepare_heatmap,
    build_te_figure,
    save_te_figure,
)

MODELS = (
    "lr_w128_k4_c0p1",
    "lr_w256_k4_c1",
    "lr_te_k6_c0p1",
    "sgd_w128_k5_a1e4",
    "cnb_w256_k5_a0p5",
    "cnb_te_k5_a0p5",
)
ENVIRONMENTS = ("Alu", "L1", "MIR", "ERV1")


def _write_fixture(directory: Path) -> dict[str, Path]:
    directory.mkdir(parents=True, exist_ok=True)
    risk_rows: list[dict[str, object]] = []
    for partition in ("select", "test"):
        for weighting in ("observation", "cluster"):
            for environment_index, environment in enumerate(ENVIRONMENTS):
                for model_index, model in enumerate(MODELS):
                    interaction = 0.0025 * abs(model_index - environment_index)
                    partition_shift = 0.001 if partition == "test" else 0.0
                    weighting_shift = 0.0004 if weighting == "cluster" else 0.0
                    risk_rows.append(
                        {
                            "partition": partition,
                            "root_definition": "homology_root",
                            "weighting": weighting,
                            "environment_id": environment,
                            "model_id": model,
                            "risk": 0.13 + interaction + partition_shift + weighting_shift,
                            "rank": model_index + 1,
                        }
                    )
    ranking = pd.DataFrame(risk_rows)
    ranking_path = directory / "ranking_heatmap.csv"
    ranking.to_csv(ranking_path, index=False, lineterminator="\n")

    selected_by_method = {
        "uniform_mean": MODELS[0],
        "worst_environment": MODELS[1],
        "point_dro_risk": MODELS[2],
        "point_dro_regret": MODELS[1],
        "hrms": MODELS[1],
    }
    selected_rows = []
    for method, _, _ in METHOD_SPECS:
        selected_rows.append(
            {
                "method_id": f"root_observation__{method}",
                "analysis_id": "root_observation",
                "root_definition": "homology_root",
                "weighting": "observation",
                "method": method,
                "selection_radius": "0.25" if method == "hrms" else "not_applicable",
                "evaluation_radius": "0.25",
                "selected_model": selected_by_method[method],
                "selection_score": 0.01,
                "score_definition": "fixture",
                "primary_comparison": True,
            }
        )
    selected_path = directory / "selected_models.csv"
    pd.DataFrame(selected_rows).to_csv(selected_path, index=False, lineterminator="\n")

    radius_settings = (
        ("0", MODELS[0], 2),
        ("0.1", MODELS[1], 3),
        ("0.25", MODELS[1], 4),
        ("0.5", MODELS[2], 4),
        ("1", MODELS[2], 5),
        ("full", MODELS[4], 6),
    )
    path_rows = []
    for radius_index, (radius, selected, set_size) in enumerate(radius_settings):
        included = set(MODELS[:set_size]) | {selected}
        # Keep the declared set size exact even if the selected model was added.
        if len(included) > set_size:
            removable = next(model for model in reversed(MODELS[:set_size]) if model != selected)
            included.remove(removable)
        for model_index, model in enumerate(MODELS):
            upper = 0.008 + 0.0015 * radius_index + 0.0004 * model_index
            path_rows.append(
                {
                    "analysis_id": "root_observation",
                    "root_definition": "homology_root",
                    "weighting": "observation",
                    "radius": radius,
                    "model": model,
                    "lower": max(0.0, upper - 0.006),
                    "upper": upper,
                    "selected": model == selected,
                    "in_confidence_set": model in included,
                    "confidence_set_size": set_size,
                }
            )
    robustness_path = directory / "robustness_path.csv"
    pd.DataFrame(path_rows).to_csv(robustness_path, index=False, lineterminator="\n")

    scopes = (
        "nominal",
        "inside_primary",
        "inside_primary",
        "inside_primary",
        "inside_primary",
        "primary_boundary",
        "primary_boundary",
        "primary_boundary",
        "outside_primary",
        "simplex_vertex",
    )
    regret_rows = []
    for method_index, (method, _, _) in enumerate(METHOD_SPECS):
        for mixture_index, scope in enumerate(scopes):
            regret_rows.append(
                {
                    "mixture_id": f"mixture_{mixture_index:05d}",
                    "scope": scope,
                    "chi_square_divergence": 0.03 * mixture_index,
                    "method_id": f"root_observation__{method}",
                    "analysis_id": "root_observation",
                    "root_definition": "homology_root",
                    "weighting": "observation",
                    "method": method,
                    "selection_radius": "0.25" if method == "hrms" else "not_applicable",
                    "selected_model": selected_by_method[method],
                    "selected_test_risk": 0.14,
                    "oracle_test_risk": 0.13,
                    "test_regret": 0.0004 * method_index + 0.0002 * mixture_index,
                    "selected_is_mixture_optimal": mixture_index == 0,
                    "primary_comparison": True,
                }
            )
    regrets_path = directory / "target_mixture_regrets.csv.gz"
    pd.DataFrame(regret_rows).to_csv(
        regrets_path,
        index=False,
        compression={"method": "gzip", "mtime": 0},
        lineterminator="\n",
    )

    diagnostics_rows = []
    for index, environment in enumerate(ENVIRONMENTS):
        diagnostics_rows.append(
            {
                "analysis_id": "root_observation",
                "root_definition": "homology_root",
                "weighting": "observation",
                "environment_id": environment,
                "independent_root_clusters": 42 + 5 * index,
                "effective_root_clusters": 35.5 + 4 * index,
                "max_cluster_leverage": 0.035 + 0.004 * index,
            }
        )
    diagnostics_path = directory / "cluster_diagnostics.csv"
    pd.DataFrame(diagnostics_rows).to_csv(
        diagnostics_path, index=False, lineterminator="\n"
    )
    return {
        "ranking": ranking_path,
        "selected": selected_path,
        "path": robustness_path,
        "regrets": regrets_path,
        "diagnostics": diagnostics_path,
    }


def test_te_figure_reads_tidy_contract_and_exports_vector_and_preview(tmp_path: Path) -> None:
    paths = _write_fixture(tmp_path / "fixture")
    figure = build_te_figure(
        paths["ranking"],
        paths["regrets"],
        paths["path"],
        paths["selected"],
        paths["diagnostics"],
    )
    panel_titles = {text.get_text() for axis in figure.axes for text in axis.texts}
    assert any("Candidate performance varies" in value for value in panel_titles)
    assert any("Mixture regret" in value for value in panel_titles)
    assert any("Radius sensitivity" in value for value in panel_titles)
    assert any("Independent evidence" in value for value in panel_titles)
    plt.close(figure)

    pdf, png = save_te_figure(
        paths["ranking"],
        paths["regrets"],
        paths["path"],
        paths["selected"],
        paths["diagnostics"],
        tmp_path / "figures" / "te_public",
        png_dpi=100,
    )
    pdf_bytes = pdf.read_bytes()
    assert pdf_bytes.startswith(b"%PDF")
    assert png.read_bytes().startswith(b"\x89PNG\r\n\x1a\n")
    assert pdf.stat().st_size > 10_000
    assert png.stat().st_size > 10_000
    assert b"Anonymous artifact" in pdf_bytes
    assert b"/Subtype /Image" not in pdf_bytes


def test_heatmap_order_and_excess_brier_are_result_order_invariant(tmp_path: Path) -> None:
    paths = _write_fixture(tmp_path / "fixture")
    table = pd.read_csv(paths["ranking"])
    shuffled = table.sample(frac=1.0, random_state=91).reset_index(drop=True)
    first, first_displays, first_codes = _prepare_heatmap(table, paths["ranking"])
    second, second_displays, second_codes = _prepare_heatmap(
        shuffled, tmp_path / "shuffled.csv"
    )
    pd.testing.assert_frame_equal(first, second)
    assert [value.model_id for value in first_displays] == [
        value.model_id for value in second_displays
    ]
    assert first_codes == second_codes
    np.testing.assert_allclose(first.min(axis=0), 0.0)


def test_te_figure_rejects_incomplete_risk_grid(tmp_path: Path) -> None:
    paths = _write_fixture(tmp_path / "fixture")
    table = pd.read_csv(paths["ranking"])
    target = (
        table["partition"].eq("test")
        & table["weighting"].eq("observation")
        & table["model_id"].eq(MODELS[0])
        & table["environment_id"].eq(ENVIRONMENTS[0])
    )
    table = table.loc[~target]
    table.to_csv(paths["ranking"], index=False, lineterminator="\n")
    with pytest.raises(ValueError, match="incomplete candidate-by-environment grid"):
        build_te_figure(
            paths["ranking"],
            paths["regrets"],
            paths["path"],
            paths["selected"],
            paths["diagnostics"],
        )


def test_te_figure_rejects_inconsistent_non_exclusion_set(tmp_path: Path) -> None:
    paths = _write_fixture(tmp_path / "fixture")
    table = pd.read_csv(paths["path"], dtype={"radius": str})
    table.loc[table["radius"].eq("0.25"), "confidence_set_size"] = 1
    table.to_csv(paths["path"], index=False, lineterminator="\n")
    with pytest.raises(ValueError, match="inconsistent non-exclusion-set size"):
        build_te_figure(
            paths["ranking"],
            paths["regrets"],
            paths["path"],
            paths["selected"],
            paths["diagnostics"],
        )


def test_te_figure_can_omit_optional_root_diagnostics(tmp_path: Path) -> None:
    paths = _write_fixture(tmp_path / "fixture")
    figure = build_te_figure(
        paths["ranking"],
        paths["regrets"],
        paths["path"],
        paths["selected"],
        None,
    )
    panel_titles = {text.get_text() for axis in figure.axes for text in axis.texts}
    assert not any("Independent evidence" in value for value in panel_titles)
    plt.close(figure)
