"""Contract and smoke tests for the controlled synthetic experiments."""

from __future__ import annotations

from dataclasses import replace

import numpy as np
import pandas as pd

from hrms import HRMS
from hrms.ambiguity import ChiSquareBall
from hrms.schema import validate_loss_table
from simulations.calibration import (
    CalibrationConfig,
    calibration_population_risks,
    generate_calibration_data,
)
from simulations.rank_reversal import (
    RankReversalConfig,
    generate_rank_reversal_data,
    rank_reversal_population_risks,
)
from simulations.run_grid import run_grid


def _assert_long_table_contract(table: pd.DataFrame, n_candidates: int) -> None:
    required = {
        "observation_id",
        "environment_id",
        "root_cluster_id",
        "model_id",
        "loss",
        "sample_weight",
    }
    assert required.issubset(table.columns)
    assert table.groupby("observation_id")["model_id"].nunique().eq(n_candidates).all()
    assert table.groupby("root_cluster_id")["environment_id"].nunique().eq(1).all()
    assert table["loss"].between(0.0, 1.0, inclusive="both").all()
    validated = validate_loss_table(table)
    assert validated.n_models == n_candidates


def test_calibration_generator_is_paired_nested_bounded_and_reproducible() -> None:
    config = CalibrationConfig(
        n_environments=3,
        n_candidates=4,
        clusters_per_environment=(3, 4, 5),
        mean_cluster_size=4,
        icc=0.4,
        candidate_correlation=0.75,
        cluster_size_imbalance="moderate",
        random_state=19,
    )
    first = generate_calibration_data(config)
    second = generate_calibration_data(config)

    _assert_long_table_contract(first.loss_table, n_candidates=4)
    pd.testing.assert_frame_equal(first.loss_table, second.loss_table)
    pd.testing.assert_frame_equal(
        first.true_environment_risks, second.true_environment_risks
    )
    observed_counts = first.cluster_sizes.groupby("environment_id").size().to_numpy()
    np.testing.assert_array_equal(observed_counts, [3, 4, 5])
    assert first.cluster_sizes["root_cluster_id"].nunique() == 12


def test_calibration_latent_icc_one_is_constant_within_cluster() -> None:
    simulation = generate_calibration_data(
        n_environments=2,
        n_candidates=3,
        clusters_per_environment=3,
        mean_cluster_size=5,
        cluster_size_imbalance="none",
        icc=1.0,
        candidate_correlation=0.4,
        random_state=7,
    )
    within_cluster_unique = simulation.loss_table.groupby(
        ["root_cluster_id", "model_id"]
    )["loss"].nunique()
    assert within_cluster_unique.eq(1).all()


def test_calibration_quadrature_agrees_with_large_monte_carlo_truth() -> None:
    config = CalibrationConfig(
        n_environments=2,
        n_candidates=3,
        clusters_per_environment=3,
        mean_cluster_size=2,
        latent_scale=1.2,
        population_size=250_000,
    )
    exact = calibration_population_risks(config, method="quadrature")
    monte_carlo = calibration_population_risks(
        config, method="monte_carlo", random_state=31
    )
    np.testing.assert_allclose(monte_carlo, exact, atol=2.5e-3)


def test_rank_reversal_generator_has_brier_losses_and_distinct_winners() -> None:
    config = RankReversalConfig(
        n_environments=5,
        n_candidates=8,
        clusters_per_environment=4,
        mean_cluster_size=4,
        cluster_size_imbalance="none",
        random_state=23,
        population_size=100_000,
    )
    simulation = generate_rank_reversal_data(config)

    _assert_long_table_contract(simulation.loss_table, n_candidates=8)
    expected_loss = np.square(
        simulation.loss_table["prediction"] - simulation.loss_table["label"]
    )
    np.testing.assert_allclose(simulation.loss_table["loss"], expected_loss)
    winners = set(simulation.true_environment_risks.idxmin(axis=1))
    assert "stable" in winners
    assert any("positive" in winner for winner in winners)
    assert any("negative" in winner for winner in winners)
    assert len(winners) >= 3


def test_rank_reversal_population_truth_is_independent_of_sample_seed() -> None:
    first = RankReversalConfig(
        n_environments=3,
        n_candidates=6,
        clusters_per_environment=3,
        mean_cluster_size=3,
        random_state=1,
        population_size=30_000,
        population_seed=101,
    )
    second = replace(first, random_state=999)

    risks_first, errors_first = rank_reversal_population_risks(first)
    risks_second, errors_second = rank_reversal_population_risks(second)

    pd.testing.assert_frame_equal(risks_first, risks_second)
    pd.testing.assert_frame_equal(errors_first, errors_second)
    assert (errors_first.to_numpy() > 0).all()


def test_generated_table_runs_through_public_hrms_api() -> None:
    simulation = generate_calibration_data(
        n_environments=3,
        n_candidates=4,
        clusters_per_environment=5,
        mean_cluster_size=3,
        cluster_size_imbalance="none",
        random_state=81,
    )
    result = HRMS(
        ambiguity=ChiSquareBall(center="uniform", radius=0.25),
        n_bootstrap=19,
        random_state=82,
        stability_replicates=10,
        robustness_radii=(0.0, 0.25, None),
    ).fit(simulation.loss_table)

    assert result.selected_model in simulation.true_environment_risks.columns
    assert result.selected_model in result.model_confidence_set
    assert result.pairwise_bounds.shape[0] == 18


def test_smoke_grid_writes_replicate_and_summary_csvs(tmp_path) -> None:
    outputs = run_grid(
        preset_name="smoke",
        output_dir=tmp_path,
        random_state=13,
        n_bootstrap=9,
        repetitions=1,
    )

    assert set(outputs) == {
        "calibration_runs",
        "calibration_summary",
        "rank_reversal_runs",
        "rank_reversal_summary",
    }
    for path in outputs.values():
        assert path.exists()
        assert not pd.read_csv(path).empty
    calibration_runs = pd.read_csv(outputs["calibration_runs"])
    rank_runs = pd.read_csv(outputs["rank_reversal_runs"])
    assert len(calibration_runs) == 3
    assert len(rank_runs) == 2
    assert {
        "joint_pairwise_coverage",
        "confidence_set_coverage",
        "selected_true_robust_regret",
    }.issubset(calibration_runs.columns)
