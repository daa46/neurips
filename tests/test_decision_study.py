from pathlib import Path

import pandas as pd
import pytest

from simulations.decision_study import (
    METHODS,
    environment_signals,
    load_decision_config,
    run_study,
)
from simulations.rank_reversal import RankReversalConfig, candidate_library


def test_environment_signal_path_is_frozen_and_reverses_one_environment():
    baseline = environment_signals(0.0)
    severe = environment_signals(2.5)

    assert len(baseline) == len(severe) == 8
    assert baseline[0] == pytest.approx(0.4)
    assert severe[0] == pytest.approx(-2.6)
    assert baseline[1:] == severe[1:]


@pytest.mark.parametrize("invalid", [-1.0, float("inf"), float("nan")])
def test_environment_signal_path_rejects_invalid_strength(invalid):
    with pytest.raises(ValueError):
        environment_signals(invalid)


def test_frozen_headline_config_records_every_oracle_determining_default():
    path = Path("configs/simulation/decision_headline.json")
    settings, declared = load_decision_config(path)

    assert declared["status"] == "frozen_before_headline_runs"
    assert settings["population_size"] == 500_000
    assert settings["population_seed"] == 29_082_026
    assert settings["calibration"] == "jackknife_bootstrap_t"
    assert settings["rank_reversal_severities"] == (0.0, 0.5, 1.0, 1.5, 2.0, 2.5)
    assert tuple(spec.model_id for spec in settings["candidate_specs"]) == (
        "stable",
        "stable_regularized",
        "positive_specialist",
        "negative_specialist",
        "positive_blend",
        "negative_blend",
        "positive_sharp",
        "stable_soft",
    )
    defaults = candidate_library(
        RankReversalConfig(
            n_environments=8,
            n_candidates=8,
            spurious_signals=environment_signals(0.0),
            population_size=2,
        )
    )
    assert settings["candidate_specs"] == defaults


def test_parallel_schedule_preserves_seeded_results_and_output_order():
    settings = {
        "repetitions": 2,
        "n_bootstrap": 9,
        "clusters_per_environment": 3,
        "mean_cluster_size": 2,
        "population_size": 2_000,
        "random_state": 11,
        "rank_reversal_severities": (0.0,),
    }

    serial_runs, serial_truth = run_study(**settings, jobs=1)
    parallel_runs, parallel_truth = run_study(**settings, jobs=2)

    pd.testing.assert_frame_equal(serial_runs, parallel_runs)
    pd.testing.assert_frame_equal(serial_truth, parallel_truth)
    assert serial_runs["method"].tolist() == list(METHODS) * 2
