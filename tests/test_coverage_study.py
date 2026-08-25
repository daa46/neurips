import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from experiments.simulations.coverage_study import (
    _separate_risk_bootstrap_t_bounds,
    load_coverage_config,
    main,
    run_one,
)
from hrms.bootstrap import JackknifeBlock


def test_separate_risk_bounds_use_interval_arithmetic_and_are_reproducible():
    values = np.asarray(
        [
            [[0.10, 0.13, 0.21]],
            [[0.18, 0.20, 0.24]],
            [[0.22, 0.24, 0.30]],
            [[0.26, 0.27, 0.32]],
            [[0.30, 0.31, 0.35]],
        ]
    )
    blocks = (
        JackknifeBlock(
            environment_indices=np.asarray([0]),
            risk_pseudovalues=values,
        ),
    )
    point = values.mean(axis=0)
    first = _separate_risk_bootstrap_t_bounds(
        blocks,
        point,
        n_bootstrap=199,
        random_state=7,
    )
    second = _separate_risk_bootstrap_t_bounds(
        blocks,
        point,
        n_bootstrap=199,
        random_state=7,
    )

    np.testing.assert_array_equal(first.max_statistics, second.max_statistics)
    np.testing.assert_allclose(
        first.pairwise_lower[0, 1],
        first.risk_lower[:, 0] - first.risk_upper[:, 1],
    )
    np.testing.assert_allclose(
        first.pairwise_upper[0, 1],
        first.risk_upper[:, 0] - first.risk_lower[:, 1],
    )
    np.testing.assert_allclose(first.pairwise_lower, -first.pairwise_upper.swapaxes(0, 1))
    np.testing.assert_array_equal(
        np.diagonal(first.pairwise_lower, axis1=0, axis2=1),
        np.zeros((1, 3)),
    )


def test_separate_risk_zero_scale_is_conservative():
    values = np.full((6, 1, 2), [[[0.2, 0.4]]])
    result = _separate_risk_bootstrap_t_bounds(
        (
            JackknifeBlock(
                environment_indices=np.asarray([0]),
                risk_pseudovalues=values,
            ),
        ),
        np.asarray([[0.2, 0.4]]),
        n_bootstrap=49,
        random_state=1,
    )

    np.testing.assert_array_equal(result.risk_lower, np.zeros((1, 2)))
    np.testing.assert_array_equal(result.risk_upper, np.ones((1, 2)))
    assert result.pairwise_lower[0, 1, 0] == -1.0
    assert result.pairwise_upper[0, 1, 0] == 1.0


def test_coverage_run_reports_separate_risk_ablation():
    rows = run_one(
        n_environments=2,
        icc=0.2,
        n_candidates=4,
        clusters_per_environment=8,
        observations_per_cluster=4,
        imbalance="moderate",
        candidate_correlation=0.8,
        n_bootstrap=49,
        seed=123,
    )

    methods = {row["method"] for row in rows}
    assert "Separate-risk bootstrap-t" in methods
    row = next(row for row in rows if row["method"] == "Separate-risk bootstrap-t")
    assert 0.0 <= row["coordinate_coverage"] <= 1.0
    assert 0.0 <= row["mean_interval_width"] <= 2.0


def test_frozen_coverage_config_exactly_records_headline_seed_schedule():
    settings, declared = load_coverage_config(Path("configs/simulation/coverage_headline.json"))

    assert declared["status"] == "frozen_headline_run_with_pairing_ablation"
    assert settings == {
        "repetitions": 500,
        "n_bootstrap": 999,
        "n_environments": 3,
        "n_candidates": 10,
        "clusters_per_environment": 25,
        "observations_per_cluster": 10,
        "imbalance": "moderate",
        "candidate_correlation": 0.7,
        "iccs": (0.0, 0.1, 0.3, 0.5),
        "random_state": 42,
        "alpha": 0.05,
    }
    assert declared["seed_rule"] == "42 + 100000 * icc_index + repetition"


def test_config_mode_executes_serially_and_rejects_overrides(tmp_path, monkeypatch):
    declared = json.loads(
        Path("configs/simulation/coverage_headline.json").read_text(encoding="utf-8")
    )
    declared.update(
        repetitions_per_icc=1,
        n_bootstrap=9,
        n_environments=2,
        n_candidates=3,
        clusters_per_environment=4,
        mean_observations_per_cluster=2,
        iccs=[0.0, 0.2],
        base_seed=17,
        seed_rule="17 + 100000 * icc_index + repetition",
    )
    config = tmp_path / "coverage.json"
    config.write_text(json.dumps(declared), encoding="utf-8")
    output = tmp_path / "output"
    monkeypatch.setattr(
        sys,
        "argv",
        ["coverage_study.py", "--config", str(config), "--output-dir", str(output)],
    )
    main()

    runs = pd.read_csv(output / "coverage_runs.csv")
    assert len(runs) == 2 * 7
    assert runs.groupby("icc")["seed"].first().to_dict() == {0.0: 17, 0.2: 100_017}
    assert (output / "coverage_summary.csv").is_file()

    monkeypatch.setattr(
        sys,
        "argv",
        [
            "coverage_study.py",
            "--config",
            str(config),
            "--bootstrap",
            "11",
            "--output-dir",
            str(tmp_path / "conflict"),
        ],
    )
    with pytest.raises(SystemExit, match="2"):
        main()
