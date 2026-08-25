from __future__ import annotations

import pandas as pd

from scripts.make_figures import make_figures


def test_figure_export_reads_result_contract_and_writes_pdf_png(tmp_path):
    simulations = tmp_path / "simulations"
    simulations.mkdir()
    pd.DataFrame(
        {
            "icc": [0.0, 0.3],
            "n_candidates": [5, 5],
            "cluster_size_imbalance": ["none", "moderate"],
            "joint_pairwise_coverage": [0.9, 0.8],
            "coordinate_coverage": [0.98, 0.97],
            "n_repetitions": [10, 10],
        }
    ).to_csv(simulations / "calibration_summary.csv", index=False)
    pd.DataFrame(
        {
            "shift_strength": [0.5, 1.0],
            "mean_selected_true_robust_regret": [0.02, 0.03],
            "point_dro_selected_true_robust_regret": [0.02, 0.025],
            "worst_environment_selected_true_robust_regret": [0.03, 0.04],
            "selected_true_robust_regret": [0.015, 0.02],
            "n_repetitions": [10, 10],
        }
    ).to_csv(simulations / "rank_reversal_summary.csv", index=False)

    outputs = make_figures(simulations_dir=simulations, output_dir=tmp_path / "figures")
    assert set(outputs) == {
        "calibration_pdf",
        "calibration_png",
        "rank_reversal_pdf",
        "rank_reversal_png",
    }
    assert all(path.is_file() and path.stat().st_size > 0 for path in outputs.values())


def test_coverage_figure_does_not_drop_methods_beyond_base_palette(tmp_path, monkeypatch):
    simulations = tmp_path / "simulations"
    simulations.mkdir()
    pd.DataFrame(
        {
            "icc": [0.0],
            "n_candidates": [5],
            "cluster_size_imbalance": ["none"],
            "joint_pairwise_coverage": [0.9],
            "coordinate_coverage": [0.98],
            "n_repetitions": [10],
        }
    ).to_csv(simulations / "calibration_summary.csv", index=False)
    pd.DataFrame(
        {
            "shift_strength": [0.5],
            "mean_selected_true_robust_regret": [0.02],
            "point_dro_selected_true_robust_regret": [0.02],
            "worst_environment_selected_true_robust_regret": [0.03],
            "selected_true_robust_regret": [0.015],
            "n_repetitions": [10],
        }
    ).to_csv(simulations / "rank_reversal_summary.csv", index=False)
    methods = [f"method_{index}" for index in range(6)]
    coverage = pd.DataFrame(
        {
            "icc": [0.0] * len(methods),
            "method": methods,
            "joint_coverage_mean": [0.9] * len(methods),
            "joint_coverage_sem": [0.01] * len(methods),
        }
    )
    coverage_path = tmp_path / "coverage.csv"
    coverage.to_csv(coverage_path, index=False)

    import matplotlib.axes

    labels: list[str] = []
    original = matplotlib.axes.Axes.errorbar

    def recording_errorbar(self, *args, **kwargs):
        labels.append(kwargs["label"])
        return original(self, *args, **kwargs)

    monkeypatch.setattr(matplotlib.axes.Axes, "errorbar", recording_errorbar)
    outputs = make_figures(
        simulations_dir=simulations,
        output_dir=tmp_path / "figures",
        coverage_summary=coverage_path,
    )
    assert labels == methods
    assert outputs["coverage_pdf"].is_file()
