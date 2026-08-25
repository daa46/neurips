from __future__ import annotations

import pandas as pd

from scripts.make_simulation_figure import save_simulation_figure


def test_simulation_figure_contract_and_anonymous_pdf(tmp_path):
    coverage_methods = [
        "IID marginal",
        "Cluster marginal",
        "Paired cluster max-t",
        "Paired cluster jackknife bootstrap-t",
        "Paired cluster bootstrap-t",
        "Separate-risk bootstrap-t",
    ]
    coverage_rows = []
    for icc in (0.0, 0.3):
        for index, method in enumerate(coverage_methods):
            coverage_rows.append(
                {
                    "icc": icc,
                    "method": method,
                    "joint_coverage_mean": min(0.99, 0.75 + 0.04 * index),
                    "joint_coverage_sem": 0.02,
                    "mean_interval_width_mean": 0.04 + 0.01 * index + 0.1 * icc,
                    "mean_interval_width_sem": 0.002,
                }
            )
    coverage_path = tmp_path / "coverage.csv"
    pd.DataFrame(coverage_rows).to_csv(coverage_path, index=False)

    decision_rows = []
    for shift in (0.0, 1.0):
        for index, method in enumerate(
            ("Mean validation", "Point robust regret", "HRMS (rho=0)", "HRMS")
        ):
            decision_rows.append(
                {
                    "shift_strength": shift,
                    "method": method,
                    "selected_true_robust_regret_mean": 0.02 + 0.01 * index + 0.01 * shift,
                    "selected_true_robust_regret_sem": 0.002,
                    "confidence_set_size_mean": 3.0 if method == "HRMS" else float("nan"),
                    "confidence_set_size_sem": 0.1 if method == "HRMS" else float("nan"),
                    "confidence_set_covers_mean": 0.95 if method == "HRMS" else float("nan"),
                    "confidence_set_covers_sem": 0.02 if method == "HRMS" else float("nan"),
                }
            )
    decision_path = tmp_path / "decision.csv"
    pd.DataFrame(decision_rows).to_csv(decision_path, index=False)

    pdf, png = save_simulation_figure(
        coverage_path,
        decision_path,
        tmp_path / "simulation",
        png_dpi=72,
    )

    assert pdf.is_file() and pdf.stat().st_size > 1_000
    assert png.is_file() and png.stat().st_size > 1_000
    pdf_bytes = pdf.read_bytes()
    assert b"personal_username" not in pdf_bytes.lower()
    assert b"/home/" not in pdf_bytes.lower()
