"""Immutable public result object and tidy exports."""

from __future__ import annotations

from collections.abc import Hashable
from dataclasses import dataclass

import numpy as np
import pandas as pd

from .diagnostics import Diagnostic


@dataclass(frozen=True)
class HRMSResult:
    """Outputs from one HRMS fit.

    ``selection_stability`` is the raw root-cluster-bootstrap frequency with
    which each candidate minimizes point-estimated robust regret. It is a
    descriptive point-winner diagnostic, not a nested HRMS refit or posterior
    probability.
    """

    selected_model: Hashable
    robust_regret_lower: pd.Series
    robust_regret_upper: pd.Series
    model_confidence_set: tuple[Hashable, ...]
    environment_risks: pd.DataFrame
    pairwise_bounds: pd.DataFrame
    selection_stability: pd.Series
    robustness_path: pd.DataFrame
    cluster_counts: pd.Series
    effective_cluster_counts: pd.Series
    max_cluster_leverage: pd.Series
    diagnostics: tuple[Diagnostic, ...]
    critical_value: float
    calibration_method: str
    alpha: float
    resampling: str
    weighting: str

    def certificate_frame(self) -> pd.DataFrame:
        return pd.DataFrame(
            {
                "lower": self.robust_regret_lower,
                "upper": self.robust_regret_upper,
                "selected": [idx == self.selected_model for idx in self.robust_regret_lower.index],
                "in_confidence_set": [
                    idx in self.model_confidence_set for idx in self.robust_regret_lower.index
                ],
            }
        )


def tidy_pairwise_bounds(
    model_ids: tuple[Hashable, ...],
    environment_ids: tuple[Hashable, ...],
    estimate: np.ndarray,
    lower: np.ndarray,
    upper: np.ndarray,
    standard_error: np.ndarray,
) -> pd.DataFrame:
    rows: list[dict] = []
    for first, first_id in enumerate(model_ids):
        for second in range(first + 1, len(model_ids)):
            for environment, environment_id in enumerate(environment_ids):
                rows.append(
                    {
                        "model": first_id,
                        "comparator": model_ids[second],
                        "environment": environment_id,
                        "estimate": estimate[first, second, environment],
                        "lower": lower[first, second, environment],
                        "upper": upper[first, second, environment],
                        "standard_error": standard_error[first, second, environment],
                    }
                )
    return pd.DataFrame.from_records(rows)
