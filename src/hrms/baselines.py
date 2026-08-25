"""Prespecified point-estimate model-selection baselines."""

from __future__ import annotations

import numpy as np

from .ambiguity import ChiSquareBall
from .confidence_set import select_minimum_upper
from .pairwise import pairwise_differences
from .regret import point_robust_regret


def nominal_mean_scores(risks: np.ndarray, mixture: np.ndarray | None = None) -> np.ndarray:
    """Environment-mixture average risk for every model."""

    risks = np.asarray(risks, dtype=float)
    if risks.ndim != 2:
        raise ValueError("risks must have shape (environment, model)")
    if mixture is None:
        mixture = np.full(risks.shape[0], 1.0 / risks.shape[0])
    mixture = np.asarray(mixture, dtype=float)
    if mixture.shape != (risks.shape[0],) or np.any(mixture < 0):
        raise ValueError("mixture must be a non-negative vector over environments")
    if not np.isclose(mixture.sum(), 1.0):
        raise ValueError("mixture must sum to one")
    return mixture @ risks


def worst_environment_scores(risks: np.ndarray) -> np.ndarray:
    """Maximum absolute environment risk for every model."""

    risks = np.asarray(risks, dtype=float)
    if risks.ndim != 2:
        raise ValueError("risks must have shape (environment, model)")
    return risks.max(axis=0)


def environment_cvar_scores(risks: np.ndarray, tail_fraction: float = 0.25) -> np.ndarray:
    """Uniform-environment upper-tail CVaR with fractional boundary weight."""

    risks = np.asarray(risks, dtype=float)
    if risks.ndim != 2:
        raise ValueError("risks must have shape (environment, model)")
    if not 0 < tail_fraction <= 1:
        raise ValueError("tail_fraction must lie in (0, 1]")
    count = tail_fraction * risks.shape[0]
    whole = int(np.floor(count))
    fraction = count - whole
    ordered = np.sort(risks, axis=0)[::-1]
    total = ordered[:whole].sum(axis=0) if whole else np.zeros(risks.shape[1])
    if fraction > 0:
        total += fraction * ordered[whole]
    return total / count


def point_dro_risk_scores(risks: np.ndarray, ambiguity: ChiSquareBall) -> np.ndarray:
    """Worst-mixture absolute risks, without validation uncertainty."""

    risks = np.asarray(risks, dtype=float)
    if risks.ndim != 2:
        raise ValueError("risks must have shape (environment, model)")
    return np.asarray([ambiguity.support(risks[:, model]) for model in range(risks.shape[1])])


def point_dro_regret_scores(risks: np.ndarray, ambiguity: ChiSquareBall) -> np.ndarray:
    """Worst-mixture library regret, without validation uncertainty."""

    return point_robust_regret(pairwise_differences(risks), ambiguity)


def select_from_scores(scores: np.ndarray) -> int:
    return select_minimum_upper(np.asarray(scores, dtype=float))
