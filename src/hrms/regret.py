"""Point and certified robust library regret."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

import numpy as np


class AmbiguitySet(Protocol):
    def support(self, values: np.ndarray, *, return_weights: bool = False): ...


@dataclass(frozen=True)
class RegretCertificates:
    lower: np.ndarray
    upper: np.ndarray
    lower_comparator: np.ndarray
    upper_comparator: np.ndarray


def point_robust_regret(pairwise: np.ndarray, ambiguity: AmbiguitySet) -> np.ndarray:
    """Return robust regret for each candidate from exact pairwise differences."""

    pairwise = np.asarray(pairwise, dtype=float)
    if pairwise.ndim != 3 or pairwise.shape[0] != pairwise.shape[1]:
        raise ValueError("pairwise must have shape (model, model, environment)")
    n_models = pairwise.shape[0]
    values = np.empty(n_models, dtype=float)
    for model in range(n_models):
        values[model] = max(
            float(ambiguity.support(pairwise[model, comparator])) for comparator in range(n_models)
        )
    return np.maximum(values, 0.0)


def robust_regret_certificates(
    lower_pairwise: np.ndarray,
    upper_pairwise: np.ndarray,
    ambiguity: AmbiguitySet,
) -> RegretCertificates:
    """Optimize simultaneous pairwise bounds over comparators and mixtures."""

    lower_pairwise = np.asarray(lower_pairwise, dtype=float)
    upper_pairwise = np.asarray(upper_pairwise, dtype=float)
    if lower_pairwise.shape != upper_pairwise.shape:
        raise ValueError("lower and upper pairwise arrays must have equal shape")
    if lower_pairwise.ndim != 3 or lower_pairwise.shape[0] != lower_pairwise.shape[1]:
        raise ValueError("pairwise arrays must have shape (model, model, environment)")
    if np.any(lower_pairwise > upper_pairwise + 1e-12):
        raise ValueError("lower pairwise bounds exceed upper bounds")

    n_models = lower_pairwise.shape[0]
    lower = np.empty(n_models, dtype=float)
    upper = np.empty(n_models, dtype=float)
    lower_comparator = np.empty(n_models, dtype=int)
    upper_comparator = np.empty(n_models, dtype=int)
    for model in range(n_models):
        lower_by_comparator = np.asarray(
            [
                ambiguity.support(lower_pairwise[model, comparator])
                for comparator in range(n_models)
            ],
            dtype=float,
        )
        upper_by_comparator = np.asarray(
            [
                ambiguity.support(upper_pairwise[model, comparator])
                for comparator in range(n_models)
            ],
            dtype=float,
        )
        lower_comparator[model] = int(np.argmax(lower_by_comparator))
        upper_comparator[model] = int(np.argmax(upper_by_comparator))
        lower[model] = max(float(lower_by_comparator[lower_comparator[model]]), 0.0)
        upper[model] = max(float(upper_by_comparator[upper_comparator[model]]), 0.0)

    if np.any(lower > upper + 1e-10):
        raise RuntimeError("numerical failure: a lower robust-regret bound exceeds its upper bound")
    return RegretCertificates(
        lower=lower,
        upper=upper,
        lower_comparator=lower_comparator,
        upper_comparator=upper_comparator,
    )
