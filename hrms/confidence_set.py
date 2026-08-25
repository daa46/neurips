"""Confidence set for robust-optimal candidates."""

from __future__ import annotations

import numpy as np


def select_minimum_upper(
    upper: np.ndarray,
    *,
    tie_breaker: np.ndarray | None = None,
    tolerance: float = 1e-12,
) -> int:
    """Select an upper-bound minimizer, breaking ties by a point criterion."""

    upper = np.asarray(upper, dtype=float)
    if upper.ndim != 1 or len(upper) == 0 or not np.isfinite(upper).all():
        raise ValueError("upper must be a non-empty finite vector")
    minimum = upper.min()
    candidates = np.flatnonzero(upper <= minimum + tolerance)
    if tie_breaker is not None:
        tie_breaker = np.asarray(tie_breaker, dtype=float)
        if tie_breaker.shape != upper.shape or not np.isfinite(tie_breaker).all():
            raise ValueError("tie_breaker must be a finite vector matching upper")
        best_point = tie_breaker[candidates].min()
        candidates = candidates[tie_breaker[candidates] <= best_point + tolerance]
    return int(candidates[0])


def robust_model_confidence_set(
    lower: np.ndarray,
    upper: np.ndarray,
    *,
    tolerance: float = 1e-12,
) -> np.ndarray:
    """Return indices not excluded from robust optimality."""

    lower = np.asarray(lower, dtype=float)
    upper = np.asarray(upper, dtype=float)
    if lower.shape != upper.shape or lower.ndim != 1 or len(lower) == 0:
        raise ValueError("lower and upper must be equal-length non-empty vectors")
    if np.any(lower > upper + tolerance):
        raise ValueError("lower bounds exceed upper bounds")
    threshold = upper.min()
    return np.flatnonzero(lower <= threshold + tolerance)
