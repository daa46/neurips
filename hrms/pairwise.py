"""Paired model-risk differences."""

from __future__ import annotations

import numpy as np


def pairwise_differences(risks: np.ndarray) -> np.ndarray:
    """Convert risks to ``model x model x environment`` differences.

    Input may be ``environment x model`` or
    ``bootstrap x environment x model``. The output respectively has shape
    ``model x model x environment`` or
    ``bootstrap x model x model x environment``.
    """

    risks = np.asarray(risks, dtype=float)
    if risks.ndim == 2:
        transposed = risks.T
        return transposed[:, None, :] - transposed[None, :, :]
    if risks.ndim == 3:
        transposed = np.transpose(risks, (0, 2, 1))
        return transposed[:, :, None, :] - transposed[:, None, :, :]
    raise ValueError(
        "risks must have shape (environment, model) or (bootstrap, environment, model)"
    )


def identical_candidate_pairs(losses: np.ndarray, tolerance: float = 1e-14) -> np.ndarray:
    """Return a symmetric mask for observation-wise identical candidates."""

    losses = np.asarray(losses, dtype=float)
    if losses.ndim != 2:
        raise ValueError("losses must have shape (observation, model)")
    difference = np.max(np.abs(losses[:, :, None] - losses[:, None, :]), axis=0)
    return difference <= tolerance
