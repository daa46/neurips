"""Environment-risk estimands and cluster-cell summaries."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import numpy as np

from .schema import LossData

Weighting = Literal["observation", "cluster"]


@dataclass(frozen=True)
class EnvironmentCells:
    environment_index: int
    cluster_indices: np.ndarray
    weighted_loss_sums: np.ndarray
    total_weights: np.ndarray
    cluster_means: np.ndarray


def cluster_cells(data: LossData) -> tuple[EnvironmentCells, ...]:
    """Aggregate observations into environment-by-root-cluster cells."""

    cells: list[EnvironmentCells] = []
    for environment in range(len(data.environment_ids)):
        observation_indices = np.flatnonzero(data.observation_environment == environment)
        represented, inverse = np.unique(
            data.observation_cluster[observation_indices], return_inverse=True
        )
        weights = data.weights[observation_indices]
        totals = np.bincount(inverse, weights=weights, minlength=len(represented))
        sums = np.zeros((len(represented), data.n_models), dtype=float)
        np.add.at(
            sums,
            inverse,
            data.losses[observation_indices] * weights[:, None],
        )
        means = sums / totals[:, None]
        cells.append(
            EnvironmentCells(
                environment_index=environment,
                cluster_indices=represented,
                weighted_loss_sums=sums,
                total_weights=totals,
                cluster_means=means,
            )
        )
    return tuple(cells)


def estimate_environment_risks(data: LossData, weighting: Weighting = "observation") -> np.ndarray:
    """Return an ``environment x model`` risk matrix."""

    if weighting not in {"observation", "cluster"}:
        raise ValueError("weighting must be 'observation' or 'cluster'")
    risks = np.empty((len(data.environment_ids), data.n_models), dtype=float)
    for cell in cluster_cells(data):
        if weighting == "observation":
            risks[cell.environment_index] = (
                cell.weighted_loss_sums.sum(axis=0) / cell.total_weights.sum()
            )
        else:
            risks[cell.environment_index] = cell.cluster_means.mean(axis=0)
    return risks
