"""Paired root-cluster bootstrap for environment-specific risks."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import numpy as np

from .estimands import Weighting, cluster_cells, estimate_environment_risks
from .schema import LossData

Resampling = Literal["auto", "stratified", "global"]


class BootstrapError(RuntimeError):
    """Raised when the requested root-cluster bootstrap is not identifiable."""


@dataclass(frozen=True)
class BootstrapResult:
    point_risks: np.ndarray
    bootstrap_risks: np.ndarray
    pairwise_standard_error: np.ndarray
    studentized_max_statistics: np.ndarray | None
    cluster_counts: np.ndarray
    effective_cluster_counts: np.ndarray
    max_cluster_leverage: np.ndarray
    resampling: str
    weighting: str


@dataclass(frozen=True)
class JackknifeBlock:
    """Jointly resampled delete-root-cluster pseudovalues.

    ``risk_pseudovalues`` has shape ``cluster x environment x model``.  A
    stratified design has one single-environment block per environment; a
    global design has one block spanning all environments.
    """

    environment_indices: np.ndarray
    risk_pseudovalues: np.ndarray


def _clusters_are_nested(data: LossData) -> bool:
    count = len(data.cluster_ids)
    minimum = np.full(count, len(data.environment_ids), dtype=np.int64)
    maximum = np.full(count, -1, dtype=np.int64)
    np.minimum.at(minimum, data.observation_cluster, data.observation_environment)
    np.maximum.at(maximum, data.observation_cluster, data.observation_environment)
    return bool(np.array_equal(minimum, maximum))


def _resolve_resampling(data: LossData, resampling: Resampling) -> str:
    nested = _clusters_are_nested(data)
    if resampling == "auto":
        return "stratified" if nested else "global"
    if resampling == "stratified" and not nested:
        raise BootstrapError(
            "stratified resampling is invalid because at least one root cluster spans environments"
        )
    if resampling not in {"stratified", "global"}:
        raise ValueError("resampling must be 'auto', 'stratified', or 'global'")
    return resampling


def _pairwise_cluster_standard_error(
    loss_sums: np.ndarray,
    total_weights: np.ndarray,
    risk: np.ndarray,
) -> np.ndarray:
    """Cluster-sandwich SE for all paired differences of a ratio estimator.

    A cluster contributes ``loss_sums[c]`` to the numerator and
    ``total_weights[c]`` to the denominator.  Setting every weight to one and
    passing cluster means therefore gives the usual SE of a mean of paired
    cluster contrasts.
    """

    count = len(total_weights)
    if count < 2:
        raise BootstrapError("at least two sampled root clusters are required")
    mean_weight = float(np.mean(total_weights))
    if mean_weight <= 0:
        raise BootstrapError("sampled root clusters contain no observations")
    residual = loss_sums - total_weights[:, None] * risk[None, :]
    paired_residual = residual[:, :, None] - residual[:, None, :]
    return (
        np.sqrt(np.square(paired_residual).sum(axis=0) / (count * (count - 1)))
        / mean_weight
    )


def _cell_contributions(
    cell,
    *,
    weighting: Weighting,
) -> tuple[np.ndarray, np.ndarray]:
    if weighting == "observation":
        return cell.weighted_loss_sums, cell.total_weights
    return cell.cluster_means, np.ones(len(cell.cluster_indices), dtype=float)


def _global_contributions(
    cell,
    *,
    n_clusters: int,
    weighting: Weighting,
) -> tuple[np.ndarray, np.ndarray]:
    loss_sums = np.zeros((n_clusters, cell.cluster_means.shape[1]), dtype=float)
    total_weights = np.zeros(n_clusters, dtype=float)
    cell_sums, cell_weights = _cell_contributions(cell, weighting=weighting)
    loss_sums[cell.cluster_indices] = cell_sums
    total_weights[cell.cluster_indices] = cell_weights
    return loss_sums, total_weights


def cluster_bootstrap(
    data: LossData,
    *,
    n_bootstrap: int,
    weighting: Weighting = "observation",
    resampling: Resampling = "auto",
    studentize: bool = False,
    random_state: int | np.random.Generator | None = None,
) -> BootstrapResult:
    """Resample independent root clusters while retaining paired model losses."""

    if n_bootstrap < 2:
        raise ValueError("n_bootstrap must be at least 2")
    if weighting not in {"observation", "cluster"}:
        raise ValueError("weighting must be 'observation' or 'cluster'")
    resampling = _resolve_resampling(data, resampling)

    rng = (
        random_state
        if isinstance(random_state, np.random.Generator)
        else np.random.default_rng(random_state)
    )
    cells = cluster_cells(data)
    cluster_counts = np.asarray([len(cell.cluster_indices) for cell in cells], dtype=int)
    if weighting == "observation":
        normalized_cluster_weights = [
            cell.total_weights / cell.total_weights.sum() for cell in cells
        ]
    else:
        normalized_cluster_weights = [
            np.full(len(cell.cluster_indices), 1.0 / len(cell.cluster_indices))
            for cell in cells
        ]
    effective_cluster_counts = np.asarray(
        [1.0 / np.square(weights).sum() for weights in normalized_cluster_weights]
    )
    max_cluster_leverage = np.asarray(
        [weights.max() for weights in normalized_cluster_weights]
    )
    if np.any(cluster_counts < 2):
        bad = [
            data.environment_ids[idx]
            for idx, count in enumerate(cluster_counts)
            if count < 2
        ]
        raise BootstrapError(f"at least two root clusters are required per environment: {bad}")

    point = estimate_environment_risks(data, weighting)
    n_environments = len(data.environment_ids)
    n_models = data.n_models
    replicates = np.empty(
        (n_bootstrap, n_environments, n_models), dtype=float
    )
    pairwise_standard_error = np.empty((n_models, n_models, n_environments), dtype=float)
    studentized_max_statistics = (
        np.empty(n_bootstrap, dtype=float) if studentize else None
    )
    point_pairwise = point.T[:, None, :] - point.T[None, :, :]
    upper_triangle = np.triu(np.ones((n_models, n_models), dtype=bool), k=1)

    if resampling == "stratified":
        contributions = [
            _cell_contributions(cell, weighting=weighting) for cell in cells
        ]
    else:
        contributions = [
            _global_contributions(
                cell,
                n_clusters=len(data.cluster_ids),
                weighting=weighting,
            )
            for cell in cells
        ]
    for cell, (loss_sums, total_weights) in zip(cells, contributions, strict=True):
        pairwise_standard_error[:, :, cell.environment_index] = (
            _pairwise_cluster_standard_error(
                loss_sums,
                total_weights,
                point[cell.environment_index],
            )
        )

    if resampling == "stratified":
        for bootstrap_index in range(n_bootstrap):
            replicate_standard_error = (
                np.empty_like(pairwise_standard_error) if studentize else None
            )
            for cell, (loss_sums, total_weights) in zip(
                cells, contributions, strict=True
            ):
                count = len(cell.cluster_indices)
                sampled = rng.integers(0, count, size=count)
                sampled_sums = loss_sums[sampled]
                sampled_weights = total_weights[sampled]
                risk = sampled_sums.sum(axis=0) / sampled_weights.sum()
                replicates[bootstrap_index, cell.environment_index] = risk
                if replicate_standard_error is not None:
                    replicate_standard_error[:, :, cell.environment_index] = (
                        _pairwise_cluster_standard_error(
                            sampled_sums,
                            sampled_weights,
                            risk,
                        )
                    )
            if replicate_standard_error is not None:
                replicate_pairwise = (
                    replicates[bootstrap_index].T[:, None, :]
                    - replicates[bootstrap_index].T[None, :, :]
                )
                numerator = np.abs(replicate_pairwise - point_pairwise)
                active = np.broadcast_to(
                    upper_triangle[:, :, None], pairwise_standard_error.shape
                ) & (pairwise_standard_error > 1e-14)
                ratios = np.zeros_like(numerator)
                positive_scale = active & (replicate_standard_error > 1e-14)
                np.divide(
                    numerator,
                    replicate_standard_error,
                    out=ratios,
                    where=positive_scale,
                )
                ratios[active & ~positive_scale & (numerator > 1e-14)] = np.inf
                studentized_max_statistics[bootstrap_index] = ratios[active].max(
                    initial=0.0
                )
    else:
        n_clusters = len(data.cluster_ids)
        for bootstrap_index in range(n_bootstrap):
            sampled = rng.integers(0, n_clusters, size=n_clusters)
            replicate_standard_error = (
                np.empty_like(pairwise_standard_error) if studentize else None
            )
            for cell, (loss_sums, total_weights) in zip(
                cells, contributions, strict=True
            ):
                sampled_sums = loss_sums[sampled]
                sampled_weights = total_weights[sampled]
                if sampled_weights.sum() == 0:
                    raise BootstrapError(
                        "a global bootstrap replicate omitted an entire environment; "
                        "use more spanning clusters or revise the environment definition"
                    )
                risk = sampled_sums.sum(axis=0) / sampled_weights.sum()
                replicates[bootstrap_index, cell.environment_index] = risk
                if replicate_standard_error is not None:
                    replicate_standard_error[:, :, cell.environment_index] = (
                        _pairwise_cluster_standard_error(
                            sampled_sums,
                            sampled_weights,
                            risk,
                        )
                    )
            if replicate_standard_error is not None:
                replicate_pairwise = (
                    replicates[bootstrap_index].T[:, None, :]
                    - replicates[bootstrap_index].T[None, :, :]
                )
                numerator = np.abs(replicate_pairwise - point_pairwise)
                active = np.broadcast_to(
                    upper_triangle[:, :, None], pairwise_standard_error.shape
                ) & (pairwise_standard_error > 1e-14)
                ratios = np.zeros_like(numerator)
                positive_scale = active & (replicate_standard_error > 1e-14)
                np.divide(
                    numerator,
                    replicate_standard_error,
                    out=ratios,
                    where=positive_scale,
                )
                ratios[active & ~positive_scale & (numerator > 1e-14)] = np.inf
                studentized_max_statistics[bootstrap_index] = ratios[active].max(
                    initial=0.0
                )

    return BootstrapResult(
        point_risks=point,
        bootstrap_risks=replicates,
        pairwise_standard_error=pairwise_standard_error,
        studentized_max_statistics=studentized_max_statistics,
        cluster_counts=cluster_counts,
        effective_cluster_counts=effective_cluster_counts,
        max_cluster_leverage=max_cluster_leverage,
        resampling=resampling,
        weighting=weighting,
    )


def cluster_jackknife_pseudovalues(
    data: LossData,
    *,
    weighting: Weighting = "observation",
    resampling: Resampling = "auto",
) -> tuple[JackknifeBlock, ...]:
    """Construct delete-root-cluster pseudovalues for smooth risk estimators.

    Pseudovalues linearize the observation-weighted ratio estimator while
    retaining every model in the same vector.  Blocks mirror the bootstrap
    design, so a root spanning environments is deleted jointly and its
    cross-environment dependence is preserved.
    """

    if weighting not in {"observation", "cluster"}:
        raise ValueError("weighting must be 'observation' or 'cluster'")
    resolved = _resolve_resampling(data, resampling)
    cells = cluster_cells(data)
    point = estimate_environment_risks(data, weighting)

    if resolved == "stratified":
        blocks: list[JackknifeBlock] = []
        for cell in cells:
            count = len(cell.cluster_indices)
            if count < 2:
                raise BootstrapError("at least two root clusters are required per environment")
            loss_sums, total_weights = _cell_contributions(
                cell, weighting=weighting
            )
            total_loss = loss_sums.sum(axis=0)
            total_weight = total_weights.sum()
            leave_one_out = (total_loss[None, :] - loss_sums) / (
                total_weight - total_weights
            )[:, None]
            pseudo = count * point[cell.environment_index][None, :] - (
                count - 1
            ) * leave_one_out
            blocks.append(
                JackknifeBlock(
                    environment_indices=np.asarray([cell.environment_index], dtype=int),
                    risk_pseudovalues=pseudo[:, None, :],
                )
            )
        return tuple(blocks)

    count = len(data.cluster_ids)
    pseudo = np.empty((count, len(data.environment_ids), data.n_models), dtype=float)
    for cell in cells:
        loss_sums, total_weights = _global_contributions(
            cell,
            n_clusters=count,
            weighting=weighting,
        )
        total_loss = loss_sums.sum(axis=0)
        total_weight = total_weights.sum()
        remaining_weight = total_weight - total_weights
        if np.any(remaining_weight <= 0):
            raise BootstrapError(
                "delete-cluster calibration requires at least two contributing roots "
                "per environment"
            )
        leave_one_out = (total_loss[None, :] - loss_sums) / remaining_weight[:, None]
        pseudo[:, cell.environment_index, :] = (
            count * point[cell.environment_index][None, :] - (count - 1) * leave_one_out
        )
    return (
        JackknifeBlock(
            environment_indices=np.arange(len(data.environment_ids), dtype=int),
            risk_pseudovalues=pseudo,
        ),
    )
