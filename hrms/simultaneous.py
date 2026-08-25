"""Simultaneous max-standardized bootstrap bounds."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from .bootstrap import JackknifeBlock


@dataclass(frozen=True)
class SimultaneousBounds:
    estimate: np.ndarray
    lower: np.ndarray
    upper: np.ndarray
    standard_error: np.ndarray
    critical_value: float
    max_statistics: np.ndarray
    degenerate_coordinates: tuple[tuple[int, int, int], ...]
    calibration_method: str


@dataclass(frozen=True)
class StudentizedCalibration:
    standard_error: np.ndarray
    max_statistics: np.ndarray
    method: str


def jackknife_bootstrap_t_calibration(
    blocks: tuple[JackknifeBlock, ...],
    *,
    n_models: int,
    n_environments: int,
    n_bootstrap: int,
    random_state: int | np.random.Generator | None = None,
    zero_tolerance: float = 1e-14,
) -> StudentizedCalibration:
    """Calibrate a paired max statistic from delete-cluster pseudovalues.

    Each replicate resamples root-cluster pseudovalues, recomputes its own
    paired standard errors, and contributes the maximum absolute bootstrap-t
    statistic.  Joint blocks preserve dependence from roots spanning multiple
    environments.
    """

    if n_models < 2 or n_environments < 1:
        raise ValueError("at least two models and one environment are required")
    if n_bootstrap < 2:
        raise ValueError("n_bootstrap must be at least two")
    rng = (
        random_state
        if isinstance(random_state, np.random.Generator)
        else np.random.default_rng(random_state)
    )
    expected_environments = set(range(n_environments))
    represented_environments: set[int] = set()
    standard_error = np.zeros((n_models, n_models, n_environments), dtype=float)
    first_model, second_model = np.triu_indices(n_models, k=1)
    paired_blocks: list[np.ndarray] = []
    for block in blocks:
        values = np.asarray(block.risk_pseudovalues, dtype=float)
        environments = np.asarray(block.environment_indices, dtype=int)
        if values.ndim != 3 or values.shape[1:] != (len(environments), n_models):
            raise ValueError(
                "risk_pseudovalues must have shape (cluster, block environment, model)"
            )
        if len(values) < 2 or not np.isfinite(values).all():
            raise ValueError("each jackknife block needs at least two finite clusters")
        if any(int(environment) in represented_environments for environment in environments):
            raise ValueError("each environment must occur in exactly one jackknife block")
        represented_environments.update(int(environment) for environment in environments)
        paired = values[:, :, first_model] - values[:, :, second_model]
        centered = paired - paired.mean(axis=0, keepdims=True)
        block_standard_error = paired.std(axis=0, ddof=1) / np.sqrt(len(values))
        for position, environment in enumerate(environments):
            se = block_standard_error[position]
            standard_error[first_model, second_model, environment] = se
            standard_error[second_model, first_model, environment] = se
        paired_blocks.append(centered.reshape(len(values), -1))
    if represented_environments != expected_environments:
        raise ValueError("jackknife blocks must cover every environment exactly once")

    max_statistics = np.zeros(n_bootstrap, dtype=float)
    for centered in paired_blocks:
        count, coordinate_count = centered.shape
        original_standard_error = centered.std(axis=0, ddof=1) / np.sqrt(count)
        active = original_standard_error > zero_tolerance
        squared = np.square(centered)
        batch_size = max(1, min(128, 2_000_000 // count))
        for start in range(0, n_bootstrap, batch_size):
            stop = min(start + batch_size, n_bootstrap)
            size = stop - start
            sampled = rng.integers(0, count, size=(size, count))
            multiplicity = np.zeros((size, count), dtype=float)
            rows = np.repeat(np.arange(size), count)
            np.add.at(multiplicity, (rows, sampled.ravel()), 1.0)
            sums = multiplicity @ centered
            square_sums = multiplicity @ squared
            means = sums / count
            variances = (square_sums - np.square(sums) / count) / (count - 1)
            replicate_standard_error = np.sqrt(np.maximum(variances, 0.0) / count)
            ratios = np.zeros((size, coordinate_count), dtype=float)
            positive_scale = active[None, :] & (
                replicate_standard_error > zero_tolerance
            )
            np.divide(
                np.abs(means),
                replicate_standard_error,
                out=ratios,
                where=positive_scale,
            )
            ratios[
                active[None, :]
                & ~positive_scale
                & (np.abs(means) > zero_tolerance)
            ] = np.inf
            block_maxima = ratios[:, active].max(axis=1, initial=0.0)
            max_statistics[start:stop] = np.maximum(
                max_statistics[start:stop], block_maxima
            )
    return StudentizedCalibration(
        standard_error=standard_error,
        max_statistics=max_statistics,
        method="jackknife-pseudovalue cluster bootstrap-t",
    )


def max_t_bounds(
    estimate: np.ndarray,
    bootstrap_estimates: np.ndarray,
    *,
    alpha: float = 0.05,
    identical_pairs: np.ndarray | None = None,
    zero_tolerance: float = 1e-14,
    contrast_bounds: tuple[float, float] = (-1.0, 1.0),
    standard_error: np.ndarray | None = None,
    max_statistics: np.ndarray | None = None,
    calibration_method: str | None = None,
) -> SimultaneousBounds:
    """Construct one two-sided envelope over all unique pairs and environments.

    By default this reproduces the ordinary fixed-denominator bootstrap max-t
    envelope.  Passing both ``standard_error`` and ``max_statistics`` uses a
    studentized bootstrap: the caller supplies the original cluster-sandwich
    SE and maxima calculated with a replicate-specific cluster-sandwich SE.
    """

    estimate = np.asarray(estimate, dtype=float)
    bootstrap_estimates = np.asarray(bootstrap_estimates, dtype=float)
    if estimate.ndim != 3 or estimate.shape[0] != estimate.shape[1]:
        raise ValueError("estimate must have shape (model, model, environment)")
    if bootstrap_estimates.ndim != 4 or bootstrap_estimates.shape[1:] != estimate.shape:
        raise ValueError(
            "bootstrap_estimates must have shape (bootstrap, model, model, environment)"
        )
    if not 0 < alpha < 1:
        raise ValueError("alpha must lie strictly between zero and one")
    if contrast_bounds[0] > contrast_bounds[1]:
        raise ValueError("contrast_bounds must be ordered")
    n_models, _, n_environments = estimate.shape
    if identical_pairs is None:
        identical_pairs = np.eye(n_models, dtype=bool)
    identical_pairs = np.asarray(identical_pairs, dtype=bool)
    if identical_pairs.shape != (n_models, n_models):
        raise ValueError("identical_pairs must have shape (model, model)")

    supplied_studentization = standard_error is not None or max_statistics is not None
    if supplied_studentization and (standard_error is None or max_statistics is None):
        raise ValueError(
            "standard_error and max_statistics must be supplied together"
        )
    if standard_error is None:
        standard_error = bootstrap_estimates.std(axis=0, ddof=1)
    else:
        standard_error = np.asarray(standard_error, dtype=float)
        if standard_error.shape != estimate.shape:
            raise ValueError("standard_error must have the same shape as estimate")
        if np.any(standard_error < 0) or not np.isfinite(standard_error).all():
            raise ValueError("standard_error must be finite and non-negative")
    upper_triangle = np.triu(np.ones((n_models, n_models), dtype=bool), k=1)
    unique_mask = np.broadcast_to(upper_triangle[:, :, None], estimate.shape)
    active = unique_mask & (standard_error > zero_tolerance)

    if supplied_studentization:
        max_statistics = np.asarray(max_statistics, dtype=float)
        if max_statistics.shape != (bootstrap_estimates.shape[0],):
            raise ValueError("max_statistics must have one entry per bootstrap replicate")
        if np.any(max_statistics < 0) or np.isnan(max_statistics).any():
            raise ValueError("max_statistics must be non-negative and not NaN")
    else:
        centered = bootstrap_estimates - estimate[None, ...]
        standardized = np.zeros_like(centered)
        np.divide(
            np.abs(centered),
            standard_error[None, ...],
            out=standardized,
            where=standard_error[None, ...] > zero_tolerance,
        )
        max_statistics = (
            standardized[:, active].max(axis=1)
            if np.any(active)
            else np.zeros(bootstrap_estimates.shape[0], dtype=float)
        )
    if np.any(active):
        critical_value = float(
            np.quantile(max_statistics, 1 - alpha, method="higher")
        )
    else:
        max_statistics = np.zeros(bootstrap_estimates.shape[0], dtype=float)
        critical_value = 0.0

    lower = np.zeros_like(estimate)
    upper = np.zeros_like(estimate)
    degenerate: list[tuple[int, int, int]] = []
    for first in range(n_models):
        for second in range(first + 1, n_models):
            for environment in range(n_environments):
                if identical_pairs[first, second]:
                    lo = hi = 0.0
                else:
                    se = standard_error[first, second, environment]
                    if se <= zero_tolerance:
                        degenerate.append((first, second, environment))
                        lo, hi = contrast_bounds
                    else:
                        lo = max(
                            contrast_bounds[0],
                            estimate[first, second, environment] - critical_value * se,
                        )
                        hi = min(
                            contrast_bounds[1],
                            estimate[first, second, environment] + critical_value * se,
                        )
                lower[first, second, environment] = lo
                upper[first, second, environment] = hi
                lower[second, first, environment] = -hi
                upper[second, first, environment] = -lo

    return SimultaneousBounds(
        estimate=estimate,
        lower=lower,
        upper=upper,
        standard_error=standard_error,
        critical_value=critical_value,
        max_statistics=max_statistics,
        degenerate_coordinates=tuple(degenerate),
        calibration_method=(
            calibration_method
            or (
                "studentized cluster bootstrap-t"
                if supplied_studentization
                else "ordinary cluster bootstrap max-t"
            )
        ),
    )
