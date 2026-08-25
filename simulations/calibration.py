"""Controlled correlated-loss simulation for interval-calibration experiments.

The generator follows the paper's direct-loss experiment: every observation has
a bounded loss for every candidate, candidates are evaluated on the same
observations, and root clusters are the independent sampling units.  The latent
construction has unit marginal variance, an ICC-like within-candidate cluster
correlation, and a separately configurable cross-candidate correlation.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import numpy as np
import pandas as pd
from scipy.special import expit

ClusterImbalance = Literal["none", "moderate", "severe"]
PopulationMethod = Literal["quadrature", "monte_carlo"]


@dataclass(frozen=True)
class CalibrationConfig:
    """Parameters for one correlated-loss calibration sample.

    ``icc`` is the latent correlation between two observations from the same
    candidate and root cluster. ``candidate_correlation`` is the latent
    correlation between two candidates evaluated on the same observation.
    Cluster sizes are generated independently of all latent effects, so the
    population observation- and cluster-weighted risks coincide.
    """

    n_environments: int = 3
    n_candidates: int = 5
    clusters_per_environment: int | tuple[int, ...] = 25
    mean_cluster_size: float = 10.0
    icc: float = 0.3
    candidate_correlation: float = 0.7
    cluster_size_imbalance: ClusterImbalance = "moderate"
    latent_scale: float = 1.0
    locations: tuple[tuple[float, ...], ...] | None = None
    random_state: int = 0
    population_method: PopulationMethod = "quadrature"
    population_size: int = 200_000
    population_seed: int = 1_904_2026

    def __post_init__(self) -> None:
        if self.n_environments < 1:
            raise ValueError("n_environments must be positive")
        if self.n_candidates < 2:
            raise ValueError("n_candidates must be at least two")
        _cluster_counts(self)
        if not np.isfinite(self.mean_cluster_size) or self.mean_cluster_size < 1:
            raise ValueError("mean_cluster_size must be finite and at least one")
        if not 0.0 <= self.icc <= 1.0:
            raise ValueError("icc must lie in [0, 1]")
        if not 0.0 <= self.candidate_correlation <= 1.0:
            raise ValueError("candidate_correlation must lie in [0, 1]")
        if self.cluster_size_imbalance not in {"none", "moderate", "severe"}:
            raise ValueError(
                "cluster_size_imbalance must be 'none', 'moderate', or 'severe'"
            )
        if not np.isfinite(self.latent_scale) or self.latent_scale < 0:
            raise ValueError("latent_scale must be finite and non-negative")
        if self.population_method not in {"quadrature", "monte_carlo"}:
            raise ValueError("population_method must be 'quadrature' or 'monte_carlo'")
        if self.population_size < 1:
            raise ValueError("population_size must be positive")
        _location_matrix(self)


@dataclass(frozen=True)
class CalibrationSimulation:
    """A sampled long loss table together with its population truth."""

    loss_table: pd.DataFrame
    true_environment_risks: pd.DataFrame
    latent_locations: pd.DataFrame
    cluster_sizes: pd.DataFrame
    config: CalibrationConfig

    @property
    def population_risks(self) -> pd.DataFrame:
        """Alias emphasizing that truth is independent of the finite sample."""

        return self.true_environment_risks


def _cluster_counts(config: CalibrationConfig) -> np.ndarray:
    raw = config.clusters_per_environment
    if isinstance(raw, (int, np.integer)):
        counts = np.full(config.n_environments, int(raw), dtype=int)
    else:
        counts = np.asarray(tuple(raw), dtype=int)
        if counts.shape != (config.n_environments,):
            raise ValueError(
                "clusters_per_environment must be an integer or have one entry "
                "per environment"
            )
    if np.any(counts < 2):
        raise ValueError("at least two root clusters are required per environment")
    return counts


def _environment_ids(config: CalibrationConfig) -> tuple[str, ...]:
    return tuple(f"environment_{index:03d}" for index in range(config.n_environments))


def _model_ids(config: CalibrationConfig) -> tuple[str, ...]:
    return tuple(f"model_{index:03d}" for index in range(config.n_candidates))


def _default_locations(n_environments: int, n_candidates: int) -> np.ndarray:
    """Create mildly heterogeneous risks without making calibration trivial."""

    environment_axis = np.arange(n_environments, dtype=float)[:, None]
    candidate_axis = np.arange(n_candidates, dtype=float)[None, :]
    candidate_effect = np.linspace(-0.28, 0.28, n_candidates)[None, :]
    environment_effect = 0.10 * np.cos(
        2.0 * np.pi * environment_axis / max(n_environments, 2)
    )
    interaction = 0.22 * np.sin(
        2.0
        * np.pi
        * (candidate_axis / max(n_candidates, 2) + environment_axis / max(n_environments, 2))
    )
    return -1.0 + candidate_effect + environment_effect + interaction


def _location_matrix(config: CalibrationConfig) -> np.ndarray:
    if config.locations is None:
        return _default_locations(config.n_environments, config.n_candidates)
    locations = np.asarray(config.locations, dtype=float)
    expected = (config.n_environments, config.n_candidates)
    if locations.shape != expected:
        raise ValueError(f"locations must have shape {expected}")
    if not np.isfinite(locations).all():
        raise ValueError("locations must be finite")
    return locations.copy()


def _draw_cluster_sizes(
    rng: np.random.Generator,
    count: int,
    mean: float,
    imbalance: ClusterImbalance,
) -> np.ndarray:
    if imbalance == "none":
        return np.full(count, max(1, int(round(mean))), dtype=int)

    sigma = 0.55 if imbalance == "moderate" else 1.15
    # The location adjustment gives the log-normal multiplier mean one.  A cap
    # protects development grids from a single pathological draw while still
    # leaving the severe setting strongly unbalanced.
    multipliers = rng.lognormal(mean=-0.5 * sigma**2, sigma=sigma, size=count)
    upper = max(2, int(np.ceil(mean * (5 if imbalance == "moderate" else 12))))
    return np.clip(np.rint(mean * multipliers), 1, upper).astype(int)


def calibration_population_risks(
    config: CalibrationConfig,
    *,
    method: PopulationMethod | None = None,
    n_samples: int | None = None,
    random_state: int | None = None,
) -> pd.DataFrame:
    """Return population environment risks, independent of the sampled table.

    Gauss--Hermite quadrature is effectively exact for the logistic-normal
    marginal used here.  ``method='monte_carlo'`` is supplied for stress tests
    and uses common random numbers across all model-environment coordinates.
    """

    method = config.population_method if method is None else method
    if method not in {"quadrature", "monte_carlo"}:
        raise ValueError("method must be 'quadrature' or 'monte_carlo'")
    locations = _location_matrix(config)

    if method == "quadrature":
        nodes, weights = np.polynomial.hermite.hermgauss(64)
        standard_normal = np.sqrt(2.0) * nodes
        values = expit(
            locations[..., None] + config.latent_scale * standard_normal[None, None, :]
        )
        risks = np.sum(values * weights[None, None, :], axis=-1) / np.sqrt(np.pi)
    else:
        size = config.population_size if n_samples is None else int(n_samples)
        if size < 1:
            raise ValueError("n_samples must be positive")
        seed = config.population_seed if random_state is None else random_state
        rng = np.random.default_rng(seed)
        standard_normal = rng.normal(size=size)
        risks = np.empty_like(locations)
        # Loop over environments to bound the peak allocation for large grids.
        for environment in range(config.n_environments):
            latent = (
                locations[environment, :, None]
                + config.latent_scale * standard_normal[None, :]
            )
            risks[environment] = expit(latent).mean(axis=1)

    return pd.DataFrame(
        risks,
        index=pd.Index(_environment_ids(config), name="environment_id"),
        columns=pd.Index(_model_ids(config), name="model_id"),
    )


def generate_calibration_data(
    config: CalibrationConfig | None = None,
    **config_overrides,
) -> CalibrationSimulation:
    """Generate one paired long-form loss table and its population risks.

    Pass either a :class:`CalibrationConfig` or keyword arguments accepted by
    that dataclass.  Root identifiers include their environment identifier and
    are globally unique, making environment nesting explicit and auditable.
    """

    if config is not None and config_overrides:
        raise TypeError("pass either config or keyword overrides, not both")
    config = CalibrationConfig(**config_overrides) if config is None else config
    if not isinstance(config, CalibrationConfig):
        raise TypeError("config must be a CalibrationConfig")

    rng = np.random.default_rng(config.random_state)
    locations = _location_matrix(config)
    environment_ids = _environment_ids(config)
    model_ids = _model_ids(config)
    cluster_counts = _cluster_counts(config)

    metadata_parts: list[pd.DataFrame] = []
    loss_parts: list[np.ndarray] = []
    cluster_rows: list[dict[str, object]] = []
    observation_counter = 0
    shared_cluster_scale = np.sqrt(config.icc)
    residual_scale = np.sqrt(1.0 - config.icc)
    common_candidate_scale = np.sqrt(config.candidate_correlation)
    idiosyncratic_candidate_scale = np.sqrt(1.0 - config.candidate_correlation)

    for environment, environment_id in enumerate(environment_ids):
        sizes = _draw_cluster_sizes(
            rng,
            int(cluster_counts[environment]),
            config.mean_cluster_size,
            config.cluster_size_imbalance,
        )
        for cluster, size in enumerate(sizes):
            root_id = f"{environment_id}__root_{cluster:05d}"
            cluster_common = rng.normal()
            cluster_by_candidate = rng.normal(size=config.n_candidates)
            cluster_latent = (
                common_candidate_scale * cluster_common
                + idiosyncratic_candidate_scale * cluster_by_candidate
            )
            observation_common = rng.normal(size=(int(size), 1))
            observation_by_candidate = rng.normal(
                size=(int(size), config.n_candidates)
            )
            observation_latent = (
                common_candidate_scale * observation_common
                + idiosyncratic_candidate_scale * observation_by_candidate
            )
            standardized_latent = (
                shared_cluster_scale * cluster_latent[None, :]
                + residual_scale * observation_latent
            )
            losses = expit(
                locations[environment][None, :]
                + config.latent_scale * standardized_latent
            )

            indices = np.arange(observation_counter, observation_counter + int(size))
            metadata_parts.append(
                pd.DataFrame(
                    {
                        "observation_id": [f"calibration_obs_{index:08d}" for index in indices],
                        "environment_id": environment_id,
                        "root_cluster_id": root_id,
                        "sample_weight": 1.0,
                    }
                )
            )
            loss_parts.append(losses)
            cluster_rows.append(
                {
                    "environment_id": environment_id,
                    "root_cluster_id": root_id,
                    "n_observations": int(size),
                }
            )
            observation_counter += int(size)

    metadata = pd.concat(metadata_parts, ignore_index=True)
    loss_matrix = np.concatenate(loss_parts, axis=0)
    wide = pd.concat(
        [metadata, pd.DataFrame(loss_matrix, columns=model_ids)], axis=1
    )
    loss_table = wide.melt(
        id_vars=[
            "observation_id",
            "environment_id",
            "root_cluster_id",
            "sample_weight",
        ],
        value_vars=list(model_ids),
        var_name="model_id",
        value_name="loss",
    )
    loss_table = loss_table.sort_values(
        ["observation_id", "model_id"], kind="stable", ignore_index=True
    )

    location_frame = pd.DataFrame(
        locations,
        index=pd.Index(environment_ids, name="environment_id"),
        columns=pd.Index(model_ids, name="model_id"),
    )
    return CalibrationSimulation(
        loss_table=loss_table,
        true_environment_risks=calibration_population_risks(config),
        latent_locations=location_frame,
        cluster_sizes=pd.DataFrame.from_records(cluster_rows),
        config=config,
    )


def generate_calibration_loss_table(**config_overrides) -> pd.DataFrame:
    """Convenience wrapper returning only the long loss table."""

    return generate_calibration_data(**config_overrides).loss_table


# Short, discoverable alias used by notebooks and grid code.
simulate_calibration = generate_calibration_data


__all__ = [
    "CalibrationConfig",
    "CalibrationSimulation",
    "calibration_population_risks",
    "generate_calibration_data",
    "generate_calibration_loss_table",
    "simulate_calibration",
]
