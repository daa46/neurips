"""Feature-based simulation with environment-dependent model-rank reversal."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy.special import expit

from experiments.simulations.calibration import ClusterImbalance


@dataclass(frozen=True)
class CandidateSpec:
    """A fixed probabilistic candidate in the synthetic library."""

    model_id: str
    stable_weight: float
    spurious_weight: float
    temperature: float = 1.0
    bias: float = 0.0

    def __post_init__(self) -> None:
        values = (self.stable_weight, self.spurious_weight, self.temperature, self.bias)
        if not all(np.isfinite(value) for value in values):
            raise ValueError("candidate parameters must be finite")
        if self.temperature <= 0:
            raise ValueError("candidate temperature must be positive")
        if not self.model_id:
            raise ValueError("model_id must be non-empty")


@dataclass(frozen=True)
class RankReversalConfig:
    """Parameters for clustered binary prediction under spurious shift."""

    n_environments: int = 5
    n_candidates: int = 8
    clusters_per_environment: int | tuple[int, ...] = 25
    mean_cluster_size: float = 10.0
    cluster_size_imbalance: ClusterImbalance = "moderate"
    stable_signal: float = 1.0
    spurious_signals: tuple[float, ...] | None = None
    cluster_effect_sd: float = 0.65
    observation_noise_sd: float = 1.0
    positive_prevalence: float = 0.5
    candidate_specs: tuple[CandidateSpec, ...] | None = None
    random_state: int = 0
    population_size: int = 200_000
    population_seed: int = 2_908_2026

    def __post_init__(self) -> None:
        if self.n_environments < 2:
            raise ValueError("n_environments must be at least two")
        if self.n_candidates < 4:
            raise ValueError("n_candidates must be at least four")
        _cluster_counts(self)
        if not np.isfinite(self.mean_cluster_size) or self.mean_cluster_size < 1:
            raise ValueError("mean_cluster_size must be finite and at least one")
        if self.cluster_size_imbalance not in {"none", "moderate", "severe"}:
            raise ValueError("cluster_size_imbalance must be 'none', 'moderate', or 'severe'")
        if not np.isfinite(self.stable_signal) or self.stable_signal <= 0:
            raise ValueError("stable_signal must be finite and positive")
        signals = _spurious_signals(self)
        if not np.isfinite(signals).all():
            raise ValueError("spurious_signals must be finite")
        if not np.isfinite(self.cluster_effect_sd) or self.cluster_effect_sd < 0:
            raise ValueError("cluster_effect_sd must be finite and non-negative")
        if not np.isfinite(self.observation_noise_sd) or self.observation_noise_sd <= 0:
            raise ValueError("observation_noise_sd must be finite and positive")
        if not 0 < self.positive_prevalence < 1:
            raise ValueError("positive_prevalence must lie strictly between zero and one")
        if self.population_size < 1:
            raise ValueError("population_size must be positive")
        specs = candidate_library(self)
        if len({spec.model_id for spec in specs}) != len(specs):
            raise ValueError("candidate model identifiers must be unique")


@dataclass(frozen=True)
class RankReversalSimulation:
    """Sample, frozen candidate registry, and deterministic population truth."""

    loss_table: pd.DataFrame
    true_environment_risks: pd.DataFrame
    population_standard_errors: pd.DataFrame
    candidate_registry: pd.DataFrame
    cluster_sizes: pd.DataFrame
    config: RankReversalConfig

    @property
    def population_risks(self) -> pd.DataFrame:
        return self.true_environment_risks


def _cluster_counts(config: RankReversalConfig) -> np.ndarray:
    raw = config.clusters_per_environment
    if isinstance(raw, (int, np.integer)):
        counts = np.full(config.n_environments, int(raw), dtype=int)
    else:
        counts = np.asarray(tuple(raw), dtype=int)
        if counts.shape != (config.n_environments,):
            raise ValueError(
                "clusters_per_environment must be an integer or have one entry per environment"
            )
    if np.any(counts < 2):
        raise ValueError("at least two root clusters are required per environment")
    return counts


def _spurious_signals(config: RankReversalConfig) -> np.ndarray:
    if config.spurious_signals is None:
        return np.linspace(-1.8, 1.8, config.n_environments)
    signals = np.asarray(config.spurious_signals, dtype=float)
    if signals.shape != (config.n_environments,):
        raise ValueError("spurious_signals must have one entry per environment")
    return signals


def _environment_ids(config: RankReversalConfig) -> tuple[str, ...]:
    return tuple(f"environment_{index:03d}" for index in range(config.n_environments))


def _default_candidate_library(n_candidates: int) -> tuple[CandidateSpec, ...]:
    base = [
        CandidateSpec("stable", stable_weight=1.35, spurious_weight=0.0),
        CandidateSpec(
            "stable_regularized",
            stable_weight=0.95,
            spurious_weight=0.10,
            temperature=1.15,
        ),
        CandidateSpec("positive_specialist", stable_weight=0.25, spurious_weight=1.45),
        CandidateSpec("negative_specialist", stable_weight=0.25, spurious_weight=-1.45),
        CandidateSpec("positive_blend", stable_weight=0.85, spurious_weight=0.70),
        CandidateSpec("negative_blend", stable_weight=0.85, spurious_weight=-0.70),
        CandidateSpec(
            "positive_sharp",
            stable_weight=0.45,
            spurious_weight=1.05,
            temperature=0.65,
        ),
        CandidateSpec("stable_soft", stable_weight=1.30, spurious_weight=0.0, temperature=1.55),
    ]
    if n_candidates <= len(base):
        return tuple(base[:n_candidates])

    extras: list[CandidateSpec] = []
    for index in range(n_candidates - len(base)):
        fraction = (index + 1) / (n_candidates - len(base) + 1)
        extras.append(
            CandidateSpec(
                model_id=f"continuum_{index:03d}",
                stable_weight=0.55 + 0.65 * (1.0 - abs(2.0 * fraction - 1.0)),
                spurious_weight=-1.20 + 2.40 * fraction,
                temperature=(0.8, 1.0, 1.3)[index % 3],
            )
        )
    return tuple(base + extras)


def candidate_library(config: RankReversalConfig) -> tuple[CandidateSpec, ...]:
    """Resolve and validate the frozen candidate library."""

    if config.candidate_specs is None:
        return _default_candidate_library(config.n_candidates)
    specs = tuple(config.candidate_specs)
    if len(specs) != config.n_candidates:
        raise ValueError("candidate_specs length must equal n_candidates")
    if not all(isinstance(spec, CandidateSpec) for spec in specs):
        raise TypeError("candidate_specs entries must be CandidateSpec instances")
    return specs


def _draw_cluster_sizes(
    rng: np.random.Generator,
    count: int,
    mean: float,
    imbalance: ClusterImbalance,
) -> np.ndarray:
    if imbalance == "none":
        return np.full(count, max(1, int(round(mean))), dtype=int)
    sigma = 0.55 if imbalance == "moderate" else 1.15
    multipliers = rng.lognormal(mean=-0.5 * sigma**2, sigma=sigma, size=count)
    upper = max(2, int(np.ceil(mean * (5 if imbalance == "moderate" else 12))))
    return np.clip(np.rint(mean * multipliers), 1, upper).astype(int)


def _predict(
    stable_feature: np.ndarray,
    spurious_feature: np.ndarray,
    specs: tuple[CandidateSpec, ...],
) -> np.ndarray:
    stable_weights = np.asarray([spec.stable_weight for spec in specs])
    spurious_weights = np.asarray([spec.spurious_weight for spec in specs])
    temperatures = np.asarray([spec.temperature for spec in specs])
    biases = np.asarray([spec.bias for spec in specs])
    scores = (
        stable_feature[:, None] * stable_weights[None, :]
        + spurious_feature[:, None] * spurious_weights[None, :]
        + biases[None, :]
    ) / temperatures[None, :]
    return expit(scores)


def rank_reversal_population_risks(
    config: RankReversalConfig,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Estimate true risks with a large, seed-fixed Monte Carlo population.

    Common random numbers are reused across environments and candidates.  This
    makes pairwise population-risk differences much more precise and guarantees
    that truth does not change when only ``config.random_state`` changes.
    """

    rng = np.random.default_rng(config.population_seed)
    size = config.population_size
    labels = rng.binomial(1, config.positive_prevalence, size=size).astype(float)
    signed_labels = 2.0 * labels - 1.0
    stable_cluster_effect = rng.normal(scale=config.cluster_effect_sd, size=size)
    spurious_cluster_effect = rng.normal(scale=config.cluster_effect_sd, size=size)
    stable_noise = rng.normal(scale=config.observation_noise_sd, size=size)
    spurious_noise = rng.normal(scale=config.observation_noise_sd, size=size)
    stable_feature = signed_labels * config.stable_signal + stable_cluster_effect + stable_noise

    specs = candidate_library(config)
    risks = np.empty((config.n_environments, config.n_candidates), dtype=float)
    standard_errors = np.empty_like(risks)
    for environment, gamma in enumerate(_spurious_signals(config)):
        spurious_feature = signed_labels * gamma + spurious_cluster_effect + spurious_noise
        predictions = _predict(stable_feature, spurious_feature, specs)
        losses = np.square(predictions - labels[:, None])
        risks[environment] = losses.mean(axis=0)
        standard_errors[environment] = losses.std(axis=0, ddof=1) / np.sqrt(size)

    environment_ids = _environment_ids(config)
    model_ids = tuple(spec.model_id for spec in specs)
    index = pd.Index(environment_ids, name="environment_id")
    columns = pd.Index(model_ids, name="model_id")
    return (
        pd.DataFrame(risks, index=index, columns=columns),
        pd.DataFrame(standard_errors, index=index, columns=columns),
    )


def generate_rank_reversal_data(
    config: RankReversalConfig | None = None,
    **config_overrides,
) -> RankReversalSimulation:
    """Generate clustered labels, features, predictions, and Brier losses."""

    if config is not None and config_overrides:
        raise TypeError("pass either config or keyword overrides, not both")
    config = RankReversalConfig(**config_overrides) if config is None else config
    if not isinstance(config, RankReversalConfig):
        raise TypeError("config must be a RankReversalConfig")

    rng = np.random.default_rng(config.random_state)
    specs = candidate_library(config)
    environment_ids = _environment_ids(config)
    signals = _spurious_signals(config)
    cluster_counts = _cluster_counts(config)
    model_ids = tuple(spec.model_id for spec in specs)

    metadata_parts: list[pd.DataFrame] = []
    prediction_parts: list[np.ndarray] = []
    loss_parts: list[np.ndarray] = []
    cluster_rows: list[dict[str, object]] = []
    observation_counter = 0

    for environment, environment_id in enumerate(environment_ids):
        sizes = _draw_cluster_sizes(
            rng,
            int(cluster_counts[environment]),
            config.mean_cluster_size,
            config.cluster_size_imbalance,
        )
        for cluster, size in enumerate(sizes):
            root_id = f"{environment_id}__root_{cluster:05d}"
            labels = rng.binomial(1, config.positive_prevalence, size=int(size)).astype(float)
            signed_labels = 2.0 * labels - 1.0
            stable_cluster_effect = rng.normal(scale=config.cluster_effect_sd)
            spurious_cluster_effect = rng.normal(scale=config.cluster_effect_sd)
            stable_feature = (
                signed_labels * config.stable_signal
                + stable_cluster_effect
                + rng.normal(scale=config.observation_noise_sd, size=int(size))
            )
            spurious_feature = (
                signed_labels * signals[environment]
                + spurious_cluster_effect
                + rng.normal(scale=config.observation_noise_sd, size=int(size))
            )
            predictions = _predict(stable_feature, spurious_feature, specs)
            losses = np.square(predictions - labels[:, None])

            indices = np.arange(observation_counter, observation_counter + int(size))
            metadata_parts.append(
                pd.DataFrame(
                    {
                        "observation_id": [f"rank_obs_{index:08d}" for index in indices],
                        "environment_id": environment_id,
                        "root_cluster_id": root_id,
                        "label": labels.astype(int),
                        "x_stable": stable_feature,
                        "x_spurious": spurious_feature,
                        "sample_weight": 1.0,
                    }
                )
            )
            prediction_parts.append(predictions)
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
    prediction_matrix = np.concatenate(prediction_parts, axis=0)
    loss_matrix = np.concatenate(loss_parts, axis=0)
    observation_index = np.repeat(np.arange(len(metadata)), config.n_candidates)
    model_index = np.tile(np.arange(config.n_candidates), len(metadata))
    loss_table = metadata.iloc[observation_index].reset_index(drop=True)
    loss_table["model_id"] = np.asarray(model_ids, dtype=object)[model_index]
    loss_table["prediction"] = prediction_matrix.reshape(-1)
    loss_table["loss"] = loss_matrix.reshape(-1)

    population_risks, population_standard_errors = rank_reversal_population_risks(config)
    registry = pd.DataFrame.from_records(
        [
            {
                "model_id": spec.model_id,
                "model_family": "synthetic_linear",
                "stable_weight": spec.stable_weight,
                "spurious_weight": spec.spurious_weight,
                "temperature": spec.temperature,
                "bias": spec.bias,
            }
            for spec in specs
        ]
    )
    return RankReversalSimulation(
        loss_table=loss_table,
        true_environment_risks=population_risks,
        population_standard_errors=population_standard_errors,
        candidate_registry=registry,
        cluster_sizes=pd.DataFrame.from_records(cluster_rows),
        config=config,
    )


def generate_rank_reversal_loss_table(**config_overrides) -> pd.DataFrame:
    """Convenience wrapper returning only the long Brier-loss table."""

    return generate_rank_reversal_data(**config_overrides).loss_table


simulate_rank_reversal = generate_rank_reversal_data


__all__ = [
    "CandidateSpec",
    "RankReversalConfig",
    "RankReversalSimulation",
    "candidate_library",
    "generate_rank_reversal_data",
    "generate_rank_reversal_loss_table",
    "rank_reversal_population_risks",
    "simulate_rank_reversal",
]
