"""Paper-focused comparison of IID, marginal, and simultaneous intervals."""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

if __package__ in {None, ""}:  # Permit the documented direct-script invocation.
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import pandas as pd
from scipy.stats import norm

from hrms.bootstrap import (  # noqa: E402
    cluster_bootstrap,
    cluster_jackknife_pseudovalues,
)
from hrms.pairwise import pairwise_differences  # noqa: E402
from hrms.schema import validate_loss_table  # noqa: E402
from hrms.simultaneous import (  # noqa: E402
    jackknife_bootstrap_t_calibration,
    max_t_bounds,
)
from simulations.calibration import (  # noqa: E402
    CalibrationConfig,
    generate_calibration_data,
)

ICC_SEED_STRIDE = 100_000


def _truth_array(simulation, data) -> np.ndarray:
    truth = simulation.true_environment_risks.reindex(
        index=data.environment_ids, columns=data.model_ids
    ).to_numpy()
    return pairwise_differences(truth)


def _iid_standard_errors(data) -> np.ndarray:
    standard_errors = np.zeros((data.n_models, data.n_models, len(data.environment_ids)))
    for environment in range(len(data.environment_ids)):
        losses = data.losses[data.observation_environment == environment]
        differences = losses[:, :, None] - losses[:, None, :]
        standard_errors[:, :, environment] = differences.std(axis=0, ddof=1) / np.sqrt(
            len(losses)
        )
    return standard_errors


@dataclass(frozen=True)
class _SeparateRiskBounds:
    """Simultaneous absolute-risk intervals and their derived contrasts."""

    risk_lower: np.ndarray
    risk_upper: np.ndarray
    pairwise_lower: np.ndarray
    pairwise_upper: np.ndarray
    risk_standard_error: np.ndarray
    critical_value: float
    max_statistics: np.ndarray


def _separate_risk_bootstrap_t_bounds(
    blocks,
    point_risks: np.ndarray,
    *,
    n_bootstrap: int,
    random_state: int | np.random.Generator | None,
    alpha: float = 0.05,
    zero_tolerance: float = 1e-14,
) -> _SeparateRiskBounds:
    """Calibrate absolute risks separately, then derive pairwise intervals.

    Root-cluster pseudovalues are resampled jointly wherever the design calls
    for joint resampling.  The max statistic therefore remains cluster-aware
    and simultaneous over all environment--model risks.  Pairwise intervals
    deliberately ignore the paired covariance: if ``[L_m, U_m]`` and
    ``[L_k, U_k]`` are the two risk intervals, the resulting contrast interval
    is ``[L_m - U_k, U_m - L_k]``.
    """

    point_risks = np.asarray(point_risks, dtype=float)
    if point_risks.ndim != 2 or not np.isfinite(point_risks).all():
        raise ValueError("point_risks must be a finite environment-by-model array")
    if n_bootstrap < 2:
        raise ValueError("n_bootstrap must be at least two")
    if not 0 < alpha < 1:
        raise ValueError("alpha must lie strictly between zero and one")

    n_environments, n_models = point_risks.shape
    expected_environments = set(range(n_environments))
    represented_environments: set[int] = set()
    risk_standard_error = np.zeros_like(point_risks)
    centered_blocks: list[np.ndarray] = []
    block_active: list[np.ndarray] = []
    rng = (
        random_state
        if isinstance(random_state, np.random.Generator)
        else np.random.default_rng(random_state)
    )

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

        standard_error = values.std(axis=0, ddof=1) / np.sqrt(len(values))
        risk_standard_error[environments] = standard_error
        centered = values - values.mean(axis=0, keepdims=True)
        centered_blocks.append(centered.reshape(len(values), -1))
        block_active.append((standard_error > zero_tolerance).reshape(-1))

    if represented_environments != expected_environments:
        raise ValueError("jackknife blocks must cover every environment exactly once")

    max_statistics = np.zeros(n_bootstrap, dtype=float)
    for centered, active in zip(centered_blocks, block_active, strict=True):
        count, coordinate_count = centered.shape
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

    active_risks = risk_standard_error > zero_tolerance
    critical_value = (
        float(np.quantile(max_statistics, 1 - alpha, method="higher"))
        if np.any(active_risks)
        else 0.0
    )
    risk_lower = np.zeros_like(point_risks)
    risk_upper = np.ones_like(point_risks)
    risk_lower[active_risks] = np.maximum(
        0.0,
        point_risks[active_risks]
        - critical_value * risk_standard_error[active_risks],
    )
    risk_upper[active_risks] = np.minimum(
        1.0,
        point_risks[active_risks]
        + critical_value * risk_standard_error[active_risks],
    )

    model_lower = risk_lower.T
    model_upper = risk_upper.T
    pairwise_lower = model_lower[:, None, :] - model_upper[None, :, :]
    pairwise_upper = model_upper[:, None, :] - model_lower[None, :, :]
    diagonal = np.arange(n_models)
    pairwise_lower[diagonal, diagonal, :] = 0.0
    pairwise_upper[diagonal, diagonal, :] = 0.0
    return _SeparateRiskBounds(
        risk_lower=risk_lower,
        risk_upper=risk_upper,
        pairwise_lower=pairwise_lower,
        pairwise_upper=pairwise_upper,
        risk_standard_error=risk_standard_error,
        critical_value=critical_value,
        max_statistics=max_statistics,
    )


def _bounds_metrics(
    *,
    method: str,
    lower: np.ndarray,
    upper: np.ndarray,
    truth: np.ndarray,
    critical_value: float,
) -> dict[str, float | bool | str]:
    models = lower.shape[0]
    mask = np.broadcast_to(
        np.triu(np.ones((models, models), dtype=bool), k=1)[:, :, None],
        lower.shape,
    )
    covered = (lower[mask] <= truth[mask]) & (truth[mask] <= upper[mask])
    return {
        "method": method,
        "joint_coverage": bool(covered.all()),
        "coordinate_coverage": float(covered.mean()),
        "familywise_error": bool(not covered.all()),
        "mean_interval_width": float(np.mean((upper - lower)[mask])),
        "critical_value": float(critical_value),
    }


def _metrics(
    *,
    method: str,
    estimate: np.ndarray,
    standard_error: np.ndarray,
    critical_value: float,
    truth: np.ndarray,
) -> dict[str, float | bool | str]:
    lower = np.maximum(-1.0, estimate - critical_value * standard_error)
    upper = np.minimum(1.0, estimate + critical_value * standard_error)
    return _bounds_metrics(
        method=method,
        lower=lower,
        upper=upper,
        truth=truth,
        critical_value=critical_value,
    )


def run_one(
    *,
    n_environments: int,
    icc: float,
    n_candidates: int,
    clusters_per_environment: int,
    observations_per_cluster: int,
    imbalance: str,
    candidate_correlation: float,
    n_bootstrap: int,
    seed: int,
    alpha: float = 0.05,
) -> list[dict]:
    simulation = generate_calibration_data(
        CalibrationConfig(
            n_environments=n_environments,
            n_candidates=n_candidates,
            clusters_per_environment=clusters_per_environment,
            mean_cluster_size=observations_per_cluster,
            icc=icc,
            candidate_correlation=candidate_correlation,
            cluster_size_imbalance=imbalance,
            random_state=seed,
        )
    )
    data = validate_loss_table(simulation.loss_table)
    bootstrap = cluster_bootstrap(
        data,
        n_bootstrap=n_bootstrap,
        studentize=True,
        random_state=seed + 1_000_000,
    )
    estimate = pairwise_differences(bootstrap.point_risks)
    replicates = pairwise_differences(bootstrap.bootstrap_risks)
    cluster_se = replicates.std(axis=0, ddof=1)
    iid_se = _iid_standard_errors(data)
    truth = _truth_array(simulation, data)
    n_coordinates = len(data.environment_ids) * n_candidates * (n_candidates - 1) // 2
    simultaneous = max_t_bounds(estimate, replicates, alpha=alpha)
    studentized = max_t_bounds(
        estimate,
        replicates,
        alpha=alpha,
        standard_error=bootstrap.pairwise_standard_error,
        max_statistics=bootstrap.studentized_max_statistics,
    )
    jackknife_blocks = cluster_jackknife_pseudovalues(data)
    jackknife = jackknife_bootstrap_t_calibration(
        jackknife_blocks,
        n_models=data.n_models,
        n_environments=len(data.environment_ids),
        n_bootstrap=n_bootstrap,
        random_state=seed + 2_000_000,
    )
    separate_risk = _separate_risk_bootstrap_t_bounds(
        jackknife_blocks,
        bootstrap.point_risks,
        n_bootstrap=n_bootstrap,
        random_state=seed + 3_000_000,
        alpha=alpha,
    )

    methods = [
        _metrics(
            method="IID marginal",
            estimate=estimate,
            standard_error=iid_se,
            critical_value=float(norm.ppf(1 - alpha / 2)),
            truth=truth,
        ),
        _metrics(
            method="IID Bonferroni",
            estimate=estimate,
            standard_error=iid_se,
            critical_value=float(norm.ppf(1 - alpha / (2 * n_coordinates))),
            truth=truth,
        ),
        _metrics(
            method="Cluster marginal",
            estimate=estimate,
            standard_error=cluster_se,
            critical_value=float(norm.ppf(1 - alpha / 2)),
            truth=truth,
        ),
        _metrics(
            method="Paired cluster max-t",
            estimate=estimate,
            standard_error=cluster_se,
            critical_value=simultaneous.critical_value,
            truth=truth,
        ),
        _metrics(
            method="Paired cluster jackknife bootstrap-t",
            estimate=estimate,
            standard_error=jackknife.standard_error,
            critical_value=float(
                np.quantile(jackknife.max_statistics, 1 - alpha, method="higher")
            ),
            truth=truth,
        ),
        _bounds_metrics(
            method="Separate-risk bootstrap-t",
            lower=separate_risk.pairwise_lower,
            upper=separate_risk.pairwise_upper,
            critical_value=separate_risk.critical_value,
            truth=truth,
        ),
        _metrics(
            method="Paired cluster bootstrap-t",
            estimate=estimate,
            standard_error=studentized.standard_error,
            critical_value=studentized.critical_value,
            truth=truth,
        ),
    ]
    common = {
        "icc": icc,
        "n_environments": n_environments,
        "n_candidates": n_candidates,
        "clusters_per_environment": clusters_per_environment,
        "observations_per_cluster": observations_per_cluster,
        "imbalance": imbalance,
        "candidate_correlation": candidate_correlation,
        "n_bootstrap": n_bootstrap,
        "seed": seed,
        "n_observations": data.n_observations,
    }
    return [{**common, **method} for method in methods]


def run_study(
    *,
    repetitions: int,
    n_bootstrap: int,
    n_environments: int = 3,
    n_candidates: int = 10,
    clusters_per_environment: int = 25,
    observations_per_cluster: int = 10,
    imbalance: str = "moderate",
    candidate_correlation: float = 0.7,
    iccs: tuple[float, ...] = (0.0, 0.1, 0.3, 0.5),
    random_state: int = 42,
    alpha: float = 0.05,
) -> pd.DataFrame:
    rows: list[dict] = []
    for icc_index, icc in enumerate(iccs):
        for repetition in range(repetitions):
            rows.extend(
                run_one(
                    icc=icc,
                    n_environments=n_environments,
                    n_candidates=n_candidates,
                    clusters_per_environment=clusters_per_environment,
                    observations_per_cluster=observations_per_cluster,
                    imbalance=imbalance,
                    candidate_correlation=candidate_correlation,
                    n_bootstrap=n_bootstrap,
                    seed=random_state + ICC_SEED_STRIDE * icc_index + repetition,
                    alpha=alpha,
                )
            )
    return pd.DataFrame.from_records(rows)


def load_coverage_config(path: str | Path) -> tuple[dict[str, Any], dict[str, Any]]:
    """Load a frozen coverage-study config into :func:`run_study` arguments."""

    source = Path(path)
    raw = json.loads(source.read_text(encoding="utf-8"))
    if not isinstance(raw, dict):
        raise ValueError("coverage-study config must contain one JSON object")
    if raw.get("schema_version") != 1:
        raise ValueError("coverage-study config schema_version must be 1")
    if raw.get("generator") != "simulations.coverage_study":
        raise ValueError("config generator must be 'simulations.coverage_study'")
    if raw.get("population_truth") != "64-node Gauss-Hermite quadrature":
        raise ValueError("config population truth does not match the implementation")
    if raw.get("default_calibration") != "jackknife_bootstrap_t":
        raise ValueError("config default calibration does not match HRMS")

    icc_values = raw.get("iccs")
    if not isinstance(icc_values, list) or not icc_values:
        raise ValueError("config must declare a non-empty ICC list")
    iccs = tuple(float(value) for value in icc_values)
    base_seed = int(raw["base_seed"])
    expected_seed_rule = (
        f"{base_seed} + {ICC_SEED_STRIDE} * icc_index + repetition"
    )
    if raw.get("seed_rule") != expected_seed_rule:
        raise ValueError(
            "config seed_rule must match the implemented ICC-index seed schedule"
        )

    settings: dict[str, Any] = {
        "repetitions": int(raw["repetitions_per_icc"]),
        "n_bootstrap": int(raw["n_bootstrap"]),
        "n_environments": int(raw["n_environments"]),
        "n_candidates": int(raw["n_candidates"]),
        "clusters_per_environment": int(raw["clusters_per_environment"]),
        "observations_per_cluster": int(raw["mean_observations_per_cluster"]),
        "imbalance": str(raw["cluster_size_imbalance"]),
        "candidate_correlation": float(raw["candidate_correlation"]),
        "iccs": iccs,
        "random_state": base_seed,
        "alpha": float(raw["alpha"]),
    }
    _validate_settings(settings)
    return settings, raw


def _default_settings(args: argparse.Namespace) -> dict[str, Any]:
    return {
        "repetitions": 100 if args.repetitions is None else args.repetitions,
        "n_bootstrap": 399 if args.bootstrap is None else args.bootstrap,
        "n_environments": 3 if args.environments is None else args.environments,
        "n_candidates": 10 if args.candidates is None else args.candidates,
        "clusters_per_environment": 25 if args.clusters is None else args.clusters,
        "observations_per_cluster": 10 if args.cluster_size is None else args.cluster_size,
        "imbalance": "moderate" if args.imbalance is None else args.imbalance,
        "candidate_correlation": (
            0.7 if args.candidate_correlation is None else args.candidate_correlation
        ),
        "iccs": (0.0, 0.1, 0.3, 0.5) if args.iccs is None else tuple(args.iccs),
        "random_state": 42 if args.seed is None else args.seed,
        "alpha": 0.05 if args.alpha is None else args.alpha,
    }


def _validate_settings(settings: dict[str, Any]) -> None:
    if settings["repetitions"] < 1 or settings["n_bootstrap"] < 2:
        raise ValueError("repetitions must be positive and bootstrap at least two")
    if not 0 < settings["alpha"] < 1:
        raise ValueError("alpha must lie strictly between zero and one")
    iccs = tuple(settings["iccs"])
    if len(set(iccs)) != len(iccs):
        raise ValueError("ICC values must be unique")
    # Reuse the generator's validation for every result-determining design
    # choice, including cluster counts, imbalance, and candidate correlation.
    for icc in iccs:
        CalibrationConfig(
            n_environments=settings["n_environments"],
            n_candidates=settings["n_candidates"],
            clusters_per_environment=settings["clusters_per_environment"],
            mean_cluster_size=settings["observations_per_cluster"],
            icc=icc,
            candidate_correlation=settings["candidate_correlation"],
            cluster_size_imbalance=settings["imbalance"],
        )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=Path("results/coverage"))
    parser.add_argument("--config", type=Path)
    parser.add_argument("--repetitions", type=int)
    parser.add_argument("--bootstrap", type=int)
    parser.add_argument("--candidates", type=int)
    parser.add_argument("--environments", type=int)
    parser.add_argument("--clusters", type=int)
    parser.add_argument("--cluster-size", type=int)
    parser.add_argument("--imbalance", choices=["none", "moderate", "severe"])
    parser.add_argument("--iccs", nargs="+", type=float)
    parser.add_argument("--candidate-correlation", type=float)
    parser.add_argument("--alpha", type=float)
    parser.add_argument("--seed", type=int)
    args = parser.parse_args()

    statistical_overrides = {
        "--repetitions": args.repetitions,
        "--bootstrap": args.bootstrap,
        "--candidates": args.candidates,
        "--environments": args.environments,
        "--clusters": args.clusters,
        "--cluster-size": args.cluster_size,
        "--imbalance": args.imbalance,
        "--iccs": args.iccs,
        "--candidate-correlation": args.candidate_correlation,
        "--alpha": args.alpha,
        "--seed": args.seed,
    }
    if args.config is not None:
        conflicts = [name for name, value in statistical_overrides.items() if value is not None]
        if conflicts:
            parser.error(
                "--config enforces frozen statistical settings; remove overrides: "
                + ", ".join(conflicts)
            )
        settings, _ = load_coverage_config(args.config)
    else:
        settings = _default_settings(args)
        try:
            _validate_settings(settings)
        except ValueError as error:
            parser.error(str(error))

    runs = run_study(**settings)
    summary = (
        runs.groupby(["icc", "method"], sort=False)[
            ["joint_coverage", "coordinate_coverage", "familywise_error", "mean_interval_width"]
        ]
        .agg(["mean", "sem"])
    )
    summary.columns = ["_".join(column) for column in summary.columns]
    summary = summary.reset_index()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    runs.to_csv(args.output_dir / "coverage_runs.csv", index=False)
    summary.to_csv(args.output_dir / "coverage_summary.csv", index=False)


if __name__ == "__main__":
    main()
