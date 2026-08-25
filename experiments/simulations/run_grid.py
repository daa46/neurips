"""Lightweight reproducible grids for the two synthetic HRMS experiments.

Examples
--------
``python -m experiments.simulations.run_grid --preset smoke --output-dir results/smoke``
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from experiments.simulations.calibration import CalibrationConfig, generate_calibration_data
from experiments.simulations.rank_reversal import RankReversalConfig, generate_rank_reversal_data
from hrms import HRMS
from hrms.ambiguity import ChiSquareBall
from hrms.regret import point_robust_regret


@dataclass(frozen=True)
class GridPreset:
    n_bootstrap: int
    repetitions: int
    calibration_settings: tuple[dict[str, Any], ...]
    rank_shift_strengths: tuple[float, ...]
    calibration_population_size: int
    rank_population_size: int


PRESETS: dict[str, GridPreset] = {
    "smoke": GridPreset(
        n_bootstrap=59,
        repetitions=2,
        calibration_settings=(
            {"icc": 0.0, "n_candidates": 5, "cluster_size_imbalance": "none"},
            {
                "icc": 0.3,
                "n_candidates": 5,
                "cluster_size_imbalance": "moderate",
            },
            {
                "icc": 0.5,
                "n_candidates": 10,
                "cluster_size_imbalance": "severe",
            },
        ),
        rank_shift_strengths=(1.0, 1.8),
        calibration_population_size=20_000,
        rank_population_size=40_000,
    ),
    "development": GridPreset(
        n_bootstrap=199,
        repetitions=5,
        calibration_settings=tuple(
            {
                "icc": icc,
                "n_candidates": n_candidates,
                "cluster_size_imbalance": imbalance,
            }
            for icc in (0.0, 0.1, 0.3, 0.5)
            for n_candidates in (5, 20)
            for imbalance in ("none", "severe")
        ),
        rank_shift_strengths=(0.5, 1.0, 1.8, 2.5),
        calibration_population_size=100_000,
        rank_population_size=150_000,
    ),
}


def _pairwise_from_risks(risks: pd.DataFrame) -> np.ndarray:
    model_by_environment = risks.to_numpy(dtype=float).T
    return model_by_environment[:, None, :] - model_by_environment[None, :, :]


def _ordered_truth(result, truth: pd.DataFrame) -> pd.DataFrame:
    return truth.reindex(
        index=result.environment_risks.index,
        columns=result.environment_risks.columns,
    )


def _coverage_metrics(result, truth: pd.DataFrame, ambiguity: ChiSquareBall) -> dict[str, Any]:
    ordered = _ordered_truth(result, truth)
    bounds = result.pairwise_bounds
    true_differences = np.asarray(
        [
            ordered.loc[row.environment, row.model] - ordered.loc[row.environment, row.comparator]
            for row in bounds.itertuples(index=False)
        ]
    )
    coordinate_coverage = (bounds["lower"].to_numpy() <= true_differences) & (
        true_differences <= bounds["upper"].to_numpy()
    )

    true_regret = point_robust_regret(_pairwise_from_risks(ordered), ambiguity)
    minimum = float(np.min(true_regret))
    oracle_indices = np.flatnonzero(np.isclose(true_regret, minimum, atol=1e-12, rtol=1e-10))
    model_ids = tuple(ordered.columns)
    oracle_models = tuple(model_ids[index] for index in oracle_indices)
    confidence = set(result.model_confidence_set)
    lower = result.robust_regret_lower.reindex(model_ids).to_numpy()
    upper = result.robust_regret_upper.reindex(model_ids).to_numpy()
    selected_index = model_ids.index(result.selected_model)
    return {
        "joint_pairwise_coverage": bool(np.all(coordinate_coverage)),
        "coordinate_coverage": float(np.mean(coordinate_coverage)),
        "mean_interval_width": float(np.mean(bounds["upper"] - bounds["lower"])),
        "certificate_coverage": bool(np.all((lower <= true_regret) & (true_regret <= upper))),
        "confidence_set_coverage": bool(all(model_id in confidence for model_id in oracle_models)),
        "confidence_set_size": len(confidence),
        "selected_model": result.selected_model,
        "oracle_model": oracle_models[0],
        "selected_is_oracle": result.selected_model in oracle_models,
        "selected_true_robust_regret": float(true_regret[selected_index]),
        "selected_upper_certificate": float(result.robust_regret_upper.loc[result.selected_model]),
        "critical_value": result.critical_value,
    }


def _fit(loss_table: pd.DataFrame, *, n_bootstrap: int, seed: int, radius: float):
    selector = HRMS(
        ambiguity=ChiSquareBall(center="uniform", radius=radius),
        alpha=0.05,
        n_bootstrap=n_bootstrap,
        random_state=seed,
        stability_replicates=min(n_bootstrap, 100),
        robustness_radii=(0.0, radius, None),
    )
    return selector.fit(loss_table)


def run_calibration_grid(
    preset: GridPreset,
    *,
    n_bootstrap: int,
    repetitions: int,
    random_state: int,
    radius: float = 0.25,
) -> pd.DataFrame:
    """Run the direct correlated-loss conditions and return replicate rows."""

    rows: list[dict[str, Any]] = []
    for condition, setting in enumerate(preset.calibration_settings):
        for repetition in range(repetitions):
            seed = random_state + condition * 10_000 + repetition
            config = CalibrationConfig(
                n_environments=3 if preset is PRESETS["smoke"] else 5,
                n_candidates=int(setting["n_candidates"]),
                clusters_per_environment=12 if preset is PRESETS["smoke"] else 25,
                mean_cluster_size=5 if preset is PRESETS["smoke"] else 10,
                icc=float(setting["icc"]),
                candidate_correlation=0.7,
                cluster_size_imbalance=setting["cluster_size_imbalance"],
                random_state=seed,
                population_method="quadrature",
                population_size=preset.calibration_population_size,
            )
            simulation = generate_calibration_data(config)
            result = _fit(
                simulation.loss_table,
                n_bootstrap=n_bootstrap,
                seed=seed + 1_000_000,
                radius=radius,
            )
            row = {
                "condition": condition,
                "repetition": repetition,
                "seed": seed,
                "n_environments": config.n_environments,
                "n_candidates": config.n_candidates,
                "clusters_per_environment": int(config.clusters_per_environment),
                "mean_cluster_size": config.mean_cluster_size,
                "icc": config.icc,
                "candidate_correlation": config.candidate_correlation,
                "cluster_size_imbalance": config.cluster_size_imbalance,
                "n_observations": simulation.loss_table["observation_id"].nunique(),
                "n_bootstrap": n_bootstrap,
                "radius": radius,
            }
            row.update(
                _coverage_metrics(
                    result,
                    simulation.true_environment_risks,
                    ChiSquareBall(center="uniform", radius=radius),
                )
            )
            rows.append(row)
    return pd.DataFrame.from_records(rows)


def _selection_index(values: np.ndarray) -> int:
    return int(np.flatnonzero(values == np.min(values))[0])


def run_rank_reversal_grid(
    preset: GridPreset,
    *,
    n_bootstrap: int,
    repetitions: int,
    random_state: int,
    radius: float = 0.5,
) -> pd.DataFrame:
    """Run feature-based rank-reversal conditions and selection baselines."""

    rows: list[dict[str, Any]] = []
    for condition, strength in enumerate(preset.rank_shift_strengths):
        for repetition in range(repetitions):
            seed = random_state + 500_000 + condition * 10_000 + repetition
            n_environments = 5
            signals = tuple(np.linspace(-strength, strength, n_environments))
            config = RankReversalConfig(
                n_environments=n_environments,
                n_candidates=8,
                clusters_per_environment=12 if preset is PRESETS["smoke"] else 25,
                mean_cluster_size=6 if preset is PRESETS["smoke"] else 10,
                spurious_signals=signals,
                random_state=seed,
                population_size=preset.rank_population_size,
            )
            simulation = generate_rank_reversal_data(config)
            ambiguity = ChiSquareBall(center="uniform", radius=radius)
            result = _fit(
                simulation.loss_table,
                n_bootstrap=n_bootstrap,
                seed=seed + 1_000_000,
                radius=radius,
            )
            metrics = _coverage_metrics(result, simulation.true_environment_risks, ambiguity)

            estimated = result.environment_risks
            truth = _ordered_truth(result, simulation.true_environment_risks)
            estimated_pairwise = _pairwise_from_risks(estimated)
            true_robust_regret = point_robust_regret(_pairwise_from_risks(truth), ambiguity)
            point_dro_index = _selection_index(point_robust_regret(estimated_pairwise, ambiguity))
            mean_index = _selection_index(estimated.mean(axis=0).to_numpy())
            worst_environment_index = _selection_index(estimated.max(axis=0).to_numpy())
            per_environment_winners = {
                truth.columns[index] for index in np.argmin(truth.to_numpy(), axis=1)
            }
            model_ids = tuple(truth.columns)
            row = {
                "condition": condition,
                "repetition": repetition,
                "seed": seed,
                "shift_strength": strength,
                "n_environments": config.n_environments,
                "n_candidates": config.n_candidates,
                "clusters_per_environment": int(config.clusters_per_environment),
                "mean_cluster_size": config.mean_cluster_size,
                "n_observations": simulation.loss_table["observation_id"].nunique(),
                "n_bootstrap": n_bootstrap,
                "radius": radius,
                "n_environment_winners": len(per_environment_winners),
                "mean_selected_model": model_ids[mean_index],
                "mean_selected_true_robust_regret": float(true_robust_regret[mean_index]),
                "point_dro_selected_model": model_ids[point_dro_index],
                "point_dro_selected_true_robust_regret": float(true_robust_regret[point_dro_index]),
                "worst_environment_selected_model": model_ids[worst_environment_index],
                "worst_environment_selected_true_robust_regret": float(
                    true_robust_regret[worst_environment_index]
                ),
            }
            row.update(metrics)
            rows.append(row)
    return pd.DataFrame.from_records(rows)


def _summarize(runs: pd.DataFrame, group_columns: list[str]) -> pd.DataFrame:
    metrics = [
        column
        for column in (
            "joint_pairwise_coverage",
            "coordinate_coverage",
            "mean_interval_width",
            "certificate_coverage",
            "confidence_set_coverage",
            "confidence_set_size",
            "selected_is_oracle",
            "selected_true_robust_regret",
            "selected_upper_certificate",
            "critical_value",
            "n_environment_winners",
            "mean_selected_true_robust_regret",
            "point_dro_selected_true_robust_regret",
            "worst_environment_selected_true_robust_regret",
        )
        if column in runs.columns
    ]
    summary = runs.groupby(group_columns, dropna=False)[metrics].mean().reset_index()
    counts = runs.groupby(group_columns, dropna=False).size().rename("n_repetitions")
    return summary.merge(counts.reset_index(), on=group_columns, validate="one_to_one")


def run_grid(
    *,
    preset_name: str,
    output_dir: str | Path,
    random_state: int = 42,
    n_bootstrap: int | None = None,
    repetitions: int | None = None,
) -> dict[str, Path]:
    """Run a named grid, write replicate and summary CSVs, and return paths."""

    try:
        preset = PRESETS[preset_name]
    except KeyError as error:
        raise ValueError(f"unknown preset {preset_name!r}; choose from {tuple(PRESETS)}") from error
    effective_bootstrap = preset.n_bootstrap if n_bootstrap is None else n_bootstrap
    effective_repetitions = preset.repetitions if repetitions is None else repetitions
    if effective_bootstrap < 2:
        raise ValueError("n_bootstrap must be at least two")
    if effective_repetitions < 1:
        raise ValueError("repetitions must be positive")

    calibration_runs = run_calibration_grid(
        preset,
        n_bootstrap=effective_bootstrap,
        repetitions=effective_repetitions,
        random_state=random_state,
    )
    rank_runs = run_rank_reversal_grid(
        preset,
        n_bootstrap=effective_bootstrap,
        repetitions=effective_repetitions,
        random_state=random_state,
    )
    calibration_summary = _summarize(
        calibration_runs,
        [
            "n_environments",
            "n_candidates",
            "clusters_per_environment",
            "mean_cluster_size",
            "icc",
            "candidate_correlation",
            "cluster_size_imbalance",
            "n_bootstrap",
            "radius",
        ],
    )
    rank_summary = _summarize(
        rank_runs,
        [
            "shift_strength",
            "n_environments",
            "n_candidates",
            "clusters_per_environment",
            "mean_cluster_size",
            "n_bootstrap",
            "radius",
        ],
    )

    directory = Path(output_dir)
    directory.mkdir(parents=True, exist_ok=True)
    outputs = {
        "calibration_runs": directory / "calibration_runs.csv",
        "calibration_summary": directory / "calibration_summary.csv",
        "rank_reversal_runs": directory / "rank_reversal_runs.csv",
        "rank_reversal_summary": directory / "rank_reversal_summary.csv",
    }
    calibration_runs.to_csv(outputs["calibration_runs"], index=False)
    calibration_summary.to_csv(outputs["calibration_summary"], index=False)
    rank_runs.to_csv(outputs["rank_reversal_runs"], index=False)
    rank_summary.to_csv(outputs["rank_reversal_summary"], index=False)
    return outputs


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--preset", choices=tuple(PRESETS), default="smoke")
    parser.add_argument("--output-dir", type=Path, default=Path("results/simulations"))
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--n-bootstrap", type=int)
    parser.add_argument("--repetitions", type=int)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    outputs = run_grid(
        preset_name=args.preset,
        output_dir=args.output_dir,
        random_state=args.seed,
        n_bootstrap=args.n_bootstrap,
        repetitions=args.repetitions,
    )
    for name, path in outputs.items():
        print(f"{name}: {path}")
    return 0


if __name__ == "__main__":  # pragma: no cover - exercised through the CLI
    raise SystemExit(main())
