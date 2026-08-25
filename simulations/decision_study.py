"""Prespecified rank-reversal decision study for the paper.

One environment's spurious association reverses progressively while the other
seven retain positive associations.  The design therefore contains a no-
reversal control and increasingly severe rank reversal without changing the
candidate library, ambiguity radius, or analysis after seeing finite-sample
results.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from concurrent.futures import ProcessPoolExecutor
from dataclasses import asdict, dataclass
from multiprocessing import get_context
from pathlib import Path
from typing import Any

if __package__ in {None, ""}:  # Permit the documented direct-script invocation.
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import pandas as pd

from hrms import HRMS  # noqa: E402
from hrms.ambiguity import ChiSquareBall  # noqa: E402
from hrms.baselines import (  # noqa: E402
    nominal_mean_scores,
    point_dro_regret_scores,
    point_dro_risk_scores,
    worst_environment_scores,
)
from hrms.pairwise import pairwise_differences  # noqa: E402
from hrms.regret import point_robust_regret  # noqa: E402
from simulations.rank_reversal import (  # noqa: E402
    CandidateSpec,
    RankReversalConfig,
    candidate_library,
    generate_rank_reversal_data,
    rank_reversal_population_risks,
)

SHIFT_STRENGTHS = (0.0, 0.5, 1.0, 1.5, 2.0, 2.5)
POSITIVE_ENVIRONMENT_SIGNALS = (0.6, 0.8, 1.0, 1.2, 1.4, 1.6, 1.8)
DEFAULT_POPULATION_SEED = 2_908_2026
METHODS = (
    "Mean validation",
    "Worst environment",
    "Point robust risk",
    "Point robust regret",
    "HRMS (rho=0)",
    "HRMS",
)


@dataclass(frozen=True)
class _ReplicateTask:
    """All state needed for one deterministic Monte Carlo repetition."""

    shift_index: int
    shift_strength: float
    repetition: int
    seed: int
    n_bootstrap: int
    clusters_per_environment: int
    mean_cluster_size: float
    radius: float
    alpha: float
    candidate_specs: tuple[CandidateSpec, ...]
    cluster_size_imbalance: str
    stable_signal: float
    cluster_effect_sd: float
    observation_noise_sd: float
    positive_prevalence: float
    population_seed: int
    calibration: str
    weighting: str
    resampling: str
    stability_replicates: int
    loss_bounds: tuple[float, float]
    model_ids: tuple[str, ...]
    true_regret: tuple[float, ...]
    oracle_value: float
    oracle_models: tuple[str, ...]


def environment_signals(shift_strength: float) -> tuple[float, ...]:
    """Return the prespecified eight-environment rank-reversal path."""

    if not np.isfinite(shift_strength) or shift_strength < 0:
        raise ValueError("shift_strength must be finite and non-negative")
    reversing_signal = 0.4 - 1.2 * shift_strength
    return (reversing_signal, *POSITIVE_ENVIRONMENT_SIGNALS)


def _select(scores: np.ndarray) -> int:
    scores = np.asarray(scores, dtype=float)
    return int(np.flatnonzero(scores <= scores.min() + 1e-12)[0])


def _truth_for_shift(
    shift_strength: float,
    population_size: int,
    *,
    population_seed: int = DEFAULT_POPULATION_SEED,
    candidate_specs: tuple[CandidateSpec, ...] | None = None,
    stable_signal: float = 1.0,
    cluster_effect_sd: float = 0.65,
    observation_noise_sd: float = 1.0,
    positive_prevalence: float = 0.5,
) -> pd.DataFrame:
    config = RankReversalConfig(
        n_environments=8,
        n_candidates=8,
        spurious_signals=environment_signals(shift_strength),
        population_size=population_size,
        population_seed=population_seed,
        candidate_specs=candidate_specs,
        stable_signal=stable_signal,
        cluster_effect_sd=cluster_effect_sd,
        observation_noise_sd=observation_noise_sd,
        positive_prevalence=positive_prevalence,
    )
    risks, _ = rank_reversal_population_risks(config)
    return risks


def _run_repetition(task: _ReplicateTask) -> list[dict[str, object]]:
    """Run one repetition; its seed is assigned before any parallel scheduling."""

    ambiguity = ChiSquareBall(center="uniform", radius=task.radius)
    sample = generate_rank_reversal_data(
        RankReversalConfig(
            n_environments=8,
            n_candidates=8,
            clusters_per_environment=task.clusters_per_environment,
            mean_cluster_size=task.mean_cluster_size,
            cluster_size_imbalance=task.cluster_size_imbalance,
            stable_signal=task.stable_signal,
            spurious_signals=environment_signals(task.shift_strength),
            cluster_effect_sd=task.cluster_effect_sd,
            observation_noise_sd=task.observation_noise_sd,
            positive_prevalence=task.positive_prevalence,
            candidate_specs=task.candidate_specs,
            random_state=task.seed,
            # Population truth is computed once in the parent process.  A
            # two-row placeholder avoids repeating that costly calculation.
            population_size=2,
            population_seed=task.population_seed,
        )
    )
    result = HRMS(
        ambiguity=ambiguity,
        alpha=task.alpha,
        n_bootstrap=task.n_bootstrap,
        calibration=task.calibration,
        weighting=task.weighting,
        resampling=task.resampling,
        random_state=task.seed + 1_000_000,
        stability_replicates=task.stability_replicates,
        robustness_radii=(0.0, task.radius, None),
        loss_bounds=task.loss_bounds,
    ).fit(sample.loss_table)
    estimated = result.environment_risks.reindex(columns=task.model_ids).to_numpy()
    radius_zero_selected = result.robustness_path.loc[
        (result.robustness_path["radius"] == 0.0)
        & result.robustness_path["selected"],
        "model",
    ]
    if len(radius_zero_selected) != 1:
        raise RuntimeError("the robustness path did not return one rho=0 selection")
    model_ids = task.model_ids
    selections = {
        "Mean validation": _select(nominal_mean_scores(estimated)),
        "Worst environment": _select(worst_environment_scores(estimated)),
        "Point robust risk": _select(point_dro_risk_scores(estimated, ambiguity)),
        "Point robust regret": _select(point_dro_regret_scores(estimated, ambiguity)),
        "HRMS (rho=0)": model_ids.index(radius_zero_selected.item()),
        "HRMS": model_ids.index(result.selected_model),
    }
    true_regret = np.asarray(task.true_regret, dtype=float)
    oracle_models = set(task.oracle_models)
    confidence = set(result.model_confidence_set)
    set_covers = oracle_models.issubset(confidence)
    rows: list[dict[str, object]] = []
    for method in METHODS:
        selected_index = selections[method]
        selected_regret = float(true_regret[selected_index])
        rows.append(
            {
                # Keep this stable machine-readable column for compatibility;
                # figures call it rank-reversal severity.
                "shift_strength": task.shift_strength,
                "reversing_signal": environment_signals(task.shift_strength)[0],
                "method": method,
                "repetition": task.repetition,
                "seed": task.seed,
                "selected_model": model_ids[selected_index],
                "selected_true_robust_regret": selected_regret,
                "excess_robust_regret": selected_regret - task.oracle_value,
                "selected_is_oracle": model_ids[selected_index] in oracle_models,
                "oracle_models": "|".join(task.oracle_models),
                "oracle_robust_regret": task.oracle_value,
                "confidence_set_size": len(confidence) if method == "HRMS" else np.nan,
                "confidence_set_covers": set_covers if method == "HRMS" else np.nan,
                "false_singleton": (
                    len(confidence) == 1 and not set_covers if method == "HRMS" else np.nan
                ),
                "upper_certificate": (
                    float(result.robust_regret_upper.loc[result.selected_model])
                    if method == "HRMS"
                    else np.nan
                ),
                "critical_value": result.critical_value if method == "HRMS" else np.nan,
                "calibration_method": result.calibration_method if method == "HRMS" else "point",
                "n_bootstrap": task.n_bootstrap,
                "clusters_per_environment": task.clusters_per_environment,
                "mean_cluster_size": task.mean_cluster_size,
                "radius": task.radius,
                "n_observations": sample.loss_table["observation_id"].nunique(),
            }
        )
    return rows


def run_study(
    *,
    repetitions: int,
    n_bootstrap: int,
    clusters_per_environment: int = 25,
    mean_cluster_size: float = 10,
    radius: float = 1.0,
    population_size: int = 500_000,
    random_state: int = 42,
    population_seed: int = DEFAULT_POPULATION_SEED,
    alpha: float = 0.05,
    cluster_size_imbalance: str = "moderate",
    stable_signal: float = 1.0,
    cluster_effect_sd: float = 0.65,
    observation_noise_sd: float = 1.0,
    positive_prevalence: float = 0.5,
    candidate_specs: tuple[CandidateSpec, ...] | None = None,
    calibration: str = "jackknife_bootstrap_t",
    weighting: str = "observation",
    resampling: str = "auto",
    stability_replicates: int = 100,
    loss_bounds: tuple[float, float] = (0.0, 1.0),
    rank_reversal_severities: tuple[float, ...] = SHIFT_STRENGTHS,
    jobs: int = 1,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Return replicate-level selections and the population risk surfaces."""

    if repetitions < 1:
        raise ValueError("repetitions must be positive")
    if n_bootstrap < 2:
        raise ValueError("n_bootstrap must be at least two")
    if jobs < 1:
        raise ValueError("jobs must be positive")
    if not 0 < alpha < 1:
        raise ValueError("alpha must lie strictly between zero and one")
    severities = tuple(float(value) for value in rank_reversal_severities)
    if not severities:
        raise ValueError("rank_reversal_severities must not be empty")
    for severity in severities:
        environment_signals(severity)
    library_config = RankReversalConfig(
        n_environments=8,
        n_candidates=8,
        clusters_per_environment=clusters_per_environment,
        mean_cluster_size=mean_cluster_size,
        cluster_size_imbalance=cluster_size_imbalance,
        stable_signal=stable_signal,
        spurious_signals=environment_signals(severities[0]),
        cluster_effect_sd=cluster_effect_sd,
        observation_noise_sd=observation_noise_sd,
        positive_prevalence=positive_prevalence,
        candidate_specs=candidate_specs,
        population_size=population_size,
        population_seed=population_seed,
    )
    resolved_specs = candidate_library(library_config)
    ambiguity = ChiSquareBall(center="uniform", radius=radius)
    tasks: list[_ReplicateTask] = []
    truth_rows: list[pd.DataFrame] = []

    for shift_index, shift_strength in enumerate(severities):
        truth = _truth_for_shift(
            shift_strength,
            population_size,
            population_seed=population_seed,
            candidate_specs=resolved_specs,
            stable_signal=stable_signal,
            cluster_effect_sd=cluster_effect_sd,
            observation_noise_sd=observation_noise_sd,
            positive_prevalence=positive_prevalence,
        )
        true_regret = point_robust_regret(pairwise_differences(truth.to_numpy()), ambiguity)
        oracle_value = float(true_regret.min())
        oracle_indices = np.flatnonzero(
            np.isclose(true_regret, oracle_value, atol=1e-10, rtol=1e-8)
        )
        oracle_models = {truth.columns[index] for index in oracle_indices}
        truth_long = (
            truth.rename_axis(index="environment_id", columns="model_id")
            .stack()
            .rename("true_risk")
            .reset_index()
        )
        truth_long["shift_strength"] = shift_strength
        truth_long["spurious_signals"] = repr(environment_signals(shift_strength))
        truth_rows.append(truth_long)

        for repetition in range(repetitions):
            seed = random_state + shift_index * 100_000 + repetition
            tasks.append(
                _ReplicateTask(
                    shift_index=shift_index,
                    shift_strength=shift_strength,
                    repetition=repetition,
                    seed=seed,
                    n_bootstrap=n_bootstrap,
                    clusters_per_environment=clusters_per_environment,
                    mean_cluster_size=mean_cluster_size,
                    radius=radius,
                    alpha=alpha,
                    candidate_specs=resolved_specs,
                    cluster_size_imbalance=cluster_size_imbalance,
                    stable_signal=stable_signal,
                    cluster_effect_sd=cluster_effect_sd,
                    observation_noise_sd=observation_noise_sd,
                    positive_prevalence=positive_prevalence,
                    population_seed=population_seed,
                    calibration=calibration,
                    weighting=weighting,
                    resampling=resampling,
                    stability_replicates=min(n_bootstrap, stability_replicates),
                    loss_bounds=loss_bounds,
                    model_ids=tuple(truth.columns),
                    true_regret=tuple(float(value) for value in true_regret),
                    oracle_value=oracle_value,
                    oracle_models=tuple(sorted(oracle_models)),
                )
            )

    if jobs == 1:
        task_rows = map(_run_repetition, tasks)
        rows = [row for batch in task_rows for row in batch]
    else:
        # ``map`` preserves task order.  Because every task carries independent,
        # preassigned sample and bootstrap seeds, scheduling cannot change output.
        with ProcessPoolExecutor(
            max_workers=jobs,
            mp_context=get_context("spawn"),
        ) as executor:
            rows = [row for batch in executor.map(_run_repetition, tasks) for row in batch]

    return pd.DataFrame.from_records(rows), pd.concat(truth_rows, ignore_index=True)


def summarize(runs: pd.DataFrame) -> pd.DataFrame:
    metrics = [
        "selected_true_robust_regret",
        "excess_robust_regret",
        "selected_is_oracle",
        "confidence_set_size",
        "confidence_set_covers",
        "false_singleton",
        "upper_certificate",
        "critical_value",
    ]
    summary = runs.groupby(["shift_strength", "method"], sort=False)[metrics].agg(["mean", "sem"])
    summary.columns = ["_".join(column) for column in summary.columns]
    return summary.reset_index()


def load_decision_config(path: str | Path) -> tuple[dict[str, Any], dict[str, Any]]:
    """Load and validate a frozen decision-study configuration.

    The returned keyword arguments can be passed directly to :func:`run_study`.
    Operational choices such as output location and worker count deliberately do
    not live in the frozen statistical configuration.
    """

    source = Path(path)
    raw = json.loads(source.read_text(encoding="utf-8"))
    if not isinstance(raw, dict):
        raise ValueError("decision-study config must contain one JSON object")
    if raw.get("generator") != "simulations.decision_study":
        raise ValueError("config generator must be 'simulations.decision_study'")
    if raw.get("n_environments") != 8 or raw.get("n_candidates") != 8:
        raise ValueError("the frozen decision study requires eight environments and candidates")
    if raw.get("ambiguity_center") != "uniform":
        raise ValueError("the frozen decision study requires a uniform ambiguity center")

    signal_rule = raw.get("environment_signal_rule", {})
    if tuple(signal_rule.get("other_environments", ())) != POSITIVE_ENVIRONMENT_SIGNALS:
        raise ValueError("config positive-environment signals do not match the implementation")
    if signal_rule.get("reversing_environment") != "0.4 - 1.2 * severity":
        raise ValueError("config reversing-environment rule does not match the implementation")

    severity_values = raw.get("rank_reversal_severities")
    if severity_values is None:
        # Compatibility with the first internal draft of the configuration.
        severity_values = raw.get("shift_strengths")
    if not isinstance(severity_values, list) or not severity_values:
        raise ValueError("config must declare rank_reversal_severities")
    severities = tuple(float(value) for value in severity_values)
    for severity in severities:
        environment_signals(severity)

    generator_defaults = raw.get("generator_defaults")
    if not isinstance(generator_defaults, dict):
        raise ValueError("config must declare generator_defaults")
    candidate_records = raw.get("candidate_library")
    if not isinstance(candidate_records, list) or len(candidate_records) != 8:
        raise ValueError("config must declare all eight frozen candidate specifications")
    candidate_specs = tuple(
        CandidateSpec(
            model_id=str(record["model_id"]),
            stable_weight=float(record["stable_weight"]),
            spurious_weight=float(record["spurious_weight"]),
            temperature=float(record["temperature"]),
            bias=float(record["bias"]),
        )
        for record in candidate_records
    )

    settings: dict[str, Any] = {
        "repetitions": int(raw["repetitions"]),
        "n_bootstrap": int(raw["n_bootstrap"]),
        "clusters_per_environment": int(raw["clusters_per_environment"]),
        "mean_cluster_size": float(raw["mean_cluster_size"]),
        "radius": float(raw["ambiguity_radius"]),
        "population_size": int(raw["population_size"]),
        "random_state": int(raw["random_state"]),
        "population_seed": int(raw["population_seed"]),
        "alpha": float(raw["alpha"]),
        "cluster_size_imbalance": str(generator_defaults["cluster_size_imbalance"]),
        "stable_signal": float(generator_defaults["stable_signal"]),
        "cluster_effect_sd": float(generator_defaults["cluster_effect_sd"]),
        "observation_noise_sd": float(generator_defaults["observation_noise_sd"]),
        "positive_prevalence": float(generator_defaults["positive_prevalence"]),
        "candidate_specs": candidate_specs,
        "calibration": str(raw["calibration"]),
        "weighting": str(raw["weighting"]),
        "resampling": str(raw["resampling"]),
        "stability_replicates": int(raw["stability_replicates"]),
        "loss_bounds": tuple(float(value) for value in raw["loss_bounds"]),
        "rank_reversal_severities": severities,
    }
    return settings, raw


def _default_settings(args: argparse.Namespace) -> dict[str, Any]:
    return {
        "repetitions": 100 if args.repetitions is None else args.repetitions,
        "n_bootstrap": 499 if args.bootstrap is None else args.bootstrap,
        "clusters_per_environment": 25 if args.clusters is None else args.clusters,
        "mean_cluster_size": 10 if args.cluster_size is None else args.cluster_size,
        "radius": 1.0 if args.radius is None else args.radius,
        "population_size": 500_000 if args.population_size is None else args.population_size,
        "random_state": 42 if args.seed is None else args.seed,
        "population_seed": (
            DEFAULT_POPULATION_SEED if args.population_seed is None else args.population_seed
        ),
        "alpha": 0.05 if args.alpha is None else args.alpha,
        "cluster_size_imbalance": "moderate",
        "stable_signal": 1.0,
        "cluster_effect_sd": 0.65,
        "observation_noise_sd": 1.0,
        "positive_prevalence": 0.5,
        "candidate_specs": None,
        "calibration": "jackknife_bootstrap_t",
        "weighting": "observation",
        "resampling": "auto",
        "stability_replicates": 100,
        "loss_bounds": (0.0, 1.0),
        "rank_reversal_severities": SHIFT_STRENGTHS,
    }


def _effective_configuration(settings: dict[str, Any]) -> dict[str, Any]:
    """Return a JSON-serializable record of every result-determining setting."""

    effective = dict(settings)
    raw_specs = effective.pop("candidate_specs")
    if raw_specs is None:
        config = RankReversalConfig(
            n_environments=8,
            n_candidates=8,
            spurious_signals=environment_signals(effective["rank_reversal_severities"][0]),
            population_size=2,
        )
        raw_specs = candidate_library(config)
    effective["candidate_library"] = [asdict(spec) for spec in raw_specs]
    effective["rank_reversal_severities"] = list(effective["rank_reversal_severities"])
    effective["loss_bounds"] = list(effective["loss_bounds"])
    effective["n_environments"] = 8
    effective["n_candidates"] = 8
    effective["ambiguity_center"] = "uniform"
    effective["environment_signal_rule"] = {
        "reversing_environment": "0.4 - 1.2 * severity",
        "other_environments": list(POSITIVE_ENVIRONMENT_SIGNALS),
    }
    return effective


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _implementation_hashes() -> dict[str, str]:
    source_root = Path(__file__).resolve().parents[1]
    implementation_files = (
        "pyproject.toml",
        "uv.lock",
        "simulations/decision_study.py",
        "simulations/rank_reversal.py",
        "hrms/ambiguity.py",
        "hrms/baselines.py",
        "hrms/bootstrap.py",
        "hrms/estimands.py",
        "hrms/pairwise.py",
        "hrms/regret.py",
        "hrms/schema.py",
        "hrms/selector.py",
        "hrms/simultaneous.py",
    )
    return {filename: _sha256(source_root / filename) for filename in implementation_files}


def _write_manifest(
    *,
    output_dir: Path,
    settings: dict[str, Any],
    jobs: int,
    source_config: Path | None,
    source_config_sha256: str | None,
    declared_config: dict[str, Any] | None,
    implementation_sha256: dict[str, str],
) -> Path:
    output_files = ("decision_runs.csv", "decision_summary.csv", "decision_truth.csv")
    manifest: dict[str, Any] = {
        "schema_version": 1,
        "study": "rank-reversal decision study",
        "effective_configuration": _effective_configuration(settings),
        "execution": {
            "jobs": jobs,
            "ordering": "shift-major, repetition-major, frozen-selector order",
            "determinism": "worker scheduling does not alter preassigned seeds or output order",
        },
        # Captured before Monte Carlo execution, so later edits cannot be
        # mistakenly attributed to an already-running result.
        "implementation_sha256": implementation_sha256,
        "outputs_sha256": {
            filename: _sha256(output_dir / filename) for filename in output_files
        },
    }
    if source_config is not None:
        assert source_config_sha256 is not None
        manifest["source_config"] = {
            "path": str(source_config),
            "sha256": source_config_sha256,
            "declared": declared_config,
        }
    destination = output_dir / "decision_manifest.json"
    destination.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return destination


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=Path("results/decision"))
    parser.add_argument("--config", type=Path)
    parser.add_argument("--jobs", type=int, default=1)
    parser.add_argument("--repetitions", type=int)
    parser.add_argument("--bootstrap", type=int)
    parser.add_argument("--clusters", type=int)
    parser.add_argument("--cluster-size", type=float)
    parser.add_argument("--radius", type=float)
    parser.add_argument("--population-size", type=int)
    parser.add_argument("--population-seed", type=int)
    parser.add_argument("--alpha", type=float)
    parser.add_argument("--seed", type=int)
    args = parser.parse_args()

    statistical_overrides = {
        "--repetitions": args.repetitions,
        "--bootstrap": args.bootstrap,
        "--clusters": args.clusters,
        "--cluster-size": args.cluster_size,
        "--radius": args.radius,
        "--population-size": args.population_size,
        "--population-seed": args.population_seed,
        "--alpha": args.alpha,
        "--seed": args.seed,
    }
    declared_config: dict[str, Any] | None = None
    if args.config is not None:
        conflicts = [name for name, value in statistical_overrides.items() if value is not None]
        if conflicts:
            parser.error(
                "--config enforces frozen statistical settings; remove overrides: "
                + ", ".join(conflicts)
            )
        settings, declared_config = load_decision_config(args.config)
    else:
        settings = _default_settings(args)

    implementation_sha256 = _implementation_hashes()
    source_config_sha256 = _sha256(args.config) if args.config is not None else None
    runs, truth = run_study(**settings, jobs=args.jobs)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    runs.to_csv(args.output_dir / "decision_runs.csv", index=False)
    summarize(runs).to_csv(args.output_dir / "decision_summary.csv", index=False)
    truth.to_csv(args.output_dir / "decision_truth.csv", index=False)
    _write_manifest(
        output_dir=args.output_dir,
        settings=settings,
        jobs=args.jobs,
        source_config=args.config,
        source_config_sha256=source_config_sha256,
        declared_config=declared_config,
        implementation_sha256=implementation_sha256,
    )


if __name__ == "__main__":
    main()
