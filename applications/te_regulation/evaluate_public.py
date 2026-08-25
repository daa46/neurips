#!/usr/bin/env python3
"""Sealed selection and evaluation for the public TE--cCRE benchmark.

Selection is persisted and hashed before the held-out loss table is opened.
This keeps model selection, test evaluation, and figure-ready exports auditable.
"""

# ruff: noqa: E402 -- direct-script support requires path setup before imports.

from __future__ import annotations

import argparse
import json
import math
import sys
from collections.abc import Iterable, Mapping
from pathlib import Path

_PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

import numpy as np
import pandas as pd
from scipy.stats import kendalltau, qmc

from applications.te_regulation.cluster_and_split import TEBenchmarkError, read_table
from hrms import HRMS
from hrms.ambiguity import ChiSquareBall, chi_square_divergence
from hrms.artifacts import sha256_file, write_artifact_manifest
from hrms.baselines import (
    nominal_mean_scores,
    point_dro_regret_scores,
    point_dro_risk_scores,
    select_from_scores,
    worst_environment_scores,
)
from hrms.estimands import estimate_environment_risks
from hrms.pairwise import pairwise_differences
from hrms.registry import assert_prediction_library_matches, validate_model_registry
from hrms.regret import point_robust_regret
from hrms.schema import validate_loss_table

RADIUS_PATH: tuple[float | None, ...] = (0.0, 0.1, 0.25, 0.5, 1.0, None)


def _write_json(path: Path, payload: object) -> None:
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _write_csv(frame: pd.DataFrame, path: Path) -> None:
    options: dict[str, object] = {"index": False, "lineterminator": "\n"}
    if path.suffix == ".gz":
        options["compression"] = {"method": "gzip", "mtime": 0}
    frame.to_csv(path, **options)


def _artifact_record(manifest: Mapping, key: str) -> Mapping:
    if manifest.get("schema_version") != 1:
        raise TEBenchmarkError("training manifest must use schema version 1")
    artifacts = manifest.get("artifacts")
    if not isinstance(artifacts, Mapping) or key not in artifacts:
        raise TEBenchmarkError(f"training manifest lacks artifact record {key!r}")
    record = artifacts[key]
    if not isinstance(record, Mapping):
        raise TEBenchmarkError(f"invalid artifact record for {key!r}")
    digest, size = record.get("sha256"), record.get("size_bytes")
    if not isinstance(digest, str) or len(digest) != 64 or not isinstance(size, int):
        raise TEBenchmarkError(f"incomplete artifact record for {key!r}")
    return record


def _verify_file_record(path: Path, record: Mapping, *, label: str) -> None:
    if not path.is_file():
        raise FileNotFoundError(f"{label} does not exist: {path}")
    if path.stat().st_size != int(record["size_bytes"]):
        raise TEBenchmarkError(f"{label} size differs from its immutable manifest")
    if sha256_file(path) != str(record["sha256"]):
        raise TEBenchmarkError(f"{label} hash differs from its immutable manifest")


def _verify_manifest_artifacts(manifest_path: Path) -> dict:
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("schema_version") != 1:
        raise TEBenchmarkError("selection manifest must use schema version 1")
    for name, record in manifest.get("artifacts", {}).items():
        artifact = Path(record["path"])
        if not artifact.is_absolute():
            artifact = manifest_path.parent.parent / artifact
        _verify_file_record(artifact, record, label=f"selection artifact {name}")
    return manifest


def _load_registry(path: Path, training_manifest: Mapping) -> pd.DataFrame:
    _verify_file_record(path, _artifact_record(training_manifest, "registry"), label="registry")
    registry = pd.read_csv(path)
    validate_model_registry(registry)
    registry = registry.copy()
    registry["model_id"] = registry["model_id"].astype(str)
    return registry


def _load_loss_table(
    path: Path,
    training_manifest: Mapping,
    artifact_key: str,
    registry: pd.DataFrame,
) -> pd.DataFrame:
    _verify_file_record(
        path, _artifact_record(training_manifest, artifact_key), label=artifact_key
    )
    table = read_table(path).copy()
    for column in ("observation_id", "environment_id", "root_cluster_id", "model_id"):
        if column in table:
            table[column] = table[column].astype(str)
    assert_prediction_library_matches(registry, table["model_id"].unique().tolist())
    data = validate_loss_table(table)
    if np.any(data.losses < 0.0) or np.any(data.losses > 1.0):
        raise TEBenchmarkError("public-TE losses must be bounded Brier losses in [0, 1]")
    return table.sort_values(["observation_id", "model_id"], kind="stable").reset_index(
        drop=True
    )


def _radius_label(radius: float | None | str) -> str:
    if radius is None or str(radius).lower() in {"full", "full_simplex", "none"}:
        return "full"
    return f"{float(radius):g}"


def _radius_token(radius: float | None | str) -> str:
    return _radius_label(radius).replace(".", "p")


def _parse_radius(radius: object) -> float | None:
    if radius is None or str(radius).lower() in {"full", "full_simplex", "none", "nan"}:
        return None
    return float(radius)


def _analysis_frame(result, analysis_id: str, root_definition: str, weighting: str) -> dict:
    risks = (
        result.environment_risks.rename_axis(index="environment_id", columns="model_id")
        .stack(future_stack=True)
        .rename("risk")
        .reset_index()
    )
    risks.insert(0, "analysis_id", analysis_id)
    risks.insert(1, "root_definition", root_definition)
    risks.insert(2, "weighting", weighting)
    pairwise = result.pairwise_bounds.copy()
    pairwise.insert(0, "analysis_id", analysis_id)
    pairwise.insert(1, "root_definition", root_definition)
    pairwise.insert(2, "weighting", weighting)
    path = result.robustness_path.copy()
    path["radius"] = path["radius"].map(_radius_label)
    path.insert(0, "analysis_id", analysis_id)
    path.insert(1, "root_definition", root_definition)
    path.insert(2, "weighting", weighting)
    stability = result.selection_stability.rename("frequency").rename_axis("model_id").reset_index()
    stability.insert(0, "analysis_id", analysis_id)
    stability.insert(1, "root_definition", root_definition)
    stability.insert(2, "weighting", weighting)
    diagnostics = [
        {
            "analysis_id": analysis_id,
            "root_definition": root_definition,
            "weighting": weighting,
            "code": item.code,
            "severity": item.severity,
            "message": item.message,
        }
        for item in result.diagnostics
    ]
    cluster = pd.concat(
        [result.cluster_counts, result.effective_cluster_counts, result.max_cluster_leverage],
        axis=1,
    ).rename_axis("environment_id").reset_index()
    cluster.insert(0, "analysis_id", analysis_id)
    cluster.insert(1, "root_definition", root_definition)
    cluster.insert(2, "weighting", weighting)
    return {
        "analysis_id": analysis_id,
        "root_definition": root_definition,
        "weighting": weighting,
        "result": result,
        "risks": risks,
        "pairwise": pairwise,
        "path": path,
        "stability": stability,
        "diagnostics": diagnostics,
        "cluster": cluster,
    }


def _fit_analysis(
    table: pd.DataFrame,
    *,
    analysis_id: str,
    root_definition: str,
    weighting: str,
    radius: float,
    alpha: float,
    n_bootstrap: int,
    seed: int,
) -> dict:
    working = table.copy()
    if root_definition == "observation_iid":
        working["root_cluster_id"] = working["observation_id"]
    result = HRMS(
        ambiguity=ChiSquareBall(center="uniform", radius=radius),
        alpha=alpha,
        n_bootstrap=n_bootstrap,
        calibration="jackknife_bootstrap_t",
        weighting=weighting,
        resampling="auto",
        random_state=seed,
        stability_replicates=min(500, n_bootstrap),
        robustness_radii=RADIUS_PATH,
    ).fit(working)
    return _analysis_frame(result, analysis_id, root_definition, weighting)


def _selection_rows(
    analyses: Mapping[str, dict], *, default_radius: float
) -> tuple[pd.DataFrame, pd.DataFrame]:
    selected_rows: list[dict] = []
    score_rows: list[dict] = []
    for analysis_id in ("root_observation", "root_cluster"):
        analysis = analyses[analysis_id]
        result = analysis["result"]
        models = [str(value) for value in result.environment_risks.columns]
        risks = result.environment_risks.to_numpy(dtype=float)
        ambiguity = ChiSquareBall(center="uniform", radius=default_radius)
        scores_by_method = {
            "uniform_mean": nominal_mean_scores(risks),
            "worst_environment": worst_environment_scores(risks),
            "point_dro_risk": point_dro_risk_scores(risks, ambiguity),
            "point_dro_regret": point_dro_regret_scores(risks, ambiguity),
        }
        for method, scores in scores_by_method.items():
            selected_index = select_from_scores(scores)
            method_id = f"{analysis_id}__{method}"
            selected_rows.append(
                {
                    "method_id": method_id,
                    "analysis_id": analysis_id,
                    "root_definition": analysis["root_definition"],
                    "weighting": analysis["weighting"],
                    "method": method,
                    "selection_radius": "not_applicable",
                    "evaluation_radius": _radius_label(default_radius),
                    "selected_model": models[selected_index],
                    "selection_score": float(scores[selected_index]),
                    "score_definition": "point_estimate",
                    "primary_comparison": analysis_id == "root_observation",
                }
            )
            for model, score in zip(models, scores, strict=True):
                score_rows.append(
                    {
                        "method_id": method_id,
                        "analysis_id": analysis_id,
                        "weighting": analysis["weighting"],
                        "method": method,
                        "selection_radius": "not_applicable",
                        "model_id": model,
                        "score": float(score),
                    }
                )
    for analysis_id, analysis in analyses.items():
        for radius, frame in analysis["path"].groupby("radius", sort=False, observed=True):
            winners = frame.loc[frame["selected"], "model"]
            if len(winners) != 1:
                raise RuntimeError(f"{analysis_id} has {len(winners)} HRMS winners at rho={radius}")
            selected_model = str(winners.iloc[0])
            method_id = f"{analysis_id}__hrms__rho_{_radius_token(radius)}"
            selected_upper = float(
                frame.loc[frame["model"].astype(str).eq(selected_model), "upper"].iloc[0]
            )
            selected_rows.append(
                {
                    "method_id": method_id,
                    "analysis_id": analysis_id,
                    "root_definition": analysis["root_definition"],
                    "weighting": analysis["weighting"],
                    "method": "hrms",
                    "selection_radius": _radius_label(radius),
                    "evaluation_radius": _radius_label(radius),
                    "selected_model": selected_model,
                    "selection_score": selected_upper,
                    "score_definition": "simultaneous_upper_regret_certificate",
                    "primary_comparison": (
                        analysis_id == "root_observation"
                        and _radius_label(radius) == _radius_label(default_radius)
                    ),
                }
            )
            for row in frame.itertuples(index=False):
                score_rows.append(
                    {
                        "method_id": method_id,
                        "analysis_id": analysis_id,
                        "weighting": analysis["weighting"],
                        "method": "hrms",
                        "selection_radius": _radius_label(radius),
                        "model_id": str(row.model),
                        "score": float(row.upper),
                    }
                )
    selected = pd.DataFrame(selected_rows).sort_values("method_id", kind="stable")
    scores = pd.DataFrame(score_rows).sort_values(["method_id", "model_id"], kind="stable")
    return selected.reset_index(drop=True), scores.reset_index(drop=True)


def _risk_ranks(risks: pd.DataFrame, *, partition: str) -> pd.DataFrame:
    output = risks.copy()
    output["partition"] = partition
    output["rank"] = output.groupby(["analysis_id", "environment_id"], observed=True)[
        "risk"
    ].rank(method="average", ascending=True)
    return output[
        [
            "partition",
            "analysis_id",
            "root_definition",
            "weighting",
            "environment_id",
            "model_id",
            "risk",
            "rank",
        ]
    ]


def select_public_models(
    selection_loss_path: Path,
    registry_path: Path,
    training_manifest_path: Path,
    output_dir: Path,
    *,
    radius: float = 0.25,
    alpha: float = 0.05,
    n_bootstrap: int = 1_999,
    seed: int = 42,
) -> Path:
    """Fit prespecified selectors using selection losses only and seal outputs."""

    if radius < 0 or not math.isfinite(radius):
        raise ValueError("default radius must be a finite non-negative number")
    training_manifest = json.loads(training_manifest_path.read_text(encoding="utf-8"))
    registry = _load_registry(registry_path, training_manifest)
    selection = _load_loss_table(
        selection_loss_path, training_manifest, "losses_select", registry
    )
    expected_count = training_manifest.get("training", {}).get("candidate_count")
    if expected_count is not None and int(expected_count) != len(registry):
        raise TEBenchmarkError("registry size differs from training-manifest candidate count")

    analyses = {
        "root_observation": _fit_analysis(
            selection,
            analysis_id="root_observation",
            root_definition="homology_root",
            weighting="observation",
            radius=radius,
            alpha=alpha,
            n_bootstrap=n_bootstrap,
            seed=seed,
        ),
        "root_cluster": _fit_analysis(
            selection,
            analysis_id="root_cluster",
            root_definition="homology_root",
            weighting="cluster",
            radius=radius,
            alpha=alpha,
            n_bootstrap=n_bootstrap,
            seed=seed + 1,
        ),
        "iid_observation": _fit_analysis(
            selection,
            analysis_id="iid_observation",
            root_definition="observation_iid",
            weighting="observation",
            radius=radius,
            alpha=alpha,
            n_bootstrap=n_bootstrap,
            seed=seed + 2,
        ),
    }
    selected, scores = _selection_rows(analyses, default_radius=radius)
    output_dir.mkdir(parents=True, exist_ok=True)
    environment_risks = pd.concat(
        [analysis["risks"] for analysis in analyses.values()], ignore_index=True
    )
    outputs = {
        "selected_models": selected,
        "selection_scores": scores,
        "environment_risks": environment_risks,
        "ranking_heatmap": _risk_ranks(environment_risks, partition="select"),
        "pairwise_bounds": pd.concat(
            [analysis["pairwise"] for analysis in analyses.values()], ignore_index=True
        ),
        "robustness_path": pd.concat(
            [analysis["path"] for analysis in analyses.values()], ignore_index=True
        ),
        "selection_stability": pd.concat(
            [analysis["stability"] for analysis in analyses.values()], ignore_index=True
        ),
        "cluster_diagnostics": pd.concat(
            [analysis["cluster"] for analysis in analyses.values()], ignore_index=True
        ),
    }
    artifacts = {
        "selected_models": output_dir / "selected_models.csv",
        "selection_scores": output_dir / "selection_scores.csv",
        "environment_risks": output_dir / "selection_environment_risks.csv",
        "ranking_heatmap": output_dir / "selection_ranking_heatmap.csv",
        "pairwise_bounds": output_dir / "pairwise_bounds.csv",
        "robustness_path": output_dir / "robustness_path.csv",
        "selection_stability": output_dir / "selection_stability.csv",
        "cluster_diagnostics": output_dir / "cluster_diagnostics.csv",
        "diagnostics": output_dir / "diagnostics.json",
        "selection_summary": output_dir / "selection_summary.json",
    }
    for name, frame in outputs.items():
        _write_csv(frame, artifacts[name])
    diagnostics = [row for analysis in analyses.values() for row in analysis["diagnostics"]]
    _write_json(artifacts["diagnostics"], diagnostics)
    primary_id = f"root_observation__hrms__rho_{_radius_token(radius)}"
    primary_row = selected.loc[selected["method_id"].eq(primary_id)]
    if len(primary_row) != 1:
        raise RuntimeError("primary HRMS selection is missing from the sealed table")
    primary_result = analyses["root_observation"]["result"]
    _write_json(
        artifacts["selection_summary"],
        {
            "schema_version": 1,
            "task": "public TE-associated enhancer-like cCRE prediction",
            "selection_partition_only": True,
            "test_table_accessed": False,
            "primary_loss": "Brier",
            "default_radius": radius,
            "radius_path": [_radius_label(value) for value in RADIUS_PATH],
            "alpha": alpha,
            "n_bootstrap": n_bootstrap,
            "seed": seed,
            "candidate_count": len(registry),
            "selection_observations": int(selection["observation_id"].nunique()),
            "selection_root_clusters": int(selection["root_cluster_id"].nunique()),
            "environments": sorted(selection["environment_id"].unique().tolist()),
            "primary_method_id": primary_id,
            "primary_selected_model": str(primary_row.iloc[0]["selected_model"]),
            "primary_confidence_set": [
                str(value) for value in primary_result.model_confidence_set
            ],
            "primary_confidence_set_size": len(primary_result.model_confidence_set),
            "calibration_method": primary_result.calibration_method,
            "resampling": primary_result.resampling,
            "iid_ablation_is_misspecified": True,
            "cluster_weighting_is_sensitivity_analysis": True,
        },
    )
    test_record = _artifact_record(training_manifest, "losses_test")
    manifest_path = output_dir / "selection_manifest.json"
    write_artifact_manifest(
        manifest_path,
        inputs={
            "training_manifest": training_manifest_path,
            "registry": registry_path,
            "selection_losses": selection_loss_path,
        },
        artifacts=artifacts,
        parameters={
            "alpha": alpha,
            "default_radius": radius,
            "n_bootstrap": n_bootstrap,
            "seed": seed,
            "test_table_accessed": False,
            "expected_test_loss_sha256": test_record["sha256"],
            "expected_test_loss_size_bytes": test_record["size_bytes"],
        },
        source_files=[Path(__file__)],
        source_base_dir=_PROJECT_ROOT,
        base_dir=output_dir.parent,
    )
    return manifest_path


def validate_partition_separation(selection: pd.DataFrame, test: pd.DataFrame) -> None:
    """Require disjoint observation and homology-root IDs across partitions."""

    select_observations = set(selection["observation_id"].astype(str))
    test_observations = set(test["observation_id"].astype(str))
    if overlap := select_observations & test_observations:
        raise TEBenchmarkError(
            f"selection and test observations overlap; examples={sorted(overlap)[:5]}"
        )
    select_roots = set(selection["root_cluster_id"].astype(str))
    test_roots = set(test["root_cluster_id"].astype(str))
    if overlap := select_roots & test_roots:
        raise TEBenchmarkError(
            f"selection and test homology roots overlap; examples={sorted(overlap)[:5]}"
        )
    select_environments = set(selection["environment_id"].astype(str))
    test_environments = set(test["environment_id"].astype(str))
    if select_environments != test_environments:
        raise TEBenchmarkError(
            "selection and test environment sets differ; "
            f"selection_only={sorted(select_environments - test_environments)}, "
            f"test_only={sorted(test_environments - select_environments)}"
        )


def _point_risk_frame(table: pd.DataFrame, weighting: str) -> pd.DataFrame:
    data = validate_loss_table(table)
    values = estimate_environment_risks(data, weighting=weighting)
    return pd.DataFrame(
        values,
        index=pd.Index(data.environment_ids, name="environment_id"),
        columns=pd.Index(data.model_ids, name="model_id"),
    )


def deterministic_target_mixtures(
    environments: Iterable[str],
    *,
    primary_radius: float = 0.25,
    n_halton: int = 512,
) -> tuple[pd.DataFrame, pd.DataFrame, np.ndarray]:
    """Build a fixed, label-free mixture design over represented environments."""

    environment_ids = tuple(sorted(str(value) for value in environments))
    count = len(environment_ids)
    if count < 2:
        raise ValueError("at least two environments are required for mixture stress tests")
    if n_halton < 1:
        raise ValueError("n_halton must be positive")
    center = np.full(count, 1.0 / count)
    rows: list[tuple[np.ndarray, str, str]] = []

    def add(weights: np.ndarray, design: str, detail: str) -> None:
        vector = np.maximum(np.asarray(weights, dtype=float), 0.0)
        vector /= vector.sum()
        rows.append((vector, design, detail))

    add(center, "nominal", "uniform")
    vertices = np.eye(count)
    for environment, vertex in zip(environment_ids, vertices, strict=True):
        add(vertex, "simplex_vertex", environment)
    for path_radius in (0.1, 0.25, 0.5, 1.0):
        for environment, vertex in zip(environment_ids, vertices, strict=True):
            divergence = chi_square_divergence(vertex, center)
            scale = min(1.0, math.sqrt(path_radius / divergence))
            add(
                center + scale * (vertex - center),
                "vertex_ray",
                f"rho={path_radius:g};target={environment}",
            )
    for first in range(count):
        for second in range(first + 1, count):
            target = 0.5 * (vertices[first] + vertices[second])
            divergence = chi_square_divergence(target, center)
            scale = min(1.0, math.sqrt(primary_radius / divergence))
            add(
                center + scale * (target - center),
                "pair_ray_primary_boundary",
                f"{environment_ids[first]}+{environment_ids[second]}",
            )
    sampler = qmc.Halton(d=count, scramble=False)
    sampler.fast_forward(1)
    unit = np.clip(sampler.random(n_halton), np.finfo(float).tiny, 1.0)
    raw = -np.log(unit)
    raw /= raw.sum(axis=1, keepdims=True)
    fractions = np.linspace(0.05, 0.95, num=19)
    for index, target in enumerate(raw):
        add(target, "halton_simplex", f"index={index}")
        divergence = chi_square_divergence(target, center)
        desired = primary_radius * fractions[index % len(fractions)]
        scale = 0.0 if divergence == 0.0 else min(1.0, math.sqrt(desired / divergence))
        add(
            center + scale * (target - center),
            "halton_primary_interior",
            f"index={index};fraction={fractions[index % len(fractions)]:g}",
        )
    unique: list[tuple[np.ndarray, str, str]] = []
    seen: set[tuple[float, ...]] = set()
    for weights, design, detail in rows:
        key = tuple(np.round(weights, 14))
        if key not in seen:
            unique.append((weights, design, detail))
            seen.add(key)
    matrix = np.vstack([row[0] for row in unique])
    metadata_rows, weight_rows = [], []
    for index, (weights, design, detail) in enumerate(unique):
        mixture_id = f"mixture_{index:05d}"
        divergence = chi_square_divergence(weights, center)
        if np.allclose(weights, center, atol=1e-12):
            scope = "nominal"
        elif design == "simplex_vertex":
            scope = "simplex_vertex"
        elif math.isclose(divergence, primary_radius, rel_tol=1e-8, abs_tol=1e-10):
            scope = "primary_boundary"
        elif divergence < primary_radius:
            scope = "inside_primary"
        else:
            scope = "outside_primary"
        metadata_rows.append(
            {
                "mixture_id": mixture_id,
                "design": design,
                "design_detail": detail,
                "chi_square_divergence": divergence,
                "scope": scope,
                "inside_primary_ball": divergence <= primary_radius + 1e-10,
            }
        )
        for environment, weight in zip(environment_ids, weights, strict=True):
            weight_rows.append(
                {"mixture_id": mixture_id, "environment_id": environment, "weight": weight}
            )
    return pd.DataFrame(metadata_rows), pd.DataFrame(weight_rows), matrix


def _certificate_audit(
    path: pd.DataFrame, test_risks: Mapping[str, pd.DataFrame]
) -> pd.DataFrame:
    rows: list[dict] = []
    for (analysis_id, radius_label), frame in path.groupby(
        ["analysis_id", "radius"], sort=False, observed=True
    ):
        weighting = str(frame["weighting"].iloc[0])
        risk = test_risks[weighting]
        models = [str(value) for value in risk.columns]
        ambiguity = ChiSquareBall(center="uniform", radius=_parse_radius(radius_label))
        regret = point_robust_regret(pairwise_differences(risk.to_numpy(dtype=float)), ambiguity)
        optimum = float(regret.min())
        optima = {models[index] for index in np.flatnonzero(np.isclose(regret, optimum))}
        confidence_mask = frame["in_confidence_set"].astype(str).str.lower().eq("true")
        selected_mask = frame["selected"].astype(str).str.lower().eq("true")
        confidence = set(frame.loc[confidence_mask, "model"].astype(str))
        selected = set(frame.loc[selected_mask, "model"].astype(str))
        for row in frame.itertuples(index=False):
            model = str(row.model)
            index = models.index(model)
            rows.append(
                {
                    "analysis_id": analysis_id,
                    "root_definition": row.root_definition,
                    "weighting": weighting,
                    "radius": _radius_label(radius_label),
                    "model_id": model,
                    "lower": float(row.lower),
                    "upper": float(row.upper),
                    "test_reference_robust_regret": float(regret[index]),
                    "interval_contains_test_reference": (
                        float(row.lower) <= regret[index] + 1e-12
                        and float(row.upper) + 1e-12 >= regret[index]
                    ),
                    "selected": model in selected,
                    "in_confidence_set": model in confidence,
                    "test_reference_robust_optimal": model in optima,
                    "confidence_set_covers_any_test_optimum": bool(confidence & optima),
                    "confidence_set_covers_all_test_optima": optima.issubset(confidence),
                }
            )
    return pd.DataFrame(rows)


def evaluate_sealed_test(
    test_loss_path: Path,
    selection_loss_path: Path,
    registry_path: Path,
    training_manifest_path: Path,
    selection_manifest_path: Path,
    output_dir: Path,
    *,
    n_mixtures: int = 512,
) -> Path:
    """Open held-out losses only after verifying the persisted selection bundle."""

    selection_manifest = _verify_manifest_artifacts(selection_manifest_path)
    if selection_manifest.get("parameters", {}).get("test_table_accessed") is not False:
        raise TEBenchmarkError("selection manifest does not attest that test data stayed sealed")
    training_manifest = json.loads(training_manifest_path.read_text(encoding="utf-8"))
    expected_training_hash = selection_manifest["inputs"]["training_manifest"]["sha256"]
    if sha256_file(training_manifest_path) != expected_training_hash:
        raise TEBenchmarkError("training manifest changed after selection")
    registry = _load_registry(registry_path, training_manifest)
    expected_test = _artifact_record(training_manifest, "losses_test")
    if expected_test["sha256"] != selection_manifest["parameters"][
        "expected_test_loss_sha256"
    ]:
        raise TEBenchmarkError(
            "expected test hash differs between training and selection manifests"
        )

    # First semantic read of the held-out table happens only after the checks above.
    test = _load_loss_table(test_loss_path, training_manifest, "losses_test", registry)
    selection = _load_loss_table(
        selection_loss_path, training_manifest, "losses_select", registry
    )
    validate_partition_separation(selection, test)

    selection_dir = selection_manifest_path.parent
    selected = pd.read_csv(selection_dir / "selected_models.csv", dtype=str)
    selection_scores = pd.read_csv(selection_dir / "selection_scores.csv", dtype=str)
    robustness_path = pd.read_csv(selection_dir / "robustness_path.csv", dtype=str)
    selection_ranking = pd.read_csv(selection_dir / "selection_ranking_heatmap.csv")
    registered_models = set(registry["model_id"])
    if not set(selected["selected_model"]).issubset(registered_models):
        raise TEBenchmarkError("sealed selection names a model outside the frozen registry")

    test_risks = {
        "observation": _point_risk_frame(test, "observation"),
        "cluster": _point_risk_frame(test, "cluster"),
    }
    environments = list(test_risks["observation"].index.astype(str))
    models = list(test_risks["observation"].columns.astype(str))
    if list(test_risks["cluster"].index.astype(str)) != environments:
        raise RuntimeError("test estimands disagree on environment order")
    risk_rows, ranking_rows = [], []
    for weighting, risk in test_risks.items():
        long = risk.stack(future_stack=True).rename("risk").reset_index()
        long["weighting"] = weighting
        long["root_definition"] = "homology_root"
        long["partition"] = "test"
        long["rank"] = long.groupby("environment_id", observed=True)["risk"].rank(
            method="average", ascending=True
        )
        risk_rows.append(
            long[["partition", "weighting", "environment_id", "model_id", "risk"]]
        )
        ranking_rows.append(
            long[
                [
                    "partition",
                    "root_definition",
                    "weighting",
                    "environment_id",
                    "model_id",
                    "risk",
                    "rank",
                ]
            ]
        )
    test_environment_risks = pd.concat(risk_rows, ignore_index=True)
    test_ranking = pd.concat(ranking_rows, ignore_index=True)
    selection_ranking = selection_ranking.loc[
        selection_ranking["analysis_id"].isin(["root_observation", "root_cluster"])
    ].drop(columns="analysis_id")
    ranking_heatmap = pd.concat([selection_ranking, test_ranking], ignore_index=True)

    default_radius = float(selection_manifest["parameters"]["default_radius"])
    mixture_metadata, mixture_weights, mixture_matrix = deterministic_target_mixtures(
        environments, primary_radius=default_radius, n_halton=n_mixtures
    )
    metadata_lookup = mixture_metadata.set_index("mixture_id")
    mixture_ids = mixture_metadata["mixture_id"].tolist()
    model_index = {model: index for index, model in enumerate(models)}
    regret_rows: list[dict] = []
    method_summary_rows: list[dict] = []
    scores_numeric = selection_scores.copy()
    scores_numeric["score"] = pd.to_numeric(scores_numeric["score"], errors="raise")
    for row in selected.itertuples(index=False):
        weighting = str(row.weighting)
        risk = test_risks[weighting].reindex(index=environments, columns=models)
        risk_array = risk.to_numpy(dtype=float)
        selected_index = model_index[str(row.selected_model)]
        mixture_risk = mixture_matrix @ risk_array
        oracle = mixture_risk.min(axis=1)
        regrets = mixture_risk[:, selected_index] - oracle
        ambiguity = ChiSquareBall(
            center="uniform", radius=_parse_radius(row.evaluation_radius)
        )
        robust_regret = point_robust_regret(pairwise_differences(risk_array), ambiguity)
        uniform_risk = nominal_mean_scores(risk_array)
        simplex_regret = point_robust_regret(
            pairwise_differences(risk_array), ChiSquareBall.full_simplex(center="uniform")
        )
        method_scores = scores_numeric.loc[
            scores_numeric["method_id"].eq(row.method_id)
        ].set_index("model_id")["score"].reindex(models)
        tau = float(kendalltau(method_scores.to_numpy(), uniform_risk).statistic)
        primary = str(row.primary_comparison).lower() == "true"
        method_summary_rows.append(
            {
                "method_id": row.method_id,
                "analysis_id": row.analysis_id,
                "root_definition": row.root_definition,
                "weighting": weighting,
                "method": row.method,
                "selection_radius": row.selection_radius,
                "evaluation_radius": row.evaluation_radius,
                "selected_model": row.selected_model,
                "test_uniform_regret": float(
                    uniform_risk[selected_index] - uniform_risk.min()
                ),
                "test_robust_regret_at_evaluation_radius": float(
                    robust_regret[selected_index]
                ),
                "test_full_simplex_regret": float(simplex_regret[selected_index]),
                "selected_is_test_robust_optimal_at_evaluation_radius": bool(
                    np.isclose(robust_regret[selected_index], robust_regret.min())
                ),
                "kendall_tau_selection_score_vs_test_uniform_risk": tau,
                "primary_comparison": primary,
            }
        )
        for mixture_id, regret, selected_risk, oracle_risk in zip(
            mixture_ids,
            regrets,
            mixture_risk[:, selected_index],
            oracle,
            strict=True,
        ):
            mixture_row = metadata_lookup.loc[mixture_id]
            regret_rows.append(
                {
                    "mixture_id": mixture_id,
                    "scope": mixture_row["scope"],
                    "chi_square_divergence": mixture_row["chi_square_divergence"],
                    "method_id": row.method_id,
                    "analysis_id": row.analysis_id,
                    "root_definition": row.root_definition,
                    "weighting": weighting,
                    "method": row.method,
                    "selection_radius": row.selection_radius,
                    "selected_model": row.selected_model,
                    "selected_test_risk": selected_risk,
                    "oracle_test_risk": oracle_risk,
                    "test_regret": regret,
                    "selected_is_mixture_optimal": bool(np.isclose(regret, 0.0)),
                    "primary_comparison": primary,
                }
            )
    mixture_regrets = pd.DataFrame(regret_rows)
    mixture_summary = (
        mixture_regrets.groupby(
            [
                "method_id",
                "analysis_id",
                "root_definition",
                "weighting",
                "method",
                "selection_radius",
                "selected_model",
                "scope",
                "primary_comparison",
            ],
            observed=True,
            dropna=False,
        )
        .agg(
            mixture_count=("test_regret", "size"),
            mean_test_regret=("test_regret", "mean"),
            median_test_regret=("test_regret", "median"),
            p95_test_regret=("test_regret", lambda values: values.quantile(0.95)),
            worst_test_regret=("test_regret", "max"),
            mixture_optimal_frequency=("selected_is_mixture_optimal", "mean"),
        )
        .reset_index()
    )
    certificate_audit = _certificate_audit(robustness_path, test_risks)
    method_summary = pd.DataFrame(method_summary_rows)
    output_dir.mkdir(parents=True, exist_ok=True)
    artifacts = {
        "test_environment_risks": output_dir / "test_environment_risks.csv",
        "ranking_heatmap": output_dir / "ranking_heatmap.csv",
        "target_mixture_metadata": output_dir / "target_mixture_metadata.csv",
        "target_mixture_weights": output_dir / "target_mixture_weights.csv.gz",
        "target_mixture_regrets": output_dir / "target_mixture_regrets.csv.gz",
        "target_mixture_summary": output_dir / "target_mixture_summary.csv",
        "test_method_summary": output_dir / "test_method_summary.csv",
        "certificate_audit": output_dir / "certificate_audit.csv",
        "test_summary": output_dir / "test_summary.json",
    }
    for name, frame in (
        ("test_environment_risks", test_environment_risks),
        ("ranking_heatmap", ranking_heatmap),
        ("target_mixture_metadata", mixture_metadata),
        ("target_mixture_weights", mixture_weights),
        ("target_mixture_regrets", mixture_regrets),
        ("target_mixture_summary", mixture_summary),
        ("test_method_summary", method_summary),
        ("certificate_audit", certificate_audit),
    ):
        _write_csv(frame, artifacts[name])
    primary_methods = method_summary.loc[method_summary["primary_comparison"]]
    _write_json(
        artifacts["test_summary"],
        {
            "schema_version": 1,
            "task": "public TE-associated enhancer-like cCRE prediction",
            "selection_was_persisted_before_test_table_opened": True,
            "test_hash_matched_training_manifest": True,
            "selection_test_observations_disjoint": True,
            "selection_test_homology_roots_disjoint": True,
            "primary_loss": "Brier",
            "default_radius": default_radius,
            "candidate_count": len(models),
            "environment_count": len(environments),
            "environments": environments,
            "test_observations": int(test["observation_id"].nunique()),
            "test_root_clusters": int(test["root_cluster_id"].nunique()),
            "deterministic_mixture_count": len(mixture_metadata),
            "primary_method_ids": primary_methods["method_id"].tolist(),
            "test_reference_is_finite_sample_not_population_truth": True,
            "iid_ablation_is_known_misspecification": True,
            "cluster_weighting_is_sensitivity_analysis": True,
        },
    )
    manifest_path = output_dir / "evaluation_manifest.json"
    write_artifact_manifest(
        manifest_path,
        inputs={
            "training_manifest": training_manifest_path,
            "selection_manifest": selection_manifest_path,
            "registry": registry_path,
            "selection_losses": selection_loss_path,
            "test_losses": test_loss_path,
        },
        artifacts=artifacts,
        parameters={
            "default_radius": default_radius,
            "n_halton": n_mixtures,
            "deterministic_mixture_count": len(mixture_metadata),
            "selection_preceded_test_access": True,
        },
        source_files=[Path(__file__)],
        source_base_dir=_PROJECT_ROOT,
        base_dir=output_dir.parent,
    )
    return manifest_path


def run_public_evaluation(
    selection_loss_path: Path,
    test_loss_path: Path,
    registry_path: Path,
    training_manifest_path: Path,
    output_dir: Path,
    *,
    radius: float = 0.25,
    alpha: float = 0.05,
    n_bootstrap: int = 1_999,
    seed: int = 42,
    n_mixtures: int = 512,
) -> dict[str, Path]:
    """Run sealed selection, then held-out evaluation."""

    selection_manifest = select_public_models(
        selection_loss_path,
        registry_path,
        training_manifest_path,
        output_dir / "selection",
        radius=radius,
        alpha=alpha,
        n_bootstrap=n_bootstrap,
        seed=seed,
    )
    evaluation_manifest = evaluate_sealed_test(
        test_loss_path,
        selection_loss_path,
        registry_path,
        training_manifest_path,
        selection_manifest,
        output_dir / "test_evaluation",
        n_mixtures=n_mixtures,
    )
    return {
        "selection_manifest": selection_manifest,
        "evaluation_manifest": evaluation_manifest,
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--selection-losses",
        type=Path,
        default=Path("results/te_public/losses_select.parquet"),
    )
    parser.add_argument(
        "--test-losses",
        type=Path,
        default=Path("results/te_public/losses_test.parquet"),
    )
    parser.add_argument(
        "--registry", type=Path, default=Path("results/te_public/model_registry.csv")
    )
    parser.add_argument(
        "--training-manifest",
        type=Path,
        default=Path("results/te_public/training_manifest.json"),
    )
    parser.add_argument(
        "--output-dir", type=Path, default=Path("results/te_public/evaluation")
    )
    parser.add_argument("--radius", type=float, default=0.25)
    parser.add_argument("--alpha", type=float, default=0.05)
    parser.add_argument("--bootstrap", type=int, default=1_999)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--mixtures", type=int, default=512)
    args = parser.parse_args(argv)
    outputs = run_public_evaluation(
        args.selection_losses,
        args.test_losses,
        args.registry,
        args.training_manifest,
        args.output_dir,
        radius=args.radius,
        alpha=args.alpha,
        n_bootstrap=args.bootstrap,
        seed=args.seed,
        n_mixtures=args.mixtures,
    )
    print(json.dumps({name: path.as_posix() for name, path in outputs.items()}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
