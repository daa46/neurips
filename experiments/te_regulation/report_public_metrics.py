#!/usr/bin/env python3
"""Reporting-only descriptive metrics for the sealed public-TE benchmark."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from collections.abc import Mapping
from pathlib import Path

import numpy as np
import pandas as pd
import sklearn
from sklearn.metrics import average_precision_score, roc_auc_score

from experiments.te_regulation.cluster_and_split import TEBenchmarkError, read_table
from experiments.te_regulation.evaluate_public import validate_partition_separation
from hrms.artifacts import sha256_file, write_artifact_manifest
from hrms.predictions import PREDICTION_PREFIX
from hrms.registry import assert_prediction_library_matches, validate_model_registry

_PROJECT_ROOT = Path(__file__).resolve().parents[2]
METRICS = ("brier", "auroc", "auprc", "ece", "balanced_accuracy")
DEFAULT_TRAINING_DIR = Path("results/te_public/training")
DEFAULT_EVALUATION_MANIFEST = Path(
    "results/te_public/evaluation/test_evaluation/evaluation_manifest.json"
)
DEFAULT_OUTPUT_DIR = Path("results/te_public/evaluation/descriptive_metrics")


def _write_json(path: Path, payload: object) -> None:
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _write_csv(frame: pd.DataFrame, path: Path) -> None:
    frame.to_csv(path, index=False, lineterminator="\n", float_format="%.17g")


def _record(container: Mapping, section: str, key: str, *, label: str) -> Mapping:
    if container.get("schema_version") != 1:
        raise TEBenchmarkError(f"{label} must use schema version 1")
    records = container.get(section)
    if not isinstance(records, Mapping) or key not in records:
        raise TEBenchmarkError(f"{label} lacks {section} record {key!r}")
    record = records[key]
    if not isinstance(record, Mapping):
        raise TEBenchmarkError(f"invalid {label} record {key!r}")
    digest, size = record.get("sha256"), record.get("size_bytes")
    if (
        not isinstance(digest, str)
        or len(digest) != 64
        or any(character not in "0123456789abcdef" for character in digest.lower())
        or not isinstance(size, int)
        or isinstance(size, bool)
        or size < 0
    ):
        raise TEBenchmarkError(f"incomplete {label} record {key!r}")
    return record


def _verify(path: Path, record: Mapping, *, label: str) -> None:
    if not path.is_file():
        raise FileNotFoundError(f"{label} does not exist: {path}")
    if path.stat().st_size != int(record["size_bytes"]):
        raise TEBenchmarkError(f"{label} size differs from its immutable manifest")
    if sha256_file(path) != str(record["sha256"]):
        raise TEBenchmarkError(f"{label} hash differs from its immutable manifest")


def _evaluation_artifact(manifest_path: Path, record: Mapping) -> Path:
    path = Path(str(record.get("path", "")))
    if not str(path):
        raise TEBenchmarkError("sealed-evaluation artifact has no path")
    return path if path.is_absolute() else manifest_path.parent.parent / path


def _verify_sealed_evaluation(
    manifest_path: Path, training_manifest_path: Path, registry_path: Path
) -> dict:
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("schema_version") != 1:
        raise TEBenchmarkError("sealed evaluation manifest must use schema version 1")
    if manifest.get("parameters", {}).get("selection_preceded_test_access") is not True:
        raise TEBenchmarkError(
            "descriptive metrics require a sealed evaluation attesting selection-before-test"
        )
    for key, path in (("training_manifest", training_manifest_path), ("registry", registry_path)):
        _verify(path, _record(manifest, "inputs", key, label="evaluation manifest"), label=key)
    artifacts = manifest.get("artifacts")
    if not isinstance(artifacts, Mapping) or not artifacts:
        raise TEBenchmarkError("sealed evaluation manifest has no artifacts")
    for key in artifacts:
        record = _record(manifest, "artifacts", key, label="evaluation manifest")
        _verify(_evaluation_artifact(manifest_path, record), record, label=f"evaluation {key}")
    summary_record = _record(manifest, "artifacts", "test_summary", label="evaluation manifest")
    summary = json.loads(
        _evaluation_artifact(manifest_path, summary_record).read_text(encoding="utf-8")
    )
    if summary.get("selection_was_persisted_before_test_table_opened") is not True:
        raise TEBenchmarkError("sealed test summary lacks the required ordering attestation")
    if summary.get("test_hash_matched_training_manifest") is not True:
        raise TEBenchmarkError("sealed test summary does not attest the held-out test hash")
    return manifest


def _booleans(series: pd.Series, *, label: str) -> pd.Series:
    mapping = {True: True, False: False, "true": True, "false": False}
    normalized = series.map(lambda value: value.lower() if isinstance(value, str) else value)
    output = normalized.map(mapping)
    if output.isna().any():
        raise TEBenchmarkError(f"{label} must contain explicit Boolean values")
    return output.astype(bool)


def _load_registry(
    path: Path, manifest: Mapping, *, artifact_key: str, eligible: bool
) -> pd.DataFrame:
    _verify(
        path,
        _record(manifest, "artifacts", artifact_key, label="training manifest"),
        label=artifact_key,
    )
    registry = pd.read_csv(path)
    validate_model_registry(registry)
    if "eligible_for_hrms_selection" not in registry:
        raise TEBenchmarkError(f"{artifact_key} lacks eligible_for_hrms_selection")
    observed = _booleans(
        registry["eligible_for_hrms_selection"],
        label=f"{artifact_key}.eligible_for_hrms_selection",
    )
    if not observed.eq(eligible).all():
        raise TEBenchmarkError(f"{artifact_key} has an incorrect HRMS eligibility role")
    registry = registry.copy()
    registry["model_id"] = registry["model_id"].astype(str)
    registry["eligible_for_hrms_selection"] = observed
    return registry


def _load_predictions(
    path: Path, manifest: Mapping, *, artifact_key: str, registry: pd.DataFrame
) -> pd.DataFrame:
    _verify(
        path,
        _record(manifest, "artifacts", artifact_key, label="training manifest"),
        label=artifact_key,
    )
    frame = read_table(path).copy()
    required = {
        "observation_id",
        "environment_id",
        "root_cluster_id",
        "label",
        "sample_weight",
    }
    missing = required.difference(frame.columns)
    if missing:
        raise TEBenchmarkError(f"{artifact_key} is missing columns: {sorted(missing)}")
    if frame.empty or frame[list(required)].isna().any().any():
        raise TEBenchmarkError(f"{artifact_key} required values must be nonempty and non-null")
    if frame.columns.duplicated().any() or frame["observation_id"].duplicated().any():
        raise TEBenchmarkError(f"{artifact_key} has duplicate columns or observations")
    for column in ("observation_id", "environment_id", "root_cluster_id"):
        frame[column] = frame[column].astype(str)
    labels = pd.to_numeric(frame["label"], errors="coerce").to_numpy(dtype=float)
    weights = pd.to_numeric(frame["sample_weight"], errors="coerce").to_numpy(dtype=float)
    if not np.isfinite(labels).all() or not np.isin(labels, [0, 1]).all():
        raise TEBenchmarkError(f"{artifact_key} labels must be binary")
    if not np.isfinite(weights).all() or np.any(weights <= 0):
        raise TEBenchmarkError(f"{artifact_key} weights must be finite and positive")
    frame["label"], frame["sample_weight"] = labels.astype(np.int8), weights
    prediction_columns = [
        column
        for column in frame
        if isinstance(column, str) and column.startswith(PREDICTION_PREFIX)
    ]
    assert_prediction_library_matches(
        registry, [column.removeprefix(PREDICTION_PREFIX) for column in prediction_columns]
    )
    probabilities = frame[prediction_columns].to_numpy(dtype=float)
    if not np.isfinite(probabilities).all() or np.any((probabilities < 0) | (probabilities > 1)):
        raise TEBenchmarkError(f"{artifact_key} predictions must lie in [0, 1]")
    return frame.sort_values("observation_id", kind="stable").reset_index(drop=True)


def _verify_bundle_hash(
    registry: pd.DataFrame, selection_path: Path, test_path: Path, *, label: str
) -> None:
    if "prediction_file_hash" not in registry:
        raise TEBenchmarkError(f"{label} registry lacks prediction_file_hash")
    hashes = registry["prediction_file_hash"].astype(str)
    if hashes.nunique() != 1:
        raise TEBenchmarkError(f"{label} registry has inconsistent prediction hashes")
    digest = hashlib.sha256()
    digest.update(bytes.fromhex(sha256_file(selection_path)))
    digest.update(bytes.fromhex(sha256_file(test_path)))
    if digest.hexdigest() != hashes.iloc[0]:
        raise TEBenchmarkError(f"{label} prediction bundle hash differs from its registry")


def _matching_metadata(left: pd.DataFrame, right: pd.DataFrame, *, partition: str) -> None:
    optional = ("nested_cluster_id", "pair_id")
    columns = [
        "observation_id",
        "environment_id",
        "root_cluster_id",
        "label",
        "sample_weight",
        *[column for column in optional if column in left or column in right],
    ]
    if any(column not in left or column not in right for column in columns):
        raise TEBenchmarkError(f"candidate and diagnostic {partition} metadata differ")
    try:
        pd.testing.assert_frame_equal(left[columns], right[columns], check_dtype=False)
    except AssertionError as error:
        raise TEBenchmarkError(f"candidate and diagnostic {partition} metadata differ") from error


def expected_calibration_error(
    labels: np.ndarray,
    probabilities: np.ndarray,
    weights: np.ndarray,
    *,
    n_bins: int = 10,
) -> float:
    """Return weighted ECE using fixed-width bins on ``[0, 1]``."""

    if isinstance(n_bins, bool) or not isinstance(n_bins, int) or not 2 <= n_bins <= 100:
        raise ValueError("n_bins must be an integer in [2, 100]")
    labels = np.asarray(labels, dtype=float)
    probabilities = np.asarray(probabilities, dtype=float)
    weights = np.asarray(weights, dtype=float)
    if not (labels.shape == probabilities.shape == weights.shape) or labels.ndim != 1:
        raise ValueError("ECE inputs must be one-dimensional arrays of equal length")
    if len(labels) == 0 or not np.isin(labels, [0, 1]).all():
        raise ValueError("ECE requires nonempty binary labels")
    if (
        not np.isfinite(probabilities).all()
        or np.any((probabilities < 0) | (probabilities > 1))
        or not np.isfinite(weights).all()
        or np.any(weights <= 0)
    ):
        raise ValueError("ECE probabilities and weights are invalid")
    bin_index = np.minimum((probabilities * n_bins).astype(np.int64), n_bins - 1)
    total_weight = float(weights.sum())
    value = 0.0
    for index in range(n_bins):
        mask = bin_index == index
        if np.any(mask):
            bin_weight = float(weights[mask].sum())
            confidence = float(np.average(probabilities[mask], weights=weights[mask]))
            frequency = float(np.average(labels[mask], weights=weights[mask]))
            value += bin_weight / total_weight * abs(confidence - frequency)
    return value


def _metric_values(
    labels: np.ndarray,
    probabilities: np.ndarray,
    weights: np.ndarray,
    *,
    ece_bins: int,
) -> tuple[dict[str, float], dict[str, tuple[bool, str]]]:
    values = {
        "brier": float(np.average(np.square(probabilities - labels), weights=weights)),
        "auroc": math.nan,
        "auprc": math.nan,
        "ece": expected_calibration_error(labels, probabilities, weights, n_bins=ece_bins),
        "balanced_accuracy": math.nan,
    }
    status = {"brier": (True, "defined"), "ece": (True, "defined")}
    if np.unique(labels).size < 2:
        reason = "undefined_single_class_environment: both labels are required"
        status.update({metric: (False, reason) for metric in METRICS if metric not in status})
        return values, status
    values["auroc"] = float(roc_auc_score(labels, probabilities, sample_weight=weights))
    values["auprc"] = float(average_precision_score(labels, probabilities, sample_weight=weights))
    predicted = probabilities >= 0.5
    positive = labels == 1
    negative = ~positive
    sensitivity = float(np.average(predicted[positive], weights=weights[positive]))
    specificity = float(np.average(~predicted[negative], weights=weights[negative]))
    values["balanced_accuracy"] = (sensitivity + specificity) / 2
    status.update({metric: (True, "defined") for metric in METRICS if metric not in status})
    return values, status


def compute_environment_metrics(
    predictions: pd.DataFrame,
    registry: pd.DataFrame,
    *,
    partition: str,
    role: str,
    eligible_for_hrms_selection: bool,
    ece_bins: int = 10,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Compute environment cells, cell statuses, and environment-macro summaries."""

    if partition not in {"select", "test"}:
        raise ValueError("partition must be 'select' or 'test'")
    environment_rows, status_rows = [], []
    model_ids = registry["model_id"].astype(str).tolist()
    for environment_id, frame in predictions.groupby("environment_id", sort=True, observed=True):
        labels = frame["label"].to_numpy(dtype=np.int8)
        weights = frame["sample_weight"].to_numpy(dtype=float)
        positive = labels == 1
        common = {
            "partition": partition,
            "model_role": role,
            "eligible_for_hrms_selection": eligible_for_hrms_selection,
            "environment_id": str(environment_id),
            "n_observations": len(frame),
            "n_positive": int(positive.sum()),
            "n_negative": int((~positive).sum()),
            "total_weight": float(weights.sum()),
            "positive_weight": float(weights[positive].sum()),
            "negative_weight": float(weights[~positive].sum()),
        }
        for model_id in model_ids:
            probability = frame[f"{PREDICTION_PREFIX}{model_id}"].to_numpy(dtype=float)
            values, statuses = _metric_values(labels, probability, weights, ece_bins=ece_bins)
            environment_rows.append({**common, "model_id": model_id, **values})
            for metric in METRICS:
                defined, message = statuses[metric]
                status_rows.append(
                    {
                        "partition": partition,
                        "model_role": role,
                        "eligible_for_hrms_selection": eligible_for_hrms_selection,
                        "environment_id": str(environment_id),
                        "model_id": model_id,
                        "metric": metric,
                        "defined": defined,
                        "status": message,
                    }
                )
    environment = pd.DataFrame(environment_rows).sort_values(
        ["partition", "environment_id", "model_id"], kind="stable"
    )
    statuses = pd.DataFrame(status_rows).sort_values(
        ["partition", "environment_id", "model_id", "metric"], kind="stable"
    )
    macro_rows = []
    for model_id, frame in environment.groupby("model_id", sort=True, observed=True):
        for metric in METRICS:
            defined = frame[metric].notna()
            macro_rows.append(
                {
                    "partition": partition,
                    "model_role": role,
                    "eligible_for_hrms_selection": eligible_for_hrms_selection,
                    "model_id": str(model_id),
                    "metric": metric,
                    "macro_value": (
                        float(frame.loc[defined, metric].mean()) if defined.any() else math.nan
                    ),
                    "environments_defined": int(defined.sum()),
                    "environments_total": len(frame),
                    "undefined_environment_ids": "|".join(
                        frame.loc[~defined, "environment_id"].astype(str)
                    ),
                    "aggregation": "unweighted_mean_across_environments",
                }
            )
    macro = pd.DataFrame(macro_rows).sort_values(["partition", "model_id", "metric"], kind="stable")
    return tuple(frame.reset_index(drop=True) for frame in (environment, statuses, macro))


def _metric_bundle(
    selection: pd.DataFrame,
    test: pd.DataFrame,
    registry: pd.DataFrame,
    *,
    role: str,
    eligible: bool,
    ece_bins: int,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    parts = [
        compute_environment_metrics(
            frame,
            registry,
            partition=partition,
            role=role,
            eligible_for_hrms_selection=eligible,
            ece_bins=ece_bins,
        )
        for partition, frame in (("select", selection), ("test", test))
    ]
    return tuple(
        pd.concat([part[index] for part in parts], ignore_index=True) for index in range(3)
    )


def run_public_metrics_report(
    predictions_select_path: Path,
    predictions_test_path: Path,
    registry_path: Path,
    training_manifest_path: Path,
    evaluation_manifest_path: Path,
    output_dir: Path,
    *,
    diagnostic_registry_path: Path | None = None,
    diagnostic_predictions_select_path: Path | None = None,
    diagnostic_predictions_test_path: Path | None = None,
    ece_bins: int = 10,
) -> dict[str, Path]:
    """Report metrics only after verifying the sealed test-evaluation bundle."""

    if isinstance(ece_bins, bool) or not isinstance(ece_bins, int) or not 2 <= ece_bins <= 100:
        raise ValueError("ece_bins must be an integer in [2, 100]")
    diagnostic_paths = (
        diagnostic_registry_path,
        diagnostic_predictions_select_path,
        diagnostic_predictions_test_path,
    )
    if any(path is None for path in diagnostic_paths) and not all(
        path is None for path in diagnostic_paths
    ):
        raise ValueError("diagnostic registry and both prediction files are all-or-none")
    if not training_manifest_path.is_file():
        raise FileNotFoundError(f"training manifest does not exist: {training_manifest_path}")
    training_manifest = json.loads(training_manifest_path.read_text(encoding="utf-8"))
    if training_manifest.get("schema_version") != 1:
        raise TEBenchmarkError("training manifest must use schema version 1")
    _verify_sealed_evaluation(evaluation_manifest_path, training_manifest_path, registry_path)

    registry = _load_registry(
        registry_path, training_manifest, artifact_key="registry", eligible=True
    )
    expected_count = training_manifest.get("training", {}).get("candidate_count")
    if expected_count is not None and int(expected_count) != len(registry):
        raise TEBenchmarkError("candidate count differs from the training manifest")
    selection = _load_predictions(
        predictions_select_path,
        training_manifest,
        artifact_key="predictions_select",
        registry=registry,
    )
    test = _load_predictions(
        predictions_test_path,
        training_manifest,
        artifact_key="predictions_test",
        registry=registry,
    )
    _verify_bundle_hash(registry, predictions_select_path, predictions_test_path, label="candidate")
    validate_partition_separation(selection, test)
    candidate_frames = _metric_bundle(
        selection,
        test,
        registry,
        role="hrms_candidate",
        eligible=True,
        ece_bins=ece_bins,
    )

    outputs = {
        "candidate_environment_metrics": output_dir / "candidate_environment_metrics.csv",
        "candidate_metric_status": output_dir / "candidate_metric_status.csv",
        "candidate_macro_metrics": output_dir / "candidate_macro_metrics.csv",
        "summary": output_dir / "report_summary.json",
        "manifest": output_dir / "descriptive_metrics_manifest.json",
    }
    inputs: dict[str, Path] = {
        "training_manifest": training_manifest_path,
        "evaluation_manifest": evaluation_manifest_path,
        "registry": registry_path,
        "predictions_select": predictions_select_path,
        "predictions_test": predictions_test_path,
    }
    diagnostic_frames = None
    if all(path is not None for path in diagnostic_paths):
        assert diagnostic_registry_path is not None
        assert diagnostic_predictions_select_path is not None
        assert diagnostic_predictions_test_path is not None
        diagnostic_registry = _load_registry(
            diagnostic_registry_path,
            training_manifest,
            artifact_key="diagnostic_registry",
            eligible=False,
        )
        if set(registry["model_id"]) & set(diagnostic_registry["model_id"]):
            raise TEBenchmarkError("diagnostic controls overlap the HRMS candidate library")
        diagnostic_selection = _load_predictions(
            diagnostic_predictions_select_path,
            training_manifest,
            artifact_key="diagnostic_predictions_select",
            registry=diagnostic_registry,
        )
        diagnostic_test = _load_predictions(
            diagnostic_predictions_test_path,
            training_manifest,
            artifact_key="diagnostic_predictions_test",
            registry=diagnostic_registry,
        )
        _verify_bundle_hash(
            diagnostic_registry,
            diagnostic_predictions_select_path,
            diagnostic_predictions_test_path,
            label="diagnostic",
        )
        _matching_metadata(selection, diagnostic_selection, partition="select")
        _matching_metadata(test, diagnostic_test, partition="test")
        diagnostic_frames = _metric_bundle(
            diagnostic_selection,
            diagnostic_test,
            diagnostic_registry,
            role="diagnostic_only",
            eligible=False,
            ece_bins=ece_bins,
        )
        outputs.update(
            {
                "diagnostic_environment_metrics": output_dir / "diagnostic_environment_metrics.csv",
                "diagnostic_metric_status": output_dir / "diagnostic_metric_status.csv",
                "diagnostic_macro_metrics": output_dir / "diagnostic_macro_metrics.csv",
            }
        )
        inputs.update(
            {
                "diagnostic_registry": diagnostic_registry_path,
                "diagnostic_predictions_select": diagnostic_predictions_select_path,
                "diagnostic_predictions_test": diagnostic_predictions_test_path,
            }
        )

    existing = sorted(path.name for path in outputs.values() if path.exists())
    if existing:
        raise TEBenchmarkError(
            "descriptive-metrics outputs already exist; use a new directory: " + ", ".join(existing)
        )
    output_dir.mkdir(parents=True, exist_ok=True)
    candidate_keys = (
        "candidate_environment_metrics",
        "candidate_metric_status",
        "candidate_macro_metrics",
    )
    for frame, key in zip(candidate_frames, candidate_keys, strict=True):
        _write_csv(frame, outputs[key])
    diagnostic_keys = (
        "diagnostic_environment_metrics",
        "diagnostic_metric_status",
        "diagnostic_macro_metrics",
    )
    if diagnostic_frames is not None:
        for frame, key in zip(diagnostic_frames, diagnostic_keys, strict=True):
            _write_csv(frame, outputs[key])
    _write_json(
        outputs["summary"],
        {
            "schema_version": 1,
            "task": "public TE-associated enhancer-like cCRE prediction",
            "reporting_only": True,
            "executed_after_sealed_evaluation": True,
            "selection_or_tuning_performed": False,
            "partitions_reported": ["select", "test"],
            "candidate_count": len(registry),
            "diagnostic_only_controls_included": diagnostic_frames is not None,
            "diagnostic_only_controls_eligible_for_hrms_selection": False,
            "metrics": list(METRICS),
            "brier_definition": "sample-weighted mean squared probability error",
            "auprc_definition": "sample-weighted non-interpolated average precision",
            "ece_definition": (f"sample-weighted ECE with {ece_bins} fixed-width bins on [0,1]"),
            "balanced_accuracy_threshold": 0.5,
            "single_class_policy": (
                "AUROC, AUPRC, and balanced accuracy undefined; Brier and ECE defined"
            ),
            "macro_definition": "unweighted mean of defined environment-specific cells",
        },
    )
    artifacts = {key: outputs[key] for key in candidate_keys}
    if diagnostic_frames is not None:
        artifacts.update({key: outputs[key] for key in diagnostic_keys})
    artifacts["summary"] = outputs["summary"]
    write_artifact_manifest(
        outputs["manifest"],
        inputs=inputs,
        artifacts=artifacts,
        parameters={
            "reporting_only": True,
            "selection_or_tuning_performed": False,
            "requires_sealed_evaluation": True,
            "ece_bins": ece_bins,
            "balanced_accuracy_threshold": 0.5,
            "diagnostic_controls_separate": True,
            "scikit_learn": sklearn.__version__,
        },
        source_files=[Path(__file__)],
        source_base_dir=_PROJECT_ROOT,
        base_dir=output_dir.parent,
    )
    return outputs


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--predictions-select",
        type=Path,
        default=DEFAULT_TRAINING_DIR / "predictions_select.parquet",
    )
    parser.add_argument(
        "--predictions-test",
        type=Path,
        default=DEFAULT_TRAINING_DIR / "predictions_test.parquet",
    )
    parser.add_argument(
        "--registry", type=Path, default=DEFAULT_TRAINING_DIR / "model_registry.csv"
    )
    parser.add_argument(
        "--training-manifest",
        type=Path,
        default=DEFAULT_TRAINING_DIR / "training_manifest.json",
    )
    parser.add_argument("--evaluation-manifest", type=Path, default=DEFAULT_EVALUATION_MANIFEST)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument(
        "--diagnostic-registry",
        type=Path,
        default=DEFAULT_TRAINING_DIR / "diagnostic_model_registry.csv",
    )
    parser.add_argument(
        "--diagnostic-predictions-select",
        type=Path,
        default=DEFAULT_TRAINING_DIR / "diagnostic_predictions_select.parquet",
    )
    parser.add_argument(
        "--diagnostic-predictions-test",
        type=Path,
        default=DEFAULT_TRAINING_DIR / "diagnostic_predictions_test.parquet",
    )
    parser.add_argument(
        "--without-diagnostics",
        action="store_true",
        help="omit the separate diagnostic-only control report",
    )
    parser.add_argument("--ece-bins", type=int, default=10)
    args = parser.parse_args(argv)
    diagnostic = (
        (None, None, None)
        if args.without_diagnostics
        else (
            args.diagnostic_registry,
            args.diagnostic_predictions_select,
            args.diagnostic_predictions_test,
        )
    )
    outputs = run_public_metrics_report(
        args.predictions_select,
        args.predictions_test,
        args.registry,
        args.training_manifest,
        args.evaluation_manifest,
        args.output_dir,
        diagnostic_registry_path=diagnostic[0],
        diagnostic_predictions_select_path=diagnostic[1],
        diagnostic_predictions_test_path=diagnostic[2],
        ece_bins=args.ece_bins,
    )
    print(json.dumps({name: path.as_posix() for name, path in outputs.items()}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
