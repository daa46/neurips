from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from experiments.te_regulation.cluster_and_split import TEBenchmarkError
from experiments.te_regulation.report_public_metrics import (
    expected_calibration_error,
    run_public_metrics_report,
)
from hrms.artifacts import sha256_file


def _record(path: Path, *, recorded_path: str | None = None) -> dict:
    return {
        "path": path.as_posix() if recorded_path is None else recorded_path,
        "size_bytes": path.stat().st_size,
        "sha256": sha256_file(path),
    }


def _prediction_frame(partition: str, model_ids: list[str]) -> pd.DataFrame:
    rows = []
    definitions = {
        "env_both": ([0, 1, 0, 1], [0.1, 0.9, 0.2, 0.8]),
        "env_single": ([1, 1], [0.7, 0.8]),
    }
    for environment, (labels, good_probabilities) in definitions.items():
        for index, (label, good) in enumerate(zip(labels, good_probabilities, strict=True)):
            row = {
                "observation_id": f"{partition}_{environment}_{index}",
                "environment_id": environment,
                "root_cluster_id": f"{partition}_{environment}_root_{index}",
                "nested_cluster_id": f"{partition}_{environment}_pair_{index}",
                "label": label,
                "sample_weight": 1.0,
            }
            for model_id in model_ids:
                if model_id == "model_good":
                    probability = good
                elif model_id == "model_bad":
                    probability = 1 - good
                else:
                    probability = 0.55 if label else 0.45
                row[f"prediction__{model_id}"] = probability
            rows.append(row)
    return pd.DataFrame(rows)


def _registry(model_ids: list[str], *, eligible: bool, prediction_hash: str) -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "model_id": model_id,
                "model_family": "fixture",
                "representation": "fixture",
                "pooling": "fixture",
                "regularization": "fixture",
                "seed": 42,
                "calibration": "training_only",
                "checkpoint_hash": hashlib.sha256(model_id.encode()).hexdigest(),
                "training_data_hash": hashlib.sha256(b"training fixture").hexdigest(),
                "prediction_file_hash": prediction_hash,
                "eligible_for_hrms_selection": eligible,
            }
            for model_id in model_ids
        ]
    )


def _bundle_hash(selection: Path, test: Path) -> str:
    digest = hashlib.sha256()
    digest.update(bytes.fromhex(sha256_file(selection)))
    digest.update(bytes.fromhex(sha256_file(test)))
    return digest.hexdigest()


def _fixture(tmp_path: Path) -> dict[str, Path]:
    training = tmp_path / "training"
    training.mkdir(parents=True)
    candidates = ["model_good", "model_bad"]
    diagnostics = ["diagnostic_gc"]
    paths = {
        "predictions_select": training / "predictions_select.parquet",
        "predictions_test": training / "predictions_test.parquet",
        "diagnostic_predictions_select": training / "diagnostic_predictions_select.parquet",
        "diagnostic_predictions_test": training / "diagnostic_predictions_test.parquet",
        "registry": training / "model_registry.csv",
        "diagnostic_registry": training / "diagnostic_model_registry.csv",
        "training_manifest": training / "training_manifest.json",
    }
    _prediction_frame("select", candidates).to_parquet(paths["predictions_select"], index=False)
    _prediction_frame("test", candidates).to_parquet(paths["predictions_test"], index=False)
    _prediction_frame("select", diagnostics).to_parquet(
        paths["diagnostic_predictions_select"], index=False
    )
    _prediction_frame("test", diagnostics).to_parquet(
        paths["diagnostic_predictions_test"], index=False
    )
    _registry(
        candidates,
        eligible=True,
        prediction_hash=_bundle_hash(paths["predictions_select"], paths["predictions_test"]),
    ).to_csv(paths["registry"], index=False, lineterminator="\n")
    _registry(
        diagnostics,
        eligible=False,
        prediction_hash=_bundle_hash(
            paths["diagnostic_predictions_select"],
            paths["diagnostic_predictions_test"],
        ),
    ).to_csv(paths["diagnostic_registry"], index=False, lineterminator="\n")
    training_manifest = {
        "schema_version": 1,
        "training": {"candidate_count": 2, "diagnostic_only_count": 1},
        "artifacts": {
            key: _record(path) for key, path in paths.items() if key != "training_manifest"
        },
    }
    paths["training_manifest"].write_text(
        json.dumps(training_manifest, sort_keys=True), encoding="utf-8"
    )

    evaluation_root = tmp_path / "evaluation"
    evaluation = evaluation_root / "test_evaluation"
    evaluation.mkdir(parents=True)
    summary_path = evaluation / "test_summary.json"
    summary_path.write_text(
        json.dumps(
            {
                "selection_was_persisted_before_test_table_opened": True,
                "test_hash_matched_training_manifest": True,
            }
        ),
        encoding="utf-8",
    )
    paths["evaluation_manifest"] = evaluation / "evaluation_manifest.json"
    paths["evaluation_manifest"].write_text(
        json.dumps(
            {
                "schema_version": 1,
                "parameters": {"selection_preceded_test_access": True},
                "inputs": {
                    "training_manifest": _record(paths["training_manifest"]),
                    "registry": _record(paths["registry"]),
                },
                "artifacts": {
                    "test_summary": _record(
                        summary_path,
                        recorded_path="test_evaluation/test_summary.json",
                    )
                },
            },
            sort_keys=True,
        ),
        encoding="utf-8",
    )
    return paths


def _run(paths: dict[str, Path], output: Path) -> dict[str, Path]:
    return run_public_metrics_report(
        paths["predictions_select"],
        paths["predictions_test"],
        paths["registry"],
        paths["training_manifest"],
        paths["evaluation_manifest"],
        output,
        diagnostic_registry_path=paths["diagnostic_registry"],
        diagnostic_predictions_select_path=paths["diagnostic_predictions_select"],
        diagnostic_predictions_test_path=paths["diagnostic_predictions_test"],
    )


def test_post_evaluation_report_is_hashed_separate_and_explicit(tmp_path: Path) -> None:
    paths = _fixture(tmp_path)
    output = tmp_path / "descriptive"
    outputs = _run(paths, output)

    candidate = pd.read_csv(outputs["candidate_environment_metrics"])
    assert len(candidate) == 8
    assert candidate["model_role"].eq("hrms_candidate").all()
    assert candidate["eligible_for_hrms_selection"].all()
    good = candidate.loc[
        candidate["partition"].eq("test")
        & candidate["environment_id"].eq("env_both")
        & candidate["model_id"].eq("model_good")
    ].iloc[0]
    assert good["brier"] == pytest.approx(0.025)
    assert good["auroc"] == pytest.approx(1.0)
    assert good["auprc"] == pytest.approx(1.0)
    assert good["ece"] == pytest.approx(0.15)
    assert good["balanced_accuracy"] == pytest.approx(1.0)

    status = pd.read_csv(outputs["candidate_metric_status"])
    undefined = status.loc[
        status["partition"].eq("test")
        & status["environment_id"].eq("env_single")
        & ~status["defined"]
    ]
    assert set(undefined["metric"]) == {"auroc", "auprc", "balanced_accuracy"}
    assert undefined["status"].str.contains("single_class").all()
    macro = pd.read_csv(outputs["candidate_macro_metrics"])
    macro_auroc = macro.loc[
        macro["partition"].eq("test")
        & macro["model_id"].eq("model_good")
        & macro["metric"].eq("auroc")
    ].iloc[0]
    assert macro_auroc["environments_defined"] == 1
    assert macro_auroc["environments_total"] == 2
    assert macro_auroc["undefined_environment_ids"] == "env_single"

    diagnostic = pd.read_csv(outputs["diagnostic_environment_metrics"])
    assert diagnostic["model_role"].eq("diagnostic_only").all()
    assert not diagnostic["eligible_for_hrms_selection"].any()
    assert set(candidate["model_id"]).isdisjoint(diagnostic["model_id"])
    summary = json.loads(outputs["summary"].read_text())
    assert summary["executed_after_sealed_evaluation"] is True
    assert summary["selection_or_tuning_performed"] is False
    assert summary["balanced_accuracy_threshold"] == 0.5

    manifest = json.loads(outputs["manifest"].read_text())
    assert manifest["parameters"]["reporting_only"] is True
    assert manifest["parameters"]["diagnostic_controls_separate"] is True
    for record in manifest["artifacts"].values():
        artifact = output.parent / record["path"]
        assert artifact.stat().st_size == record["size_bytes"]
        assert sha256_file(artifact) == record["sha256"]


def test_report_rejects_prediction_tampering_and_unsealed_evaluation(tmp_path: Path) -> None:
    paths = _fixture(tmp_path)
    changed = pd.read_parquet(paths["predictions_test"])
    changed.loc[0, "prediction__model_good"] += 0.001
    changed.to_parquet(paths["predictions_test"], index=False)
    with pytest.raises(TEBenchmarkError, match="immutable manifest"):
        _run(paths, tmp_path / "tampered")

    paths = _fixture(tmp_path / "second")
    evaluation = json.loads(paths["evaluation_manifest"].read_text())
    evaluation["parameters"]["selection_preceded_test_access"] = False
    paths["evaluation_manifest"].write_text(json.dumps(evaluation), encoding="utf-8")
    with pytest.raises(TEBenchmarkError, match="selection-before-test"):
        _run(paths, tmp_path / "unsealed")


def test_diagnostic_bundle_is_optional_but_not_partial(tmp_path: Path) -> None:
    paths = _fixture(tmp_path)
    outputs = run_public_metrics_report(
        paths["predictions_select"],
        paths["predictions_test"],
        paths["registry"],
        paths["training_manifest"],
        paths["evaluation_manifest"],
        tmp_path / "without_diagnostics",
    )
    assert "diagnostic_environment_metrics" not in outputs
    with pytest.raises(ValueError, match="all-or-none"):
        run_public_metrics_report(
            paths["predictions_select"],
            paths["predictions_test"],
            paths["registry"],
            paths["training_manifest"],
            paths["evaluation_manifest"],
            tmp_path / "partial",
            diagnostic_registry_path=paths["diagnostic_registry"],
        )


def test_weighted_ece_has_fixed_bins_and_handles_probability_one() -> None:
    labels = np.asarray([0, 1, 1], dtype=float)
    probabilities = np.asarray([0.1, 0.8, 1.0], dtype=float)
    weights = np.asarray([1.0, 2.0, 1.0])
    assert expected_calibration_error(labels, probabilities, weights, n_bins=10) == pytest.approx(
        0.125
    )
