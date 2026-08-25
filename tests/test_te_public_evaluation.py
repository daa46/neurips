from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from applications.te_regulation.cluster_and_split import TEBenchmarkError
from applications.te_regulation.evaluate_public import (
    deterministic_target_mixtures,
    evaluate_sealed_test,
    run_public_evaluation,
    validate_partition_separation,
)
from hrms.artifacts import sha256_file


def _registry() -> pd.DataFrame:
    rows = []
    for index in range(3):
        model_id = f"model_{index}"
        rows.append(
            {
                "model_id": model_id,
                "model_family": "fixture",
                "representation": f"fixture_{index}",
                "pooling": "none",
                "regularization": f"setting_{index}",
                "seed": 42,
                "calibration": "training_only",
                "checkpoint_hash": hashlib.sha256(model_id.encode()).hexdigest(),
                "training_data_hash": hashlib.sha256(b"fixture_training").hexdigest(),
            }
        )
    return pd.DataFrame(rows)


def _losses(partition: str) -> pd.DataFrame:
    rows = []
    selection_effect = np.asarray(
        [[0.00, 0.025, 0.045], [0.035, 0.005, 0.03], [0.04, 0.03, 0.00]]
    )
    test_effect = np.asarray(
        [[0.035, 0.00, 0.04], [0.01, 0.035, 0.02], [0.045, 0.025, 0.00]]
    )
    effects = selection_effect if partition == "select" else test_effect
    for environment in range(3):
        for cluster in range(7):
            observation_count = 1 + cluster % 3
            for observation in range(observation_count):
                observation_id = f"{partition}_e{environment}_c{cluster}_o{observation}"
                shared = 0.12 + 0.006 * cluster + 0.003 * observation
                for model in range(3):
                    rows.append(
                        {
                            "observation_id": observation_id,
                            "environment_id": f"env_{environment}",
                            "root_cluster_id": f"{partition}_e{environment}_root_{cluster}",
                            "model_id": f"model_{model}",
                            "loss": shared + effects[environment, model],
                            "sample_weight": 1.0,
                        }
                    )
    return pd.DataFrame(rows)


def _record(path: Path) -> dict:
    return {
        "path": path.as_posix(),
        "size_bytes": path.stat().st_size,
        "sha256": sha256_file(path),
    }


def _fixture(tmp_path: Path) -> tuple[Path, Path, Path, Path]:
    registry_path = tmp_path / "model_registry.csv"
    selection_path = tmp_path / "losses_select.parquet"
    test_path = tmp_path / "losses_test.parquet"
    manifest_path = tmp_path / "training_manifest.json"
    _registry().to_csv(registry_path, index=False, lineterminator="\n")
    _losses("select").to_parquet(selection_path, index=False)
    _losses("test").to_parquet(test_path, index=False)
    manifest = {
        "schema_version": 1,
        "training": {"candidate_count": 3},
        "artifacts": {
            "registry": _record(registry_path),
            "losses_select": _record(selection_path),
            "losses_test": _record(test_path),
        },
    }
    manifest_path.write_text(json.dumps(manifest, sort_keys=True), encoding="utf-8")
    return selection_path, test_path, registry_path, manifest_path


def test_public_te_evaluation_is_sealed_complete_and_hashed(tmp_path: Path) -> None:
    selection, test, registry, training_manifest = _fixture(tmp_path)
    output = tmp_path / "evaluation"
    manifests = run_public_evaluation(
        selection,
        test,
        registry,
        training_manifest,
        output,
        n_bootstrap=31,
        n_mixtures=12,
    )

    selection_manifest = json.loads(manifests["selection_manifest"].read_text())
    assert selection_manifest["parameters"]["test_table_accessed"] is False
    evaluation_manifest = json.loads(manifests["evaluation_manifest"].read_text())
    for record in evaluation_manifest["artifacts"].values():
        artifact = output / record["path"]
        assert artifact.stat().st_size == record["size_bytes"]
        assert sha256_file(artifact) == record["sha256"]

    selected = pd.read_csv(output / "selection" / "selected_models.csv")
    assert set(selected["method"]) == {
        "uniform_mean",
        "worst_environment",
        "point_dro_risk",
        "point_dro_regret",
        "hrms",
    }
    assert set(selected["analysis_id"]) == {
        "root_observation",
        "root_cluster",
        "iid_observation",
    }
    path = pd.read_csv(output / "selection" / "robustness_path.csv", dtype=str)
    assert set(path["radius"]) == {"0", "0.1", "0.25", "0.5", "1", "full"}

    mixture_metadata = pd.read_csv(
        output / "test_evaluation" / "target_mixture_metadata.csv"
    )
    required_scopes = {
        "nominal",
        "inside_primary",
        "primary_boundary",
        "outside_primary",
        "simplex_vertex",
    }
    assert required_scopes.issubset(set(mixture_metadata["scope"]))
    weights = pd.read_csv(
        output / "test_evaluation" / "target_mixture_weights.csv.gz"
    )
    assert np.allclose(weights.groupby("mixture_id")["weight"].sum(), 1.0)
    audit = pd.read_csv(output / "test_evaluation" / "certificate_audit.csv")
    assert {
        "interval_contains_test_reference",
        "confidence_set_covers_any_test_optimum",
        "test_reference_robust_regret",
    }.issubset(audit.columns)
    ranking = pd.read_csv(output / "test_evaluation" / "ranking_heatmap.csv")
    assert set(ranking["partition"]) == {"select", "test"}
    assert set(ranking["weighting"]) == {"observation", "cluster"}

    changed = pd.read_parquet(test)
    changed.loc[0, "loss"] += 0.001
    changed.to_parquet(test, index=False)
    with pytest.raises(TEBenchmarkError, match="manifest"):
        evaluate_sealed_test(
            test,
            selection,
            registry,
            training_manifest,
            manifests["selection_manifest"],
            tmp_path / "tampered_evaluation",
            n_mixtures=4,
        )


def test_target_mixture_design_is_deterministic_and_contains_stress_cases() -> None:
    first = deterministic_target_mixtures(
        ["env_c", "env_a", "env_b"], primary_radius=0.25, n_halton=16
    )
    second = deterministic_target_mixtures(
        ["env_b", "env_c", "env_a"], primary_radius=0.25, n_halton=16
    )
    pd.testing.assert_frame_equal(first[0], second[0])
    pd.testing.assert_frame_equal(first[1], second[1])
    np.testing.assert_allclose(first[2], second[2])
    assert np.all(first[2] >= 0)
    assert np.allclose(first[2].sum(axis=1), 1.0)


def test_partition_validation_rejects_homology_leakage() -> None:
    selection = _losses("select")
    test = _losses("test")
    leaked_root = selection["root_cluster_id"].iloc[0]
    test.loc[test["observation_id"].eq(test["observation_id"].iloc[0]), "root_cluster_id"] = (
        leaked_root
    )
    with pytest.raises(TEBenchmarkError, match="homology roots overlap"):
        validate_partition_separation(selection, test)
