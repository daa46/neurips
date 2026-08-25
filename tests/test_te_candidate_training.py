from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from applications.te_regulation.cluster_and_split import TEBenchmarkError
from applications.te_regulation.train_public_candidates import (
    DEFAULT_CONFIG,
    load_frozen_config,
    train_public_library,
    validate_cohort,
)
from hrms.registry import validate_model_registry
from hrms.splits import split_assignment_hash


def test_public_candidate_config_is_frozen_and_complete() -> None:
    config = load_frozen_config(DEFAULT_CONFIG)
    assert len(config["candidates"]) == 24
    assert {item["kind"] for item in config["diagnostic_only"]} == {
        "gc_length",
        "environment_only",
    }


def _cohort() -> pd.DataFrame:
    rng = np.random.default_rng(19)
    rows = []
    split_sizes = {"train": 40, "select": 16, "test": 16}
    offset = 0
    for split, size in split_sizes.items():
        for index in range(size):
            label = index % 2
            environment = f"env_{(index // 2) % 2}"
            window = rng.choice(np.asarray(list("ACGT")), size=512)
            window[248:264] = list("AAAAAAAAAAAAAAAA" if label else "CCCCCCCCCCCCCCCC")
            sequence = "".join(window.tolist())
            te_sequence = sequence[96:416]
            rows.append(
                {
                    "observation_id": f"obs_{offset + index:04d}",
                    "environment_id": environment,
                    "root_cluster_id": f"root_{offset + index // 2:04d}",
                    "pair_id": f"pair_{offset + index // 2:04d}",
                    "split": split,
                    "label": label,
                    "sample_weight": 1.0,
                    "te_sequence": te_sequence,
                    "window_sequence": sequence,
                    "te_gc_fraction": sum(base in "GC" for base in te_sequence)
                    / len(te_sequence),
                    "window_gc_fraction": sum(base in "GC" for base in sequence)
                    / len(sequence),
                    "te_length": len(te_sequence),
                }
            )
        offset += size
    cohort = pd.DataFrame(rows)
    cohort["split_seed"] = 17
    cohort["split_hash"] = split_assignment_hash(cohort)
    return cohort


def _mini_config(path: Path) -> Path:
    config = {
        "schema_version": 1,
        "frozen_before_selection": True,
        "seed": 42,
        "calibration": {
            "method": "heldout_platt",
            "fraction_of_training_roots": 0.2,
            "split_trials": 1024,
            "solver": "liblinear",
            "C": 1000.0,
            "max_iter": 2000,
        },
        "vectorizer": {
            "analyzer": "char",
            "lowercase": False,
            "min_df": 3,
            "max_df": 0.995,
            "sublinear_tf": True,
            "norm": "l2",
            "dtype": "float32",
        },
        "candidates": [
            {
                "model_id": "fixture_lr_w128_k4",
                "family": "logistic",
                "view": "window_128",
                "k": 4,
                "C": 1.0,
            }
        ],
        "diagnostic_only": [
            {
                "model_id": "diagnostic_gc_length",
                "kind": "gc_length",
                "family": "logistic",
                "features": [
                    "te_gc_fraction",
                    "window_gc_fraction",
                    "log1p_te_length",
                ],
                "standardize": True,
                "C": 1.0,
            },
            {
                "model_id": "diagnostic_environment_only",
                "kind": "environment_only",
                "family": "logistic",
                "features": ["environment_id"],
                "encoding": "one_hot_ignore_unknown",
                "C": 1.0,
            },
        ],
    }
    path.write_text(json.dumps(config), encoding="utf-8")
    return path


def test_frozen_candidate_training_is_reproducible_and_root_safe(tmp_path: Path) -> None:
    cohort_path = tmp_path / "cohort.parquet"
    config_path = _mini_config(tmp_path / "candidates.json")
    _cohort().to_parquet(cohort_path, index=False)

    first = train_public_library(
        cohort_path,
        tmp_path / "first",
        config_path=config_path,
        minimum_candidates=1,
    )
    second = train_public_library(
        cohort_path,
        tmp_path / "second",
        config_path=config_path,
        minimum_candidates=1,
    )

    first_registry = pd.read_csv(first["registry"])
    second_registry = pd.read_csv(second["registry"])
    validate_model_registry(first_registry)
    assert first_registry["eligible_for_hrms_selection"].all()
    pd.testing.assert_frame_equal(first_registry, second_registry)
    first_predictions = pd.read_parquet(first["predictions_select"])
    second_predictions = pd.read_parquet(second["predictions_select"])
    pd.testing.assert_frame_equal(first_predictions, second_predictions)
    probability = first_predictions["prediction__fixture_lr_w128_k4"]
    assert probability.between(0, 1).all()
    assert len(pd.read_parquet(first["losses_select"])) == 16
    assert len(pd.read_parquet(first["losses_test"])) == 16
    assert "pair_id" in first_predictions
    assert "nested_cluster_id" in pd.read_parquet(first["losses_select"])

    first_diagnostic_registry = pd.read_csv(first["diagnostic_registry"])
    second_diagnostic_registry = pd.read_csv(second["diagnostic_registry"])
    pd.testing.assert_frame_equal(first_diagnostic_registry, second_diagnostic_registry)
    assert not first_diagnostic_registry["eligible_for_hrms_selection"].any()
    assert set(first_registry["model_id"]).isdisjoint(first_diagnostic_registry["model_id"])
    diagnostic_predictions = pd.read_parquet(first["diagnostic_predictions_select"])
    assert {
        "prediction__diagnostic_gc_length",
        "prediction__diagnostic_environment_only",
    }.issubset(diagnostic_predictions)
    assert not any(column.startswith("prediction__diagnostic") for column in first_predictions)
    assert len(pd.read_parquet(first["diagnostic_losses_select"])) == 32
    assert len(pd.read_parquet(first["diagnostic_losses_test"])) == 32
    manifest = json.loads(first["manifest"].read_text())
    assert manifest["training"]["cohort_split_hash_recomputed"] == split_assignment_hash(
        _cohort()
    )
    assert manifest["training"]["cohort_split_seed_recorded"] == 17
    assert manifest["training"]["diagnostic_only_models_excluded_from_hrms_registry"] == [
        "diagnostic_gc_length",
        "diagnostic_environment_only",
    ]
    assert "timings" not in manifest["artifacts"]
    assert not (tmp_path / "first/.hrms-candidate-training.lock").exists()
    with pytest.raises(TEBenchmarkError, match="output already exists"):
        train_public_library(
            cohort_path,
            tmp_path / "first",
            config_path=config_path,
            minimum_candidates=1,
        )

    internal = pd.read_csv(first["calibration_roots"])
    fit_roots = set(internal.loc[internal["internal_partition"].eq("fit"), "root_cluster_id"])
    calibration_roots = set(
        internal.loc[internal["internal_partition"].eq("calibration"), "root_cluster_id"]
    )
    assert fit_roots.isdisjoint(calibration_roots)
    assert fit_roots | calibration_roots == set(
        _cohort().loc[lambda frame: frame["split"].eq("train"), "root_cluster_id"]
    )


def test_candidate_training_rejects_root_leakage() -> None:
    cohort = _cohort()
    select_index = cohort.index[cohort["split"].eq("select")][0]
    training_root = cohort.loc[cohort["split"].eq("train"), "root_cluster_id"].iloc[0]
    cohort.loc[select_index, "root_cluster_id"] = training_root

    with pytest.raises(ValueError, match="root clusters cross"):
        validate_cohort(cohort)


def test_candidate_training_rejects_invalid_window() -> None:
    cohort = _cohort()
    cohort.loc[0, "window_sequence"] = "A" * 511
    with pytest.raises(TEBenchmarkError, match="512-bp"):
        validate_cohort(cohort)


def test_candidate_training_rejects_pair_members_in_different_roots() -> None:
    cohort = _cohort()
    first_pair = cohort["pair_id"].iloc[0]
    member_indices = cohort.index[cohort["pair_id"].eq(first_pair)]
    cohort.loc[member_indices[1], "root_cluster_id"] = "different_root_same_split"

    with pytest.raises(TEBenchmarkError, match="matched pair"):
        validate_cohort(cohort)


def test_candidate_training_rejects_tampered_split_hash(tmp_path: Path) -> None:
    cohort = _cohort()
    cohort["split_hash"] = "0" * 64
    cohort_path = tmp_path / "cohort.parquet"
    cohort.to_parquet(cohort_path, index=False)

    with pytest.raises(TEBenchmarkError, match="split_hash does not match"):
        train_public_library(
            cohort_path,
            tmp_path / "output",
            config_path=_mini_config(tmp_path / "candidates.json"),
            minimum_candidates=1,
        )


def test_candidate_training_rejects_tampered_sequence_diagnostics(tmp_path: Path) -> None:
    cohort = _cohort()
    cohort.loc[0, "window_gc_fraction"] = 0.0
    cohort_path = tmp_path / "cohort.parquet"
    cohort.to_parquet(cohort_path, index=False)

    with pytest.raises(TEBenchmarkError, match="GC fractions do not match"):
        train_public_library(
            cohort_path,
            tmp_path / "output",
            config_path=_mini_config(tmp_path / "candidates.json"),
            minimum_candidates=1,
        )


def test_frozen_config_rejects_duplicate_candidate_configurations(tmp_path: Path) -> None:
    config_path = _mini_config(tmp_path / "candidates.json")
    config = json.loads(config_path.read_text())
    duplicate = dict(config["candidates"][0])
    duplicate["model_id"] = "duplicate_name_same_configuration"
    config["candidates"].append(duplicate)
    config_path.write_text(json.dumps(config), encoding="utf-8")

    with pytest.raises(TEBenchmarkError, match="duplicate configurations"):
        load_frozen_config(config_path, minimum_candidates=1)
