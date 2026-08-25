from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from hrms.predictions import predictions_to_brier_losses
from hrms.schema import SchemaError
from scripts.build_loss_table import main


def _registry() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "model_id": ["a", "b"],
            "model_family": ["linear", "linear"],
            "representation": ["x", "x"],
            "pooling": ["none", "none"],
            "regularization": ["1", "2"],
            "seed": [1, 2],
            "calibration": ["temperature", "temperature"],
            "checkpoint_hash": ["a" * 64, "b" * 64],
            "training_data_hash": ["c" * 64, "c" * 64],
        }
    )


def _predictions() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "observation_id": ["o1", "o2"],
            "environment_id": ["e1", "e2"],
            "root_cluster_id": ["c1", "c2"],
            "nested_cluster_id": ["n1", "n2"],
            "label": [1, 0],
            "sample_weight": [1.0, 2.0],
            "prediction__a": [0.8, 0.3],
            "prediction__b": [0.6, 0.1],
        }
    )


def test_wide_predictions_convert_to_paired_brier_loss_contract():
    losses = predictions_to_brier_losses(_predictions(), _registry())
    assert len(losses) == 4
    assert set(losses["model_id"]) == {"a", "b"}
    np.testing.assert_allclose(losses["loss"], [0.04, 0.16, 0.09, 0.01])
    assert losses.groupby("observation_id")["sample_weight"].first().to_dict() == {
        "o1": 1.0,
        "o2": 2.0,
    }


def test_prediction_export_rejects_library_mismatch_and_invalid_probabilities():
    missing = _predictions().drop(columns="prediction__b")
    with pytest.raises(ValueError, match="differs from registry"):
        predictions_to_brier_losses(missing, _registry())

    invalid = _predictions()
    invalid.loc[0, "prediction__a"] = 1.1
    with pytest.raises(SchemaError, match="probabilities"):
        predictions_to_brier_losses(invalid, _registry())


def test_build_loss_table_cli_writes_losses_and_manifest(tmp_path):
    predictions_path = tmp_path / "predictions.csv"
    registry_path = tmp_path / "registry.csv"
    output_path = tmp_path / "losses.csv"
    _predictions().to_csv(predictions_path, index=False)
    _registry().to_csv(registry_path, index=False)

    assert (
        main(
            [
                str(predictions_path),
                str(registry_path),
                "--output",
                str(output_path),
            ]
        )
        == 0
    )
    assert len(pd.read_csv(output_path)) == 4
    assert output_path.with_suffix(".csv.manifest.json").is_file()
