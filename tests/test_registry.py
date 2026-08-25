import pandas as pd
import pytest

from hrms.registry import assert_prediction_library_matches, validate_model_registry


def _registry():
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


def test_valid_registry_matches_predictions():
    registry = _registry()
    validate_model_registry(registry)
    assert_prediction_library_matches(registry, ["b", "a"])


def test_registry_rejects_pending_hashes():
    registry = _registry()
    registry.loc[0, "checkpoint_hash"] = "pending"
    with pytest.raises(ValueError, match="content hash"):
        validate_model_registry(registry)


def test_predictions_must_match_frozen_library():
    with pytest.raises(ValueError, match="differs from registry"):
        assert_prediction_library_matches(_registry(), ["a"])
