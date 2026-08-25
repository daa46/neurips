"""Frozen candidate-library validation."""

from __future__ import annotations

import re

import pandas as pd

REQUIRED_REGISTRY_COLUMNS = (
    "model_id",
    "model_family",
    "representation",
    "pooling",
    "regularization",
    "seed",
    "calibration",
    "checkpoint_hash",
    "training_data_hash",
)

_HASH = re.compile(r"^[0-9a-f]{32,128}$")


def validate_model_registry(registry: pd.DataFrame, *, require_hashes: bool = True) -> None:
    """Validate that the prespecified candidate library is immutable and complete."""

    missing = set(REQUIRED_REGISTRY_COLUMNS).difference(registry.columns)
    if missing:
        raise ValueError(f"model registry is missing columns: {sorted(missing)}")
    if registry.empty:
        raise ValueError("model registry is empty")
    if registry[list(REQUIRED_REGISTRY_COLUMNS)].isna().any().any():
        raise ValueError("model registry contains null required values")
    if registry["model_id"].duplicated().any():
        duplicates = registry.loc[registry["model_id"].duplicated(), "model_id"].tolist()
        raise ValueError(f"model identifiers are not unique: {duplicates}")
    if require_hashes:
        for column in ("checkpoint_hash", "training_data_hash"):
            invalid = ~registry[column].astype(str).str.lower().map(
                lambda value: bool(_HASH.fullmatch(value))
            )
            if invalid.any():
                models = registry.loc[invalid, "model_id"].tolist()
                raise ValueError(f"{column} is not a hexadecimal content hash for {models}")


def assert_prediction_library_matches(
    registry: pd.DataFrame,
    prediction_models: list | tuple | set,
) -> None:
    """Require prediction columns/models to match the frozen registry exactly."""

    validate_model_registry(registry)
    registered = set(registry["model_id"])
    observed = set(prediction_models)
    if registered != observed:
        raise ValueError(
            "prediction candidate library differs from registry; "
            f"missing={sorted(registered - observed)}, extra={sorted(observed - registered)}"
        )
