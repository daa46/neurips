"""Validation of immutable wide prediction exports and Brier-loss conversion."""

from __future__ import annotations

import numpy as np
import pandas as pd

from .registry import assert_prediction_library_matches
from .schema import SchemaError, validate_loss_table

PREDICTION_PREFIX = "prediction__"


def predictions_to_brier_losses(
    predictions: pd.DataFrame,
    registry: pd.DataFrame,
    *,
    observation_col: str = "observation_id",
    environment_col: str = "environment_id",
    cluster_col: str = "root_cluster_id",
    nested_cluster_col: str = "nested_cluster_id",
    label_col: str = "label",
    weight_col: str = "sample_weight",
    prediction_prefix: str = PREDICTION_PREFIX,
) -> pd.DataFrame:
    """Convert a complete wide probability export to the HRMS long loss schema.

    Candidate identity is checked against a fully hashed frozen registry before
    any losses are calculated. The function never calibrates or modifies model
    probabilities beyond applying the Brier-loss formula.
    """

    required = {observation_col, environment_col, cluster_col, label_col}
    missing = required.difference(predictions.columns)
    if missing:
        raise SchemaError(f"prediction table is missing columns: {sorted(missing)}")
    if predictions.empty:
        raise SchemaError("prediction table is empty")
    if predictions.columns.duplicated().any():
        duplicates = predictions.columns[predictions.columns.duplicated()].tolist()
        raise SchemaError(f"prediction table contains duplicate columns: {duplicates}")
    if predictions[observation_col].duplicated().any():
        examples = predictions.loc[
            predictions[observation_col].duplicated(keep=False), observation_col
        ].head(5)
        raise SchemaError(f"observation identifiers are not unique: {examples.tolist()}")

    prediction_columns = [
        column
        for column in predictions.columns
        if isinstance(column, str) and column.startswith(prediction_prefix)
    ]
    if not prediction_columns:
        raise SchemaError(f"no columns begin with prediction prefix {prediction_prefix!r}")
    model_ids = [column.removeprefix(prediction_prefix) for column in prediction_columns]
    if any(not model_id for model_id in model_ids):
        raise SchemaError("prediction column has an empty model identifier")
    assert_prediction_library_matches(registry, model_ids)

    required_values = [observation_col, environment_col, cluster_col, label_col]
    if predictions[required_values].isna().any().any():
        raise SchemaError("prediction metadata and labels cannot contain null values")
    labels = predictions[label_col].to_numpy(dtype=float)
    if not np.isfinite(labels).all() or not np.isin(labels, [0.0, 1.0]).all():
        raise SchemaError("Brier-loss labels must be binary values in {0, 1}")
    probabilities = predictions[prediction_columns].to_numpy(dtype=float)
    if not np.isfinite(probabilities).all() or np.any((probabilities < 0) | (probabilities > 1)):
        raise SchemaError("candidate predictions must be finite probabilities in [0, 1]")

    metadata = [observation_col, environment_col, cluster_col]
    if nested_cluster_col in predictions.columns:
        metadata.append(nested_cluster_col)
    working = predictions[metadata].copy().reset_index(drop=True)
    if weight_col in predictions.columns:
        weights = predictions[weight_col].to_numpy(dtype=float)
        if not np.isfinite(weights).all() or np.any(weights <= 0):
            raise SchemaError("sample weights must be finite and strictly positive")
        working[weight_col] = weights
    else:
        working[weight_col] = 1.0

    repeated = working.loc[working.index.repeat(len(model_ids))].reset_index(drop=True)
    repeated["model_id"] = np.tile(np.asarray(model_ids, dtype=object), len(working))
    repeated["loss"] = np.square(probabilities - labels[:, None]).reshape(-1)
    validate_loss_table(repeated)
    return repeated
