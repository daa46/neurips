"""Input schema validation and canonical array representation."""

from __future__ import annotations

from collections.abc import Hashable
from dataclasses import dataclass

import numpy as np
import pandas as pd


class SchemaError(ValueError):
    """Raised when a loss table violates the frozen-candidate contract."""


def _sorted(values: list[Hashable]) -> tuple[Hashable, ...]:
    """Sort identifiers deterministically without requiring a shared Python type."""

    return tuple(sorted(values, key=lambda value: (type(value).__name__, str(value))))


@dataclass(frozen=True)
class ColumnSpec:
    observation: str = "observation_id"
    environment: str = "environment_id"
    cluster: str = "root_cluster_id"
    model: str = "model_id"
    loss: str = "loss"
    weight: str | None = "sample_weight"


@dataclass(frozen=True)
class LossData:
    """Canonical paired loss matrix plus observation metadata."""

    observation_ids: tuple[Hashable, ...]
    environment_ids: tuple[Hashable, ...]
    cluster_ids: tuple[Hashable, ...]
    model_ids: tuple[Hashable, ...]
    observation_environment: np.ndarray
    observation_cluster: np.ndarray
    losses: np.ndarray
    weights: np.ndarray

    @property
    def n_observations(self) -> int:
        return self.losses.shape[0]

    @property
    def n_models(self) -> int:
        return self.losses.shape[1]


def validate_loss_table(table: pd.DataFrame, columns: ColumnSpec | None = None) -> LossData:
    """Validate a long loss table and return deterministic paired arrays.

    Each observation must occur exactly once for every model. Environment,
    cluster, and sample weight must be invariant across a given observation's
    model rows.
    """

    columns = columns or ColumnSpec()
    required = {
        columns.observation,
        columns.environment,
        columns.cluster,
        columns.model,
        columns.loss,
    }
    if columns.weight is not None and columns.weight in table.columns:
        required.add(columns.weight)
    missing = required.difference(table.columns)
    if missing:
        raise SchemaError(f"loss table is missing columns: {sorted(missing)}")
    if table.empty:
        raise SchemaError("loss table is empty")
    if table[list(required)].isna().any().any():
        bad = table[list(required)].columns[table[list(required)].isna().any()].tolist()
        raise SchemaError(f"null values are not allowed in required columns: {bad}")

    key = [columns.observation, columns.model]
    if table.duplicated(key).any():
        examples = table.loc[table.duplicated(key, keep=False), key].head(5).to_dict("records")
        raise SchemaError(f"duplicate observation/model rows: {examples}")

    model_ids = _sorted(table[columns.model].unique().tolist())
    observation_ids = _sorted(table[columns.observation].unique().tolist())
    if not model_ids:
        raise SchemaError("candidate library is empty")

    counts = table.groupby(columns.observation, sort=False)[columns.model].nunique()
    incomplete = counts[counts != len(model_ids)]
    if not incomplete.empty or len(table) != len(observation_ids) * len(model_ids):
        examples = incomplete.head(5).to_dict()
        raise SchemaError(
            "every observation must contain the complete frozen candidate library; "
            f"violations: {examples}"
        )

    invariant_columns = [columns.environment, columns.cluster]
    if columns.weight is not None and columns.weight in table.columns:
        invariant_columns.append(columns.weight)
    for name in invariant_columns:
        varying = table.groupby(columns.observation, sort=False)[name].nunique(dropna=False)
        if (varying != 1).any():
            examples = varying[varying != 1].head(5).index.tolist()
            raise SchemaError(f"{name!r} varies across model rows for observations {examples}")

    metadata_columns = [columns.observation, columns.environment, columns.cluster]
    if columns.weight is not None and columns.weight in table.columns:
        metadata_columns.append(columns.weight)
    metadata = (
        table[metadata_columns]
        .drop_duplicates(columns.observation)
        .set_index(columns.observation)
        .reindex(observation_ids)
    )

    wide = table.pivot(index=columns.observation, columns=columns.model, values=columns.loss)
    wide = wide.reindex(index=observation_ids, columns=model_ids)
    losses = wide.to_numpy(dtype=float)
    if not np.isfinite(losses).all():
        raise SchemaError("losses must be finite")

    if columns.weight is not None and columns.weight in metadata.columns:
        weights = metadata[columns.weight].to_numpy(dtype=float)
    else:
        weights = np.ones(len(observation_ids), dtype=float)
    if not np.isfinite(weights).all() or np.any(weights <= 0):
        raise SchemaError("sample weights must be finite and strictly positive")

    environment_ids = _sorted(metadata[columns.environment].unique().tolist())
    cluster_ids = _sorted(metadata[columns.cluster].unique().tolist())
    environment_index = {identifier: idx for idx, identifier in enumerate(environment_ids)}
    cluster_index = {identifier: idx for idx, identifier in enumerate(cluster_ids)}

    return LossData(
        observation_ids=observation_ids,
        environment_ids=environment_ids,
        cluster_ids=cluster_ids,
        model_ids=model_ids,
        observation_environment=np.fromiter(
            (environment_index[value] for value in metadata[columns.environment]),
            dtype=np.int64,
        ),
        observation_cluster=np.fromiter(
            (cluster_index[value] for value in metadata[columns.cluster]),
            dtype=np.int64,
        ),
        losses=losses,
        weights=weights,
    )
