"""Root-cluster train/select/test governance."""

from __future__ import annotations

import hashlib
from collections.abc import Mapping

import numpy as np
import pandas as pd

from .schema import SchemaError

DEFAULT_PROPORTIONS = {"train": 0.6, "select": 0.2, "test": 0.2}


def split_assignment_hash(
    splits: pd.DataFrame,
    *,
    cluster_col: str = "root_cluster_id",
    split_col: str = "split",
) -> str:
    """Hash the canonical root-cluster-to-partition assignment.

    The hash ignores observation order and repeated observation rows. Identifier
    type names are included so, for example, integer ``1`` and string ``"1"``
    cannot silently collide.
    """

    required = {cluster_col, split_col}
    missing = required.difference(splits.columns)
    if missing:
        raise SchemaError(f"split table is missing columns: {sorted(missing)}")
    assert_no_cluster_leakage(splits, cluster_col=cluster_col, split_col=split_col)
    assignments = splits[[cluster_col, split_col]].drop_duplicates()
    canonical = sorted(
        (
            type(cluster).__name__,
            str(cluster),
            type(split).__name__,
            str(split),
        )
        for cluster, split in assignments.itertuples(index=False, name=None)
    )
    payload = "".join("\x1f".join(row) + "\n" for row in canonical).encode()
    return hashlib.sha256(payload).hexdigest()


def assign_root_cluster_splits(
    table: pd.DataFrame,
    *,
    cluster_col: str = "root_cluster_id",
    observation_col: str = "observation_id",
    proportions: Mapping[str, float] = DEFAULT_PROPORTIONS,
    random_state: int = 42,
) -> pd.DataFrame:
    """Assign entire root clusters to reproducible partitions.

    The assignment balances the *number of root clusters*, not loci. Sparse
    applications should freeze a domain-specific group-balanced assignment after
    checking support, then validate it with :func:`assert_no_cluster_leakage`.
    """

    required = {cluster_col, observation_col}
    missing = required.difference(table.columns)
    if missing:
        raise SchemaError(f"split table is missing columns: {sorted(missing)}")
    if table[list(required)].isna().any().any():
        raise SchemaError("cluster and observation identifiers cannot be null")
    proportions = dict(proportions)
    if not proportions or any(value <= 0 for value in proportions.values()):
        raise ValueError("split proportions must be strictly positive")
    total = sum(proportions.values())
    normalized = {name: value / total for name, value in proportions.items()}

    clusters = np.asarray(
        sorted(table[cluster_col].unique().tolist(), key=lambda value: str(value)),
        dtype=object,
    )
    if len(clusters) < len(normalized):
        raise ValueError("there are fewer root clusters than requested partitions")
    rng = np.random.default_rng(random_state)
    clusters = clusters[rng.permutation(len(clusters))]

    names = list(normalized)
    raw_counts = np.asarray([normalized[name] * len(clusters) for name in names])
    counts = np.floor(raw_counts).astype(int)
    remainder = len(clusters) - counts.sum()
    order = np.argsort(-(raw_counts - counts), kind="stable")
    counts[order[:remainder]] += 1
    if np.any(counts == 0):
        raise ValueError("a requested partition received no root clusters")

    assignment: dict[object, str] = {}
    cursor = 0
    for name, count in zip(names, counts, strict=True):
        for cluster in clusters[cursor : cursor + count]:
            assignment[cluster] = name
        cursor += count

    output = table[[observation_col, cluster_col]].drop_duplicates().copy()
    output["split"] = output[cluster_col].map(assignment)
    output["split_seed"] = random_state
    assert_no_cluster_leakage(output, cluster_col=cluster_col, split_col="split")
    output["split_hash"] = split_assignment_hash(
        output, cluster_col=cluster_col, split_col="split"
    )
    return output.sort_values(observation_col, key=lambda series: series.astype(str)).reset_index(
        drop=True
    )


def assert_no_cluster_leakage(
    splits: pd.DataFrame,
    *,
    cluster_col: str = "root_cluster_id",
    split_col: str = "split",
) -> None:
    """Raise when a root cluster occurs in more than one partition."""

    required = {cluster_col, split_col}
    missing = required.difference(splits.columns)
    if missing:
        raise SchemaError(f"split table is missing columns: {sorted(missing)}")
    if splits[[cluster_col, split_col]].isna().any().any():
        raise SchemaError("root-cluster and split identifiers cannot be null")
    crossing = splits.groupby(cluster_col, sort=False)[split_col].nunique(dropna=False)
    crossing = crossing[crossing != 1]
    if not crossing.empty:
        raise SchemaError(
            "root clusters cross partitions: "
            f"{crossing.head(10).index.tolist()}"
        )
