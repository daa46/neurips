#!/usr/bin/env python3
"""Construct the disjoint eight-environment CivilComments task.

This is a derived task, not the official WILDS leaderboard evaluation.  It
keeps comments whose eight WILDS identity-category indicators contain exactly
one value at or above the prespecified threshold, while preserving the
official train/validation/test split column byte-for-byte.
"""

# ruff: noqa: E402 -- direct-script support requires path setup before imports.

from __future__ import annotations

import argparse
import sys
from pathlib import Path

_PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

import pandas as pd

from applications.civilcomments.common import (
    ENVIRONMENT_NAMES,
    IDENTITY_COLUMNS,
    SOURCE_REPOSITORY_COMMIT,
    SOURCE_SHA256,
    SOURCE_SIZE_BYTES,
    SOURCE_URL,
    SPLITS,
    sha256_file,
    write_csv_gzip,
    write_json,
)


def prepare(
    source: Path,
    output_dir: Path,
    *,
    identity_threshold: float = 0.5,
    toxicity_threshold: float = 0.5,
    verify_source: bool = True,
) -> tuple[Path, Path, Path, Path, Path]:
    """Validate the mirror and export the exact-one-identity cohort."""

    if not source.is_file():
        raise FileNotFoundError(source)
    source_hash = sha256_file(source)
    source_size = source.stat().st_size
    if verify_source and (source_hash != SOURCE_SHA256 or source_size != SOURCE_SIZE_BYTES):
        raise ValueError(
            "source bytes do not match the registered mirror: "
            f"size={source_size}, sha256={source_hash}"
        )

    usecols = [
        "id",
        "article_id",
        "comment_text",
        "split",
        "toxicity",
        *IDENTITY_COLUMNS,
    ]
    raw = pd.read_csv(source, usecols=usecols)
    if raw["id"].isna().any() or raw["id"].duplicated().any():
        raise ValueError("source comment identifiers must be complete and unique")
    observed_splits = set(raw["split"].unique())
    if observed_splits != set(SPLITS):
        raise ValueError(f"unexpected official split labels: {sorted(observed_splits)}")
    if raw["article_id"].isna().any():
        raise ValueError("source article identifiers must be complete")
    root_split_counts = raw.groupby("article_id", observed=True)["split"].nunique()
    if (root_split_counts != 1).any():
        raise ValueError("an article root crosses official train/validation/test splits")

    identity_values = raw.loc[:, IDENTITY_COLUMNS].fillna(0.0)
    identity_flags = identity_values.ge(identity_threshold)
    identity_count = identity_flags.sum(axis=1)
    keep = identity_count.eq(1)
    retained_flags = identity_flags.loc[keep]

    derived = raw.loc[
        keep, ["id", "article_id", "comment_text", "split", "toxicity"]
    ].copy()
    if derived["comment_text"].isna().any():
        raise ValueError("a retained exact-one-identity row has missing comment text")
    derived["environment_id"] = (
        retained_flags.idxmax(axis=1).map(ENVIRONMENT_NAMES).astype("string")
    )
    derived["label"] = derived["toxicity"].ge(toxicity_threshold).astype("int8")
    derived["observation_id"] = derived["id"].astype("int64")
    # Comments from one article share topic and discussion context.  Articles
    # are therefore the highest defensible dependence unit exposed here.
    derived["root_cluster_id"] = derived["article_id"].astype("int64")
    derived["sample_weight"] = 1.0
    derived = derived[
        [
            "observation_id",
            "environment_id",
            "root_cluster_id",
            "split",
            "label",
            "toxicity",
            "sample_weight",
            "comment_text",
        ]
    ].sort_values(["split", "observation_id"], kind="stable")

    if derived.empty or derived["environment_id"].nunique() != len(IDENTITY_COLUMNS):
        raise ValueError("derived cohort does not contain all eight environments")
    split_environment = derived.groupby(["split", "environment_id"], observed=True)
    if len(split_environment) != len(SPLITS) * len(IDENTITY_COLUMNS):
        raise ValueError("at least one environment is absent from an official split")
    if (split_environment["label"].nunique() != 2).any():
        raise ValueError("each split/environment cell must contain both toxicity labels")
    derived_root_splits = derived.groupby("root_cluster_id", observed=True)["split"].nunique()
    if (derived_root_splits != 1).any():
        raise ValueError("a retained article root crosses official splits")

    root_environment_counts = derived.groupby("root_cluster_id", observed=True)[
        "environment_id"
    ].nunique()
    split_root_summary: dict[str, dict[str, float | int]] = {}
    for split_name, split_frame in derived.groupby("split", observed=True):
        sizes = split_frame.groupby("root_cluster_id", observed=True).size()
        split_root_summary[str(split_name)] = {
            "root_clusters": int(len(sizes)),
            "median_observations_per_root": float(sizes.median()),
            "p95_observations_per_root": float(sizes.quantile(0.95)),
            "maximum_observations_per_root": int(sizes.max()),
        }

    output_dir.mkdir(parents=True, exist_ok=True)
    train_validation_path = output_dir / "derived_train_validation.csv.gz"
    test_features_path = output_dir / "derived_test_features.csv.gz"
    test_labels_path = output_dir / "derived_test_labels.csv.gz"
    counts_path = output_dir / "environment_counts.csv"
    metadata_path = output_dir / "source_metadata.json"
    train_validation = derived.loc[derived["split"].isin(["train", "val"])].copy()
    test = derived.loc[derived["split"].eq("test")].copy()
    write_csv_gzip(train_validation, train_validation_path)
    write_csv_gzip(
        test.drop(columns=["label", "toxicity"]),
        test_features_path,
    )
    write_csv_gzip(
        test[["observation_id", "label"]],
        test_labels_path,
    )

    counts = (
        derived.groupby(["split", "environment_id", "label"], observed=True)
        .size()
        .rename("observations")
        .reset_index()
        .sort_values(["split", "environment_id", "label"], kind="stable")
    )
    counts.to_csv(counts_path, index=False, lineterminator="\n")

    payload = {
        "schema_version": 1,
        "source": {
            "url": SOURCE_URL,
            "repository_commit": SOURCE_REPOSITORY_COMMIT,
            "license": "CC0-1.0",
            "size_bytes": source_size,
            "sha256": source_hash,
        },
        "construction": {
            "identity_columns": list(IDENTITY_COLUMNS),
            "environment_names": ENVIRONMENT_NAMES,
            "identity_threshold": identity_threshold,
            "toxicity_threshold": toxicity_threshold,
            "rule": "retain rows with exactly one of the eight indicators >= threshold",
            "root_cluster": "article_id",
            "root_cluster_rationale": (
                "comments within an article share topic and discussion context"
            ),
            "resampling": (
                "global article-cluster resampling because some articles span identity environments"
            ),
            "test_labels_sealed_from_training_and_selection": True,
            "official_splits_preserved": True,
        },
        "counts": {
            "source_rows": int(len(raw)),
            "source_rows_with_missing_text": int(raw["comment_text"].isna().sum()),
            "retained_rows": int(len(derived)),
            "zero_selected_identities": int(identity_count.eq(0).sum()),
            "multiple_selected_identities": int(identity_count.gt(1).sum()),
            "retained_root_clusters": int(derived["root_cluster_id"].nunique()),
            "root_clusters_spanning_environments": int(root_environment_counts.gt(1).sum()),
            "fraction_root_clusters_spanning_environments": float(
                root_environment_counts.gt(1).mean()
            ),
            "root_clusters_crossing_official_splits": 0,
            "root_cluster_summary_by_split": split_root_summary,
            "by_split": {
                str(key): int(value) for key, value in derived["split"].value_counts().items()
            },
        },
        "artifacts": {
            "train_validation_cohort": {
                "path": train_validation_path.as_posix(),
                "size_bytes": train_validation_path.stat().st_size,
                "sha256": sha256_file(train_validation_path),
            },
            "test_features": {
                "path": test_features_path.as_posix(),
                "size_bytes": test_features_path.stat().st_size,
                "sha256": sha256_file(test_features_path),
            },
            "sealed_test_labels": {
                "path": test_labels_path.as_posix(),
                "size_bytes": test_labels_path.stat().st_size,
                "sha256": sha256_file(test_labels_path),
            },
            "environment_counts": {
                "path": counts_path.as_posix(),
                "size_bytes": counts_path.stat().st_size,
                "sha256": sha256_file(counts_path),
            },
        },
    }
    write_json(metadata_path, payload)
    return (
        train_validation_path,
        test_features_path,
        test_labels_path,
        counts_path,
        metadata_path,
    )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--source",
        type=Path,
        default=Path("data/raw/civilcomments/all_data_with_identities.csv"),
    )
    parser.add_argument("--output-dir", type=Path, default=Path("data/civilcomments"))
    parser.add_argument("--identity-threshold", type=float, default=0.5)
    parser.add_argument("--toxicity-threshold", type=float, default=0.5)
    parser.add_argument(
        "--skip-source-verification",
        action="store_true",
        help="allow a different source file while still recording its content hash",
    )
    args = parser.parse_args(argv)
    outputs = prepare(
        args.source,
        args.output_dir,
        identity_threshold=args.identity_threshold,
        toxicity_threshold=args.toxicity_threshold,
        verify_source=not args.skip_source_verification,
    )
    for output in outputs:
        print(output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
