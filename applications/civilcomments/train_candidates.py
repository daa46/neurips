#!/usr/bin/env python3
"""Train and freeze a lightweight TF--IDF CivilComments candidate library."""

# ruff: noqa: E402 -- direct-script support requires path setup before imports.

from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path

_PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, brier_score_loss, roc_auc_score
from sklearn.model_selection import GroupShuffleSplit

from applications.civilcomments.common import read_table, sha256_file, write_csv_gzip, write_json


@dataclass(frozen=True)
class Representation:
    name: str
    analyzer: str
    ngram_range: tuple[int, int]
    max_features: int


REPRESENTATIONS = (
    Representation("word_1", "word", (1, 1), 60_000),
    Representation("word_1_2", "word", (1, 2), 100_000),
    Representation("char_3_5", "char_wb", (3, 5), 100_000),
    Representation("char_4_6", "char_wb", (4, 6), 100_000),
)


def _update_field(digest: hashlib._Hash, value: object) -> None:  # type: ignore[name-defined]
    encoded = str(value).encode("utf-8")
    digest.update(len(encoded).to_bytes(8, "little"))
    digest.update(encoded)


def _training_hash(frame: pd.DataFrame, *, seed: int, calibration_fraction: float) -> str:
    digest = hashlib.sha256()
    _update_field(digest, seed)
    _update_field(digest, calibration_fraction)
    columns = ["observation_id", "environment_id", "label", "comment_text"]
    for row in frame.sort_values("observation_id", kind="stable")[columns].itertuples(
        index=False, name=None
    ):
        for value in row:
            _update_field(digest, value)
    return digest.hexdigest()


def _identifier_hash(values: np.ndarray) -> str:
    digest = hashlib.sha256()
    for value in sorted(values.tolist(), key=lambda item: (type(item).__name__, str(item))):
        _update_field(digest, value)
    return digest.hexdigest()


def _root_group_split(
    train: pd.DataFrame,
    *,
    calibration_fraction: float,
    seed: int,
    candidates: int = 128,
) -> tuple[np.ndarray, np.ndarray]:
    """Choose a disjoint article-root split with balanced environment-label cells."""

    groups = train["root_cluster_id"].to_numpy()
    cell = train["environment_id"].astype(str) + "::" + train["label"].astype(str)
    total_counts = cell.value_counts().sort_index()
    splitter = GroupShuffleSplit(
        n_splits=candidates,
        test_size=calibration_fraction,
        random_state=seed,
    )
    best: tuple[float, np.ndarray, np.ndarray] | None = None
    for fit_indices, calibration_indices in splitter.split(train, groups=groups):
        fit_cells = cell.iloc[fit_indices].value_counts().reindex(total_counts.index, fill_value=0)
        calibration_cells = (
            cell.iloc[calibration_indices]
            .value_counts()
            .reindex(total_counts.index, fill_value=0)
        )
        if (fit_cells == 0).any() or (calibration_cells == 0).any():
            continue
        fractions = calibration_cells.to_numpy(dtype=float) / total_counts.to_numpy(dtype=float)
        row_fraction = len(calibration_indices) / len(train)
        score = float(
            np.mean(np.square(fractions - calibration_fraction))
            + np.square(row_fraction - calibration_fraction)
        )
        if best is None or score < best[0]:
            best = (score, fit_indices, calibration_indices)
    if best is None:
        raise RuntimeError("could not construct a root-disjoint calibration split")
    fit_indices = np.sort(best[1])
    calibration_indices = np.sort(best[2])
    fit_roots = set(groups[fit_indices])
    calibration_roots = set(groups[calibration_indices])
    if fit_roots & calibration_roots:
        raise RuntimeError("internal fit/calibration article roots overlap")
    if fit_roots | calibration_roots != set(groups):
        raise RuntimeError("internal split does not cover every training article root")
    return fit_indices, calibration_indices


def _update_array(digest: hashlib._Hash, array: np.ndarray) -> None:  # type: ignore[name-defined]
    contiguous = np.ascontiguousarray(array)
    _update_field(digest, contiguous.dtype.str)
    _update_field(digest, contiguous.shape)
    digest.update(contiguous.view(np.uint8))


def _checkpoint_hash(
    vectorizer: TfidfVectorizer,
    classifier: LogisticRegression,
    calibrator: LogisticRegression | None,
    config: dict,
) -> str:
    """Fingerprint every fitted parameter needed to reproduce predictions."""

    digest = hashlib.sha256(json.dumps(config, sort_keys=True).encode("utf-8"))
    for token, index in sorted(vectorizer.vocabulary_.items()):
        _update_field(digest, token)
        _update_field(digest, index)
    _update_array(digest, vectorizer.idf_)
    for fitted in (classifier, calibrator):
        if fitted is None:
            _update_field(digest, "no-calibrator")
            continue
        _update_array(digest, fitted.classes_)
        _update_array(digest, fitted.coef_)
        _update_array(digest, fitted.intercept_)
    return digest.hexdigest()


def _fit_calibrator(scores: np.ndarray, labels: np.ndarray, seed: int) -> LogisticRegression:
    calibrator = LogisticRegression(
        C=1_000.0,
        solver="liblinear",
        max_iter=1_000,
        random_state=seed,
    )
    calibrator.fit(scores.reshape(-1, 1), labels)
    return calibrator


def _probabilities(
    classifier: LogisticRegression,
    calibrator: LogisticRegression | None,
    features,
) -> np.ndarray:
    if calibrator is None:
        return classifier.predict_proba(features)[:, 1]
    scores = classifier.decision_function(features)
    return calibrator.predict_proba(scores.reshape(-1, 1))[:, 1]


def _prediction_frame(
    metadata: pd.DataFrame,
    predictions: dict[str, np.ndarray],
    *,
    include_label: bool,
) -> pd.DataFrame:
    columns = [
        "observation_id",
        "environment_id",
        "root_cluster_id",
        "sample_weight",
    ]
    if include_label:
        columns.insert(3, "label")
    output = metadata[columns].reset_index(drop=True)
    for model_id, values in predictions.items():
        output[f"prediction__{model_id}"] = values
    return output


def _loss_frame(predictions: pd.DataFrame, model_ids: list[str]) -> pd.DataFrame:
    metadata = [
        "observation_id",
        "environment_id",
        "root_cluster_id",
        "label",
        "sample_weight",
    ]
    long = predictions.melt(
        id_vars=metadata,
        value_vars=[f"prediction__{model}" for model in model_ids],
        var_name="model_id",
        value_name="prediction",
    )
    long["model_id"] = long["model_id"].str.removeprefix("prediction__")
    long["loss"] = np.square(long["prediction"] - long["label"])
    return long[
        [
            "observation_id",
            "environment_id",
            "root_cluster_id",
            "model_id",
            "loss",
            "sample_weight",
            "label",
            "prediction",
        ]
    ].sort_values(["observation_id", "model_id"], kind="stable")


def train_library(
    train_validation_path: Path,
    test_features_path: Path,
    output_dir: Path,
    *,
    representations: tuple[str, ...] | None = None,
    c_values: tuple[float, ...] = (0.1, 1.0, 10.0),
    calibration_fraction: float = 0.15,
    calibrate: bool = True,
    seed: int = 1701,
) -> dict[str, Path]:
    """Fit the registered candidates using only the official training split."""

    cohort = read_table(train_validation_path)
    test = read_table(test_features_path)
    required = {
        "observation_id",
        "environment_id",
        "root_cluster_id",
        "split",
        "label",
        "sample_weight",
        "comment_text",
    }
    missing = required.difference(cohort.columns)
    if missing:
        raise ValueError(f"cohort is missing columns: {sorted(missing)}")
    if set(cohort["split"].unique()) != {"train", "val"}:
        raise ValueError("training/validation cohort must contain only official train and val")
    test_required = {
        "observation_id",
        "environment_id",
        "root_cluster_id",
        "split",
        "sample_weight",
        "comment_text",
    }
    test_missing = test_required.difference(test.columns)
    if test_missing:
        raise ValueError(f"test feature table is missing columns: {sorted(test_missing)}")
    if "label" in test.columns or "toxicity" in test.columns:
        raise ValueError("test labels must be sealed from candidate training")
    if set(test["split"].unique()) != {"test"}:
        raise ValueError("test feature table must contain only the official test split")
    if cohort["observation_id"].duplicated().any():
        raise ValueError("cohort observation identifiers are not unique")
    if test["observation_id"].duplicated().any():
        raise ValueError("test observation identifiers are not unique")
    split_roots = {
        split: set(frame["root_cluster_id"])
        for split, frame in pd.concat(
            [cohort, test], ignore_index=True
        ).groupby("split", observed=True)
    }
    for first, second in (("train", "val"), ("train", "test"), ("val", "test")):
        if split_roots[first] & split_roots[second]:
            raise ValueError(f"article root leakage between {first} and {second}")

    selected_representations = tuple(
        representation
        for representation in REPRESENTATIONS
        if representations is None or representation.name in representations
    )
    if not selected_representations:
        raise ValueError("no matching representations were requested")
    requested = set(representations or ())
    available = {representation.name for representation in selected_representations}
    if requested and requested != available:
        raise ValueError(f"unknown representations: {sorted(requested - available)}")
    if not c_values or any(value <= 0 for value in c_values):
        raise ValueError("C values must be positive")
    if not 0 < calibration_fraction < 0.5:
        raise ValueError("calibration_fraction must lie in (0, 0.5)")

    train = cohort.loc[cohort["split"].eq("train")].reset_index(drop=True)
    validation = cohort.loc[cohort["split"].eq("val")].reset_index(drop=True)
    test = test.reset_index(drop=True)
    fit_indices, calibration_indices = _root_group_split(
        train,
        calibration_fraction=calibration_fraction,
        seed=seed,
    )
    fit = train.iloc[np.sort(fit_indices)].reset_index(drop=True)
    calibration = train.iloc[np.sort(calibration_indices)].reset_index(drop=True)
    fit_roots = fit["root_cluster_id"].drop_duplicates().to_numpy()
    calibration_roots = calibration["root_cluster_id"].drop_duplicates().to_numpy()
    fit_root_hash = _identifier_hash(fit_roots)
    calibration_root_hash = _identifier_hash(calibration_roots)
    training_data_hash = _training_hash(
        train, seed=seed, calibration_fraction=calibration_fraction
    )

    predictions_val: dict[str, np.ndarray] = {}
    predictions_test: dict[str, np.ndarray] = {}
    registry_rows: list[dict] = []
    metric_rows: list[dict] = []
    timing_rows: list[dict] = []

    for representation in selected_representations:
        vectorizer_started = time.perf_counter()
        vectorizer = TfidfVectorizer(
            analyzer=representation.analyzer,
            ngram_range=representation.ngram_range,
            max_features=representation.max_features,
            min_df=3,
            max_df=0.995,
            sublinear_tf=True,
            strip_accents="unicode",
            lowercase=True,
            dtype=np.float32,
        )
        fit_features = vectorizer.fit_transform(fit["comment_text"])
        calibration_features = vectorizer.transform(calibration["comment_text"])
        validation_features = vectorizer.transform(validation["comment_text"])
        test_features = vectorizer.transform(test["comment_text"])
        vectorizer_seconds = time.perf_counter() - vectorizer_started

        for c_value in c_values:
            model_id = f"{representation.name}__c_{c_value:g}".replace(".", "p")
            started = time.perf_counter()
            classifier = LogisticRegression(
                C=c_value,
                solver="liblinear",
                max_iter=1_000,
                random_state=seed,
            )
            classifier.fit(fit_features, fit["label"].to_numpy())
            calibrator = None
            if calibrate:
                calibration_scores = classifier.decision_function(calibration_features)
                calibrator = _fit_calibrator(
                    calibration_scores,
                    calibration["label"].to_numpy(),
                    seed,
                )
            val_probability = _probabilities(classifier, calibrator, validation_features)
            test_probability = _probabilities(classifier, calibrator, test_features)
            elapsed = time.perf_counter() - started
            if (
                not np.isfinite(val_probability).all()
                or not np.isfinite(test_probability).all()
                or np.any((val_probability < 0) | (val_probability > 1))
                or np.any((test_probability < 0) | (test_probability > 1))
            ):
                raise RuntimeError(f"invalid probabilities for {model_id}")

            predictions_val[model_id] = val_probability
            predictions_test[model_id] = test_probability
            config = {
                "representation": asdict(representation),
                "C": c_value,
                "solver": "liblinear",
                "fit_intercept": True,
                "calibration": "heldout_platt" if calibrate else "none",
                "calibration_fraction": calibration_fraction,
                "fit_root_hash": fit_root_hash,
                "calibration_root_hash": calibration_root_hash,
                "seed": seed,
            }
            registry_rows.append(
                {
                    "model_id": model_id,
                    "model_family": "tfidf_logistic_regression",
                    "representation": representation.name,
                    "pooling": "not_applicable",
                    "regularization": f"l2_C={c_value:g}",
                    "seed": seed,
                    "calibration": "heldout_platt" if calibrate else "none",
                    "checkpoint_hash": _checkpoint_hash(
                        vectorizer, classifier, calibrator, config
                    ),
                    "training_data_hash": training_data_hash,
                    "configuration_json": json.dumps(config, sort_keys=True),
                }
            )
            labels = validation["label"].to_numpy()
            metric_rows.append(
                {
                    "split": "val",
                    "model_id": model_id,
                    "brier": brier_score_loss(labels, val_probability),
                    "auprc": average_precision_score(labels, val_probability),
                    "auroc": roc_auc_score(labels, val_probability),
                }
            )
            timing_rows.append(
                {
                    "model_id": model_id,
                    "representation_seconds_shared": vectorizer_seconds,
                    "fit_calibrate_predict_seconds": elapsed,
                    "vocabulary_size": len(vectorizer.vocabulary_),
                    "fit_rows": len(fit),
                    "calibration_rows": len(calibration),
                }
            )
            print(
                f"{model_id}: val Brier={metric_rows[-1]['brier']:.6f}; "
                f"fit/calibrate/predict={elapsed:.1f}s",
                flush=True,
            )

        del fit_features, calibration_features, validation_features, test_features

    model_ids = list(predictions_val)
    prediction_val = _prediction_frame(validation, predictions_val, include_label=True)
    prediction_test = _prediction_frame(test, predictions_test, include_label=False)
    loss_val = _loss_frame(prediction_val, model_ids)

    output_dir.mkdir(parents=True, exist_ok=True)
    outputs = {
        "predictions_val": output_dir / "predictions_val.csv.gz",
        "predictions_test": output_dir / "predictions_test.csv.gz",
        "losses_val": output_dir / "losses_val.csv.gz",
        "registry": output_dir / "model_registry.csv",
        "metrics": output_dir / "candidate_metrics.csv",
        "timings": output_dir / "training_timings.csv",
        "manifest": output_dir / "training_manifest.json",
    }
    write_csv_gzip(prediction_val, outputs["predictions_val"])
    write_csv_gzip(prediction_test, outputs["predictions_test"])
    write_csv_gzip(loss_val, outputs["losses_val"])

    prediction_hash = hashlib.sha256()
    for key in ("predictions_val", "predictions_test"):
        prediction_hash.update(bytes.fromhex(sha256_file(outputs[key])))
    registry = pd.DataFrame(registry_rows)
    registry["prediction_file_hash"] = prediction_hash.hexdigest()
    registry.to_csv(outputs["registry"], index=False, lineterminator="\n")
    pd.DataFrame(metric_rows).to_csv(outputs["metrics"], index=False, lineterminator="\n")
    pd.DataFrame(timing_rows).to_csv(outputs["timings"], index=False, lineterminator="\n")

    artifact_records = {}
    for name, path in outputs.items():
        if name == "manifest":
            continue
        artifact_records[name] = {
            "path": path.as_posix(),
            "size_bytes": path.stat().st_size,
            "sha256": sha256_file(path),
        }
    write_json(
        outputs["manifest"],
        {
            "schema_version": 1,
            "inputs": {
                "train_validation": {
                    "path": train_validation_path.as_posix(),
                    "size_bytes": train_validation_path.stat().st_size,
                    "sha256": sha256_file(train_validation_path),
                },
                "test_features_without_labels": {
                    "path": test_features_path.as_posix(),
                    "size_bytes": test_features_path.stat().st_size,
                    "sha256": sha256_file(test_features_path),
                },
            },
            "training": {
                "seed": seed,
                "official_training_rows": len(train),
                "fit_rows": len(fit),
                "calibration_rows": len(calibration),
                "fit_root_clusters": len(fit_roots),
                "calibration_root_clusters": len(calibration_roots),
                "fit_root_hash": fit_root_hash,
                "calibration_root_hash": calibration_root_hash,
                "fit_calibration_roots_disjoint": True,
                "calibration_fraction": calibration_fraction,
                "actual_calibration_row_fraction": len(calibration) / len(train),
                "calibration": "heldout_platt" if calibrate else "none",
                "candidate_count": len(model_ids),
                "test_labels_accessed": False,
                "representations": [asdict(value) for value in selected_representations],
                "c_values": list(c_values),
                "training_data_hash": training_data_hash,
                "checkpoint_hash_definition": (
                    "SHA256 over canonical configuration, fitted vocabulary/IDF, classifier "
                    "coefficients/intercepts/classes, and calibrator state"
                ),
            },
            "artifacts": artifact_records,
        },
    )
    return outputs


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--train-validation",
        type=Path,
        default=Path("data/civilcomments/derived_train_validation.csv.gz"),
    )
    parser.add_argument(
        "--test-features",
        type=Path,
        default=Path("data/civilcomments/derived_test_features.csv.gz"),
    )
    parser.add_argument("--output-dir", type=Path, default=Path("results/civilcomments"))
    parser.add_argument(
        "--representations",
        nargs="+",
        choices=[value.name for value in REPRESENTATIONS],
        help="subset for smoke tests; the default freezes all four representations",
    )
    parser.add_argument("--c-values", nargs="+", type=float, default=[0.1, 1.0, 10.0])
    parser.add_argument("--calibration-fraction", type=float, default=0.15)
    parser.add_argument("--no-calibration", action="store_true")
    parser.add_argument("--seed", type=int, default=1701)
    args = parser.parse_args(argv)
    outputs = train_library(
        args.train_validation,
        args.test_features,
        args.output_dir,
        representations=tuple(args.representations) if args.representations else None,
        c_values=tuple(args.c_values),
        calibration_fraction=args.calibration_fraction,
        calibrate=not args.no_calibration,
        seed=args.seed,
    )
    for key, output in outputs.items():
        print(f"{key}: {output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
