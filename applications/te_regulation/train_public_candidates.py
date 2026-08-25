#!/usr/bin/env python3
"""Train the frozen, CPU-only public-TE candidate configuration library."""

# ruff: noqa: E402 -- direct-script support requires path setup before imports.

from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import sys
import time
import warnings
from collections import defaultdict
from pathlib import Path

_PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

import numpy as np
import pandas as pd
import sklearn
from sklearn.exceptions import ConvergenceWarning
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression, SGDClassifier
from sklearn.naive_bayes import ComplementNB
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from applications.te_regulation.cluster_and_split import (
    DNA_ALPHABET,
    TEBenchmarkError,
    read_table,
    sha256_file,
    write_json,
)
from hrms.predictions import predictions_to_brier_losses
from hrms.registry import validate_model_registry
from hrms.splits import assert_no_cluster_leakage, split_assignment_hash

DEFAULT_CONFIG = _PROJECT_ROOT / "configs/te/candidates.public.json"
VIEWS = frozenset({"window_128", "window_256", "window_512", "te_copy"})
FAMILIES = frozenset({"logistic", "sgd_logistic", "complement_nb"})
DIAGNOSTIC_KINDS = frozenset({"gc_length", "environment_only"})
OUTPUT_FILENAMES = {
    "registry": "model_registry.csv",
    "diagnostic_registry": "diagnostic_model_registry.csv",
    "predictions_select": "predictions_select.parquet",
    "predictions_test": "predictions_test.parquet",
    "losses_select": "losses_select.parquet",
    "losses_test": "losses_test.parquet",
    "diagnostic_predictions_select": "diagnostic_predictions_select.parquet",
    "diagnostic_predictions_test": "diagnostic_predictions_test.parquet",
    "diagnostic_losses_select": "diagnostic_losses_select.parquet",
    "diagnostic_losses_test": "diagnostic_losses_test.parquet",
    "calibration_roots": "calibration_roots.csv",
    "timings": "training_timings.csv",
    "manifest": "training_manifest.json",
}


def _positive_finite_number(value: object) -> bool:
    return (
        isinstance(value, int | float)
        and not isinstance(value, bool)
        and np.isfinite(float(value))
        and float(value) > 0
    )


def _update_field(digest, value: object) -> None:
    encoded = str(value).encode()
    digest.update(len(encoded).to_bytes(8, "little"))
    digest.update(encoded)


def _update_array(digest, array: np.ndarray) -> None:
    contiguous = np.ascontiguousarray(array)
    _update_field(digest, contiguous.dtype.str)
    _update_field(digest, contiguous.shape)
    digest.update(contiguous.view(np.uint8))


def _content_hash(frame: pd.DataFrame, columns: list[str], *, extra: object) -> str:
    digest = hashlib.sha256()
    _update_field(digest, extra)
    for row in frame.sort_values("observation_id", kind="stable")[columns].itertuples(
        index=False, name=None
    ):
        for value in row:
            _update_field(digest, value)
    return digest.hexdigest()


def load_frozen_config(path: Path, *, minimum_candidates: int = 18) -> dict:
    config = json.loads(path.read_text(encoding="utf-8"))
    if config.get("schema_version") != 1 or config.get("frozen_before_selection") is not True:
        raise TEBenchmarkError("candidate config must be schema v1 and explicitly frozen")
    candidates = config.get("candidates")
    if not 1 <= minimum_candidates <= 30:
        raise TEBenchmarkError("minimum_candidates must lie in [1, 30]")
    if not isinstance(candidates, list) or not minimum_candidates <= len(candidates) <= 30:
        raise TEBenchmarkError(
            f"frozen candidate library must contain {minimum_candidates}--30 configurations"
        )
    identifiers = [candidate.get("model_id") for candidate in candidates]
    if any(not isinstance(value, str) or not value for value in identifiers):
        raise TEBenchmarkError("every candidate needs a nonempty string model_id")
    if len(set(identifiers)) != len(identifiers):
        raise TEBenchmarkError("candidate model identifiers are not unique")
    signatures: set[str] = set()
    for candidate in candidates:
        family, view, k = candidate.get("family"), candidate.get("view"), candidate.get("k")
        if family not in FAMILIES or view not in VIEWS:
            raise TEBenchmarkError(f"unsupported frozen candidate: {candidate}")
        if isinstance(k, bool) or not isinstance(k, int) or not 2 <= k <= 8:
            raise TEBenchmarkError(f"invalid k-mer size: {candidate}")
        parameter = candidate.get("C") if family == "logistic" else candidate.get("alpha")
        if not _positive_finite_number(parameter):
            raise TEBenchmarkError(f"invalid regularization: {candidate}")
        signature = json.dumps(
            {key: value for key, value in candidate.items() if key != "model_id"},
            sort_keys=True,
            separators=(",", ":"),
        )
        if signature in signatures:
            raise TEBenchmarkError("candidate library contains duplicate configurations")
        signatures.add(signature)
    calibration = config.get("calibration", {})
    fraction = calibration.get("fraction_of_training_roots")
    if calibration.get("method") != "heldout_platt" or not _positive_finite_number(fraction):
        raise TEBenchmarkError("only heldout_platt calibration is supported")
    if not 0 < fraction < 0.5:
        raise TEBenchmarkError("calibration root fraction must lie in (0, 0.5)")
    if (
        calibration.get("solver") != "liblinear"
        or not _positive_finite_number(calibration.get("C"))
        or isinstance(calibration.get("max_iter"), bool)
        or not isinstance(calibration.get("max_iter"), int)
        or calibration["max_iter"] < 1
        or isinstance(calibration.get("split_trials"), bool)
        or not isinstance(calibration.get("split_trials"), int)
        or calibration["split_trials"] < 1
    ):
        raise TEBenchmarkError("heldout_platt implementation settings are invalid")
    diagnostics = config.get("diagnostic_only")
    if not isinstance(diagnostics, list) or len(diagnostics) != len(DIAGNOSTIC_KINDS):
        raise TEBenchmarkError("both frozen diagnostic-only controls must be configured")
    diagnostic_ids = [diagnostic.get("model_id") for diagnostic in diagnostics]
    diagnostic_kinds = [diagnostic.get("kind") for diagnostic in diagnostics]
    if (
        any(not isinstance(value, str) or not value for value in diagnostic_ids)
        or len(set(diagnostic_ids)) != len(diagnostic_ids)
        or set(diagnostic_ids) & set(identifiers)
        or set(diagnostic_kinds) != DIAGNOSTIC_KINDS
    ):
        raise TEBenchmarkError(
            "diagnostic controls must be unique and outside the candidate library"
        )
    expected = {
        "gc_length": {
            "family": "logistic",
            "features": ["te_gc_fraction", "window_gc_fraction", "log1p_te_length"],
            "standardize": True,
        },
        "environment_only": {
            "family": "logistic",
            "features": ["environment_id"],
            "encoding": "one_hot_ignore_unknown",
        },
    }
    for diagnostic in diagnostics:
        required_settings = expected[diagnostic["kind"]]
        if any(diagnostic.get(key) != value for key, value in required_settings.items()):
            raise TEBenchmarkError(f"invalid frozen diagnostic configuration: {diagnostic}")
        if not _positive_finite_number(diagnostic.get("C")):
            raise TEBenchmarkError(f"invalid diagnostic regularization: {diagnostic}")
    seed = config.get("seed")
    if isinstance(seed, bool) or not isinstance(seed, int):
        raise TEBenchmarkError("frozen candidate seed must be an integer")
    return config


def _view_sequences(frame: pd.DataFrame, view: str) -> pd.Series:
    if view == "te_copy":
        return frame["te_sequence"]
    width = int(view.removeprefix("window_"))
    start = (512 - width) // 2
    return frame["window_sequence"].str.slice(start, start + width)


def validate_cohort(cohort: pd.DataFrame) -> pd.DataFrame:
    required = {
        "observation_id",
        "environment_id",
        "root_cluster_id",
        "pair_id",
        "split",
        "label",
        "sample_weight",
        "te_sequence",
        "window_sequence",
    }
    missing = required.difference(cohort.columns)
    if missing:
        raise TEBenchmarkError(f"clustered cohort is missing columns: {sorted(missing)}")
    if cohort.empty or cohort[list(required)].isna().any().any():
        raise TEBenchmarkError("clustered cohort required values must be nonempty and non-null")
    cohort = cohort.copy()
    for column in ("observation_id", "environment_id", "root_cluster_id", "pair_id"):
        cohort[column] = cohort[column].astype(str)
    if cohort["observation_id"].duplicated().any():
        raise TEBenchmarkError("cohort observation IDs are not unique")
    if set(cohort["split"].unique()) != {"train", "select", "test"}:
        raise TEBenchmarkError("cohort must contain exactly train/select/test")
    if not cohort["label"].isin([0, 1]).all():
        raise TEBenchmarkError("cohort labels must be binary")
    weights = pd.to_numeric(cohort["sample_weight"], errors="coerce")
    if weights.isna().any() or not np.isfinite(weights).all() or (weights <= 0).any():
        raise TEBenchmarkError("sample weights must be finite and positive")
    cohort["sample_weight"] = weights.astype(float)
    for window in cohort["window_sequence"]:
        if len(window) != 512 or set(window).difference(DNA_ALPHABET):
            raise TEBenchmarkError("all model-input windows must be valid 512-bp DNA")
    for sequence in cohort["te_sequence"]:
        if not sequence or set(sequence).difference(DNA_ALPHABET):
            raise TEBenchmarkError("all oriented TE-copy sequences must be valid DNA")
    assert_no_cluster_leakage(cohort)
    pair_summary = cohort.groupby("pair_id", sort=False).agg(
        observations=("observation_id", "size"),
        labels=("label", "nunique"),
        roots=("root_cluster_id", "nunique"),
        splits=("split", "nunique"),
    )
    invalid_pairs = pair_summary.loc[
        (pair_summary["observations"] != 2)
        | (pair_summary["labels"] != 2)
        | (pair_summary["roots"] != 1)
        | (pair_summary["splits"] != 1)
    ]
    if not invalid_pairs.empty:
        raise TEBenchmarkError(
            "every matched pair must contain two opposite-label loci in one root and split; "
            f"violations={invalid_pairs.head(10).index.tolist()}"
        )
    return cohort.sort_values("observation_id", kind="stable").reset_index(drop=True)


def _validate_split_provenance(cohort: pd.DataFrame) -> tuple[str, int | None]:
    """Recompute the root assignment hash and verify optional upstream metadata."""

    computed = split_assignment_hash(cohort)
    if "split_hash" in cohort:
        recorded = cohort["split_hash"]
        if recorded.isna().any() or recorded.astype(str).nunique() != 1:
            raise TEBenchmarkError("cohort split_hash must be one non-null value")
        if str(recorded.iloc[0]) != computed:
            raise TEBenchmarkError("cohort split_hash does not match root assignments")
    split_seed: int | None = None
    if "split_seed" in cohort:
        recorded_seed = pd.to_numeric(cohort["split_seed"], errors="coerce")
        if (
            recorded_seed.isna().any()
            or not np.isfinite(recorded_seed).all()
            or recorded_seed.nunique() != 1
            or float(recorded_seed.iloc[0]) != int(recorded_seed.iloc[0])
        ):
            raise TEBenchmarkError("cohort split_seed must be one finite integer")
        split_seed = int(recorded_seed.iloc[0])
    return computed, split_seed


def _validate_diagnostic_features(cohort: pd.DataFrame) -> pd.DataFrame:
    required = ("te_gc_fraction", "window_gc_fraction", "te_length")
    missing = set(required).difference(cohort.columns)
    if missing:
        raise TEBenchmarkError(f"diagnostic controls require columns: {sorted(missing)}")
    cohort = cohort.copy()
    for column in required:
        values = pd.to_numeric(cohort[column], errors="coerce")
        if values.isna().any() or not np.isfinite(values).all():
            raise TEBenchmarkError(f"diagnostic feature {column!r} must be finite")
        cohort[column] = values.astype(float)
    if not cohort["te_gc_fraction"].between(0, 1).all() or not cohort[
        "window_gc_fraction"
    ].between(0, 1).all():
        raise TEBenchmarkError("GC fractions must lie in [0, 1]")
    if (cohort["te_length"] <= 0).any():
        raise TEBenchmarkError("TE lengths must be strictly positive")
    observed_lengths = cohort["te_sequence"].str.len().to_numpy(dtype=np.int64)
    if not np.array_equal(cohort["te_length"].to_numpy(dtype=float), observed_lengths):
        raise TEBenchmarkError("reported TE lengths do not match TE-copy sequences")

    def sequence_gc(sequence: str) -> float:
        canonical = sum(sequence.count(base) for base in "ACGT")
        if canonical == 0:
            raise TEBenchmarkError("a diagnostic sequence contains no canonical DNA bases")
        return (sequence.count("G") + sequence.count("C")) / canonical

    recomputed_te_gc = np.fromiter(
        (sequence_gc(sequence) for sequence in cohort["te_sequence"]), dtype=float
    )
    recomputed_window_gc = np.fromiter(
        (sequence_gc(sequence) for sequence in cohort["window_sequence"]), dtype=float
    )
    if not np.allclose(
        cohort["te_gc_fraction"].to_numpy(dtype=float),
        recomputed_te_gc,
        atol=5e-12,
        rtol=0,
    ) or not np.allclose(
        cohort["window_gc_fraction"].to_numpy(dtype=float),
        recomputed_window_gc,
        atol=5e-12,
        rtol=0,
    ):
        raise TEBenchmarkError("reported GC fractions do not match oriented sequences")
    return cohort


def _output_paths(output_dir: Path) -> dict[str, Path]:
    return {name: output_dir / filename for name, filename in OUTPUT_FILENAMES.items()}


def _acquire_training_lock(output_dir: Path, outputs: dict[str, Path]) -> Path:
    """Prevent concurrent writers and accidental replacement of immutable exports."""

    output_dir.mkdir(parents=True, exist_ok=True)
    existing = sorted(path.name for path in outputs.values() if path.exists())
    if existing:
        raise TEBenchmarkError(
            "candidate-training output already exists; use a new directory: "
            + ", ".join(existing)
        )
    lock = output_dir / ".hrms-candidate-training.lock"
    try:
        descriptor = os.open(lock, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o644)
    except FileExistsError as error:
        raise TEBenchmarkError(
            f"candidate-training directory is already locked: {lock}"
        ) from error
    with os.fdopen(descriptor, "w") as handle:
        handle.write("exclusive candidate-training writer; inspect before removing\n")
        handle.flush()
        os.fsync(handle.fileno())
    return lock


def _training_calibration_split(
    train: pd.DataFrame,
    *,
    fraction: float,
    seed: int,
    trials: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Find a deterministic root-disjoint split preserving environment/label cells."""

    roots = sorted(train["root_cluster_id"].unique())
    environments = sorted(train["environment_id"].unique())
    root_index = {value: index for index, value in enumerate(roots)}
    environment_index = {value: index for index, value in enumerate(environments)}
    cells = np.zeros((len(roots), 2 * len(environments)), dtype=np.int64)
    for root, environment, label in train[
        ["root_cluster_id", "environment_id", "label"]
    ].itertuples(index=False, name=None):
        cells[root_index[root], 2 * environment_index[environment] + int(label)] += 1
    calibration_count = max(1, int(round(fraction * len(roots))))
    if calibration_count >= len(roots):
        raise TEBenchmarkError("not enough training roots for held-out calibration")
    rng = np.random.default_rng(seed)
    total = cells.sum(axis=0)
    best: tuple[float, tuple[int, ...], np.ndarray] | None = None
    for _ in range(trials):
        calibration_roots = np.sort(rng.choice(len(roots), calibration_count, replace=False))
        mask = np.zeros(len(roots), dtype=bool)
        mask[calibration_roots] = True
        calibration_cells, fit_cells = cells[mask].sum(axis=0), cells[~mask].sum(axis=0)
        if np.any(calibration_cells == 0) or np.any(fit_cells == 0):
            continue
        score = float(
            np.mean(np.square((calibration_cells / total) - fraction))
            + np.mean(np.square((fit_cells / total) - (1 - fraction)))
        )
        if best is None or score < best[0]:
            best = (score, tuple(int(value) for value in mask), mask.copy())
        elif score == best[0]:
            canonical = tuple(int(value) for value in mask)
            if canonical < best[1]:
                best = (score, canonical, mask.copy())
    if best is None:
        raise TEBenchmarkError("could not form a root-disjoint calibration split with all cells")
    fit_roots = {roots[index] for index in np.flatnonzero(~best[2])}
    calibration_roots = {roots[index] for index in np.flatnonzero(best[2])}
    fit_indices = np.flatnonzero(train["root_cluster_id"].isin(fit_roots).to_numpy())
    calibration_indices = np.flatnonzero(
        train["root_cluster_id"].isin(calibration_roots).to_numpy()
    )
    return fit_indices, calibration_indices


def _identifier_hash(values: set[str] | np.ndarray | pd.Series) -> str:
    digest = hashlib.sha256()
    for value in sorted(str(item) for item in values):
        _update_field(digest, value)
    return digest.hexdigest()


def _make_classifier(candidate: dict, *, seed: int):
    family = candidate["family"]
    if family == "logistic":
        return LogisticRegression(
            C=float(candidate["C"]),
            solver="liblinear",
            max_iter=2_000,
            random_state=seed,
        )
    if family == "sgd_logistic":
        return SGDClassifier(
            loss="log_loss",
            penalty="l2",
            alpha=float(candidate["alpha"]),
            max_iter=2_000,
            tol=1e-4,
            average=True,
            random_state=seed,
        )
    if family == "complement_nb":
        return ComplementNB(alpha=float(candidate["alpha"]))
    raise TEBenchmarkError(f"unsupported candidate family: {family}")


def _fit_with_convergence_guard(
    estimator,
    features,
    labels: np.ndarray,
    sample_weight: np.ndarray,
    *,
    context: str,
):
    with warnings.catch_warnings():
        warnings.simplefilter("error", ConvergenceWarning)
        try:
            estimator.fit(features, labels, sample_weight=sample_weight)
        except ConvergenceWarning as error:
            raise TEBenchmarkError(f"{context} did not converge") from error
    return estimator


def _decision_scores(classifier, features) -> np.ndarray:
    classes = np.asarray(classifier.classes_)
    if set(classes.tolist()) != {0, 1}:
        raise TEBenchmarkError(f"candidate fitted unexpected classes: {classes.tolist()}")
    if hasattr(classifier, "decision_function"):
        scores = np.asarray(classifier.decision_function(features), dtype=float)
        if scores.ndim == 2:
            positive = int(np.flatnonzero(classes == 1)[0])
            negative = int(np.flatnonzero(classes == 0)[0])
            scores = scores[:, positive] - scores[:, negative]
    else:
        log_probability = np.asarray(classifier.predict_log_proba(features), dtype=float)
        positive = int(np.flatnonzero(classes == 1)[0])
        negative = int(np.flatnonzero(classes == 0)[0])
        scores = log_probability[:, positive] - log_probability[:, negative]
    scores = np.asarray(scores, dtype=float).reshape(-1)
    if not np.isfinite(scores).all():
        raise RuntimeError("candidate produced non-finite calibration scores")
    return scores


def _fit_platt_calibrator(
    scores: np.ndarray,
    labels: np.ndarray,
    sample_weight: np.ndarray,
    *,
    seed: int,
    calibration_config: dict,
) -> LogisticRegression:
    calibrator = LogisticRegression(
        C=float(calibration_config["C"]),
        solver=calibration_config["solver"],
        max_iter=int(calibration_config["max_iter"]),
        random_state=seed,
    )
    return _fit_with_convergence_guard(
        calibrator,
        scores.reshape(-1, 1),
        labels,
        sample_weight,
        context="Platt calibrator",
    )


def _calibrated_probabilities(classifier, calibrator, features) -> np.ndarray:
    scores = _decision_scores(classifier, features)
    probabilities = calibrator.predict_proba(scores.reshape(-1, 1))[:, 1]
    probabilities = np.asarray(probabilities, dtype=float)
    if not np.isfinite(probabilities).all() or np.any(
        (probabilities < 0) | (probabilities > 1)
    ):
        raise RuntimeError("candidate produced invalid calibrated probabilities")
    return probabilities


def _update_fitted_state(digest, fitted) -> None:
    for attribute in (
        "classes_",
        "coef_",
        "intercept_",
        "n_iter_",
        "t_",
        "class_count_",
        "class_log_prior_",
        "feature_count_",
        "feature_log_prob_",
    ):
        if hasattr(fitted, attribute):
            _update_field(digest, attribute)
            _update_array(digest, np.asarray(getattr(fitted, attribute)))


def _update_preprocessor_state(digest, preprocessor) -> None:
    if isinstance(preprocessor, TfidfVectorizer):
        _update_field(digest, "TfidfVectorizer")
        for token, index in sorted(preprocessor.vocabulary_.items()):
            _update_field(digest, token)
            _update_field(digest, index)
        _update_array(digest, preprocessor.idf_)
        return
    if isinstance(preprocessor, StandardScaler):
        _update_field(digest, "StandardScaler")
        for attribute in ("mean_", "scale_", "var_", "n_samples_seen_"):
            _update_field(digest, attribute)
            _update_array(digest, np.asarray(getattr(preprocessor, attribute)))
        return
    if isinstance(preprocessor, OneHotEncoder):
        _update_field(digest, "OneHotEncoder")
        for categories in preprocessor.categories_:
            for category in categories:
                _update_field(digest, category)
        return
    raise TypeError(f"unsupported fitted preprocessor: {type(preprocessor).__name__}")


def _checkpoint_hash(
    preprocessor,
    classifier,
    calibrator: LogisticRegression,
    *,
    configuration: dict,
    fit_root_hash: str,
    calibration_root_hash: str,
) -> str:
    digest = hashlib.sha256()
    _update_field(
        digest,
        json.dumps(
            {
                "configuration": configuration,
                "fit_root_hash": fit_root_hash,
                "calibration_root_hash": calibration_root_hash,
            },
            sort_keys=True,
            separators=(",", ":"),
        ),
    )
    _update_preprocessor_state(digest, preprocessor)
    _update_fitted_state(digest, classifier)
    _update_fitted_state(digest, calibrator)
    return digest.hexdigest()


def _prediction_frame(
    metadata: pd.DataFrame,
    predictions: dict[str, np.ndarray],
) -> pd.DataFrame:
    columns = ["observation_id", "environment_id", "root_cluster_id"]
    if "pair_id" in metadata:
        columns.append("pair_id")
    if "nested_cluster_id" in metadata:
        columns.append("nested_cluster_id")
    columns.extend(["label", "sample_weight"])
    output = metadata[columns].reset_index(drop=True).copy()
    if "nested_cluster_id" not in output and "pair_id" in output:
        output["nested_cluster_id"] = output["pair_id"]
    for model_id, values in predictions.items():
        if len(values) != len(output):
            raise RuntimeError(f"prediction length mismatch for {model_id}")
        output[f"prediction__{model_id}"] = values
    return output


def _regularization_label(candidate: dict) -> str:
    if candidate["family"] == "logistic":
        return f"l2_C={float(candidate['C']):g}"
    if candidate["family"] == "sgd_logistic":
        return f"l2_alpha={float(candidate['alpha']):g}"
    return f"complement_alpha={float(candidate['alpha']):g}"


def _gc_length_features(frame: pd.DataFrame) -> np.ndarray:
    return np.column_stack(
        [
            frame["te_gc_fraction"].to_numpy(dtype=float),
            frame["window_gc_fraction"].to_numpy(dtype=float),
            np.log1p(frame["te_length"].to_numpy(dtype=float)),
        ]
    )


def _diagnostic_feature_matrices(
    diagnostic: dict,
    fit: pd.DataFrame,
    calibration: pd.DataFrame,
    selection: pd.DataFrame,
    test: pd.DataFrame,
):
    """Fit a diagnostic preprocessor on fit roots only and transform all partitions."""

    kind = diagnostic["kind"]
    if kind == "gc_length":
        preprocessor = StandardScaler()
        fit_raw = _gc_length_features(fit)
        preprocessor.fit(
            fit_raw,
            sample_weight=fit["sample_weight"].to_numpy(dtype=float),
        )
        frames = (fit, calibration, selection, test)
        matrices = tuple(preprocessor.transform(_gc_length_features(frame)) for frame in frames)
    elif kind == "environment_only":
        preprocessor = OneHotEncoder(
            handle_unknown="ignore",
            sparse_output=True,
            dtype=np.float64,
        )
        fit_raw = fit[["environment_id"]]
        preprocessor.fit(fit_raw)
        matrices = tuple(
            preprocessor.transform(frame[["environment_id"]])
            for frame in (fit, calibration, selection, test)
        )
    else:  # guarded by frozen-config validation
        raise TEBenchmarkError(f"unsupported diagnostic control: {kind}")
    return (preprocessor, *matrices)


def train_public_library(
    cohort_path: Path,
    output_dir: Path,
    *,
    config_path: Path = DEFAULT_CONFIG,
    minimum_candidates: int = 18,
) -> dict[str, Path]:
    """Fit all frozen candidates without using selection/test data for training.

    The only data-dependent calibration operation uses a root-disjoint subset of
    the training partition. Selection and test labels are carried into immutable
    prediction/loss exports, but are never supplied to a vectorizer, classifier,
    calibrator, hyperparameter choice, or candidate filter.
    """

    config = load_frozen_config(config_path, minimum_candidates=minimum_candidates)
    cohort = _validate_diagnostic_features(validate_cohort(read_table(cohort_path)))
    cohort_split_hash, cohort_split_seed = _validate_split_provenance(cohort)
    seed = config.get("seed")
    outputs = _output_paths(output_dir)
    lock = _acquire_training_lock(output_dir, outputs)
    train = cohort.loc[cohort["split"].eq("train")].reset_index(drop=True)
    selection = cohort.loc[cohort["split"].eq("select")].reset_index(drop=True)
    test = cohort.loc[cohort["split"].eq("test")].reset_index(drop=True)
    calibration_config = config["calibration"]
    fit_indices, calibration_indices = _training_calibration_split(
        train,
        fraction=float(calibration_config["fraction_of_training_roots"]),
        seed=seed,
        trials=int(calibration_config.get("split_trials", 2_048)),
    )
    fit = train.iloc[np.sort(fit_indices)].reset_index(drop=True)
    calibration = train.iloc[np.sort(calibration_indices)].reset_index(drop=True)
    fit_roots = set(fit["root_cluster_id"])
    calibration_roots = set(calibration["root_cluster_id"])
    if fit_roots & calibration_roots or fit_roots | calibration_roots != set(
        train["root_cluster_id"]
    ):
        raise RuntimeError("internal fit/calibration root partition is invalid")
    fit_root_hash = _identifier_hash(fit_roots)
    calibration_root_hash = _identifier_hash(calibration_roots)
    training_columns = [
        "observation_id",
        "environment_id",
        "root_cluster_id",
        "pair_id",
        "label",
        "sample_weight",
        "te_sequence",
        "window_sequence",
        "te_gc_fraction",
        "window_gc_fraction",
        "te_length",
    ]
    training_data_hash = _content_hash(
        train,
        training_columns,
        extra={"partition": "train", "split_assignment_hash": cohort_split_hash},
    )

    grouped: dict[tuple[str, int], list[dict]] = defaultdict(list)
    for candidate in config["candidates"]:
        grouped[(candidate["view"], int(candidate["k"]))].append(candidate)
    vectorizer_config = config.get("vectorizer", {})
    required_vectorizer = {
        "analyzer": "char",
        "lowercase": False,
        "min_df": 3,
        "max_df": 0.995,
        "sublinear_tf": True,
        "norm": "l2",
        "dtype": "float32",
    }
    if vectorizer_config != required_vectorizer:
        raise TEBenchmarkError(
            "vectorizer configuration differs from the frozen public-TE specification"
        )

    predictions_select: dict[str, np.ndarray] = {}
    predictions_test: dict[str, np.ndarray] = {}
    diagnostic_predictions_select: dict[str, np.ndarray] = {}
    diagnostic_predictions_test: dict[str, np.ndarray] = {}
    registry_rows: list[dict] = []
    diagnostic_registry_rows: list[dict] = []
    timing_rows: list[dict] = []
    for (view, k), candidates in grouped.items():
        vectorizer_started = time.perf_counter()
        vectorizer = TfidfVectorizer(
            analyzer="char",
            ngram_range=(k, k),
            lowercase=False,
            min_df=3,
            max_df=0.995,
            sublinear_tf=True,
            norm="l2",
            dtype=np.float32,
        )
        fit_features = vectorizer.fit_transform(_view_sequences(fit, view))
        calibration_features = vectorizer.transform(_view_sequences(calibration, view))
        selection_features = vectorizer.transform(_view_sequences(selection, view))
        test_features = vectorizer.transform(_view_sequences(test, view))
        vectorizer_seconds = time.perf_counter() - vectorizer_started
        if fit_features.shape[1] == 0:
            raise TEBenchmarkError(f"empty TF--IDF vocabulary for {view}, k={k}")

        for candidate in candidates:
            started = time.perf_counter()
            classifier = _make_classifier(candidate, seed=seed)
            _fit_with_convergence_guard(
                classifier,
                fit_features,
                fit["label"].to_numpy(dtype=np.int8),
                fit["sample_weight"].to_numpy(dtype=float),
                context=f"candidate {candidate['model_id']}",
            )
            calibration_scores = _decision_scores(classifier, calibration_features)
            calibrator = _fit_platt_calibrator(
                calibration_scores,
                calibration["label"].to_numpy(dtype=np.int8),
                calibration["sample_weight"].to_numpy(dtype=float),
                seed=seed,
                calibration_config=calibration_config,
            )
            model_id = candidate["model_id"]
            predictions_select[model_id] = _calibrated_probabilities(
                classifier, calibrator, selection_features
            )
            predictions_test[model_id] = _calibrated_probabilities(
                classifier, calibrator, test_features
            )
            elapsed = time.perf_counter() - started
            candidate_configuration = {
                **candidate,
                "vectorizer": vectorizer_config,
                "calibration": calibration_config,
                "seed": seed,
            }
            registry_rows.append(
                {
                    "model_id": model_id,
                    "model_family": candidate["family"],
                    "representation": f"{view}:char_{k}mer_tfidf",
                    "pooling": (
                        "full_oriented_te_copy"
                        if view == "te_copy"
                        else f"centered_{view.removeprefix('window_')}bp"
                    ),
                    "regularization": _regularization_label(candidate),
                    "seed": seed,
                    "calibration": "heldout_platt_training_roots",
                    "checkpoint_hash": _checkpoint_hash(
                        vectorizer,
                        classifier,
                        calibrator,
                        configuration=candidate_configuration,
                        fit_root_hash=fit_root_hash,
                        calibration_root_hash=calibration_root_hash,
                    ),
                    "training_data_hash": training_data_hash,
                    "configuration_json": json.dumps(
                        candidate_configuration, sort_keys=True, separators=(",", ":")
                    ),
                    "candidate_config_hash": sha256_file(config_path),
                    "fit_root_hash": fit_root_hash,
                    "calibration_root_hash": calibration_root_hash,
                    "eligible_for_hrms_selection": True,
                }
            )
            timing_rows.append(
                {
                    "model_id": model_id,
                    "role": "hrms_candidate",
                    "view": view,
                    "k": k,
                    "vectorizer_seconds_shared": vectorizer_seconds,
                    "fit_calibrate_predict_seconds": elapsed,
                    "vocabulary_size": len(vectorizer.vocabulary_),
                    "fit_rows": len(fit),
                    "calibration_rows": len(calibration),
                }
            )
            print(f"trained {model_id} in {elapsed:.2f}s", flush=True)
        del fit_features, calibration_features, selection_features, test_features

    for diagnostic in config["diagnostic_only"]:
        started = time.perf_counter()
        (
            preprocessor,
            fit_features,
            calibration_features,
            selection_features,
            test_features,
        ) = _diagnostic_feature_matrices(diagnostic, fit, calibration, selection, test)
        classifier = _make_classifier(diagnostic, seed=seed)
        _fit_with_convergence_guard(
            classifier,
            fit_features,
            fit["label"].to_numpy(dtype=np.int8),
            fit["sample_weight"].to_numpy(dtype=float),
            context=f"diagnostic {diagnostic['model_id']}",
        )
        calibration_scores = _decision_scores(classifier, calibration_features)
        calibrator = _fit_platt_calibrator(
            calibration_scores,
            calibration["label"].to_numpy(dtype=np.int8),
            calibration["sample_weight"].to_numpy(dtype=float),
            seed=seed,
            calibration_config=calibration_config,
        )
        model_id = diagnostic["model_id"]
        diagnostic_predictions_select[model_id] = _calibrated_probabilities(
            classifier, calibrator, selection_features
        )
        diagnostic_predictions_test[model_id] = _calibrated_probabilities(
            classifier, calibrator, test_features
        )
        diagnostic_configuration = {
            **diagnostic,
            "calibration": calibration_config,
            "seed": seed,
            "eligible_for_hrms_selection": False,
        }
        diagnostic_registry_rows.append(
            {
                "model_id": model_id,
                "model_family": "diagnostic_logistic",
                "representation": "+".join(diagnostic["features"]),
                "pooling": "diagnostic_covariates_only",
                "regularization": _regularization_label(diagnostic),
                "seed": seed,
                "calibration": "heldout_platt_training_roots",
                "checkpoint_hash": _checkpoint_hash(
                    preprocessor,
                    classifier,
                    calibrator,
                    configuration=diagnostic_configuration,
                    fit_root_hash=fit_root_hash,
                    calibration_root_hash=calibration_root_hash,
                ),
                "training_data_hash": training_data_hash,
                "configuration_json": json.dumps(
                    diagnostic_configuration, sort_keys=True, separators=(",", ":")
                ),
                "candidate_config_hash": sha256_file(config_path),
                "fit_root_hash": fit_root_hash,
                "calibration_root_hash": calibration_root_hash,
                "eligible_for_hrms_selection": False,
            }
        )
        elapsed = time.perf_counter() - started
        timing_rows.append(
            {
                "model_id": model_id,
                "role": "diagnostic_only",
                "view": diagnostic["kind"],
                "k": "",
                "vectorizer_seconds_shared": 0.0,
                "fit_calibrate_predict_seconds": elapsed,
                "vocabulary_size": int(fit_features.shape[1]),
                "fit_rows": len(fit),
                "calibration_rows": len(calibration),
            }
        )
        print(f"trained diagnostic-only control {model_id} in {elapsed:.2f}s", flush=True)
        del fit_features, calibration_features, selection_features, test_features

    expected_model_order = [candidate["model_id"] for candidate in config["candidates"]]
    if set(predictions_select) != set(expected_model_order):
        raise RuntimeError("trained candidate library differs from frozen configuration")
    predictions_select = {
        model_id: predictions_select[model_id] for model_id in expected_model_order
    }
    predictions_test = {model_id: predictions_test[model_id] for model_id in expected_model_order}
    expected_diagnostic_order = [
        diagnostic["model_id"] for diagnostic in config["diagnostic_only"]
    ]
    if set(diagnostic_predictions_select) != set(expected_diagnostic_order):
        raise RuntimeError("trained diagnostic controls differ from frozen configuration")
    diagnostic_predictions_select = {
        model_id: diagnostic_predictions_select[model_id]
        for model_id in expected_diagnostic_order
    }
    diagnostic_predictions_test = {
        model_id: diagnostic_predictions_test[model_id]
        for model_id in expected_diagnostic_order
    }
    registry = (
        pd.DataFrame(registry_rows)
        .set_index("model_id")
        .loc[expected_model_order]
        .reset_index()
    )
    diagnostic_registry = (
        pd.DataFrame(diagnostic_registry_rows)
        .set_index("model_id")
        .loc[expected_diagnostic_order]
        .reset_index()
    )
    validate_model_registry(registry)
    validate_model_registry(diagnostic_registry)
    if not registry["eligible_for_hrms_selection"].astype(bool).all():
        raise RuntimeError("the frozen candidate registry contains an ineligible model")
    if registry["model_id"].isin(diagnostic_registry["model_id"]).any():
        raise RuntimeError("diagnostic controls leaked into the HRMS candidate registry")
    if diagnostic_registry["eligible_for_hrms_selection"].astype(bool).any():
        raise RuntimeError("diagnostic controls must be ineligible for HRMS selection")
    prediction_select = _prediction_frame(selection, predictions_select)
    prediction_test = _prediction_frame(test, predictions_test)
    diagnostic_prediction_select = _prediction_frame(
        selection, diagnostic_predictions_select
    )
    diagnostic_prediction_test = _prediction_frame(test, diagnostic_predictions_test)
    loss_select = predictions_to_brier_losses(prediction_select, registry)
    loss_test = predictions_to_brier_losses(prediction_test, registry)
    diagnostic_loss_select = predictions_to_brier_losses(
        diagnostic_prediction_select, diagnostic_registry
    )
    diagnostic_loss_test = predictions_to_brier_losses(
        diagnostic_prediction_test, diagnostic_registry
    )

    prediction_select.to_parquet(outputs["predictions_select"], index=False)
    prediction_test.to_parquet(outputs["predictions_test"], index=False)
    loss_select.to_parquet(outputs["losses_select"], index=False)
    loss_test.to_parquet(outputs["losses_test"], index=False)
    diagnostic_prediction_select.to_parquet(
        outputs["diagnostic_predictions_select"], index=False
    )
    diagnostic_prediction_test.to_parquet(
        outputs["diagnostic_predictions_test"], index=False
    )
    diagnostic_loss_select.to_parquet(outputs["diagnostic_losses_select"], index=False)
    diagnostic_loss_test.to_parquet(outputs["diagnostic_losses_test"], index=False)
    calibration_roots_frame = pd.DataFrame(
        {
            "root_cluster_id": sorted(fit_roots | calibration_roots),
        }
    )
    calibration_roots_frame["internal_partition"] = calibration_roots_frame[
        "root_cluster_id"
    ].map(lambda root: "calibration" if root in calibration_roots else "fit")
    calibration_roots_frame.to_csv(
        outputs["calibration_roots"], index=False, lineterminator="\n"
    )
    pd.DataFrame(timing_rows).to_csv(outputs["timings"], index=False, lineterminator="\n")
    prediction_digest = hashlib.sha256()
    for key in ("predictions_select", "predictions_test"):
        prediction_digest.update(bytes.fromhex(sha256_file(outputs[key])))
    registry["prediction_file_hash"] = prediction_digest.hexdigest()
    validate_model_registry(registry)
    registry.to_csv(outputs["registry"], index=False, lineterminator="\n")
    diagnostic_prediction_digest = hashlib.sha256()
    for key in ("diagnostic_predictions_select", "diagnostic_predictions_test"):
        diagnostic_prediction_digest.update(bytes.fromhex(sha256_file(outputs[key])))
    diagnostic_registry["prediction_file_hash"] = diagnostic_prediction_digest.hexdigest()
    validate_model_registry(diagnostic_registry)
    diagnostic_registry.to_csv(
        outputs["diagnostic_registry"], index=False, lineterminator="\n"
    )

    artifacts = {
        name: {
            "path": path.as_posix(),
            "size_bytes": path.stat().st_size,
            "sha256": sha256_file(path),
        }
        for name, path in outputs.items()
        if name not in {"manifest", "timings"}
    }
    write_json(
        outputs["manifest"],
        {
            "schema_version": 1,
            "software": {
                "python": platform.python_version(),
                "numpy": np.__version__,
                "pandas": pd.__version__,
                "scikit_learn": sklearn.__version__,
            },
            "inputs": {
                "clustered_cohort": {
                    "path": cohort_path.as_posix(),
                    "sha256": sha256_file(cohort_path),
                },
                "frozen_candidate_config": {
                    "path": config_path.as_posix(),
                    "sha256": sha256_file(config_path),
                },
                "training_program": {
                    "path": Path(__file__).resolve().as_posix(),
                    "sha256": sha256_file(Path(__file__).resolve()),
                },
                "dependency_lock": {
                    "path": (_PROJECT_ROOT / "uv.lock").as_posix(),
                    "sha256": sha256_file(_PROJECT_ROOT / "uv.lock"),
                },
            },
            "training": {
                "seed": seed,
                "candidate_count": len(expected_model_order),
                "diagnostic_only_count": len(expected_diagnostic_order),
                "cohort_split_hash_recomputed": cohort_split_hash,
                "cohort_split_seed_recorded": cohort_split_seed,
                "training_rows": len(train),
                "fit_rows": len(fit),
                "calibration_rows": len(calibration),
                "fit_roots": len(fit_roots),
                "calibration_roots": len(calibration_roots),
                "fit_root_hash": fit_root_hash,
                "calibration_root_hash": calibration_root_hash,
                "fit_calibration_roots_disjoint": True,
                "training_data_hash": training_data_hash,
                "selection_or_test_used_for_vectorizer_fit": False,
                "selection_or_test_used_for_classifier_fit": False,
                "selection_or_test_used_for_calibration": False,
                "selection_or_test_used_for_candidate_choice": False,
                "test_metrics_computed_during_training": False,
                "diagnostic_only_models_excluded_from_hrms_registry": (
                    expected_diagnostic_order
                ),
                "diagnostic_preprocessors_fit_on_training_fit_roots_only": True,
                "diagnostic_classifiers_fit_on_training_fit_roots_only": True,
                "diagnostic_calibration_uses_training_calibration_roots_only": True,
            },
            "non_content_addressed_runtime_diagnostic": {
                "path": outputs["timings"].as_posix(),
                "reason": "wall-clock durations are intentionally non-deterministic",
            },
            "artifacts": artifacts,
        },
    )
    lock.unlink()
    return outputs


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--cohort",
        type=Path,
        default=Path("data/te_public/processed/labeled_loci_split.parquet"),
    )
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--output-dir", type=Path, default=Path("results/te_public"))
    args = parser.parse_args(argv)
    outputs = train_public_library(args.cohort, args.output_dir, config_path=args.config)
    for name, path in outputs.items():
        print(f"{name}: {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
