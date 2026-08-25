#!/usr/bin/env python3
"""Open CivilComments test labels only after a validation selection is sealed."""

# ruff: noqa: E402 -- direct-script support requires path setup before imports.

from __future__ import annotations

import argparse
import json
import sys
from datetime import UTC, datetime
from pathlib import Path

_PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import kendalltau
from sklearn.metrics import average_precision_score, brier_score_loss, roc_auc_score

from applications.civilcomments.common import (
    read_table,
    sha256_file,
    write_csv_gzip,
    write_json,
)
from hrms.ambiguity import ChiSquareBall, chi_square_divergence
from hrms.baselines import nominal_mean_scores
from hrms.pairwise import pairwise_differences
from hrms.registry import validate_model_registry
from hrms.regret import point_robust_regret


def _sample_stress_mixtures(
    count: int,
    environments: int,
    radius: float,
    rng: np.random.Generator,
) -> np.ndarray:
    """Generate a reproducible diagnostic design inside a chi-square ball."""

    center = np.full(environments, 1.0 / environments)
    directions = rng.dirichlet(np.ones(environments), size=count)
    output = np.empty_like(directions)
    for index, direction in enumerate(directions):
        divergence = chi_square_divergence(direction, center)
        boundary_scale = 1.0 if divergence <= radius else np.sqrt(radius / divergence)
        scale = boundary_scale * np.sqrt(rng.uniform())
        output[index] = center + scale * (direction - center)
    return output


def _test_loss_table(
    predictions: pd.DataFrame,
    labels: pd.DataFrame,
    models: list[str],
) -> pd.DataFrame:
    if "label" in predictions.columns or "toxicity" in predictions.columns:
        raise ValueError("test prediction file must not contain labels")
    expected_prediction_columns = {f"prediction__{model}" for model in models}
    observed_prediction_columns = {
        column for column in predictions if column.startswith("prediction__")
    }
    if expected_prediction_columns != observed_prediction_columns:
        raise ValueError("test prediction candidate library differs from the frozen registry")
    if labels.columns.tolist() != ["observation_id", "label"]:
        raise ValueError("sealed test labels must contain only observation_id and label")
    if (
        predictions["observation_id"].duplicated().any()
        or labels["observation_id"].duplicated().any()
    ):
        raise ValueError("test observation identifiers must be unique")
    merged = predictions.merge(labels, on="observation_id", how="left", validate="one_to_one")
    if merged["label"].isna().any() or len(merged) != len(labels):
        raise ValueError("test predictions and sealed labels do not match exactly")
    if not merged["label"].isin([0, 1]).all():
        raise ValueError("test labels must be binary")

    metadata = [
        "observation_id",
        "environment_id",
        "root_cluster_id",
        "sample_weight",
        "label",
    ]
    long = merged.melt(
        id_vars=metadata,
        value_vars=[f"prediction__{model}" for model in models],
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


def _display_model(model: str) -> str:
    return (
        model.replace("char_", "char ")
        .replace("word_", "word ")
        .replace("__c_", " C=")
        .replace("_", "–")
        .replace("p", ".")
    )


def _make_figure(
    methods: pd.DataFrame,
    stability: pd.Series,
    selected_model: str,
    confidence_set_size: int,
    output_dir: Path,
) -> None:
    method_labels = {
        "environment_balanced_mean": "Balanced mean",
        "environment_cvar_25": "Env. CVaR",
        "full_hrms": "Full HRMS",
        "hrms_rho_0": r"HRMS $\rho=0$",
        "mean_validation": "Mean validation",
        "point_dro_regret": "Point DRO-regret",
        "point_dro_risk": "Point DRO-risk",
        "worst_environment": "Worst environment",
    }
    labels = methods["method"].map(method_labels)
    values = 1_000 * methods["test_robust_regret_rho"]
    colors = ["#0072B2" if method == "full_hrms" else "#999999" for method in methods.method]
    figure, axes = plt.subplots(
        1, 2, figsize=(9.4, 3.5), gridspec_kw={"width_ratios": [1.25, 1]}
    )
    axis = axes[0]
    positions = np.arange(len(methods))
    axis.bar(positions, values, color=colors, width=0.72)
    axis.set_xticks(positions, labels, rotation=35, ha="right")
    axis.set_ylabel("Worst test regret in $\\mathcal{Q}_{0.25}$ ($\\times 10^{-3}$ Brier)")
    axis.spines[["top", "right"]].set_visible(False)
    axis.grid(axis="y", alpha=0.2)
    axis.set_title("(a) Realized mixture-robust regret", loc="left", fontsize=10)

    stability = stability.sort_values(ascending=True)
    stability_colors = [
        "#0072B2" if model == selected_model else "#999999" for model in stability.index
    ]
    axes[1].barh(
        np.arange(len(stability)),
        stability.to_numpy(),
        color=stability_colors,
        height=0.72,
    )
    axes[1].set_yticks(
        np.arange(len(stability)), [_display_model(str(model)) for model in stability.index]
    )
    axes[1].set_xlabel("Bootstrap selection frequency")
    axes[1].set_xlim(0, max(0.5, 1.05 * float(stability.max())))
    axes[1].spines[["top", "right"]].set_visible(False)
    axes[1].grid(axis="x", alpha=0.2)
    axes[1].set_title("(b) No stable unique winner", loc="left", fontsize=10)
    axes[1].text(
        0.98,
        0.05,
        f"95% confidence set: {confidence_set_size}/{len(stability)}",
        transform=axes[1].transAxes,
        ha="right",
        va="bottom",
        fontsize=8,
    )
    figure.tight_layout()
    figure.savefig(output_dir / "civilcomments_regret.pdf", bbox_inches="tight")
    figure.savefig(output_dir / "civilcomments_regret.png", dpi=220, bbox_inches="tight")
    plt.close(figure)


def evaluate(
    test_prediction_path: Path,
    test_label_path: Path,
    registry_path: Path,
    selection_manifest_path: Path,
    selection_scores_path: Path,
    certificate_path: Path,
    source_metadata_path: Path,
    output_dir: Path,
    *,
    n_mixtures: int = 2_000,
    seed: int = 43,
) -> dict:
    """Evaluate the already-sealed selections; never rerun selection here."""

    if not selection_manifest_path.is_file():
        raise FileNotFoundError(
            "selection manifest must exist before test labels may be opened: "
            f"{selection_manifest_path}"
        )
    selection = json.loads(selection_manifest_path.read_text())
    if selection.get("test_labels_accessed") is not False:
        raise ValueError("selection manifest does not attest that test labels remained sealed")

    registry = pd.read_csv(registry_path)
    validate_model_registry(registry)
    expected_registry_hash = selection["inputs"]["model_registry"]["sha256"]
    if sha256_file(registry_path) != expected_registry_hash:
        raise ValueError("model registry changed after selection was sealed")
    predictions = read_table(test_prediction_path)
    expected_scores_hash = selection["artifacts"]["validation_selection_scores"]["sha256"]
    if sha256_file(selection_scores_path) != expected_scores_hash:
        raise ValueError("validation selection scores changed after selection was sealed")
    source_metadata = json.loads(source_metadata_path.read_text())
    stability_path = selection_manifest_path.parent / "selection_stability.csv"
    stability = pd.read_csv(stability_path, index_col=0).iloc[:, 0]

    # This is deliberately the first read of the sealed label artifact, after
    # the persisted selection and frozen registry have been verified.
    labels = read_table(test_label_path)
    models = registry["model_id"].tolist()
    test = _test_loss_table(predictions, labels, models)
    output_dir.mkdir(parents=True, exist_ok=True)
    test_loss_path = output_dir / "losses_test.csv.gz"
    write_csv_gzip(test, test_loss_path)

    selected = {str(key): str(value) for key, value in selection["selected_models"].items()}
    if any(model not in set(models) for model in selected.values()):
        raise ValueError("selection manifest names a candidate outside the frozen registry")
    environments = sorted(test["environment_id"].unique().tolist())
    risks_frame = test.pivot_table(
        index="environment_id", columns="model_id", values="loss", aggfunc="mean"
    ).reindex(index=environments, columns=models)
    if risks_frame.isna().any().any():
        raise ValueError("test losses lack an environment/model cell")
    risks_frame.to_csv(output_dir / "test_environment_risks.csv")
    test_risks = risks_frame.to_numpy(dtype=float)
    model_index = {model: index for index, model in enumerate(models)}
    radius = float(selection["parameters"]["radius"])
    ambiguity = ChiSquareBall(center="uniform", radius=radius)
    test_pairwise = pairwise_differences(test_risks)
    test_robust_regret = point_robust_regret(test_pairwise, ambiguity)
    test_simplex_regret = point_robust_regret(
        test_pairwise, ChiSquareBall.full_simplex(center="uniform")
    )
    test_uniform_risk = nominal_mean_scores(test_risks)
    test_uniform_regret = test_uniform_risk - test_uniform_risk.min()
    observation_metadata = test.drop_duplicates("observation_id")
    observed_weights = (
        observation_metadata["environment_id"]
        .value_counts()
        .reindex(environments)
        .to_numpy(dtype=float)
    )
    observed_weights /= observed_weights.sum()
    observed_risk = nominal_mean_scores(test_risks, observed_weights)
    observed_regret = observed_risk - observed_risk.min()
    optimum = test_robust_regret.min()
    robust_optimal = {
        models[index]
        for index in np.flatnonzero(np.isclose(test_robust_regret, optimum, atol=1e-12))
    }

    scores = pd.read_csv(selection_scores_path)
    score_lookup = {
        method: frame.set_index("model_id")["score"].reindex(models).to_numpy()
        for method, frame in scores.groupby("method", observed=True)
    }
    method_rows = []
    for method, model in selected.items():
        index = model_index[model]
        selection_score = score_lookup.get(method)
        tau = (
            float(kendalltau(selection_score, test_uniform_risk).statistic)
            if selection_score is not None
            else np.nan
        )
        method_rows.append(
            {
                "method": method,
                "selected_model": model,
                "test_uniform_regret": test_uniform_regret[index],
                "test_observed_mixture_regret": observed_regret[index],
                "test_robust_regret_rho": test_robust_regret[index],
                "test_full_simplex_regret": test_simplex_regret[index],
                "selected_test_robust_optimal": model in robust_optimal,
                "kendall_tau_selection_score_vs_test_uniform_risk": tau,
            }
        )
    methods = pd.DataFrame(method_rows)
    methods.to_csv(output_dir / "method_results.csv", index=False, lineterminator="\n")

    candidate_metrics = []
    for model in models:
        frame = test.loc[test["model_id"].eq(model)]
        candidate_metrics.append(
            {
                "model_id": model,
                "brier": brier_score_loss(frame["label"], frame["prediction"]),
                "auprc": average_precision_score(frame["label"], frame["prediction"]),
                "auroc": roc_auc_score(frame["label"], frame["prediction"]),
            }
        )
    pd.DataFrame(candidate_metrics).to_csv(
        output_dir / "candidate_test_metrics.csv", index=False, lineterminator="\n"
    )

    mixtures = _sample_stress_mixtures(
        n_mixtures, len(environments), radius, np.random.default_rng(seed)
    )
    mixture_risks = mixtures @ test_risks
    mixture_oracle = mixture_risks.min(axis=1)
    mixture_rows = []
    for method, model in selected.items():
        regrets = mixture_risks[:, model_index[model]] - mixture_oracle
        for mixture_index, regret in enumerate(regrets):
            mixture_rows.append(
                {
                    "mixture_index": mixture_index,
                    "method": method,
                    "selected_model": model,
                    "test_regret": regret,
                }
            )
    pd.DataFrame(mixture_rows).to_csv(
        output_dir / "stress_mixture_regrets.csv.gz",
        index=False,
        compression={"method": "gzip", "mtime": 0},
    )
    pd.DataFrame(mixtures, columns=environments).to_csv(
        output_dir / "stress_mixtures.csv.gz",
        index=False,
        compression={"method": "gzip", "mtime": 0},
    )

    certificate = pd.read_csv(certificate_path, index_col=0).reindex(models)
    certificate["test_robust_regret"] = test_robust_regret
    certificate["lower_below_test"] = certificate["lower"] <= test_robust_regret
    certificate["upper_above_test"] = certificate["upper"] >= test_robust_regret
    certificate.to_csv(output_dir / "test_certificate_audit.csv")
    confidence_set = {str(value) for value in selection["hrms"]["confidence_set"]}
    confidence_covers_optimum = bool(robust_optimal & confidence_set)
    confidence_covers_all_optima = robust_optimal.issubset(confidence_set)
    root_summary = source_metadata["counts"]["root_cluster_summary_by_split"]

    rank_taus = []
    for first in range(len(environments)):
        for second in range(first + 1, len(environments)):
            rank_taus.append(float(kendalltau(test_risks[first], test_risks[second]).statistic))
    summary = {
        "schema_version": 1,
        "evaluated_at_utc": datetime.now(UTC).isoformat(),
        "task": "derived CivilComments exact-one-identity environment-mixture task",
        "not_official_wilds_leaderboard_evaluation": True,
        "selection_was_sealed_before_test_labels_opened": True,
        "selection_manifest_sha256": sha256_file(selection_manifest_path),
        "radius": radius,
        "environments": environments,
        "candidate_count": len(models),
        "test_observations": int(test["observation_id"].nunique()),
        "test_root_clusters": int(test["root_cluster_id"].nunique()),
        "validation_root_clusters": int(root_summary["val"]["root_clusters"]),
        "resampling_unit": "article_id root cluster",
        "selection_resampling": selection["parameters"]["resampling"],
        "global_resampling_detail": (
            "sample validation article roots globally with replacement and retain all "
            "identity-environment records from each sampled article"
        ),
        "hrms": {
            "selected_model": selected["full_hrms"],
            "confidence_set": sorted(confidence_set),
            "confidence_set_size": len(confidence_set),
            "test_robust_optimal_models": sorted(robust_optimal),
            "confidence_set_contains_a_test_robust_optimum": confidence_covers_optimum,
            "confidence_set_contains_all_test_robust_optima": confidence_covers_all_optima,
            "all_upper_certificates_cover_test_estimates": bool(
                certificate["upper_above_test"].all()
            ),
        },
        "rank_variation": {
            "mean_pairwise_environment_kendall_tau": float(np.mean(rank_taus)),
            "minimum_pairwise_environment_kendall_tau": float(np.min(rank_taus)),
        },
        "selection_stability": {
            "maximum_frequency": float(stability.max()),
            "most_frequent_candidate": str(stability.idxmax()),
            "frequency_is_not_a_posterior_probability": True,
        },
        "inputs": {
            "test_predictions_sha256": sha256_file(test_prediction_path),
            "sealed_test_labels_sha256": sha256_file(test_label_path),
            "model_registry_sha256": sha256_file(registry_path),
            "source_metadata_sha256": sha256_file(source_metadata_path),
            "selection_stability_sha256": sha256_file(stability_path),
        },
        "artifacts": {"test_losses_sha256": sha256_file(test_loss_path)},
    }
    write_json(output_dir / "evaluation_summary.json", summary)
    _make_figure(
        methods,
        stability,
        selected["full_hrms"],
        len(confidence_set),
        output_dir,
    )
    return summary


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--test-predictions",
        type=Path,
        default=Path("results/civilcomments/predictions_test.csv.gz"),
    )
    parser.add_argument(
        "--test-labels",
        type=Path,
        default=Path("data/civilcomments/derived_test_labels.csv.gz"),
    )
    parser.add_argument(
        "--registry",
        type=Path,
        default=Path("results/civilcomments/model_registry.csv"),
    )
    parser.add_argument(
        "--selection-manifest",
        type=Path,
        default=Path("results/civilcomments/selection/selection_manifest.json"),
    )
    parser.add_argument(
        "--selection-scores",
        type=Path,
        default=Path("results/civilcomments/selection/validation_selection_scores.csv"),
    )
    parser.add_argument(
        "--certificates",
        type=Path,
        default=Path("results/civilcomments/selection/regret_certificates.csv"),
    )
    parser.add_argument(
        "--source-metadata",
        type=Path,
        default=Path("data/civilcomments/source_metadata.json"),
    )
    parser.add_argument(
        "--output-dir", type=Path, default=Path("results/civilcomments/test_evaluation")
    )
    parser.add_argument("--mixtures", type=int, default=2_000)
    parser.add_argument("--seed", type=int, default=43)
    args = parser.parse_args(argv)
    summary = evaluate(
        args.test_predictions,
        args.test_labels,
        args.registry,
        args.selection_manifest,
        args.selection_scores,
        args.certificates,
        args.source_metadata,
        args.output_dir,
        n_mixtures=args.mixtures,
        seed=args.seed,
    )
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
