#!/usr/bin/env python3
"""Post-test sensitivity analysis that incorrectly treats comments as IID roots."""

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

import pandas as pd

from applications.civilcomments.common import read_table, sha256_file, write_json
from hrms import HRMS
from hrms.ambiguity import ChiSquareBall
from hrms.pairwise import pairwise_differences
from hrms.registry import assert_prediction_library_matches
from hrms.regret import point_robust_regret


def run_ablation(
    validation_loss_path: Path,
    registry_path: Path,
    primary_manifest_path: Path,
    primary_certificate_path: Path,
    test_environment_risk_path: Path,
    method_result_path: Path,
    output_dir: Path,
    *,
    radius: float = 0.25,
    alpha: float = 0.05,
    n_bootstrap: int = 1_999,
    seed: int = 42,
) -> dict:
    """Run the misspecified comment-root selector without altering primary artifacts."""

    validation = read_table(validation_loss_path)
    registry = pd.read_csv(registry_path)
    assert_prediction_library_matches(registry, validation["model_id"].unique().tolist())
    true_article_roots = int(validation["root_cluster_id"].nunique())
    validation = validation.copy()
    validation["root_cluster_id"] = validation["observation_id"]

    result = HRMS(
        ambiguity=ChiSquareBall(center="uniform", radius=radius),
        alpha=alpha,
        n_bootstrap=n_bootstrap,
        weighting="observation",
        resampling="auto",
        random_state=seed,
        stability_replicates=min(500, n_bootstrap),
        robustness_radii=(0.0, 0.1, 0.25, 0.5, 1.0, None),
    ).fit(validation)
    output_dir.mkdir(parents=True, exist_ok=True)
    result.environment_risks.to_csv(output_dir / "validation_environment_risks.csv")
    result.pairwise_bounds.to_csv(output_dir / "pairwise_bounds.csv", index=False)
    result.certificate_frame().to_csv(output_dir / "regret_certificates.csv")
    result.robustness_path.to_csv(output_dir / "robustness_path.csv", index=False)
    result.selection_stability.to_csv(output_dir / "selection_stability.csv")

    models = [str(value) for value in result.environment_risks.columns]
    test_risks = pd.read_csv(test_environment_risk_path, index_col=0).reindex(
        index=result.environment_risks.index, columns=models
    )
    if test_risks.isna().any().any():
        raise ValueError("test environment risks do not align with ablation candidates")
    test_regret = point_robust_regret(
        pairwise_differences(test_risks.to_numpy(dtype=float)),
        ChiSquareBall(center="uniform", radius=radius),
    )
    selected_index = models.index(str(result.selected_model))

    primary_manifest = json.loads(primary_manifest_path.read_text())
    primary_model = str(primary_manifest["hrms"]["selected_model"])
    primary_certificates = pd.read_csv(primary_certificate_path, index_col=0)
    iid_certificates = result.certificate_frame().reindex(primary_certificates.index)
    upper_change = iid_certificates["upper"] - primary_certificates["upper"]
    method_results = pd.read_csv(method_result_path).set_index("method")
    primary = {
        "root_definition": "article_id",
        "root_clusters": true_article_roots,
        "resampling": primary_manifest["parameters"]["resampling"],
        "selected_model": primary_model,
        "confidence_set_size": int(primary_manifest["hrms"]["confidence_set_size"]),
        "selected_certificate_lower": float(primary_certificates.loc[primary_model, "lower"]),
        "selected_certificate_upper": float(primary_certificates.loc[primary_model, "upper"]),
        "test_robust_regret": float(
            method_results.loc["full_hrms", "test_robust_regret_rho"]
        ),
    }
    ablation = {
        "root_definition": "individual comment (known misspecification)",
        "root_clusters": int(validation["root_cluster_id"].nunique()),
        "resampling": result.resampling,
        "selected_model": str(result.selected_model),
        "confidence_set_size": len(result.model_confidence_set),
        "confidence_set": [str(value) for value in result.model_confidence_set],
        "selected_certificate_lower": float(
            result.robust_regret_lower.loc[result.selected_model]
        ),
        "selected_certificate_upper": float(
            result.robust_regret_upper.loc[result.selected_model]
        ),
        "test_robust_regret": float(test_regret[selected_index]),
        "critical_value": result.critical_value,
    }
    summary = {
        "schema_version": 1,
        "created_at_utc": datetime.now(UTC).isoformat(),
        "analysis_role": "post-test sensitivity analysis; never used for primary selection",
        "iid_root_assumption_is_known_to_be_misspecified": True,
        "reason": "comments in one article share topic/context and some articles span environments",
        "parameters": {
            "alpha": alpha,
            "radius": radius,
            "n_bootstrap": n_bootstrap,
            "seed": seed,
            "calibration_method": result.calibration_method,
        },
        "primary_article_clustered": primary,
        "iid_comment_root_ablation": ablation,
        "differences": {
            "selected_model_changed": ablation["selected_model"] != primary["selected_model"],
            "confidence_set_changed": set(ablation["confidence_set"])
            != set(primary_manifest["hrms"]["confidence_set"]),
            "confidence_set_size_iid_minus_primary": (
                ablation["confidence_set_size"] - primary["confidence_set_size"]
            ),
            "selected_upper_certificate_iid_minus_primary": (
                ablation["selected_certificate_upper"]
                - primary["selected_certificate_upper"]
            ),
            "maximum_absolute_modelwise_upper_certificate_change": float(
                upper_change.abs().max()
            ),
            "median_absolute_modelwise_upper_certificate_change": float(
                upper_change.abs().median()
            ),
            "primary_selected_model_upper_under_iid": float(
                iid_certificates.loc[primary_model, "upper"]
            ),
            "iid_selected_model_upper_under_primary": float(
                primary_certificates.loc[ablation["selected_model"], "upper"]
            ),
        },
        "inputs": {
            "validation_losses_sha256": sha256_file(validation_loss_path),
            "registry_sha256": sha256_file(registry_path),
            "primary_selection_manifest_sha256": sha256_file(primary_manifest_path),
            "test_environment_risks_sha256": sha256_file(test_environment_risk_path),
        },
    }
    write_json(output_dir / "summary.json", summary)
    return summary


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--validation-losses",
        type=Path,
        default=Path("results/civilcomments/losses_val.csv.gz"),
    )
    parser.add_argument(
        "--registry",
        type=Path,
        default=Path("results/civilcomments/model_registry.csv"),
    )
    parser.add_argument(
        "--primary-manifest",
        type=Path,
        default=Path("results/civilcomments/selection/selection_manifest.json"),
    )
    parser.add_argument(
        "--primary-certificates",
        type=Path,
        default=Path("results/civilcomments/selection/regret_certificates.csv"),
    )
    parser.add_argument(
        "--test-environment-risks",
        type=Path,
        default=Path("results/civilcomments/test_evaluation/test_environment_risks.csv"),
    )
    parser.add_argument(
        "--method-results",
        type=Path,
        default=Path("results/civilcomments/test_evaluation/method_results.csv"),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("results/civilcomments/ablations/iid_comment_roots"),
    )
    parser.add_argument("--radius", type=float, default=0.25)
    parser.add_argument("--alpha", type=float, default=0.05)
    parser.add_argument("--bootstrap", type=int, default=1_999)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args(argv)
    summary = run_ablation(
        args.validation_losses,
        args.registry,
        args.primary_manifest,
        args.primary_certificates,
        args.test_environment_risks,
        args.method_results,
        args.output_dir,
        radius=args.radius,
        alpha=args.alpha,
        n_bootstrap=args.bootstrap,
        seed=args.seed,
    )
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
