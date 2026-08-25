#!/usr/bin/env python3
"""Select CivilComments candidates using validation data only and seal the decision."""

from __future__ import annotations

import argparse
import json
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import pandas as pd

from experiments.civilcomments.common import read_table, sha256_file, write_json
from hrms import HRMS
from hrms.ambiguity import ChiSquareBall
from hrms.baselines import (
    environment_cvar_scores,
    nominal_mean_scores,
    point_dro_regret_scores,
    point_dro_risk_scores,
    select_from_scores,
    worst_environment_scores,
)
from hrms.registry import assert_prediction_library_matches


def _save_hrms_outputs(result, output_dir: Path) -> None:
    result.environment_risks.to_csv(output_dir / "validation_environment_risks.csv")
    result.pairwise_bounds.to_csv(output_dir / "pairwise_bounds.csv", index=False)
    result.certificate_frame().to_csv(output_dir / "regret_certificates.csv")
    result.robustness_path.to_csv(output_dir / "robustness_path.csv", index=False)
    result.selection_stability.to_csv(output_dir / "selection_stability.csv")
    pd.concat(
        [
            result.cluster_counts,
            result.effective_cluster_counts,
            result.max_cluster_leverage,
        ],
        axis=1,
    ).to_csv(output_dir / "cluster_diagnostics.csv")


def select(
    validation_loss_path: Path,
    registry_path: Path,
    output_dir: Path,
    *,
    radius: float = 0.25,
    alpha: float = 0.05,
    n_bootstrap: int = 999,
    seed: int = 42,
) -> dict:
    """Run every selection rule without opening test features or labels."""

    validation = read_table(validation_loss_path)
    registry = pd.read_csv(registry_path)
    assert_prediction_library_matches(registry, validation["model_id"].unique().tolist())
    if validation.duplicated(["observation_id", "model_id"]).any():
        raise ValueError("validation loss table contains duplicate observation/model rows")
    if not np.allclose(
        validation["loss"].to_numpy(),
        np.square(validation["prediction"].to_numpy() - validation["label"].to_numpy()),
    ):
        raise ValueError("validation loss column is not Brier loss")

    ambiguity = ChiSquareBall(center="uniform", radius=radius)
    result = HRMS(
        ambiguity=ambiguity,
        alpha=alpha,
        n_bootstrap=n_bootstrap,
        weighting="observation",
        resampling="auto",
        random_state=seed,
        stability_replicates=min(500, n_bootstrap),
        robustness_radii=(0.0, 0.1, 0.25, 0.5, 1.0, None),
    ).fit(validation)
    output_dir.mkdir(parents=True, exist_ok=True)
    _save_hrms_outputs(result, output_dir)

    models = [str(value) for value in result.environment_risks.columns]
    risks = result.environment_risks.to_numpy(dtype=float)
    observed_validation = (
        validation.groupby("model_id", observed=True)["loss"].mean().reindex(models).to_numpy()
    )
    score_vectors = {
        "mean_validation": observed_validation,
        "environment_balanced_mean": nominal_mean_scores(risks),
        "worst_environment": worst_environment_scores(risks),
        "environment_cvar_25": environment_cvar_scores(risks, 0.25),
        "point_dro_risk": point_dro_risk_scores(risks, ambiguity),
        "point_dro_regret": point_dro_regret_scores(risks, ambiguity),
    }
    selected = {
        method: models[select_from_scores(scores)] for method, scores in score_vectors.items()
    }

    zero_path = result.robustness_path.loc[
        result.robustness_path["radius"].astype(str).isin({"0", "0.0"})
    ].set_index("model")
    if int(zero_path["selected"].sum()) != 1:
        raise RuntimeError("HRMS robustness path lacks a unique rho=0 selection")
    selected["hrms_rho_0"] = str(zero_path.index[zero_path["selected"]][0])
    selected["full_hrms"] = str(result.selected_model)

    score_rows = []
    for method, scores in score_vectors.items():
        for model, score in zip(models, scores, strict=True):
            score_rows.append({"method": method, "model_id": model, "score": score})
    for model in models:
        score_rows.append(
            {
                "method": "hrms_rho_0",
                "model_id": model,
                "score": float(zero_path.loc[model, "upper"]),
            }
        )
        score_rows.append(
            {
                "method": "full_hrms",
                "model_id": model,
                "score": float(result.robust_regret_upper.loc[model]),
            }
        )
    scores_path = output_dir / "validation_selection_scores.csv"
    pd.DataFrame(score_rows).to_csv(scores_path, index=False, lineterminator="\n")

    manifest = {
        "schema_version": 1,
        "sealed_at_utc": datetime.now(UTC).isoformat(),
        "test_labels_accessed": False,
        "inputs": {
            "validation_losses": {
                "path": validation_loss_path.as_posix(),
                "sha256": sha256_file(validation_loss_path),
            },
            "model_registry": {
                "path": registry_path.as_posix(),
                "sha256": sha256_file(registry_path),
            },
        },
        "parameters": {
            "alpha": alpha,
            "radius": radius,
            "n_bootstrap": n_bootstrap,
            "seed": seed,
            "primary_loss": "Brier",
            "weighting": result.weighting,
            "resampling": result.resampling,
            "calibration_method": result.calibration_method,
        },
        "candidate_count": len(models),
        "validation_observations": int(validation["observation_id"].nunique()),
        "validation_root_clusters": int(validation["root_cluster_id"].nunique()),
        "resampling_unit": "article_id root cluster",
        "global_resampling_detail": (
            "sample validation article roots globally with replacement and retain all "
            "identity-environment records from each sampled article"
        ),
        "selected_models": selected,
        "hrms": {
            "selected_model": str(result.selected_model),
            "confidence_set": [str(value) for value in result.model_confidence_set],
            "confidence_set_size": len(result.model_confidence_set),
            "critical_value": result.critical_value,
        },
        "artifacts": {
            "validation_selection_scores": {
                "path": scores_path.as_posix(),
                "sha256": sha256_file(scores_path),
            }
        },
    }
    manifest_path = output_dir / "selection_manifest.json"
    write_json(manifest_path, manifest)
    return manifest


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
    parser.add_argument("--output-dir", type=Path, default=Path("results/civilcomments/selection"))
    parser.add_argument("--radius", type=float, default=0.25)
    parser.add_argument("--alpha", type=float, default=0.05)
    parser.add_argument("--bootstrap", type=int, default=999)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args(argv)
    manifest = select(
        args.validation_losses,
        args.registry,
        args.output_dir,
        radius=args.radius,
        alpha=args.alpha,
        n_bootstrap=args.bootstrap,
        seed=args.seed,
    )
    print(json.dumps(manifest, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
