"""Run HRMS on a long-form CSV or Parquet loss table."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

from . import HRMS
from .ambiguity import ChiSquareBall
from .artifacts import write_artifact_manifest
from .registry import assert_prediction_library_matches

_PACKAGE_ROOT = Path(__file__).resolve().parent


def _read(path: Path) -> pd.DataFrame:
    if path.suffix.lower() == ".parquet":
        return pd.read_parquet(path)
    return pd.read_csv(path)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("loss_table", type=Path)
    parser.add_argument("--output", type=Path, default=Path("results/hrms"))
    parser.add_argument(
        "--model-registry",
        type=Path,
        help="optional frozen registry CSV; hashes and candidate identity are validated",
    )
    parser.add_argument("--radius", type=float, default=0.25)
    parser.add_argument("--alpha", type=float, default=0.05)
    parser.add_argument("--bootstrap", type=int, default=2_000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--calibration",
        choices=["jackknife_bootstrap_t", "raw_bootstrap_t", "fixed_max_t"],
        default="jackknife_bootstrap_t",
    )
    parser.add_argument("--weighting", choices=["observation", "cluster"], default="observation")
    args = parser.parse_args(argv)

    loss_table = _read(args.loss_table)
    if args.model_registry is not None:
        registry = pd.read_csv(args.model_registry)
        assert_prediction_library_matches(
            registry, loss_table["model_id"].drop_duplicates().tolist()
        )

    result = HRMS(
        ambiguity=ChiSquareBall(center="uniform", radius=args.radius),
        alpha=args.alpha,
        n_bootstrap=args.bootstrap,
        calibration=args.calibration,
        weighting=args.weighting,
        random_state=args.seed,
    ).fit(loss_table)
    args.output.mkdir(parents=True, exist_ok=True)
    outputs = {
        "environment_risks": args.output / "environment_risks.csv",
        "pairwise_bounds": args.output / "pairwise_bounds.csv",
        "regret_certificates": args.output / "regret_certificates.csv",
        "robustness_path": args.output / "robustness_path.csv",
        "selection_stability": args.output / "selection_stability.csv",
        "cluster_diagnostics": args.output / "cluster_diagnostics.csv",
        "summary": args.output / "summary.json",
    }
    result.environment_risks.to_csv(outputs["environment_risks"])
    result.pairwise_bounds.to_csv(outputs["pairwise_bounds"], index=False)
    result.certificate_frame().to_csv(outputs["regret_certificates"])
    result.robustness_path.to_csv(outputs["robustness_path"], index=False)
    result.selection_stability.to_csv(outputs["selection_stability"])
    pd.concat(
        [
            result.cluster_counts,
            result.effective_cluster_counts,
            result.max_cluster_leverage,
        ],
        axis=1,
    ).to_csv(outputs["cluster_diagnostics"])
    summary = {
        "selected_model": result.selected_model,
        "model_confidence_set": list(result.model_confidence_set),
        "critical_value": result.critical_value,
        "calibration_method": result.calibration_method,
        "resampling": result.resampling,
        "weighting": result.weighting,
        "diagnostics": [diagnostic.__dict__ for diagnostic in result.diagnostics],
    }
    outputs["summary"].write_text(json.dumps(summary, indent=2, default=str) + "\n")
    inputs = {"loss_table": args.loss_table}
    if args.model_registry is not None:
        inputs["model_registry"] = args.model_registry
    write_artifact_manifest(
        args.output / "manifest.json",
        inputs=inputs,
        artifacts=outputs,
        parameters={
            "alpha": args.alpha,
            "calibration": args.calibration,
            "n_bootstrap": args.bootstrap,
            "radius": args.radius,
            "random_state": args.seed,
            "weighting": args.weighting,
        },
        source_files=sorted(_PACKAGE_ROOT.glob("*.py")),
        source_base_dir=_PACKAGE_ROOT.parent,
        base_dir=args.output.parent,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
