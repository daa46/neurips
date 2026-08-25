#!/usr/bin/env python3
"""Generate the small, deterministic long-form loss-table example."""

from __future__ import annotations

import argparse
import hashlib
from pathlib import Path

import numpy as np
import pandas as pd

MODEL_EFFECTS = {
    "stable": (0.000, 0.000, 0.000),
    "nominal_average": (-0.018, 0.004, 0.026),
    "environment_0_specialist": (-0.036, 0.018, 0.046),
    "environment_2_specialist": (0.046, 0.018, -0.036),
}


def generate_synthetic_loss_table() -> pd.DataFrame:
    """Return paired bounded losses with clusters and environment rank reversals."""

    rows: list[dict[str, object]] = []
    for environment in range(3):
        environment_id = f"environment_{environment}"
        for cluster in range(6):
            root_cluster_id = f"{environment_id}__root_{cluster:02d}"
            cluster_effect = 0.012 * np.sin((environment + 1) * (cluster + 1))
            for observation in range(3):
                observation_id = f"{root_cluster_id}__observation_{observation:02d}"
                observation_effect = 0.005 * np.cos(
                    (environment + 2) * (cluster + 1) * (observation + 1)
                )
                base_loss = 0.20 + 0.01 * environment + cluster_effect + observation_effect
                sample_weight = 1.0 + 0.25 * ((cluster + observation) % 3)
                for model_index, (model_id, effects) in enumerate(MODEL_EFFECTS.items()):
                    model_cluster_interaction = 0.006 * np.sin(
                        (model_index + 1) * (cluster + 1) + 0.4 * environment
                    )
                    model_observation_interaction = 0.002 * np.cos(
                        (model_index + 2) * (observation + 1) + environment
                    )
                    loss = float(
                        np.clip(
                            base_loss
                            + effects[environment]
                            + model_cluster_interaction
                            + model_observation_interaction,
                            0.0,
                            1.0,
                        )
                    )
                    rows.append(
                        {
                            "observation_id": observation_id,
                            "environment_id": environment_id,
                            "root_cluster_id": root_cluster_id,
                            "model_id": model_id,
                            "loss": loss,
                            "sample_weight": sample_weight,
                        }
                    )
    return pd.DataFrame.from_records(rows)


def generate_synthetic_registry() -> pd.DataFrame:
    """Return a valid frozen registry for the deterministic example rules."""

    training_hash = hashlib.sha256(b"hrms_synthetic_training_population_v1").hexdigest()
    rows = []
    for seed, (model_id, effects) in enumerate(MODEL_EFFECTS.items(), start=1):
        recipe = f"hrms_synthetic_rule_v1:{model_id}:{effects}".encode()
        rows.append(
            {
                "model_id": model_id,
                "model_family": "deterministic_synthetic_rule",
                "representation": "direct_bounded_loss",
                "pooling": "none",
                "regularization": "none",
                "seed": seed,
                "calibration": "none_synthetic",
                "checkpoint_hash": hashlib.sha256(recipe).hexdigest(),
                "training_data_hash": training_hash,
                "status": "frozen_example",
            }
        )
    return pd.DataFrame.from_records(rows)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("examples/synthetic_losses.csv"),
    )
    parser.add_argument(
        "--registry",
        type=Path,
        default=Path("examples/synthetic_model_registry.csv"),
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.registry.parent.mkdir(parents=True, exist_ok=True)
    generate_synthetic_loss_table().to_csv(args.output, index=False, float_format="%.12g")
    generate_synthetic_registry().to_csv(args.registry, index=False)
    print(f"loss_table: {args.output}")
    print(f"model_registry: {args.registry}")
    return 0


if __name__ == "__main__":  # pragma: no cover - exercised through the CLI
    raise SystemExit(main())
