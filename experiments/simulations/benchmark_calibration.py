"""Small reproducible runtime benchmark for simultaneous calibrations."""

from __future__ import annotations

import argparse
import json
import time

from experiments.simulations.calibration import (
    CalibrationConfig,
    generate_calibration_data,
)
from hrms import HRMS


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--calibration",
        choices=["jackknife_bootstrap_t", "raw_bootstrap_t", "fixed_max_t"],
        required=True,
    )
    parser.add_argument("--clusters", type=int, default=25)
    parser.add_argument("--candidates", type=int, default=10)
    parser.add_argument("--bootstrap", type=int, default=399)
    args = parser.parse_args()

    simulation = generate_calibration_data(
        CalibrationConfig(
            n_environments=3,
            n_candidates=args.candidates,
            clusters_per_environment=args.clusters,
            mean_cluster_size=10,
            icc=0.3,
            candidate_correlation=0.7,
            cluster_size_imbalance="moderate",
            random_state=42,
        )
    )
    selector = HRMS(
        n_bootstrap=args.bootstrap,
        calibration=args.calibration,
        stability_replicates=1,
        robustness_radii=(0.25,),
        random_state=43,
    )
    start = time.perf_counter()
    result = selector.fit(simulation.loss_table)
    elapsed = time.perf_counter() - start
    print(
        json.dumps(
            {
                "calibration": args.calibration,
                "clusters_per_environment": args.clusters,
                "candidates": args.candidates,
                "bootstrap": args.bootstrap,
                "elapsed_seconds": elapsed,
                "critical_value": result.critical_value,
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
