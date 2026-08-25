"""Predeclared warnings for weak independent evidence and numerical pathologies."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class Diagnostic:
    code: str
    severity: str
    message: str


def build_diagnostics(
    *,
    environment_ids: tuple,
    cluster_counts: np.ndarray,
    effective_cluster_counts: np.ndarray,
    max_cluster_leverage: np.ndarray,
    degenerate_coordinates: tuple[tuple[int, int, int], ...],
    confidence_set_size: int,
    n_models: int,
) -> tuple[Diagnostic, ...]:
    diagnostics: list[Diagnostic] = []
    for index, count in enumerate(cluster_counts):
        if count < 10:
            diagnostics.append(
                Diagnostic(
                    code="very_few_clusters",
                    severity="warning",
                    message=(
                        f"environment {environment_ids[index]!r} has only {count} independent "
                        "root clusters; bootstrap calibration is likely unreliable"
                    ),
                )
            )
        elif count < 20:
            diagnostics.append(
                Diagnostic(
                    code="few_clusters",
                    severity="warning",
                    message=(
                        f"environment {environment_ids[index]!r} has {count} independent root "
                        "clusters; interpret singleton conclusions cautiously"
                    ),
                )
            )
        if effective_cluster_counts[index] < max(10.0, 0.5 * count):
            diagnostics.append(
                Diagnostic(
                    code="low_effective_cluster_count",
                    severity="warning",
                    message=(
                        f"environment {environment_ids[index]!r} has effective cluster count "
                        f"{effective_cluster_counts[index]:.1f} from {count} roots because of "
                        "cluster-size imbalance"
                    ),
                )
            )
        if max_cluster_leverage[index] > 0.2:
            diagnostics.append(
                Diagnostic(
                    code="dominating_cluster",
                    severity="warning",
                    message=(
                        f"one root contributes {max_cluster_leverage[index]:.1%} of the weight "
                        f"in environment {environment_ids[index]!r}"
                    ),
                )
            )
    if degenerate_coordinates:
        diagnostics.append(
            Diagnostic(
                code="zero_contrast_scale",
                severity="warning",
                message=(
                    f"{len(degenerate_coordinates)} non-identical pair/environment coordinates "
                    "had zero calibration scale and received the full bounded-loss interval"
                ),
            )
        )
    if confidence_set_size == n_models and n_models > 1:
        diagnostics.append(
            Diagnostic(
                code="uninformative_confidence_set",
                severity="info",
                message="the simultaneous evidence excludes no candidate from robust optimality",
            )
        )
    return tuple(diagnostics)
