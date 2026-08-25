"""End-to-end HRMS estimator."""

from __future__ import annotations

from collections.abc import Iterable

import numpy as np
import pandas as pd

from .ambiguity import ChiSquareBall
from .bootstrap import cluster_bootstrap, cluster_jackknife_pseudovalues
from .confidence_set import robust_model_confidence_set, select_minimum_upper
from .diagnostics import build_diagnostics
from .pairwise import pairwise_differences
from .regret import point_robust_regret, robust_regret_certificates
from .results import HRMSResult, tidy_pairwise_bounds
from .schema import ColumnSpec, validate_loss_table
from .simultaneous import jackknife_bootstrap_t_calibration, max_t_bounds


class HRMS:
    """Paired, cluster-aware robust model selector for a frozen library."""

    def __init__(
        self,
        *,
        environment_col: str = "environment_id",
        cluster_col: str = "root_cluster_id",
        observation_col: str = "observation_id",
        model_col: str = "model_id",
        loss_col: str = "loss",
        weight_col: str | None = "sample_weight",
        ambiguity: ChiSquareBall | None = None,
        alpha: float = 0.05,
        n_bootstrap: int = 2_000,
        calibration: str = "jackknife_bootstrap_t",
        weighting: str = "observation",
        resampling: str = "auto",
        random_state: int | None = None,
        stability_replicates: int | None = 500,
        robustness_radii: Iterable[float | None] = (0.0, 0.1, 0.25, 0.5, 1.0, None),
        loss_bounds: tuple[float, float] = (0.0, 1.0),
        structural_duplicate_pairs: Iterable[tuple[object, object]] = (),
    ) -> None:
        self.columns = ColumnSpec(
            observation=observation_col,
            environment=environment_col,
            cluster=cluster_col,
            model=model_col,
            loss=loss_col,
            weight=weight_col,
        )
        self.ambiguity = ambiguity or ChiSquareBall(center="uniform", radius=0.25)
        self.alpha = alpha
        self.n_bootstrap = n_bootstrap
        if calibration not in {"jackknife_bootstrap_t", "raw_bootstrap_t", "fixed_max_t"}:
            raise ValueError(
                "calibration must be 'jackknife_bootstrap_t', 'raw_bootstrap_t', or 'fixed_max_t'"
            )
        self.calibration = calibration
        self.weighting = weighting
        self.resampling = resampling
        self.random_state = random_state
        self.stability_replicates = stability_replicates
        self.robustness_radii = tuple(robustness_radii)
        self.loss_bounds = loss_bounds
        self.structural_duplicate_pairs = tuple(structural_duplicate_pairs)

    def fit(self, loss_table: pd.DataFrame) -> HRMSResult:
        data = validate_loss_table(loss_table, self.columns)
        if self.loss_bounds[0] > self.loss_bounds[1]:
            raise ValueError("loss_bounds must be ordered")
        if np.any(data.losses < self.loss_bounds[0]) or np.any(data.losses > self.loss_bounds[1]):
            raise ValueError(
                f"losses must lie in the declared bounds {self.loss_bounds}; "
                "Brier losses use [0, 1]"
            )
        structural_duplicates = self._validated_structural_duplicates(data.losses, data.model_ids)
        bootstrap = cluster_bootstrap(
            data,
            n_bootstrap=self.n_bootstrap,
            weighting=self.weighting,
            resampling=self.resampling,
            studentize=self.calibration == "raw_bootstrap_t",
            random_state=self.random_state,
        )
        pairwise = pairwise_differences(bootstrap.point_risks)
        bootstrap_pairwise = pairwise_differences(bootstrap.bootstrap_risks)
        simultaneous_options = {}
        if self.calibration == "raw_bootstrap_t":
            assert bootstrap.studentized_max_statistics is not None
            simultaneous_options = {
                "standard_error": bootstrap.pairwise_standard_error,
                "max_statistics": bootstrap.studentized_max_statistics,
                "calibration_method": "studentized root-cluster bootstrap-t",
            }
        elif self.calibration == "jackknife_bootstrap_t":
            jackknife_blocks = cluster_jackknife_pseudovalues(
                data,
                weighting=self.weighting,
                resampling=bootstrap.resampling,
            )
            jackknife_seed = None if self.random_state is None else self.random_state + 7_919
            jackknife = jackknife_bootstrap_t_calibration(
                jackknife_blocks,
                n_models=data.n_models,
                n_environments=len(data.environment_ids),
                n_bootstrap=self.n_bootstrap,
                random_state=jackknife_seed,
            )
            simultaneous_options = {
                "standard_error": jackknife.standard_error,
                "max_statistics": jackknife.max_statistics,
                "calibration_method": jackknife.method,
            }
        bounds = max_t_bounds(
            pairwise,
            bootstrap_pairwise,
            alpha=self.alpha,
            identical_pairs=structural_duplicates,
            contrast_bounds=(
                self.loss_bounds[0] - self.loss_bounds[1],
                self.loss_bounds[1] - self.loss_bounds[0],
            ),
            **simultaneous_options,
        )
        certificates = robust_regret_certificates(bounds.lower, bounds.upper, self.ambiguity)
        selected_index = select_minimum_upper(
            certificates.upper,
            tie_breaker=point_robust_regret(pairwise, self.ambiguity),
        )
        confidence_indices = robust_model_confidence_set(certificates.lower, certificates.upper)

        stability = self._selection_stability(bootstrap_pairwise, data.model_ids)
        path = self._robustness_path(
            bounds.estimate,
            bounds.lower,
            bounds.upper,
            data.model_ids,
        )
        diagnostics = build_diagnostics(
            environment_ids=data.environment_ids,
            cluster_counts=bootstrap.cluster_counts,
            effective_cluster_counts=bootstrap.effective_cluster_counts,
            max_cluster_leverage=bootstrap.max_cluster_leverage,
            degenerate_coordinates=bounds.degenerate_coordinates,
            confidence_set_size=len(confidence_indices),
            n_models=data.n_models,
        )

        lower = pd.Series(certificates.lower, index=data.model_ids, name="lower")
        upper = pd.Series(certificates.upper, index=data.model_ids, name="upper")
        environment_risks = pd.DataFrame(
            bootstrap.point_risks,
            index=pd.Index(data.environment_ids, name="environment"),
            columns=pd.Index(data.model_ids, name="model"),
        )
        return HRMSResult(
            selected_model=data.model_ids[selected_index],
            robust_regret_lower=lower,
            robust_regret_upper=upper,
            model_confidence_set=tuple(data.model_ids[index] for index in confidence_indices),
            environment_risks=environment_risks,
            pairwise_bounds=tidy_pairwise_bounds(
                data.model_ids,
                data.environment_ids,
                bounds.estimate,
                bounds.lower,
                bounds.upper,
                bounds.standard_error,
            ),
            selection_stability=stability,
            robustness_path=path,
            cluster_counts=pd.Series(
                bootstrap.cluster_counts,
                index=data.environment_ids,
                name="independent_root_clusters",
            ),
            effective_cluster_counts=pd.Series(
                bootstrap.effective_cluster_counts,
                index=data.environment_ids,
                name="effective_root_clusters",
            ),
            max_cluster_leverage=pd.Series(
                bootstrap.max_cluster_leverage,
                index=data.environment_ids,
                name="max_cluster_leverage",
            ),
            diagnostics=diagnostics,
            critical_value=bounds.critical_value,
            calibration_method=bounds.calibration_method,
            alpha=self.alpha,
            resampling=bootstrap.resampling,
            weighting=bootstrap.weighting,
        )

    def _validated_structural_duplicates(self, losses: np.ndarray, model_ids: tuple) -> np.ndarray:
        """Validate predeclared duplicates against every paired loss value.

        Equality is deliberately exact: a structural duplicate represents the
        same frozen predictor and loss computation, not two merely close
        candidates. Observed equality never marks a pair automatically.
        """

        duplicates = np.eye(len(model_ids), dtype=bool)
        model_index = {identifier: index for index, identifier in enumerate(model_ids)}
        for first, second in self.structural_duplicate_pairs:
            if first not in model_index or second not in model_index:
                raise ValueError(f"unknown structural-duplicate pair: {(first, second)}")
            i, j = model_index[first], model_index[second]
            if i == j:
                raise ValueError(
                    "a structural-duplicate pair must name two distinct candidates; "
                    f"got {(first, second)}"
                )
            if not np.array_equal(losses[:, i], losses[:, j]):
                raise ValueError(
                    "declared structural-duplicate candidates must have exactly "
                    "equal observation-wise loss vectors; "
                    f"pair {(first, second)} does not"
                )
            duplicates[i, j] = duplicates[j, i] = True
        return duplicates

    def _selection_stability(self, bootstrap_pairwise: np.ndarray, model_ids: tuple) -> pd.Series:
        """Raw root-bootstrap frequency of the point robust-regret winner.

        This descriptive diagnostic reuses raw paired cluster-bootstrap risk
        replicates. It is neither a nested refit of the HRMS confidence
        procedure nor a posterior model probability.
        """

        total = len(bootstrap_pairwise)
        if self.stability_replicates is None or self.stability_replicates >= total:
            indices = np.arange(total)
        else:
            indices = np.linspace(0, total - 1, num=self.stability_replicates, dtype=int)
        counts = np.zeros(len(model_ids), dtype=int)
        for index in indices:
            regret = point_robust_regret(bootstrap_pairwise[index], self.ambiguity)
            counts[select_minimum_upper(regret)] += 1
        return pd.Series(counts / len(indices), index=model_ids, name="frequency")

    def _robustness_path(
        self,
        point_pairwise: np.ndarray,
        lower_pairwise: np.ndarray,
        upper_pairwise: np.ndarray,
        model_ids: tuple,
    ) -> pd.DataFrame:
        rows: list[dict] = []
        for radius in self.robustness_radii:
            ambiguity = self.ambiguity.with_radius(radius)
            certificates = robust_regret_certificates(lower_pairwise, upper_pairwise, ambiguity)
            selected = select_minimum_upper(
                certificates.upper,
                tie_breaker=point_robust_regret(point_pairwise, ambiguity),
            )
            confidence = robust_model_confidence_set(certificates.lower, certificates.upper)
            for model, model_id in enumerate(model_ids):
                rows.append(
                    {
                        "radius": "full" if radius is None else radius,
                        "model": model_id,
                        "lower": certificates.lower[model],
                        "upper": certificates.upper[model],
                        "selected": model == selected,
                        "in_confidence_set": model in confidence,
                        "confidence_set_size": len(confidence),
                    }
                )
        return pd.DataFrame.from_records(rows)
