r"""Chi-squared-divergence ambiguity sets.

This module implements the support function

.. math::

   \sup_q q^\top v \quad\text{subject to}\quad
   q \in \Delta^E,\qquad
   \sum_e (q_e-p_e)^2/p_e \leq \rho.

The production solver is dependency-free apart from NumPy.  It uses the KKT
conditions to reduce the problem to a finite active-set search.  Independent
SciPy/SLSQP and CVXPY implementations are included only as reference solvers.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
from numpy.typing import ArrayLike, NDArray

FULL_SIMPLEX = "full_simplex"
"""Sentinel accepted as a radius when every mixture is allowed."""


@dataclass(frozen=True)
class SupportResult:
    """Value and maximizing mixture returned by an ambiguity-set solver."""

    value: float
    weights: NDArray[np.float64]
    divergence: float

    def __iter__(self):
        """Allow ``value, weights = result`` for lightweight interoperability."""

        yield self.value
        yield self.weights


def _as_values(values: ArrayLike) -> NDArray[np.float64]:
    vector = np.asarray(values, dtype=np.float64)
    if vector.ndim != 1 or vector.size == 0:
        raise ValueError("values must be a non-empty one-dimensional array")
    if not np.all(np.isfinite(vector)):
        raise ValueError("values must all be finite")
    return vector


def _as_center(center: ArrayLike | str | None, size: int) -> NDArray[np.float64]:
    if center is None or (isinstance(center, str) and center == "uniform"):
        return np.full(size, 1.0 / size, dtype=np.float64)
    if isinstance(center, str):
        raise ValueError("center must be 'uniform' or a strictly positive mixture")

    mixture = np.asarray(center, dtype=np.float64)
    if mixture.ndim != 1 or mixture.size != size:
        raise ValueError(f"center must be one-dimensional with length {size}")
    if not np.all(np.isfinite(mixture)) or np.any(mixture <= 0.0):
        raise ValueError("center entries must be finite and strictly positive")

    total = float(np.sum(mixture, dtype=np.float64))
    tolerance = 128.0 * np.finfo(np.float64).eps * max(1, size)
    if not math.isclose(total, 1.0, rel_tol=tolerance, abs_tol=tolerance):
        raise ValueError("center must sum to one")
    # Normalizing removes harmless summation error and makes prefix identities
    # used by the active-set derivation numerically exact enough.
    return mixture / total


def _parse_radius(radius: float | str | None) -> tuple[bool, float]:
    if radius is None:
        return True, math.inf
    if isinstance(radius, str):
        normalized = radius.strip().lower().replace("-", "_").replace(" ", "_")
        if normalized in {FULL_SIMPLEX, "simplex", "full"}:
            return True, math.inf
        raise ValueError("radius string must denote the full simplex")
    if isinstance(radius, (bool, np.bool_)):
        raise TypeError("radius must be a non-negative number or 'full_simplex'")

    numeric = float(radius)
    if math.isnan(numeric) or numeric < 0.0:
        raise ValueError("radius must be non-negative")
    if math.isinf(numeric):
        return True, math.inf
    return False, numeric


def chi_square_divergence(weights: ArrayLike, center: ArrayLike) -> float:
    r"""Return :math:`\sum_e (q_e-p_e)^2/p_e` after validating both mixtures."""

    q = np.asarray(weights, dtype=np.float64)
    if q.ndim != 1 or q.size == 0:
        raise ValueError("weights must be a non-empty one-dimensional array")
    p = _as_center(center, q.size)
    if not np.all(np.isfinite(q)) or np.any(q < 0.0):
        raise ValueError("weights must be finite and non-negative")
    tolerance = 128.0 * np.finfo(np.float64).eps * max(1, q.size)
    if not math.isclose(float(np.sum(q)), 1.0, rel_tol=tolerance, abs_tol=tolerance):
        raise ValueError("weights must sum to one")
    return float(np.sum(np.square(q - p) / p, dtype=np.float64))


def _objective(weights: NDArray[np.float64], values: NDArray[np.float64]) -> float:
    # longdouble avoids an avoidable intermediate overflow for large, mixed-sign
    # finite values.  A convex combination itself lies between finite endpoints.
    result = np.sum(
        weights.astype(np.longdouble) * values.astype(np.longdouble),
        dtype=np.longdouble,
    )
    return float(result)


def _simplex_solution(values: NDArray[np.float64], center: NDArray[np.float64]) -> SupportResult:
    maximum = float(np.max(values))
    maximizers = values == maximum
    weights = np.zeros_like(center)
    # This tie rule is permutation-equivariant and, among all maximizers, has
    # the smallest chi-squared divergence from p.
    weights[maximizers] = center[maximizers] / float(np.sum(center[maximizers]))
    divergence = float(np.sum(np.square(weights - center) / center))
    return SupportResult(maximum, weights, divergence)


def _scale_values(values: NDArray[np.float64]) -> NDArray[np.float64] | None:
    """Affinely scale values to [0, 1], or return None for a constant vector."""

    if np.all(values == values[0]):
        return None
    magnitude = float(np.max(np.abs(values)))
    scaled = values / magnitude if magnitude > 0.0 else values.copy()
    lower = float(np.min(scaled))
    span = float(np.max(scaled) - lower)
    if span == 0.0:
        # The inputs differ by less than representable precision after scaling;
        # treating them as tied is the only stable floating-point decision.
        return None
    return (scaled - lower) / span


def solve_chi_square_support(
    values: ArrayLike,
    center: ArrayLike | str | None = "uniform",
    radius: float | str | None = 0.0,
) -> SupportResult:
    r"""Maximize a linear functional over a chi-squared ambiguity set.

    Parameters
    ----------
    values:
        Environment-specific values in the linear objective.
    center:
        Strictly positive nominal mixture, or ``"uniform"``.
    radius:
        Non-negative chi-squared radius.  ``None``, infinity, or
        ``"full_simplex"`` removes the divergence constraint.

    Notes
    -----
    For an active set :math:`A={e:v_e>t}`, the KKT conditions give

    .. math::

       q_e = \frac{p_e}{P_A}\left(1 + \frac{v_e-\mu_A}{a}\right),

    where ``a`` is available in closed form from the divergence constraint.
    Sorting the values means only prefixes can be active, so at most ``E``
    analytic candidates are examined.
    """

    objective_values = _as_values(values)
    nominal = _as_center(center, objective_values.size)
    full_simplex, rho = _parse_radius(radius)

    if full_simplex:
        return _simplex_solution(objective_values, nominal)
    if rho == 0.0:
        return SupportResult(_objective(nominal, objective_values), nominal.copy(), 0.0)
    if objective_values.size == 1:
        return SupportResult(float(objective_values[0]), np.ones(1), 0.0)

    scaled_values = _scale_values(objective_values)
    if scaled_values is None:
        return SupportResult(float(objective_values[0]), nominal.copy(), 0.0)

    maximum = float(np.max(scaled_values))
    maximizers = scaled_values == maximum
    maximizing_mass = float(np.sum(nominal[maximizers]))
    vertex_radius = (1.0 - maximizing_mass) / maximizing_mass
    if rho >= vertex_radius:
        return _simplex_solution(objective_values, nominal)

    order = np.argsort(-scaled_values, kind="stable")
    sorted_values = scaled_values[order]
    sorted_center = nominal[order]
    eps = np.finfo(np.float64).eps
    gap_tolerance = 512.0 * eps

    # Search from the full support toward the maximizing environments.  Tied
    # values must enter and leave together because the KKT active set is {v>t}.
    for active_size in range(objective_values.size, 0, -1):
        if (
            active_size < objective_values.size
            and sorted_values[active_size - 1] == sorted_values[active_size]
        ):
            continue

        active_p = sorted_center[:active_size]
        active_v = sorted_values[:active_size]
        active_mass = float(np.sum(active_p))
        base_divergence = max(0.0, (1.0 - active_mass) / active_mass)
        residual_radius = rho - base_divergence
        if residual_radius <= 0.0:
            continue

        mean = float(np.dot(active_p, active_v) / active_mass)
        variance_numerator = float(np.dot(active_p, np.square(active_v - mean)))
        if variance_numerator <= 0.0:
            continue

        scale = math.sqrt(variance_numerator) / (active_mass * math.sqrt(residual_radius))
        distance_to_lowest_active = mean - float(active_v[-1])
        if scale + gap_tolerance < distance_to_lowest_active:
            continue
        if active_size < objective_values.size:
            distance_to_highest_inactive = mean - float(sorted_values[active_size])
            if scale > distance_to_highest_inactive + gap_tolerance:
                continue

        active_weights = (active_p / active_mass) * (1.0 + (active_v - mean) / scale)
        if float(np.min(active_weights)) < -gap_tolerance:
            continue
        active_weights = np.maximum(active_weights, 0.0)

        sorted_weights = np.zeros_like(sorted_center)
        sorted_weights[:active_size] = active_weights
        sorted_weights /= float(np.sum(sorted_weights))
        weights = np.empty_like(sorted_weights)
        weights[order] = sorted_weights

        divergence = float(np.sum(np.square(weights - nominal) / nominal))
        if divergence > rho:
            # Roundoff at an active-set boundary can put q a few ulps outside
            # the ball.  Contracting toward p preserves the simplex and restores
            # feasibility without changing the active-set answer materially.
            anchor_sorted = np.zeros_like(sorted_center)
            anchor_sorted[:active_size] = active_p / active_mass
            anchor = np.empty_like(anchor_sorted)
            anchor[order] = anchor_sorted
            contraction = math.sqrt((rho - base_divergence) / (divergence - base_divergence))
            weights = anchor + contraction * (weights - anchor)
            weights /= float(np.sum(weights))
            divergence = float(np.sum(np.square(weights - nominal) / nominal))

        return SupportResult(_objective(weights, objective_values), weights, divergence)

    raise RuntimeError("failed to identify a chi-squared active set; inputs may be ill-conditioned")


def chi_square_support(
    values: ArrayLike,
    center: ArrayLike | str | None = "uniform",
    radius: float | str | None = 0.0,
    *,
    return_weights: bool = False,
) -> float | tuple[float, NDArray[np.float64]]:
    """Convenience wrapper returning the support value, optionally with weights."""

    result = solve_chi_square_support(values, center, radius)
    if return_weights:
        return result.value, result.weights
    return result.value


def max_component_weight(
    nominal_weight: float | ArrayLike,
    radius: float | str | None,
) -> float | NDArray[np.float64]:
    """Return the largest feasible value of one or more mixture components.

    For a nominal component weight ``p_e``, allocating the remaining mass
    proportionally across other components gives the least divergence at fixed
    ``q_e``.  Consequently,

    ``q_e,max = min(1, p_e + sqrt(rho * p_e * (1 - p_e)))``.

    The function accepts either a scalar component weight or an entire nominal
    mixture and applies the expression componentwise.
    """

    full_simplex, rho = _parse_radius(radius)
    nominal = np.asarray(nominal_weight, dtype=np.float64)
    scalar = nominal.ndim == 0
    if nominal.ndim > 1 or nominal.size == 0:
        raise ValueError("nominal_weight must be a scalar or one-dimensional array")
    if not np.all(np.isfinite(nominal)) or np.any(nominal <= 0.0) or np.any(nominal > 1.0):
        raise ValueError("nominal component weights must lie in (0, 1]")
    if full_simplex:
        answer = np.ones_like(nominal, dtype=np.float64)
    else:
        answer = np.minimum(
            1.0,
            nominal + np.sqrt(rho * nominal * (1.0 - nominal)),
        )
    if scalar:
        return float(answer)
    return answer


def maximum_component_weights(
    center: ArrayLike,
    radius: float | str | None,
) -> NDArray[np.float64]:
    """Validate a nominal mixture and return every component's feasible maximum."""

    raw = np.asarray(center, dtype=np.float64)
    if raw.ndim != 1 or raw.size == 0:
        raise ValueError("center must be a non-empty one-dimensional array")
    nominal = _as_center(raw, raw.size)
    return np.asarray(max_component_weight(nominal, radius), dtype=np.float64)


def scipy_reference_support(
    values: ArrayLike,
    center: ArrayLike | str | None = "uniform",
    radius: float | str | None = 0.0,
) -> SupportResult:
    """Solve the same problem with SciPy/SLSQP for testing and validation.

    SciPy is imported lazily and is not required by the production solver.
    """

    objective_values = _as_values(values)
    nominal = _as_center(center, objective_values.size)
    full_simplex, rho = _parse_radius(radius)
    if full_simplex:
        return _simplex_solution(objective_values, nominal)
    if rho == 0.0:
        return SupportResult(_objective(nominal, objective_values), nominal.copy(), 0.0)

    try:
        from scipy.optimize import minimize
    except ImportError as error:  # pragma: no cover - environment-dependent
        raise ImportError("SciPy is required for scipy_reference_support") from error

    constraints = (
        {
            "type": "eq",
            "fun": lambda q: float(np.sum(q) - 1.0),
        },
        {
            "type": "ineq",
            "fun": lambda q: float(rho - np.sum(np.square(q - nominal) / nominal)),
        },
    )
    result = minimize(
        lambda q: -float(np.dot(q, objective_values)),
        nominal.copy(),
        method="SLSQP",
        bounds=[(0.0, 1.0)] * objective_values.size,
        constraints=constraints,
        options={"ftol": 1e-12, "maxiter": 2_000},
    )
    if not result.success:
        raise RuntimeError(f"SciPy reference solver failed: {result.message}")

    weights = np.maximum(np.asarray(result.x, dtype=np.float64), 0.0)
    weights /= float(np.sum(weights))
    divergence = float(np.sum(np.square(weights - nominal) / nominal))
    return SupportResult(_objective(weights, objective_values), weights, divergence)


def cvxpy_reference_support(
    values: ArrayLike,
    center: ArrayLike | str | None = "uniform",
    radius: float | str | None = 0.0,
    *,
    solver: str | None = None,
    solver_options: dict[str, object] | None = None,
) -> SupportResult:
    """Solve the support problem from its exact CVXPY formulation.

    This is a deliberately separate reference implementation of

    ``maximize(values @ q)`` subject to ``q >= 0``, ``sum(q) == 1``, and
    ``sum((q - center)**2 / center) <= radius``.

    CVXPY is imported lazily and remains an optional ``reference`` dependency.
    A solver can be selected explicitly; otherwise CVXPY chooses an installed
    conic solver.  ``solver_options`` are forwarded unchanged to
    :meth:`cvxpy.Problem.solve`.
    """

    objective_values = _as_values(values)
    nominal = _as_center(center, objective_values.size)
    full_simplex, rho = _parse_radius(radius)
    if full_simplex:
        return _simplex_solution(objective_values, nominal)
    if rho == 0.0:
        return SupportResult(_objective(nominal, objective_values), nominal.copy(), 0.0)

    try:
        import cvxpy as cp
    except ImportError as error:  # pragma: no cover - environment-dependent
        raise ImportError(
            "CVXPY is required for cvxpy_reference_support; install the reference extra"
        ) from error

    weights_variable = cp.Variable(objective_values.size)
    divergence_expression = cp.sum(
        cp.multiply(1.0 / nominal, cp.square(weights_variable - nominal))
    )
    problem = cp.Problem(
        cp.Maximize(objective_values @ weights_variable),
        [
            weights_variable >= 0.0,
            cp.sum(weights_variable) == 1.0,
            divergence_expression <= rho,
        ],
    )
    options = dict(solver_options or {})
    if solver is not None:
        options["solver"] = solver
    try:
        problem.solve(**options)
    except cp.error.SolverError as error:
        raise RuntimeError(f"CVXPY reference solver failed: {error}") from error
    if problem.status not in {cp.OPTIMAL, cp.OPTIMAL_INACCURATE}:
        raise RuntimeError(f"CVXPY reference solver status was {problem.status!r}")
    if weights_variable.value is None:
        raise RuntimeError("CVXPY reference solver returned no weights")

    weights = np.maximum(np.asarray(weights_variable.value, dtype=np.float64).reshape(-1), 0.0)
    weights /= float(np.sum(weights))
    divergence = float(np.sum(np.square(weights - nominal) / nominal))
    if divergence > rho:
        # Conic solvers satisfy constraints to a numerical tolerance.  Contract
        # a barely exterior answer toward p so the returned mixture is feasible.
        contraction = math.sqrt(rho / divergence)
        weights = nominal + contraction * (weights - nominal)
        weights /= float(np.sum(weights))
        divergence = float(np.sum(np.square(weights - nominal) / nominal))
    return SupportResult(_objective(weights, objective_values), weights, divergence)


@dataclass(frozen=True)
class ChiSquareBall:
    """Chi-squared ambiguity set with a reusable nominal center and radius."""

    center: ArrayLike | str | None = "uniform"
    radius: float | str | None = 0.0

    def __post_init__(self) -> None:
        _parse_radius(self.radius)
        if isinstance(self.center, str) and self.center != "uniform":
            raise ValueError("center string must be 'uniform'")
        if self.center is not None and not isinstance(self.center, str):
            raw = np.asarray(self.center, dtype=np.float64)
            if raw.ndim != 1 or raw.size == 0:
                raise ValueError("center must be a non-empty one-dimensional array")
            _as_center(raw, raw.size)

    @classmethod
    def full_simplex(cls, center: ArrayLike | str | None = "uniform") -> ChiSquareBall:
        """Construct an ambiguity set containing the entire simplex."""

        return cls(center=center, radius=FULL_SIMPLEX)

    def with_radius(self, radius: float | str | None) -> ChiSquareBall:
        """Return the same ambiguity-set family at a different radius.

        ``None`` intentionally denotes the full simplex, which makes this
        convenient for robustness paths ending in an unconstrained mixture.
        """

        return type(self)(center=self.center, radius=radius)

    def resolved_center(self, n_environments: int) -> NDArray[np.float64]:
        """Materialize and return the nominal mixture for a supplied dimension."""

        if n_environments <= 0:
            raise ValueError("n_environments must be positive")
        return _as_center(self.center, n_environments)

    def solve(
        self,
        values: ArrayLike,
        center: ArrayLike | str | None = None,
    ) -> SupportResult:
        """Return the support value, maximizing mixture, and divergence."""

        effective_center = self.center if center is None else center
        return solve_chi_square_support(values, effective_center, self.radius)

    def support(
        self,
        values: ArrayLike,
        center: ArrayLike | str | None = None,
        *,
        return_weights: bool = False,
    ) -> float | tuple[float, NDArray[np.float64]]:
        """Return the support value, optionally paired with maximizing weights."""

        result = self.solve(values, center)
        if return_weights:
            return result.value, result.weights
        return result.value

    def worst_case_weights(
        self,
        values: ArrayLike,
        center: ArrayLike | str | None = None,
    ) -> NDArray[np.float64]:
        """Return a maximizing target-environment mixture."""

        return self.solve(values, center).weights

    def maximum_component_weights(
        self,
        *,
        n_environments: int | None = None,
        center: ArrayLike | str | None = None,
    ) -> NDArray[np.float64]:
        """Return the largest permitted weight for each environment."""

        effective_center = self.center if center is None else center
        is_uniform = effective_center is None or (
            isinstance(effective_center, str) and effective_center == "uniform"
        )
        if is_uniform:
            if n_environments is None or n_environments <= 0:
                raise ValueError("n_environments is required for a uniform center")
            nominal = np.full(n_environments, 1.0 / n_environments)
        else:
            raw = np.asarray(effective_center, dtype=np.float64)
            nominal = _as_center(raw, raw.size)
        return np.asarray(max_component_weight(nominal, self.radius))

    def maximum_component_weight(
        self,
        index: int,
        *,
        n_environments: int | None = None,
        center: ArrayLike | str | None = None,
    ) -> float:
        """Return the largest permitted weight for one environment."""

        maxima = self.maximum_component_weights(n_environments=n_environments, center=center)
        try:
            return float(maxima[index])
        except IndexError as error:
            raise IndexError("environment index is out of range") from error


# Compatibility-friendly aliases for callers that prefer generic terminology.
support_function = chi_square_support
solve_support = solve_chi_square_support


__all__ = [
    "FULL_SIMPLEX",
    "ChiSquareBall",
    "SupportResult",
    "chi_square_divergence",
    "chi_square_support",
    "cvxpy_reference_support",
    "max_component_weight",
    "maximum_component_weights",
    "scipy_reference_support",
    "solve_chi_square_support",
    "solve_support",
    "support_function",
]
