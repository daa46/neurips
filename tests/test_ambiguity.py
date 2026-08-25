"""Tests for the specialized chi-squared ambiguity-set solver."""

from __future__ import annotations

import numpy as np
import pytest

from hrms.ambiguity import (
    FULL_SIMPLEX,
    ChiSquareBall,
    chi_square_divergence,
    cvxpy_reference_support,
    max_component_weight,
    maximum_component_weights,
    scipy_reference_support,
    solve_chi_square_support,
)


def test_zero_radius_is_the_nominal_mixture() -> None:
    values = np.array([0.9, -0.2, 0.4])
    center = np.array([0.2, 0.3, 0.5])

    result = solve_chi_square_support(values, center, radius=0.0)

    np.testing.assert_array_equal(result.weights, center)
    assert result.value == pytest.approx(float(center @ values))
    assert result.divergence == 0.0


def test_uniform_interior_case_has_known_analytic_solution() -> None:
    # At this radius every q_e is positive, yielding
    # support = E_p[v] + sqrt(rho * Var_p[v]).
    values = np.array([-1.0, 0.0, 1.0])
    center = np.full(3, 1.0 / 3.0)
    radius = 0.1
    variance = float(center @ np.square(values - center @ values))
    expected_weights = center * (1.0 + (values - center @ values) * np.sqrt(radius / variance))

    result = solve_chi_square_support(values, center, radius)

    assert result.value == pytest.approx(np.sqrt(radius * variance), abs=1e-13)
    np.testing.assert_allclose(result.weights, expected_weights, atol=1e-13)
    assert result.divergence == pytest.approx(radius, abs=1e-13)


def test_full_simplex_and_sufficient_finite_radius_reach_a_vertex() -> None:
    values = np.array([-2.0, 4.0, 1.0])
    center = np.array([0.2, 0.5, 0.3])
    # Moving all mass to component 1 costs 1 / p_1 - 1 = 1.
    for radius in (FULL_SIMPLEX, None, np.inf, 1.0, 2.0):
        result = solve_chi_square_support(values, center, radius)
        assert result.value == 4.0
        np.testing.assert_array_equal(result.weights, [0.0, 1.0, 0.0])


def test_full_simplex_tie_rule_is_permutation_equivariant() -> None:
    values = np.array([3.0, 1.0, 3.0])
    center = np.array([0.2, 0.3, 0.5])
    result = ChiSquareBall.full_simplex(center).solve(values)

    np.testing.assert_allclose(result.weights, [2.0 / 7.0, 0.0, 5.0 / 7.0])
    assert result.value == 3.0


def test_support_is_monotone_in_radius() -> None:
    values = np.array([-0.4, 0.1, 0.3, 1.2, -1.0])
    center = np.array([0.05, 0.2, 0.3, 0.35, 0.1])
    radii = [0.0, 0.01, 0.1, 0.4, 1.0, None]

    supports = [ChiSquareBall(center, radius).support(values) for radius in radii]

    assert np.all(np.diff(supports) >= -1e-13)
    assert supports[-1] == pytest.approx(float(np.max(values)))


def test_component_maximum_formula_and_helper_agree_with_solver() -> None:
    center = np.array([0.1, 0.3, 0.6])
    radius = 0.2
    maxima = maximum_component_weights(center, radius)

    expected = np.minimum(1.0, center + np.sqrt(radius * center * (1.0 - center)))
    np.testing.assert_allclose(maxima, expected, atol=1e-15)
    np.testing.assert_allclose(max_component_weight(center, radius), expected)

    for index in range(center.size):
        indicator = np.eye(center.size)[index]
        result = solve_chi_square_support(indicator, center, radius)
        assert result.weights[index] == pytest.approx(maxima[index], abs=2e-13)
        assert result.value == pytest.approx(maxima[index], abs=2e-13)


def test_class_radius_path_and_resolved_uniform_center() -> None:
    ball = ChiSquareBall(center="uniform", radius=0.25)
    np.testing.assert_array_equal(ball.resolved_center(4), np.full(4, 0.25))
    np.testing.assert_allclose(
        ball.maximum_component_weights(n_environments=4),
        np.full(4, 0.25 + np.sqrt(0.25 * 0.25 * 0.75)),
    )
    assert ball.with_radius(None).support([0.0, 2.0, 1.0]) == 2.0


@pytest.mark.parametrize("radius", [0.001, 0.05, 0.3, 1.1])
def test_candidate_and_environment_permutation_invariance(radius: float) -> None:
    rng = np.random.default_rng(391)
    values = rng.normal(size=7)
    center = rng.dirichlet(np.ones(7))
    permutation = rng.permutation(7)

    original = solve_chi_square_support(values, center, radius)
    permuted = solve_chi_square_support(values[permutation], center[permutation], radius)
    restored_weights = np.empty_like(permuted.weights)
    restored_weights[permutation] = permuted.weights

    assert permuted.value == pytest.approx(original.value, abs=2e-13)
    np.testing.assert_allclose(restored_weights, original.weights, atol=2e-12)


@pytest.mark.parametrize("seed", range(10))
def test_specialized_solver_agrees_with_scipy_reference(seed: int) -> None:
    pytest.importorskip("scipy.optimize")
    rng = np.random.default_rng(seed)
    size = int(rng.integers(2, 9))
    values = rng.normal(size=size)
    center = rng.dirichlet(np.full(size, 1.5))
    radius = float(10 ** rng.uniform(-2.0, 0.7))

    specialized = solve_chi_square_support(values, center, radius)
    reference = scipy_reference_support(values, center, radius)

    assert specialized.value == pytest.approx(reference.value, abs=2e-7)
    assert specialized.divergence <= radius + 2e-12
    assert reference.divergence <= radius + 2e-7


def test_specialized_solver_agrees_with_exact_cvxpy_formulation() -> None:
    cvxpy = pytest.importorskip("cvxpy", reason="install the optional reference extra")
    if not cvxpy.installed_solvers():
        pytest.skip("CVXPY has no installed conic solver")

    values = np.array([1.7, -0.8, 0.2, 1.1, -1.4])
    center = np.array([0.08, 0.17, 0.31, 0.29, 0.15])
    radius = 0.63

    specialized = solve_chi_square_support(values, center, radius)
    reference = cvxpy_reference_support(values, center, radius)

    assert specialized.value == pytest.approx(reference.value, abs=2e-5)
    assert specialized.divergence <= radius + 2e-12
    assert reference.divergence <= radius + 2e-10


def test_solution_is_feasible_with_active_set_truncation() -> None:
    values = np.array([10.0, 3.0, 0.0, -4.0])
    center = np.full(4, 0.25)
    result = solve_chi_square_support(values, center, radius=1.7)

    assert np.count_nonzero(result.weights == 0.0) >= 1
    assert np.all(result.weights >= 0.0)
    assert np.sum(result.weights) == pytest.approx(1.0)
    assert chi_square_divergence(result.weights, center) <= 1.7 + 1e-12


@pytest.mark.parametrize(
    ("center", "radius"),
    [
        ([0.0, 1.0], 0.2),
        ([0.2, 0.2], 0.2),
        ([0.5, 0.5], -0.1),
    ],
)
def test_invalid_ambiguity_sets_are_rejected(center, radius) -> None:
    with pytest.raises((TypeError, ValueError)):
        solve_chi_square_support([0.0, 1.0], center, radius)
