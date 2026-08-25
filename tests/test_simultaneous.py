import numpy as np

from hrms.pairwise import pairwise_differences
from hrms.simultaneous import max_t_bounds


def test_simultaneous_bounds_enforce_antisymmetry():
    rng = np.random.default_rng(2)
    risks = np.asarray([[0.2, 0.25, 0.3], [0.35, 0.31, 0.29]])
    boot = risks[None, :, :] + rng.normal(0, 0.02, size=(100, 2, 3))
    bounds = max_t_bounds(pairwise_differences(risks), pairwise_differences(boot))
    assert np.allclose(bounds.lower, -np.swapaxes(bounds.upper, 0, 1))
    assert np.allclose(bounds.upper, -np.swapaxes(bounds.lower, 0, 1))


def test_structural_duplicates_have_zero_interval():
    estimate = np.zeros((2, 2, 1))
    replicates = np.zeros((20, 2, 2, 1))
    structural = np.ones((2, 2), dtype=bool)
    bounds = max_t_bounds(estimate, replicates, identical_pairs=structural)
    assert np.array_equal(bounds.lower, np.zeros_like(estimate))
    assert np.array_equal(bounds.upper, np.zeros_like(estimate))
    assert not bounds.degenerate_coordinates


def test_nonstructural_zero_scale_gets_trivial_bounded_interval():
    estimate = np.zeros((2, 2, 1))
    replicates = np.zeros((20, 2, 2, 1))
    bounds = max_t_bounds(estimate, replicates, contrast_bounds=(-1, 1))
    assert bounds.lower[0, 1, 0] == -1
    assert bounds.upper[0, 1, 0] == 1
    assert bounds.degenerate_coordinates == ((0, 1, 0),)


def test_supplied_studentized_statistics_determine_critical_value():
    estimate = np.zeros((2, 2, 1))
    replicates = np.zeros((5, 2, 2, 1))
    standard_error = np.zeros_like(estimate)
    standard_error[0, 1, 0] = standard_error[1, 0, 0] = 0.2
    maxima = np.asarray([1.0, 2.0, 3.0, 4.0, 5.0])
    bounds = max_t_bounds(
        estimate,
        replicates,
        alpha=0.2,
        standard_error=standard_error,
        max_statistics=maxima,
    )
    assert bounds.critical_value == 5.0
    assert bounds.calibration_method == "studentized cluster bootstrap-t"
    assert bounds.lower[0, 1, 0] == -1.0
    assert bounds.upper[0, 1, 0] == 1.0


def test_non_degenerate_intervals_are_clipped_to_loss_contrast_range():
    estimate = np.zeros((2, 2, 1))
    estimate[0, 1, 0] = 0.8
    estimate[1, 0, 0] = -0.8
    replicates = np.repeat(estimate[None, ...], 5, axis=0)
    standard_error = np.zeros_like(estimate)
    standard_error[0, 1, 0] = standard_error[1, 0, 0] = 0.5
    bounds = max_t_bounds(
        estimate,
        replicates,
        alpha=0.2,
        contrast_bounds=(-1.0, 1.0),
        standard_error=standard_error,
        max_statistics=np.full(5, 4.0),
    )
    assert bounds.lower[0, 1, 0] == -1.0
    assert bounds.upper[0, 1, 0] == 1.0
    assert bounds.lower[1, 0, 0] == -1.0
    assert bounds.upper[1, 0, 0] == 1.0
