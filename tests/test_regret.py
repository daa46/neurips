import numpy as np

from hrms.confidence_set import robust_model_confidence_set, select_minimum_upper
from hrms.regret import point_robust_regret, robust_regret_certificates


class FullSimplex:
    @staticmethod
    def support(values, *, return_weights=False):
        values = np.asarray(values)
        index = int(np.argmax(values))
        if return_weights:
            weights = np.zeros_like(values, dtype=float)
            weights[index] = 1.0
            return float(values[index]), weights
        return float(values[index])


def test_rank_reversal_can_make_every_fixed_model_have_positive_robust_regret():
    risks = np.asarray([[0.0, 1.0], [1.0, 0.0]])  # environment x model
    pairwise = np.transpose(risks, (1, 0))[:, None, :] - np.transpose(risks, (1, 0))[None]
    regret = point_robust_regret(pairwise, FullSimplex())
    assert np.allclose(regret, [1.0, 1.0])


def test_robust_regret_sandwich_and_confidence_set_include_true_optimum():
    delta = np.asarray(
        [
            [[0.0, 0.0], [-0.2, 0.1], [-0.1, -0.1]],
            [[0.2, -0.1], [0.0, 0.0], [0.1, -0.2]],
            [[0.1, 0.1], [-0.1, 0.2], [0.0, 0.0]],
        ]
    )
    lower, upper = delta - 0.05, delta + 0.05
    for model in range(3):
        lower[model, model] = upper[model, model] = 0
    truth = point_robust_regret(delta, FullSimplex())
    bounds = robust_regret_certificates(lower, upper, FullSimplex())
    assert np.all(bounds.lower <= truth)
    assert np.all(truth <= bounds.upper)
    confidence = robust_model_confidence_set(bounds.lower, bounds.upper)
    true_optima = np.flatnonzero(truth == truth.min())
    assert set(true_optima).issubset(confidence)
    assert select_minimum_upper(bounds.upper) in confidence


def test_upper_certificate_ties_use_point_regret_then_first_index():
    upper = np.asarray([1.0, 1.0, 1.0])
    point = np.asarray([0.7, 0.2, 0.2])
    assert select_minimum_upper(upper, tie_breaker=point) == 1
