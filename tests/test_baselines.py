import numpy as np

from hrms.ambiguity import ChiSquareBall
from hrms.baselines import (
    environment_cvar_scores,
    nominal_mean_scores,
    point_dro_regret_scores,
    point_dro_risk_scores,
    worst_environment_scores,
)


def test_nominal_and_worst_environment_scores():
    risks = np.asarray([[0.1, 0.4], [0.5, 0.2]])
    assert np.allclose(nominal_mean_scores(risks), [0.3, 0.3])
    assert np.allclose(worst_environment_scores(risks), [0.5, 0.4])
    assert np.allclose(environment_cvar_scores(risks, 0.5), [0.5, 0.4])


def test_full_simplex_robust_regret_is_not_worst_absolute_risk():
    risks = np.asarray([[0.2, 0.3], [0.7, 0.6]])
    ambiguity = ChiSquareBall.full_simplex()
    absolute = point_dro_risk_scores(risks, ambiguity)
    regret = point_dro_regret_scores(risks, ambiguity)
    assert np.allclose(absolute, [0.7, 0.6])
    assert np.allclose(regret, [0.1, 0.1])
