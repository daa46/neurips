import numpy as np

from hrms.pairwise import pairwise_differences


def test_pairwise_bounds_are_antisymmetric():
    risks = np.asarray([[0.1, 0.2, 0.3], [0.4, 0.3, 0.2]])
    differences = pairwise_differences(risks)
    assert np.allclose(differences, -np.swapaxes(differences, 0, 1))
    assert np.allclose(np.diagonal(differences, axis1=0, axis2=1), 0)


def test_paired_difference_uses_common_environment_risks():
    risks = np.asarray([[0.1, 0.4], [0.2, 0.3]])
    differences = pairwise_differences(risks)
    assert np.allclose(differences[0, 1], [-0.3, -0.1])

