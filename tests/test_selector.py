import numpy as np
import pandas as pd
import pytest

from hrms import HRMS
from hrms.ambiguity import ChiSquareBall


def test_end_to_end_returns_selected_model_and_confidence_set(loss_table):
    result = HRMS(
        ambiguity=ChiSquareBall(radius=0.25),
        n_bootstrap=100,
        stability_replicates=30,
        random_state=42,
    ).fit(loss_table)
    assert result.selected_model in result.model_confidence_set
    assert (result.robust_regret_lower <= result.robust_regret_upper).all()
    assert np.isclose(result.selection_stability.sum(), 1.0)
    assert result.pairwise_bounds["environment"].nunique() == 2
    assert result.calibration_method == "jackknife-pseudovalue cluster bootstrap-t"


def test_candidate_and_environment_order_permutations_preserve_results(loss_table):
    selector = HRMS(n_bootstrap=80, stability_replicates=20, random_state=8)
    original = selector.fit(loss_table)
    permuted = selector.fit(loss_table.sample(frac=1.0, random_state=999).reset_index(drop=True))
    pd.testing.assert_series_equal(original.robust_regret_lower, permuted.robust_regret_lower)
    pd.testing.assert_series_equal(original.robust_regret_upper, permuted.robust_regret_upper)
    assert original.selected_model == permuted.selected_model


def test_upper_certificate_is_nondecreasing_in_radius(loss_table):
    result = HRMS(
        n_bootstrap=80,
        stability_replicates=20,
        robustness_radii=(0.0, 0.1, 0.25, 0.5, 1.0),
        random_state=2,
    ).fit(loss_table)
    for _, model_path in result.robustness_path.groupby("model", sort=False):
        assert np.all(np.diff(model_path["upper"].to_numpy()) >= -1e-10)


@pytest.mark.parametrize(
    ("calibration", "method"),
    [
        ("fixed_max_t", "ordinary cluster bootstrap max-t"),
        ("raw_bootstrap_t", "studentized root-cluster bootstrap-t"),
    ],
)
def test_calibration_ablation_is_explicit(loss_table, calibration, method):
    result = HRMS(
        calibration=calibration,
        n_bootstrap=30,
        stability_replicates=5,
        robustness_radii=(0.25,),
        random_state=6,
    ).fit(loss_table)
    assert result.calibration_method == method


def test_unknown_calibration_is_rejected():
    with pytest.raises(ValueError, match="calibration must be"):
        HRMS(calibration="unknown")


def test_declared_structural_duplicate_requires_exact_loss_vector(loss_table):
    duplicate_table = loss_table.copy()
    model_0 = duplicate_table["model_id"].eq("model_0")
    model_1 = duplicate_table["model_id"].eq("model_1")
    model_0_losses = duplicate_table.loc[model_0, "loss"].to_numpy()
    duplicate_table.loc[model_1, "loss"] = model_0_losses

    changed_row = duplicate_table.index[model_1][0]
    duplicate_table.loc[changed_row, "loss"] = np.nextafter(
        duplicate_table.loc[changed_row, "loss"], 1.0
    )

    with pytest.raises(ValueError, match="exactly equal observation-wise loss vectors"):
        HRMS(
            structural_duplicate_pairs=(("model_0", "model_1"),),
            n_bootstrap=20,
            random_state=9,
        ).fit(duplicate_table)


def test_exact_declared_structural_duplicate_has_zero_bounds(loss_table):
    duplicate_table = loss_table.copy()
    model_0 = duplicate_table["model_id"].eq("model_0")
    model_1 = duplicate_table["model_id"].eq("model_1")
    duplicate_table.loc[model_1, "loss"] = duplicate_table.loc[model_0, "loss"].to_numpy()

    result = HRMS(
        structural_duplicate_pairs=(("model_0", "model_1"),),
        n_bootstrap=20,
        stability_replicates=10,
        robustness_radii=(0.25,),
        random_state=9,
    ).fit(duplicate_table)
    pair = result.pairwise_bounds.query("model == 'model_0' and comparator == 'model_1'")
    assert np.count_nonzero(pair[["estimate", "lower", "upper"]].to_numpy()) == 0
