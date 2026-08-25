import numpy as np
import pandas as pd

from hrms.bootstrap import cluster_bootstrap, cluster_jackknife_pseudovalues
from hrms.estimands import cluster_cells
from hrms.schema import validate_loss_table


def test_repeated_runs_with_same_seed_are_exact(loss_table):
    data = validate_loss_table(loss_table)
    first = cluster_bootstrap(data, n_bootstrap=40, random_state=13)
    second = cluster_bootstrap(data, n_bootstrap=40, random_state=13)
    assert np.array_equal(first.bootstrap_risks, second.bootstrap_risks)


def test_global_row_duplication_does_not_add_independent_evidence(loss_table):
    duplicated = pd.concat(
        [
            loss_table.assign(
                observation_id=lambda frame: frame["observation_id"] + "_copy0"
            ),
            loss_table.assign(
                observation_id=lambda frame: frame["observation_id"] + "_copy1"
            ),
        ],
        ignore_index=True,
    )
    original = cluster_bootstrap(
        validate_loss_table(loss_table), n_bootstrap=50, random_state=21
    )
    copied = cluster_bootstrap(
        validate_loss_table(duplicated), n_bootstrap=50, random_state=21
    )
    assert np.allclose(original.point_risks, copied.point_risks)
    assert np.allclose(original.bootstrap_risks, copied.bootstrap_risks)
    assert np.array_equal(original.cluster_counts, copied.cluster_counts)


def test_spanning_root_clusters_trigger_global_resampling():
    rows = []
    for cluster in range(8):
        for environment in range(2):
            observation = f"c{cluster}_e{environment}"
            for model in range(2):
                rows.append(
                    {
                        "observation_id": observation,
                        "environment_id": environment,
                        "root_cluster_id": cluster,
                        "model_id": model,
                        "loss": 0.2 + 0.02 * model + 0.01 * environment,
                        "sample_weight": 1.0,
                    }
                )
    result = cluster_bootstrap(
        validate_loss_table(pd.DataFrame(rows)), n_bootstrap=20, random_state=4
    )
    assert result.resampling == "global"
    blocks = cluster_jackknife_pseudovalues(
        validate_loss_table(pd.DataFrame(rows))
    )
    assert len(blocks) == 1
    assert blocks[0].risk_pseudovalues.shape == (8, 2, 2)
    assert np.array_equal(blocks[0].environment_indices, np.asarray([0, 1]))


def test_studentized_point_scale_matches_paired_cluster_mean_se():
    first = np.asarray([0.1, 0.2, 0.3, 0.4])
    second = np.asarray([0.2, 0.1, 0.5, 0.4])
    rows = []
    for cluster, (loss_first, loss_second) in enumerate(zip(first, second, strict=True)):
        for model, loss in enumerate((loss_first, loss_second)):
            rows.append(
                {
                    "observation_id": f"o{cluster}",
                    "environment_id": "e0",
                    "root_cluster_id": f"c{cluster}",
                    "model_id": f"m{model}",
                    "loss": loss,
                    "sample_weight": 1.0,
                }
            )
    result = cluster_bootstrap(
        validate_loss_table(pd.DataFrame(rows)), n_bootstrap=100, random_state=9
    )
    expected = np.std(first - second, ddof=1) / np.sqrt(len(first))
    assert np.isclose(result.pairwise_standard_error[0, 1, 0], expected)
    assert result.studentized_max_statistics is None
    studentized = cluster_bootstrap(
        validate_loss_table(pd.DataFrame(rows)),
        n_bootstrap=100,
        studentize=True,
        random_state=9,
    )
    assert np.all(studentized.studentized_max_statistics >= 0)


def test_cluster_mean_jackknife_pseudovalues_equal_cluster_means(loss_table):
    data = validate_loss_table(loss_table)
    blocks = cluster_jackknife_pseudovalues(data, weighting="cluster")
    cells = cluster_cells(data)
    assert len(blocks) == len(cells)
    for block, cell in zip(blocks, cells, strict=True):
        assert np.allclose(block.risk_pseudovalues[:, 0, :], cell.cluster_means)


def test_singleton_clusters_match_flat_paired_bootstrap():
    rng = np.random.default_rng(31)
    losses = rng.uniform(0.0, 1.0, size=(12, 3))
    rows = []
    for observation in range(len(losses)):
        for model in range(losses.shape[1]):
            rows.append(
                {
                    "observation_id": f"o{observation:02d}",
                    "environment_id": "e0",
                    "root_cluster_id": f"o{observation:02d}",
                    "model_id": f"m{model}",
                    "loss": losses[observation, model],
                    "sample_weight": 1.0,
                }
            )
    data = validate_loss_table(pd.DataFrame(rows))
    result = cluster_bootstrap(data, n_bootstrap=25, random_state=44)
    flat_rng = np.random.default_rng(44)
    expected = np.stack(
        [
            losses[flat_rng.integers(0, len(losses), size=len(losses))].mean(axis=0)
            for _ in range(25)
        ]
    )
    assert np.allclose(result.bootstrap_risks[:, 0, :], expected)
