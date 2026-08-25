import pandas as pd
import pytest

from hrms.schema import SchemaError
from hrms.splits import (
    assert_no_cluster_leakage,
    assign_root_cluster_splits,
    split_assignment_hash,
)


def test_no_root_cluster_crosses_train_select_test(loss_table):
    observations = loss_table.drop_duplicates("observation_id")
    splits = assign_root_cluster_splits(observations, random_state=19)
    assert_no_cluster_leakage(splits)
    assert set(splits["split"]) == {"train", "select", "test"}
    assert (splits.groupby("root_cluster_id")["split"].nunique() == 1).all()
    assert splits["split_hash"].nunique() == 1
    assert splits["split_hash"].iat[0] == split_assignment_hash(splits)


def test_split_hash_is_invariant_to_observation_order_and_duplicates(loss_table):
    observations = loss_table.drop_duplicates("observation_id")
    splits = assign_root_cluster_splits(observations, random_state=19)
    permuted = pd.concat(
        [splits.sample(frac=1.0, random_state=99), splits.iloc[[0]]],
        ignore_index=True,
    )
    assert split_assignment_hash(permuted) == split_assignment_hash(splits)


def test_leakage_check_fails_loudly():
    bad = pd.DataFrame(
        {
            "root_cluster_id": ["c1", "c1"],
            "split": ["train", "test"],
        }
    )
    with pytest.raises(SchemaError, match="cross partitions"):
        assert_no_cluster_leakage(bad)


def test_leakage_check_rejects_null_identifiers():
    bad = pd.DataFrame({"root_cluster_id": ["c1", None], "split": ["train", "test"]})
    with pytest.raises(SchemaError, match="cannot be null"):
        assert_no_cluster_leakage(bad)
