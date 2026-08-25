import pandas as pd
import pytest

from hrms.schema import SchemaError, validate_loss_table


def test_schema_builds_complete_paired_matrix(loss_table):
    data = validate_loss_table(loss_table)
    assert data.losses.shape == (36, 3)
    assert len(data.environment_ids) == 2
    assert len(data.cluster_ids) == 12


def test_schema_rejects_missing_candidate(loss_table):
    bad = loss_table.drop(index=loss_table.index[0])
    with pytest.raises(SchemaError, match="complete frozen candidate library"):
        validate_loss_table(bad)


def test_schema_rejects_duplicate_observation_model(loss_table):
    bad = pd.concat([loss_table, loss_table.iloc[[0]]], ignore_index=True)
    with pytest.raises(SchemaError, match="duplicate"):
        validate_loss_table(bad)
