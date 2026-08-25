from __future__ import annotations

from pathlib import Path

import pandas as pd

from hrms.registry import assert_prediction_library_matches, validate_model_registry
from hrms.schema import validate_loss_table
from scripts.generate_synthetic_example import main


def test_committed_synthetic_example_is_reproducible_and_valid(tmp_path):
    loss_path = tmp_path / "losses.csv"
    registry_path = tmp_path / "registry.csv"
    assert main(["--output", str(loss_path), "--registry", str(registry_path)]) == 0

    assert loss_path.read_bytes() == Path("examples/synthetic_losses.csv").read_bytes()
    assert registry_path.read_bytes() == Path("examples/synthetic_model_registry.csv").read_bytes()

    losses = pd.read_csv(loss_path)
    registry = pd.read_csv(registry_path)
    validated = validate_loss_table(losses)
    validate_model_registry(registry)
    assert_prediction_library_matches(registry, validated.model_ids)
    assert validated.n_observations == 54
    assert validated.n_models == 4
