from __future__ import annotations

import numpy as np
import pandas as pd
import pytest


def make_loss_table(
    *,
    n_environments: int = 2,
    clusters_per_environment: int = 6,
    observations_per_cluster: int = 3,
    n_models: int = 3,
    seed: int = 7,
) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    rows = []
    for environment in range(n_environments):
        for cluster in range(clusters_per_environment):
            cluster_id = f"e{environment}_c{cluster}"
            cluster_effect = rng.normal(0.0, 0.04)
            for observation in range(observations_per_cluster):
                observation_id = f"{cluster_id}_o{observation}"
                base = 0.18 + 0.03 * environment + cluster_effect + rng.normal(0, 0.015)
                for model in range(n_models):
                    loss = np.clip(base + 0.025 * model - 0.02 * model * environment, 0, 1)
                    rows.append(
                        {
                            "observation_id": observation_id,
                            "environment_id": f"env_{environment}",
                            "root_cluster_id": cluster_id,
                            "model_id": f"model_{model}",
                            "loss": loss,
                            "sample_weight": 1.0,
                        }
                    )
    return pd.DataFrame(rows)


@pytest.fixture
def loss_table() -> pd.DataFrame:
    return make_loss_table()

