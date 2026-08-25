import numpy as np
import pandas as pd

from hrms.estimands import estimate_environment_risks
from hrms.schema import validate_loss_table


def test_observation_and_cluster_estimands_differ_under_cluster_imbalance():
    rows = []
    specifications = [("large", 8, 0.8), ("small", 1, 0.0)]
    for cluster, size, loss in specifications:
        for observation in range(size):
            for model, offset in [("a", 0.0), ("b", 0.1)]:
                rows.append(
                    {
                        "observation_id": f"{cluster}_{observation}",
                        "environment_id": "e",
                        "root_cluster_id": cluster,
                        "model_id": model,
                        "loss": min(loss + offset, 1.0),
                        "sample_weight": 1.0,
                    }
                )
    data = validate_loss_table(pd.DataFrame(rows))
    observation = estimate_environment_risks(data, "observation")
    cluster = estimate_environment_risks(data, "cluster")
    assert observation[0, 0] > cluster[0, 0]
    assert np.isclose(cluster[0, 0], 0.4)

