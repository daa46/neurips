from __future__ import annotations

import json

from hrms.artifacts import sha256_file
from scripts.run_hrms import main


def test_cli_writes_complete_content_addressed_result_bundle(tmp_path):
    output = tmp_path / "hrms"
    assert (
        main(
            [
                "examples/synthetic_losses.csv",
                "--model-registry",
                "examples/synthetic_model_registry.csv",
                "--output",
                str(output),
                "--bootstrap",
                "9",
                "--seed",
                "17",
            ]
        )
        == 0
    )

    expected = {
        "cluster_diagnostics.csv",
        "environment_risks.csv",
        "manifest.json",
        "pairwise_bounds.csv",
        "regret_certificates.csv",
        "robustness_path.csv",
        "selection_stability.csv",
        "summary.json",
    }
    assert {path.name for path in output.iterdir()} == expected
    summary = json.loads((output / "summary.json").read_text())
    assert summary["selected_model"] in summary["model_confidence_set"]
    assert summary["calibration_method"] == "jackknife-pseudovalue cluster bootstrap-t"

    manifest = json.loads((output / "manifest.json").read_text())
    assert manifest["parameters"]["n_bootstrap"] == 9
    assert manifest["parameters"]["calibration"] == "jackknife_bootstrap_t"
    record = manifest["artifacts"]["summary"]
    assert record["sha256"] == sha256_file(output / "summary.json")
