from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import pytest

from applications.te_regulation.cluster_and_split import TEBenchmarkError, read_fasta
from applications.te_regulation.freeze_cluster_sample import (
    DEFAULT_CONFIG,
    freeze_cluster_sample,
    select_final_pairs,
    validate_gate_evidence,
)
from hrms.artifacts import sha256_file


def test_proportional_capped_selection_is_permutation_invariant() -> None:
    pairs = pd.DataFrame(
        [
            {"pair_id": f"pair_{index}", "environment_id": "env", "rep_name": family}
            for index, family in enumerate(["a"] * 5 + ["b"] * 3 + ["c"])
        ]
    )
    first, audit = select_final_pairs(
        pairs,
        seed="fixture",
        target_pairs_per_environment=5,
        maximum_pairs_per_subfamily=2,
    )
    second, _ = select_final_pairs(
        pairs.sample(frac=1, random_state=8),
        seed="fixture",
        target_pairs_per_environment=5,
        maximum_pairs_per_subfamily=2,
    )

    assert set(first["pair_id"]) == set(second["pair_id"])
    assert first.groupby("rep_name").size().to_dict() == {"a": 2, "b": 2, "c": 1}
    assert audit["selected_pairs"].sum() == 5
    assert audit["selected_pairs"].max() <= 2


def _artifact(path: Path) -> dict[str, object]:
    return {
        "path": path.as_posix(),
        "size_bytes": path.stat().st_size,
        "sha256": sha256_file(path),
    }


def _window_fixture(path: Path, *, two_environments: bool = False) -> Path:
    path.mkdir()
    pair_rows = []
    locus_rows = []
    te_records = []
    window_records = []
    for index in range(4):
        pair_id = f"pair_{index}"
        family = "A" if index < 3 else "B"
        environment = "drop" if two_environments and index < 2 else (
            "keep" if two_environments else "env"
        )
        pair_rows.append(
            {"pair_id": pair_id, "environment_id": environment, "rep_name": family}
        )
        for label in (0, 1):
            observation = f"obs_{index}_{label}"
            locus_rows.append(
                {
                    "observation_id": observation,
                    "pair_id": pair_id,
                    "environment_id": environment,
                    "rep_name": family,
                    "label": label,
                    "sample_weight": 1.0,
                }
            )
            te_records.append(f">{observation}\n{'ACGT' * 30}\n")
            window_records.append(f">{observation}\n{'ACGT' * 128}\n")
    paths = {
        "cluster_input_pairs": path / "cluster_input_pairs.csv.gz",
        "cluster_input_loci": path / "cluster_input_loci.csv.gz",
        "cluster_input_te_sequences": path / "cluster_input_te_sequences.oriented.fa",
        "cluster_input_windows": path / "cluster_input_windows.oriented.fa",
    }
    pd.DataFrame(pair_rows).to_csv(paths["cluster_input_pairs"], index=False)
    pd.DataFrame(locus_rows).to_csv(paths["cluster_input_loci"], index=False)
    paths["cluster_input_te_sequences"].write_text("".join(te_records), encoding="utf-8")
    paths["cluster_input_windows"].write_text("".join(window_records), encoding="utf-8")
    manifest = {"schema_version": 1, "artifacts": {}}
    for name, artifact_path in paths.items():
        manifest["artifacts"][name] = _artifact(artifact_path)
    manifest_path = path / "cluster_input_manifest.json"
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    return path


def test_manifest_verified_preflight_sample_is_pair_complete_and_immutable(
    tmp_path: Path,
) -> None:
    window_dir = _window_fixture(tmp_path / "windows")
    output_dir = tmp_path / "sample"
    first = freeze_cluster_sample(
        DEFAULT_CONFIG,
        window_dir,
        output_dir,
        purpose="homology-preflight",
        target_pairs_per_environment=2,
    )
    second = freeze_cluster_sample(
        DEFAULT_CONFIG,
        window_dir,
        output_dir,
        purpose="homology-preflight",
        target_pairs_per_environment=2,
    )

    assert first == second
    selected_pairs = pd.read_csv(first["pairs"])
    selected_loci = pd.read_csv(first["loci"])
    assert len(selected_pairs) == 2
    assert len(selected_loci) == 4
    assert selected_loci.groupby("pair_id")["label"].nunique().eq(2).all()
    fasta_ids = set(read_fasta(first["window_fasta"]))
    assert fasta_ids == set(selected_loci["observation_id"])
    manifest = json.loads(first["manifest"].read_text(encoding="utf-8"))
    assert manifest["purpose"] == "homology-preflight"
    assert manifest["final_sample_frozen"] is False
    assert manifest["parameters"]["effective_target_pairs_per_environment"] == 2


def test_production_sample_rejects_target_override(tmp_path: Path) -> None:
    with pytest.raises(TEBenchmarkError, match="allowed only"):
        freeze_cluster_sample(
            DEFAULT_CONFIG,
            tmp_path,
            tmp_path / "output",
            purpose="production",
            target_pairs_per_environment=500,
        )


def _gate_evidence(path: Path, *, environment: str, fraction: float) -> Path:
    total = 100
    observations = int(round(total * fraction))
    pd.DataFrame(
        [
            {
                "environment_id": environment,
                "root_cluster_id": "root",
                "observations": observations,
                "design_pairs": observations // 2,
                "labels": 2,
                "environment_observations": total,
                "environment_fraction": observations / total,
            }
        ]
    ).to_csv(path, index=False, lineterminator="\n")
    return path


def test_gate_evidence_rejects_hash_mismatch_and_passing_exclusion(tmp_path: Path) -> None:
    evidence = _gate_evidence(tmp_path / "passing.csv", environment="passer", fraction=0.04)
    with pytest.raises(TEBenchmarkError, match="hash mismatch"):
        validate_gate_evidence(
            evidence,
            expected_sha256="0" * 64,
            excluded_environments=["passer"],
            available_environments={"passer"},
            maximum_root_fraction=0.05,
        )
    with pytest.raises(TEBenchmarkError, match="pass the frozen"):
        validate_gate_evidence(
            evidence,
            expected_sha256=sha256_file(evidence),
            excluded_environments=["passer"],
            available_environments={"passer"},
            maximum_root_fraction=0.05,
        )


def test_production_exclusion_is_pinned_to_failing_pilot_evidence(tmp_path: Path) -> None:
    window_dir = _window_fixture(tmp_path / "windows", two_environments=True)
    evidence = _gate_evidence(tmp_path / "gate.csv", environment="drop", fraction=0.8)
    outputs = freeze_cluster_sample(
        DEFAULT_CONFIG,
        window_dir,
        tmp_path / "production",
        purpose="production",
        gate_evidence_path=evidence,
        gate_evidence_sha256=sha256_file(evidence),
        exclude_environments=["drop"],
    )

    selected = pd.read_csv(outputs["pairs"])
    assert set(selected["environment_id"]) == {"keep"}
    assert len(selected) == 2
    decision = pd.read_csv(outputs["gate_decision"])
    assert decision.loc[decision["environment_id"].eq("drop"), "decision"].item() == (
        "exclude"
    )
    manifest = json.loads(outputs["manifest"].read_text(encoding="utf-8"))
    assert manifest["final_sample_frozen"] is True
    assert manifest["config"]["sha256"] == sha256_file(DEFAULT_CONFIG)
    assert manifest["gate_decision"]["excluded_environments"] == ["drop"]
    assert manifest["gate_decision"]["retained_environments"] == ["keep"]
    assert manifest["gate_decision"]["evidence"]["sha256"] == sha256_file(evidence)
