from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import pandas as pd
import pytest

from applications.te_regulation.cluster_and_split import (
    TEBenchmarkError,
    _audit_assignment,
    assign_stratified_root_splits,
    run_mmseqs_connected_components,
    union_sequence_components,
    validate_loci_and_sequences,
)
from hrms.splits import assert_no_cluster_leakage


def _singleton_loci(environments: int = 2, roots_per_environment: int = 120) -> pd.DataFrame:
    rows = []
    for environment in range(environments):
        for root in range(roots_per_environment):
            rows.append(
                {
                    "observation_id": f"obs_{environment}_{root:03d}",
                    "environment_id": f"env_{environment}",
                    "root_cluster_id": f"root_{environment}_{root:03d}",
                    "label": root % 2,
                }
            )
    return pd.DataFrame(rows)


def test_component_union_closes_links_from_both_sequence_views() -> None:
    membership = union_sequence_components(
        ["a", "b", "c", "d"],
        te_pairs=[("a", "b")],
        window_pairs=[("b", "c")],
    ).set_index("observation_id")

    assert membership.loc["a", "te_component_id"] == membership.loc["b", "te_component_id"]
    assert membership.loc["b", "window_component_id"] == membership.loc[
        "c", "window_component_id"
    ]
    assert membership.loc["a", "root_cluster_id"] == membership.loc[
        "c", "root_cluster_id"
    ]
    assert membership.loc["d", "root_cluster_id"] != membership.loc[
        "a", "root_cluster_id"
    ]


def test_design_pair_edges_are_part_of_root_but_keep_separate_component_id() -> None:
    membership = union_sequence_components(
        ["a", "b", "c", "d"],
        te_pairs=[("a", "b")],
        window_pairs=[("b", "c")],
        design_pairs=[("c", "d")],
    ).set_index("observation_id")

    assert membership["root_cluster_id"].nunique() == 1
    assert membership.loc["c", "design_pair_component_id"] == membership.loc[
        "d", "design_pair_component_id"
    ]
    assert membership.loc["a", "design_pair_component_id"] != membership.loc[
        "c", "design_pair_component_id"
    ]
    assert membership.loc["a", "te_component_id"] == membership.loc[
        "b", "te_component_id"
    ]
    assert membership.loc["c", "te_component_id"] != membership.loc[
        "d", "te_component_id"
    ]


def test_mmseqs_release18_command_freezes_nucleotide_80_80_settings(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    executable = tmp_path / "mmseqs"
    executable.touch()
    calls = []

    def fake_run(command, **_kwargs):
        calls.append(command)
        query_id = Path(command[2]).read_text(encoding="utf-8").splitlines()[0][1:]
        target_id = Path(command[3]).read_text(encoding="utf-8").splitlines()[0][1:]
        Path(command[4]).write_text(
            f"{query_id}\t{target_id}\t85.0\t0.9\t0.9\n",
            encoding="utf-8",
        )
        return SimpleNamespace(returncode=0, stderr="", stdout="")

    monkeypatch.setattr("applications.te_regulation.cluster_and_split.subprocess.run", fake_run)
    result = run_mmseqs_connected_components(
        {"a": "ACGT" * 30, "b": "ACGT" * 30},
        mmseqs_path=executable,
        min_identity=0.8,
        min_coverage=0.8,
        query_chunk_size=1,
        target_shard_size=1,
        checkpoint_dir=tmp_path / "checkpoint",
    )
    resumed = run_mmseqs_connected_components(
        {"a": "ACGT" * 30, "b": "ACGT" * 30},
        mmseqs_path=executable,
        min_identity=0.8,
        min_coverage=0.8,
        query_chunk_size=1,
        target_shard_size=1,
        checkpoint_dir=tmp_path / "checkpoint",
    )

    assert result.spanning_pairs == [("a", "b")]
    assert resumed.spanning_pairs == result.spanning_pairs
    assert len(calls) == 4
    command = list(result.commands[0])
    assert command[1] == "easy-search"
    assert command[command.index("--search-type") + 1] == "3"
    assert command[command.index("--dbtype") + 1] == "2"
    assert command[command.index("--min-seq-id") + 1] == "0.8"
    assert command[command.index("-c") + 1] == "0.8"
    assert command[command.index("--cov-mode") + 1] == "0"
    assert command[command.index("--alignment-mode") + 1] == "3"
    assert command[command.index("-s") + 1] == "7.5"
    assert command[command.index("--max-seqs") + 1] == "1"
    assert command[command.index("--format-output") + 1] == (
        "query,target,pident,qcov,tcov"
    )
    # Release 18 rejects easy-cluster connected-component mode for nucleotide
    # input, so the production code computes the transitive closure of these
    # audited threshold edges itself.
    assert "--cluster-mode" not in command


def test_stratified_split_is_deterministic_and_passes_hard_gates() -> None:
    loci = _singleton_loci()
    first = assign_stratified_root_splits(loci, seed=17, trials=1024)
    second = assign_stratified_root_splits(loci, seed=17, trials=1024)

    pd.testing.assert_frame_equal(first.split_table, second.split_table)
    audited = loci.merge(first.split_table, on=["observation_id", "root_cluster_id"])
    assert_no_cluster_leakage(audited)
    root_counts = first.root_counts.set_index(["environment_id", "split"])["roots"]
    assert (root_counts.xs("select", level="split") >= 20).all()
    assert (root_counts.xs("test", level="split") >= 20).all()
    labels = audited.groupby(["environment_id", "split"], observed=True)["label"].nunique()
    assert (labels == 2).all()
    assert audited.groupby("root_cluster_id")["split"].nunique().max() == 1


def test_split_rejects_dominant_root() -> None:
    loci = _singleton_loci(environments=1, roots_per_environment=120)
    duplicated = pd.concat(
        [loci, *[loci.iloc[[0]].assign(observation_id=f"duplicate_{index}") for index in range(8)]],
        ignore_index=True,
    )
    with pytest.raises(TEBenchmarkError, match="largest root"):
        assign_stratified_root_splits(duplicated, seed=1, trials=10)


def test_hard_gate_checks_dominance_inside_each_environment_split() -> None:
    rows = []
    offset = 0
    for split, root_count in (("train", 60), ("select", 21), ("test", 21)):
        for index in range(root_count):
            repeats = 2 if split == "select" and index == 0 else 1
            for repeat in range(repeats):
                rows.append(
                    {
                        "observation_id": f"obs_{offset}_{repeat}",
                        "environment_id": "env",
                        "root_cluster_id": f"root_{offset}",
                        "split": split,
                        "label": index % 2,
                    }
                )
            offset += 1
    assigned = pd.DataFrame(rows)
    assert assigned.groupby("root_cluster_id").size().max() / len(assigned) < 0.05

    with pytest.raises(TEBenchmarkError, match="environment/split"):
        _audit_assignment(
            assigned,
            min_roots_per_environment=20,
            max_root_fraction=0.05,
        )


def test_split_dominance_gate_applies_to_select_and_test_only() -> None:
    rows = []
    for root in range(105):
        repeats = 8 if root == 0 else 2
        for repeat in range(repeats):
            rows.append(
                {
                    "observation_id": f"obs_{root:03d}_{repeat}",
                    "environment_id": "env",
                    "root_cluster_id": f"root_{root:03d}",
                    "label": repeat % 2,
                }
            )
    loci = pd.DataFrame(rows)

    diagnostics = assign_stratified_root_splits(loci, seed=3, trials=1)

    largest = diagnostics.largest_root_fraction.set_index("split")["fraction"]
    assert largest["train"] >= 0.05
    assert largest["select"] < 0.05
    assert largest["test"] < 0.05


def test_sequence_contract_requires_exact_512bp_windows() -> None:
    loci = pd.DataFrame(
        {
            "observation_id": ["a", "b"],
            "pair_id": ["pair", "pair"],
            "environment_id": ["env", "env"],
            "label": [0, 1],
        }
    )
    with pytest.raises(TEBenchmarkError, match="expected 512"):
        validate_loci_and_sequences(
            loci,
            {"a": "ACGT", "b": "ACGT"},
            {"a": "A" * 511, "b": "A" * 512},
        )


def test_sequence_validation_stringifies_unbounded_sha_priority() -> None:
    priorities = [2**200, 2**200 + 1]
    loci = pd.DataFrame(
        {
            "observation_id": ["a", "b"],
            "pair_id": ["pair", "pair"],
            "environment_id": ["env", "env"],
            "label": [0, 1],
            "candidate_priority_sha256_int": priorities,
        }
    )

    validated = validate_loci_and_sequences(
        loci,
        {"a": "ACGT", "b": "ACGT"},
        {"a": "A" * 512, "b": "A" * 512},
    )

    assert validated["candidate_priority_sha256_int"].tolist() == [
        str(value) for value in priorities
    ]
