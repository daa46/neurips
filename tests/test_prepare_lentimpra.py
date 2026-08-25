from __future__ import annotations

import gzip
import hashlib
import json
from pathlib import Path

import pandas as pd
import pytest

from applications.te_regulation.prepare_lentimpra import (
    ACTIVITY_CALL_STATUS,
    ACTIVITY_COLUMNS,
    DESIGN_COLUMNS,
    LentiMPRAError,
    build_te_activity_tables,
    load_config,
    write_outputs,
)


def _config() -> dict:
    return {
        "claim_boundary": (
            "Sequence-intrinsic activity averaged over random lentiviral integration contexts; "
            "not endogenous causal regulation."
        ),
        "construction": {
            "assembly": "GRCh38",
            "minimum_single_repeat_overlap_fraction_of_assayed_element": 0.5,
            "allowed_repeat_classes": ["DNA", "LINE", "LTR", "SINE"],
            "environment_column": "te_family",
            "root_cluster_status": "unassigned_until_global_sequence_homology_clustering",
            "split_status": "forbidden_until_root_cluster_assignment",
        },
        "activity_call_fdr05": {
            "status": ACTIVITY_CALL_STATUS,
            "evidence": "fixture has no element-level inferential call",
            "parser_behavior": "emit null",
        },
        "encode": {"k562_large_scale_experiment": "ENCSR382BVV"},
    }


def _design() -> pd.DataFrame:
    adaptor = "A" * 15
    element = "C" * 200
    sequence = adaptor + element + adaptor
    rows = [
        ("element_a", "potential enhancer", "chr1", 100, 200, "+", sequence),
        ("element_a_Reversed:", "potential enhancer", "chr1", 100, 200, "-", sequence),
        ("element_b", "promoter", "chr1", 300, 400, "+", sequence),
        ("element_c", "promoter", "chr1", 500, 600, "+", sequence),
    ]
    return pd.DataFrame(rows, columns=DESIGN_COLUMNS)


def _activity() -> pd.DataFrame:
    rows = [
        ("element_a", 1.0, 2.0, 3.0, 2.0),
        ("element_a_Reversed:", 1.5, 2.5, 3.5, 2.5),
        ("element_b", -1.0, 0.0, 1.0, 0.0),
        ("element_c", -3.0, -2.0, -1.0, -2.0),
    ]
    return pd.DataFrame(rows, columns=ACTIVITY_COLUMNS)


def _repeat_row(
    *,
    start: int,
    end: int,
    subfamily: str,
    repeat_class: str,
    family: str,
    repeat_id: str,
) -> str:
    fields = [
        "0",
        "100",
        "0",
        "0",
        "0",
        "chr1",
        str(start),
        str(end),
        "0",
        "+",
        subfamily,
        repeat_class,
        family,
        "0",
        str(end - start),
        "0",
        repeat_id,
    ]
    return "\t".join(fields) + "\n"


def _write_repeats(path: Path, rows: list[str]) -> Path:
    with gzip.open(path, "wt", encoding="utf-8", newline="") as handle:
        handle.writelines(rows)
    return path


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_build_preserves_continuous_activity_and_never_invents_fdr_call(tmp_path):
    repeats = _write_repeats(
        tmp_path / "rmsk.txt.gz",
        [
            # 80% of both assayed orientations: qualifying.
            _repeat_row(
                start=120,
                end=200,
                subfamily="L1HS",
                repeat_class="LINE",
                family="L1",
                repeat_id="1",
            ),
            # Exactly 50% of element_b: qualifying by the frozen inclusive boundary.
            _repeat_row(
                start=350,
                end=400,
                subfamily="AluY",
                repeat_class="SINE",
                family="Alu",
                repeat_id="2",
            ),
            # A full overlap from a filtered class must not make an assayed element eligible.
            _repeat_row(
                start=300,
                end=400,
                subfamily="(CA)n",
                repeat_class="Simple_repeat",
                family="Simple_repeat",
                repeat_id="3",
            ),
            # 49% of element_c: non-qualifying.
            _repeat_row(
                start=551,
                end=600,
                subfamily="LTR5",
                repeat_class="LTR",
                family="ERVK",
                repeat_id="4",
            ),
        ],
    )
    elements, long, diagnostics = build_te_activity_tables(
        _design(), _activity(), repeats, _config()
    )

    assert set(elements["observation_id"]) == {
        "element_a",
        "element_a_Reversed:",
        "element_b",
    }
    by_name = elements.set_index("observation_id")
    assert by_name.loc["element_a", "activity_mean_log2_rna_dna"] == 2.0
    assert by_name.loc["element_a_Reversed:", "activity_mean_log2_rna_dna"] == 2.5
    assert by_name.loc["element_b", "repeat_overlap_fraction_of_assayed_element"] == 0.5
    assert by_name.loc["element_b", "te_subfamily"] == "AluY"
    assert (elements["activity_active_fdr05"] == "").all()
    assert (elements["activity_call_status"] == ACTIVITY_CALL_STATUS).all()
    assert (elements["root_cluster_id"] == "").all()
    assert elements["element_sequence_200nt"].str.len().eq(200).all()
    assert len(long) == 4 * len(elements)
    assert set(long["activity_statistic"]) == {
        "mean",
        "replicate_1",
        "replicate_2",
        "replicate_3",
    }
    assert diagnostics["materialized_fdr05_calls"] == 0
    assert diagnostics["assayed_elements_with_qualifying_repeat"] == 3


def test_repeat_tie_break_is_independent_of_repeatmasker_record_order(tmp_path):
    design = _design().iloc[[0]].reset_index(drop=True)
    activity = _activity().iloc[[0]].reset_index(drop=True)
    alpha = _repeat_row(
        start=120,
        end=180,
        subfamily="Alpha",
        repeat_class="LINE",
        family="L1",
        repeat_id="2",
    )
    zeta = _repeat_row(
        start=120,
        end=180,
        subfamily="Zeta",
        repeat_class="LINE",
        family="L1",
        repeat_id="1",
    )
    first = _write_repeats(tmp_path / "first.gz", [zeta, alpha])
    second = _write_repeats(tmp_path / "second.gz", [alpha, zeta])

    observed = []
    for path in (first, second):
        elements, _, _ = build_te_activity_tables(design, activity, path, _config())
        observed.append(elements.loc[0, "te_subfamily"])
        assert elements.loc[0, "qualified_repeat_hit_count"] == 2
    assert observed == ["Alpha", "Alpha"]


def test_duplicate_names_and_schema_drift_fail_closed(tmp_path):
    repeats = _write_repeats(tmp_path / "rmsk.gz", [])
    duplicate = pd.concat([_design(), _design().iloc[[0]]], ignore_index=True)
    with pytest.raises(LentiMPRAError, match="duplicate element name"):
        build_te_activity_tables(duplicate, _activity(), repeats, _config())

    changed = _activity().rename(columns={"mean": "average"})
    with pytest.raises(LentiMPRAError, match="unexpected activity columns"):
        build_te_activity_tables(_design(), changed, repeats, _config())


def test_outputs_are_immutable_deterministic_and_block_hrms(tmp_path):
    repeats = _write_repeats(
        tmp_path / "rmsk.gz",
        [
            _repeat_row(
                start=120,
                end=200,
                subfamily="L1HS",
                repeat_class="LINE",
                family="L1",
                repeat_id="1",
            )
        ],
    )
    elements, long, diagnostics = build_te_activity_tables(
        _design(), _activity(), repeats, _config()
    )
    output = tmp_path / "output"
    first = write_outputs(elements, long, diagnostics, _config(), output)
    first_hashes = {name: _sha256(path) for name, path in first.items()}
    repeated = write_outputs(elements, long, diagnostics, _config(), output)
    assert {name: _sha256(path) for name, path in repeated.items()} == first_hashes

    preflight = json.loads(first["preflight"].read_text())
    assert preflight["gates"]["ready_for_homology_clustering"] is True
    assert preflight["gates"]["ready_for_split"] is False
    assert preflight["gates"]["ready_for_hrms"] is False
    assert preflight["output_contract"]["activity_active_fdr05"]["all_values_null"] is True


def test_repository_config_pins_verified_processed_source_and_exact_headers():
    config = load_config(Path("configs/te/lentimpra_k562.json"))
    archive = config["sources"]["processed_tables_archive"]
    assert archive["size_bytes"] == 342234089
    assert archive["sha256"] == (
        "aefb1974e9ebe5c196a89a00f4e868c0abaf69e3d3952e79c74bfb42f217f6a4"
    )
    assert tuple(config["archive_members"]["design"]["expected_columns"]) == DESIGN_COLUMNS
    assert tuple(config["archive_members"]["activity"]["expected_columns"]) == ACTIVITY_COLUMNS
    assert config["activity_call_fdr05"]["status"] == ACTIVITY_CALL_STATUS
