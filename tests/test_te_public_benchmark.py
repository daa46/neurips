from __future__ import annotations

import csv
import gzip
import hashlib
import json
from pathlib import Path

import pytest

from applications.te_regulation.public_data import (
    PublicDataError,
    SourceSpec,
    build_candidate_pool,
    build_precluster_matches,
    download_source,
    extract_candidate_te_sequences,
    extract_precluster_windows,
    proportional_capped_quotas,
    read_fasta,
    run_preflight,
)


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _write_fixture_config(tmp_path: Path) -> Path:
    ccre = tmp_path / "ccre.bed"
    ccre.write_text(
        "chr1\t100\t200\ta1\te1\tpELS\n"
        "chr1\t400\t500\ta2\te2\tCA\n"
        "chr2\t100\t200\ta3\te3\tdELS\n"
    )
    repeats = tmp_path / "repeats.bed"
    repeats.write_text(
        # Exactly 50% of the pELS cCRE is covered: positive. The reciprocal
        # TE-overlap fraction is lower, guarding the direction of the rule.
        "chr1\t150\t350\tAluA\t1\t+\n"
        # The buffered interval touches a CA cCRE, so this is not a negative.
        "chr1\t600\t800\tAluA\t1\t-\n"
        "chr1\t1000\t1200\tAluA\t1\t+\n"
        "chr2\t100\t250\tAluB\t1\t-\n"
        "chr2\t1000\t1150\tAluB\t1\t+\n"
        "chr1\t1300\t1350\tAluA\t1\t+\n"
        "chr1\t1400\t1600\tOther\t1\t+\n"
        "chrUn\t100\t300\tAluA\t1\t+\n"
    )
    mapping = tmp_path / "rmsk.txt.gz"
    with gzip.open(mapping, "wt") as handle:
        for index, (name, family) in enumerate(
            [("AluA", "Alu"), ("AluB", "Alu"), ("Other", "L1")], 1
        ):
            handle.write(
                f"0\t1\t0\t0\t0\tchr1\t{index}\t{index + 1}\t0\t+\t{name}"
                f"\tSINE\t{family}\t0\t1\t0\t{index}\n"
            )
    genome = tmp_path / "hg38.2bit"
    genome.write_bytes(b"fixture-not-a-real-two-bit")
    tool = tmp_path / "fake_twoBitToFa"
    tool.write_text(
        "#!/usr/bin/env python3\n"
        "import sys\n"
        "bed = next(value[5:] for value in sys.argv if value.startswith('-bed='))\n"
        "with open(sys.argv[2], 'w') as output, open(bed) as rows:\n"
        "    for line in rows:\n"
        "        chrom, start, end, name, score, strand = line.rstrip().split('\\t')\n"
        "        length = int(end) - int(start)\n"
        "        output.write(f'>{name}\\n' + 'A' * (length - 1) + 'C\\n')\n"
    )
    tool.chmod(0o755)

    sources = {}
    for name, path in {
        "encode_registry_v4": ccre,
        "ucsc_repeatbrowser_2020": repeats,
        "ucsc_rmsk_out_current": mapping,
        "ucsc_hg38_2bit": genome,
    }.items():
        sources[name] = {
            "url": f"https://example.test/{path.name}",
            "path": path.name,
            "size_bytes": path.stat().st_size,
            "sha256": _sha(path),
        }
    config = {
        "schema_version": 1,
        "claim_boundary": "fixture annotation task",
        "sources": sources,
        "coordinates": {
            "assembly": "hg38",
            "canonical_chromosomes": ["chr1", "chr2"],
            "chromosome_sizes": {"chr1": 2000, "chr2": 2000},
        },
        "labels": {
            "positive_ccre_classes": ["pELS", "dELS"],
            "positive_minimum_fraction_of_ccre_overlapped_by_one_te": 0.5,
            "negative_no_any_ccre_buffer_bp": 250,
            "minimum_te_length_bp": 100,
        },
        "environments": ["Alu", "L1"],
        "sampling": {
            "seed": "fixture",
            "target_pairs_per_environment": 2,
            "candidate_pool_multiplier": 2,
            "maximum_candidate_pairs_per_subfamily": 2,
            "candidate_quota_rule": "fixture proportional cap",
            "maximum_ambiguous_base_fraction": 0.01,
            "gc_denominator": "A+C+G+T; ambiguous bases excluded",
            "maximum_gc_fraction_difference": 0.03,
            "maximum_relative_te_length_difference": 0.20,
            "match_exact_fields": ["environment_id", "rep_name", "chrom"],
        },
        "sequences": {
            "two_bit_to_fa": tool.name,
            "no_mask": True,
            "orientation": "fixture TE orientation",
            "model_window_bp": 64,
            "negative_anchor": "matched strand-aware normalized position",
            "boundary_rule": "drop non-exact windows",
        },
        "clustering": {},
        "splits": {},
        "gates": {},
    }
    config_path = tmp_path / "config.json"
    config_path.write_text(json.dumps(config))
    return config_path


def test_preflight_applies_directional_positive_and_strict_negative_rules(tmp_path):
    config = _write_fixture_config(tmp_path)
    outputs = run_preflight(config, tmp_path / "out", base_dir=tmp_path)

    with gzip.open(outputs["eligible_loci"], "rt") as handle:
        loci = list(csv.DictReader(handle))
    observed = {(row["rep_name"], row["chrom"], int(row["start"])): row for row in loci}
    assert set(observed) == {
        ("AluA", "chr1", 150),
        ("AluA", "chr1", 1000),
        ("AluB", "chr2", 100),
        ("AluB", "chr2", 1000),
        ("Other", "chr1", 1400),
    }
    assert observed[("AluA", "chr1", 150)]["label"] == "1"
    assert observed[("AluA", "chr1", 150)]["ccre_overlap_fraction"] == "0.5"
    assert observed[("AluA", "chr1", 1000)]["label"] == "0"
    assert ("AluA", "chr1", 600) not in observed

    family_rows = list(csv.DictReader(outputs["family_counts"].open()))
    alu = next(row for row in family_rows if row["environment_id"] == "Alu")
    assert alu["positive_loci"] == "2"
    assert alu["strict_negative_loci"] == "2"
    manifest = json.loads(outputs["manifest"].read_text())
    assert manifest["counts"]["excluded_ambiguous_or_other_ccre"] == 1
    assert manifest["counts"]["excluded_short_te"] == 1
    assert manifest["counts"]["excluded_noncanonical_chromosome"] == 1

    # Every artifact is byte-identical on a repeated run; immutable outputs
    # therefore permit exact reproduction but reject changed bytes.
    first_hashes = {name: _sha(path) for name, path in outputs.items()}
    repeated = run_preflight(config, tmp_path / "out", base_dir=tmp_path)
    assert {name: _sha(path) for name, path in repeated.items()} == first_hashes


def test_download_refuses_to_replace_an_existing_wrong_file(tmp_path):
    path = tmp_path / "source.bin"
    path.write_bytes(b"wrong")
    spec = SourceSpec(
        name="fixture",
        url="https://example.test/source.bin",
        path=path,
        size_bytes=5,
        sha256=hashlib.sha256(b"right").hexdigest(),
    )
    with pytest.raises(PublicDataError, match="wrong SHA-256"):
        download_source(spec)


def test_candidate_pool_is_balanced_by_exact_subfamily_and_not_final(tmp_path):
    config = _write_fixture_config(tmp_path)
    preflight = tmp_path / "preflight"
    run_preflight(config, preflight, base_dir=tmp_path)
    outputs = build_candidate_pool(config, preflight, tmp_path / "pool")
    with gzip.open(outputs["candidate_pool"], "rt") as handle:
        rows = list(csv.DictReader(handle))
    counts = {}
    for row in rows:
        key = (row["rep_name"], int(row["label"]))
        counts[key] = counts.get(key, 0) + 1
    assert counts == {("AluA", 0): 1, ("AluA", 1): 1, ("AluB", 0): 1, ("AluB", 1): 1}
    manifest = json.loads(outputs["manifest"].read_text())
    assert manifest["final_sample_frozen"] is False


def test_downstream_stage_rejects_a_changed_frozen_config(tmp_path):
    config = _write_fixture_config(tmp_path)
    preflight = tmp_path / "preflight"
    run_preflight(config, preflight, base_dir=tmp_path)
    payload = json.loads(config.read_text())
    payload["labels"]["negative_no_any_ccre_buffer_bp"] = 500
    config.write_text(json.dumps(payload))
    with pytest.raises(PublicDataError, match="wrong byte count|wrong SHA-256"):
        build_candidate_pool(config, preflight, tmp_path / "pool")


def test_proportional_quota_has_a_hard_diversity_cap():
    quotas = proportional_capped_quotas(
        {"dominant": 100000, "b": 100, "c": 100}, total=180, per_group_cap=60
    )
    assert quotas == {"b": 60, "c": 60, "dominant": 60}


def test_sequence_extraction_controls_strand_and_records_gc(tmp_path):
    config = _write_fixture_config(tmp_path)
    preflight = tmp_path / "preflight"
    pool = tmp_path / "pool"
    run_preflight(config, preflight, base_dir=tmp_path)
    build_candidate_pool(config, preflight, pool)
    outputs = extract_candidate_te_sequences(
        config, pool, tmp_path / "sequences", base_dir=tmp_path
    )

    sequences = dict(read_fasta(outputs["candidate_te_sequences"]))
    with gzip.open(outputs["candidate_pool_with_gc"], "rt") as handle:
        rows = {row["locus_id"]: row for row in csv.DictReader(handle)}
    plus = next(
        row for row in rows.values() if row["chrom"] == "chr1" and row["label"] == "1"
    )
    minus = next(
        row for row in rows.values() if row["chrom"] == "chr2" and row["label"] == "1"
    )
    assert sequences[plus["locus_id"]] == "A" * 199 + "C"
    assert sequences[minus["locus_id"]] == "G" + "T" * 149
    assert all(row["sequence_eligible"] == "1" for row in rows.values())
    assert float(plus["te_gc_fraction"]) == pytest.approx(1 / 200)


def test_precluster_match_is_exact_subfamily_chromosome_and_calipered(tmp_path):
    config = _write_fixture_config(tmp_path)
    preflight = tmp_path / "preflight"
    pool = tmp_path / "pool"
    sequences = tmp_path / "sequences"
    run_preflight(config, preflight, base_dir=tmp_path)
    build_candidate_pool(config, preflight, pool)
    extract_candidate_te_sequences(config, pool, sequences, base_dir=tmp_path)
    outputs = build_precluster_matches(config, sequences, tmp_path / "matches")

    with gzip.open(outputs["precluster_pairs"], "rt") as handle:
        pairs = list(csv.DictReader(handle))
    with gzip.open(outputs["precluster_loci"], "rt") as handle:
        loci = list(csv.DictReader(handle))
    assert len(pairs) == 2
    assert {row["rep_name"] for row in pairs} == {"AluA", "AluB"}
    assert all(float(row["te_gc_fraction_difference"]) == 0 for row in pairs)
    assert all(float(row["te_relative_length_difference"]) == 0 for row in pairs)
    assert len(loci) == 4
    assert all(int(row["model_window_end"]) - int(row["model_window_start"]) == 64 for row in loci)
    assert all(
        len({row["chrom"] for row in loci if row["pair_id"] == pair["pair_id"]}) == 1
        for pair in pairs
    )


def test_window_stage_emits_pair_complete_oriented_cluster_contract(tmp_path):
    config = _write_fixture_config(tmp_path)
    preflight = tmp_path / "preflight"
    pool = tmp_path / "pool"
    sequences = tmp_path / "sequences"
    matches = tmp_path / "matches"
    run_preflight(config, preflight, base_dir=tmp_path)
    build_candidate_pool(config, preflight, pool)
    extract_candidate_te_sequences(config, pool, sequences, base_dir=tmp_path)
    build_precluster_matches(config, sequences, matches)
    outputs = extract_precluster_windows(
        config, sequences, matches, tmp_path / "windows", base_dir=tmp_path
    )

    with gzip.open(outputs["cluster_input_loci"], "rt") as handle:
        loci = list(csv.DictReader(handle))
    pairs = {row["pair_id"] for row in loci}
    assert len(loci) == 4
    assert all(sum(row["pair_id"] == pair_id for row in loci) == 2 for pair_id in pairs)
    assert all(row["observation_id"] == row["locus_id"] for row in loci)
    assert all(row["sample_weight"] == "1.0" for row in loci)
    windows = dict(read_fasta(outputs["cluster_input_windows"]))
    copies = dict(read_fasta(outputs["cluster_input_te_sequences"]))
    assert set(windows) == set(copies) == {row["locus_id"] for row in loci}
    assert {len(sequence) for sequence in windows.values()} == {64}
    minus = next(row for row in loci if row["strand"] == "-")
    assert windows[minus["locus_id"]] == "G" + "T" * 63
    label_balance = list(
        csv.DictReader(outputs["window_gc_balance_by_environment_label"].open())
    )
    paired_balance = list(csv.DictReader(outputs["window_gc_paired_differences"].open()))
    assert {(row["environment_id"], row["label"]) for row in label_balance} == {
        ("Alu", "0"),
        ("Alu", "1"),
    }
    assert paired_balance[0]["pairs"] == "2"
