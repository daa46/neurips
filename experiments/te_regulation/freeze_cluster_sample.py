#!/usr/bin/env python3
"""Freeze a deterministic, pair-complete public-TE sample before clustering."""

from __future__ import annotations

import argparse
import gzip
import hashlib
import io
import json
import os
from collections.abc import Sequence
from pathlib import Path

import pandas as pd

from experiments.te_regulation.cluster_and_split import TEBenchmarkError, sha256_file
from experiments.te_regulation.public_data import (
    load_config,
    read_fasta,
    verify_recorded_artifact,
)

_PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG = _PROJECT_ROOT / "configs/te/public_ccre.json"
DEFAULT_WINDOW_DIR = _PROJECT_ROOT / "data/te_public/windows_verified_v2"


def _priority(seed: str, pair_id: str) -> str:
    return hashlib.sha256(f"{seed}:final-pair:{pair_id}".encode()).hexdigest()


def _proportional_capped_quotas(
    counts: pd.Series,
    *,
    target: int,
    cap: int,
) -> dict[str, int]:
    """Allocate a capped target by deterministic largest remainder."""

    capacities = {str(name): min(int(value), cap) for name, value in counts.items()}
    allocated_total = min(target, sum(capacities.values()))
    if allocated_total == 0:
        return {name: 0 for name in capacities}
    denominator = sum(capacities.values())
    divisions = {
        name: divmod(allocated_total * capacity, denominator)
        for name, capacity in capacities.items()
    }
    quotas = {name: min(capacities[name], divisions[name][0]) for name in capacities}
    remaining = allocated_total - sum(quotas.values())
    order = sorted(
        capacities,
        key=lambda name: (-divisions[name][1], name),
    )
    while remaining:
        progressed = False
        for name in order:
            if quotas[name] < capacities[name]:
                quotas[name] += 1
                remaining -= 1
                progressed = True
                if remaining == 0:
                    break
        if not progressed:
            raise RuntimeError("capped largest-remainder allocation stalled")
    return quotas


def select_final_pairs(
    pairs: pd.DataFrame,
    *,
    seed: str,
    target_pairs_per_environment: int,
    maximum_pairs_per_subfamily: int,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Allocate frozen subfamily quotas, then select by pair-ID content hash."""

    required = {"pair_id", "environment_id", "rep_name"}
    missing = required.difference(pairs.columns)
    if missing:
        raise TEBenchmarkError(f"cluster-input pair table is missing: {sorted(missing)}")
    if pairs.empty or pairs[list(required)].isna().any().any():
        raise TEBenchmarkError("cluster-input pair identifiers must be nonempty and non-null")
    if pairs["pair_id"].duplicated().any():
        raise TEBenchmarkError("cluster-input pair identifiers are not unique")
    if target_pairs_per_environment < 1 or maximum_pairs_per_subfamily < 1:
        raise TEBenchmarkError("pair target and subfamily cap must be positive")
    working = pairs.copy()
    for column in required:
        working[column] = working[column].astype(str)
    working["final_pair_priority_sha256"] = working["pair_id"].map(
        lambda pair_id: _priority(seed, pair_id)
    )
    selected_groups = []
    audit_rows = []
    for environment, group in working.groupby("environment_id", sort=True, observed=True):
        availability = group.groupby("rep_name", observed=True).size().sort_index()
        quotas = _proportional_capped_quotas(
            availability,
            target=target_pairs_per_environment,
            cap=maximum_pairs_per_subfamily,
        )
        for subfamily, subgroup in group.groupby("rep_name", sort=True, observed=True):
            chosen = subgroup.sort_values(
                ["final_pair_priority_sha256", "pair_id"], kind="stable"
            ).head(quotas[str(subfamily)])
            selected_groups.append(chosen)
            audit_rows.append(
                {
                    "environment_id": environment,
                    "rep_name": subfamily,
                    "available_pairs": len(subgroup),
                    "capped_capacity": min(len(subgroup), maximum_pairs_per_subfamily),
                    "allocated_quota": quotas[str(subfamily)],
                    "selected_pairs": len(chosen),
                }
            )
    selected = pd.concat(selected_groups, ignore_index=True).sort_values(
        ["environment_id", "rep_name", "final_pair_priority_sha256", "pair_id"],
        kind="stable",
    )
    return selected.reset_index(drop=True), pd.DataFrame(audit_rows)


def _deterministic_gzip_bytes(text: str) -> bytes:
    buffer = io.BytesIO()
    with gzip.GzipFile(filename="", fileobj=buffer, mode="wb", mtime=0) as compressed:
        compressed.write(text.encode())
    return buffer.getvalue()


def _install_bytes(path: Path, payload: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        if path.read_bytes() != payload:
            raise TEBenchmarkError(f"refusing to replace non-identical frozen sample: {path}")
        return
    partial = path.with_name(path.name + ".part")
    if partial.exists():
        partial.unlink()
    with partial.open("xb") as handle:
        handle.write(payload)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(partial, path)


def _install_text(path: Path, text: str) -> None:
    _install_bytes(path, text.encode())


def _artifact(path: Path) -> dict[str, object]:
    return {
        "path": path.as_posix(),
        "size_bytes": path.stat().st_size,
        "sha256": sha256_file(path),
    }


def validate_gate_evidence(
    evidence_path: Path,
    *,
    expected_sha256: str,
    excluded_environments: Sequence[str],
    available_environments: set[str],
    maximum_root_fraction: float,
) -> pd.DataFrame:
    """Verify an immutable pilot diagnostic and authorize only gate failures."""

    expected_sha256 = expected_sha256.lower()
    if len(expected_sha256) != 64 or set(expected_sha256).difference("0123456789abcdef"):
        raise TEBenchmarkError("gate evidence SHA-256 must be 64 lowercase hex characters")
    actual_sha256 = sha256_file(evidence_path)
    if actual_sha256 != expected_sha256:
        raise TEBenchmarkError(
            f"gate evidence hash mismatch: expected {expected_sha256}, observed {actual_sha256}"
        )
    excluded = [str(value) for value in excluded_environments]
    if not excluded or len(set(excluded)) != len(excluded):
        raise TEBenchmarkError("excluded environments must be nonempty and unique")
    unknown = set(excluded).difference(available_environments)
    if unknown:
        raise TEBenchmarkError(f"excluded environments are absent upstream: {sorted(unknown)}")
    evidence = pd.read_csv(evidence_path)
    required = {
        "environment_id",
        "root_cluster_id",
        "observations",
        "environment_observations",
        "environment_fraction",
    }
    missing = required.difference(evidence.columns)
    if evidence.empty or missing:
        raise TEBenchmarkError(f"gate evidence is empty or missing columns: {sorted(missing)}")
    if evidence[list(required)].isna().any().any():
        raise TEBenchmarkError("gate evidence required values cannot be null")
    evidence["environment_id"] = evidence["environment_id"].astype(str)
    for column in ("observations", "environment_observations", "environment_fraction"):
        evidence[column] = pd.to_numeric(evidence[column], errors="coerce")
        if evidence[column].isna().any():
            raise TEBenchmarkError(f"gate evidence column {column!r} must be numeric")
    if (
        (evidence["observations"] <= 0).any()
        or (evidence["environment_observations"] <= 0).any()
        or not evidence["environment_fraction"].between(0, 1).all()
    ):
        raise TEBenchmarkError("gate evidence contains invalid counts or fractions")
    recomputed = evidence["observations"] / evidence["environment_observations"]
    if not (recomputed - evidence["environment_fraction"]).abs().le(1e-12).all():
        raise TEBenchmarkError("gate evidence fractions do not agree with its counts")
    maxima = (
        evidence.sort_values(
            ["environment_id", "environment_fraction", "root_cluster_id"],
            ascending=[True, False, True],
            kind="stable",
        )
        .groupby("environment_id", observed=True)
        .first()
        .reset_index()
    )
    missing_evidence = set(excluded).difference(maxima["environment_id"])
    if missing_evidence:
        raise TEBenchmarkError(
            f"excluded environments are absent from gate evidence: {sorted(missing_evidence)}"
        )
    maxima["violates_frozen_gate"] = maxima["environment_fraction"].ge(maximum_root_fraction)
    maxima["decision"] = maxima["environment_id"].map(
        lambda environment: "exclude" if environment in set(excluded) else "retain"
    )
    invalid = maxima["decision"].eq("exclude") & ~maxima["violates_frozen_gate"]
    if invalid.any():
        raise TEBenchmarkError(
            "cannot exclude environments that pass the frozen dominance gate: "
            f"{maxima.loc[invalid, 'environment_id'].tolist()}"
        )
    return maxima


def freeze_cluster_sample(
    config_path: Path,
    window_dir: Path,
    output_dir: Path,
    *,
    purpose: str = "production",
    target_pairs_per_environment: int | None = None,
    gate_evidence_path: Path | None = None,
    gate_evidence_sha256: str | None = None,
    exclude_environments: Sequence[str] = (),
) -> dict[str, Path]:
    """Verify the final window stage and emit immutable cluster-ready files."""

    config = load_config(config_path)
    if purpose not in {"production", "homology-preflight"}:
        raise TEBenchmarkError("purpose must be production or homology-preflight")
    if purpose == "production" and target_pairs_per_environment is not None:
        raise TEBenchmarkError(
            "an explicit pair target is allowed only for purpose=homology-preflight"
        )
    if purpose == "homology-preflight" and target_pairs_per_environment is None:
        raise TEBenchmarkError("homology-preflight requires an explicit pair target")
    gate_values_present = (
        gate_evidence_path is not None,
        gate_evidence_sha256 is not None,
        bool(exclude_environments),
    )
    if any(gate_values_present) and not all(gate_values_present):
        raise TEBenchmarkError(
            "gate evidence path, SHA-256, and excluded environment are required together"
        )
    if purpose != "production" and any(gate_values_present):
        raise TEBenchmarkError("pilot gate decisions may be applied only to production sampling")
    manifest_path = window_dir / "cluster_input_manifest.json"
    pair_path = window_dir / "cluster_input_pairs.csv.gz"
    locus_path = window_dir / "cluster_input_loci.csv.gz"
    te_fasta_path = window_dir / "cluster_input_te_sequences.oriented.fa"
    window_fasta_path = window_dir / "cluster_input_windows.oriented.fa"
    for path in (manifest_path, pair_path, locus_path, te_fasta_path, window_fasta_path):
        if not path.is_file():
            raise FileNotFoundError(path)
    for name, path in (
        ("cluster_input_pairs", pair_path),
        ("cluster_input_loci", locus_path),
        ("cluster_input_te_sequences", te_fasta_path),
        ("cluster_input_windows", window_fasta_path),
    ):
        verify_recorded_artifact(manifest_path, path, artifact_name=name)

    pairs = pd.read_csv(pair_path, dtype=str)
    loci = pd.read_csv(locus_path)
    required_loci = {
        "observation_id",
        "pair_id",
        "environment_id",
        "rep_name",
        "label",
        "sample_weight",
    }
    missing = required_loci.difference(loci.columns)
    if missing:
        raise TEBenchmarkError(f"cluster-input locus table is missing: {sorted(missing)}")
    if loci.empty or loci[list(required_loci)].isna().any().any():
        raise TEBenchmarkError("cluster-input locus fields must be nonempty and non-null")
    loci["observation_id"] = loci["observation_id"].astype(str)
    loci["pair_id"] = loci["pair_id"].astype(str)
    if loci["observation_id"].duplicated().any():
        raise TEBenchmarkError("cluster-input observation identifiers are not unique")
    upstream_locus_ids = set(loci["observation_id"])
    labels = pd.to_numeric(loci["label"], errors="coerce")
    weights = pd.to_numeric(loci["sample_weight"], errors="coerce")
    if labels.isna().any() or not labels.isin([0, 1]).all():
        raise TEBenchmarkError("cluster-input labels must be binary")
    if weights.isna().any() or (weights <= 0).any():
        raise TEBenchmarkError("cluster-input sample weights must be positive")
    loci["label"] = labels.astype(int)
    loci["sample_weight"] = weights.astype(float)
    pair_cells = loci.groupby("pair_id", observed=True).agg(
        loci=("observation_id", "size"),
        labels=("label", "nunique"),
        environments=("environment_id", "nunique"),
        subfamilies=("rep_name", "nunique"),
    )
    valid_pairs = (
        pair_cells["loci"].eq(2)
        & pair_cells["labels"].eq(2)
        & pair_cells["environments"].eq(1)
        & pair_cells["subfamilies"].eq(1)
    )
    if not valid_pairs.all():
        raise TEBenchmarkError("every cluster-input pair must be one-positive/one-negative")
    if set(pair_cells.index) != set(pairs["pair_id"]):
        raise TEBenchmarkError("pair and locus tables disagree on pair identifiers")
    pair_metadata = pairs.set_index("pair_id")[["environment_id", "rep_name"]].astype(str)
    locus_metadata = (
        loci.groupby("pair_id", observed=True)[["environment_id", "rep_name"]].first().astype(str)
    )
    if not pair_metadata.sort_index().equals(locus_metadata.sort_index()):
        raise TEBenchmarkError("pair and locus environment/subfamily metadata disagree")

    input_pair_count = len(pairs)
    gate_decision: pd.DataFrame | None = None
    gate_evidence_artifact: dict[str, object] | None = None
    excluded = tuple(str(value) for value in exclude_environments)
    if gate_evidence_path is not None and gate_evidence_sha256 is not None:
        if not gate_evidence_path.is_file():
            raise FileNotFoundError(gate_evidence_path)
        frozen_threshold = float(
            config["gates"]["maximum_fraction_of_environment_observations_in_one_root"]
        )
        gate_decision = validate_gate_evidence(
            gate_evidence_path,
            expected_sha256=gate_evidence_sha256,
            excluded_environments=excluded,
            available_environments=set(pairs["environment_id"].astype(str)),
            maximum_root_fraction=frozen_threshold,
        )
        gate_evidence_artifact = _artifact(gate_evidence_path)
        pairs = pairs.loc[~pairs["environment_id"].astype(str).isin(excluded)].copy()
        retained_pair_ids = set(pairs["pair_id"].astype(str))
        loci = loci.loc[loci["pair_id"].isin(retained_pair_ids)].copy()
        if pairs.empty or loci.empty:
            raise TEBenchmarkError("gate decision removed every upstream environment")

    sampling = config["sampling"]
    configured_target = int(sampling["target_pairs_per_environment"])
    effective_target = (
        configured_target
        if target_pairs_per_environment is None
        else int(target_pairs_per_environment)
    )
    selected_pairs, audit = select_final_pairs(
        pairs,
        seed=str(sampling["seed"]),
        target_pairs_per_environment=effective_target,
        maximum_pairs_per_subfamily=int(sampling["maximum_final_pairs_per_subfamily"]),
    )
    selected_pair_ids = set(selected_pairs["pair_id"])
    selected_loci = loci.loc[loci["pair_id"].isin(selected_pair_ids)].copy()
    selected_loci = selected_loci.sort_values("observation_id", kind="stable")
    if len(selected_loci) != 2 * len(selected_pairs):
        raise TEBenchmarkError("final frozen sample is not pair complete")
    selected_ids = set(selected_loci["observation_id"])
    te_sequences = dict(read_fasta(te_fasta_path))
    window_sequences = dict(read_fasta(window_fasta_path))
    if set(te_sequences) != upstream_locus_ids or set(window_sequences) != upstream_locus_ids:
        raise TEBenchmarkError("verified window-stage FASTAs disagree with its locus table")
    if any(len(sequence) != 512 for sequence in window_sequences.values()):
        raise TEBenchmarkError("verified model-input windows must all be exactly 512 bp")

    output_dir.mkdir(parents=True, exist_ok=True)
    outputs = {
        "loci": output_dir / "cluster_sample_loci.csv.gz",
        "pairs": output_dir / "cluster_sample_pairs.csv.gz",
        "te_fasta": output_dir / "cluster_sample_te_sequences.oriented.fa",
        "window_fasta": output_dir / "cluster_sample_windows.oriented.fa",
        "audit": output_dir / "cluster_sample_quota_audit.csv",
        "manifest": output_dir / "cluster_sample_manifest.json",
    }
    if gate_decision is not None:
        outputs["gate_decision"] = output_dir / "cluster_sample_gate_decision.csv"
    _install_bytes(
        outputs["pairs"],
        _deterministic_gzip_bytes(selected_pairs.to_csv(index=False, lineterminator="\n")),
    )
    _install_bytes(
        outputs["loci"],
        _deterministic_gzip_bytes(selected_loci.to_csv(index=False, lineterminator="\n")),
    )
    _install_text(
        outputs["te_fasta"],
        "".join(
            f">{identifier}\n{te_sequences[identifier]}\n" for identifier in sorted(selected_ids)
        ),
    )
    _install_text(
        outputs["window_fasta"],
        "".join(
            f">{identifier}\n{window_sequences[identifier]}\n"
            for identifier in sorted(selected_ids)
        ),
    )
    _install_text(outputs["audit"], audit.to_csv(index=False, lineterminator="\n"))
    if gate_decision is not None:
        _install_text(
            outputs["gate_decision"],
            gate_decision.to_csv(index=False, lineterminator="\n"),
        )
    artifacts = {name: _artifact(path) for name, path in outputs.items() if name != "manifest"}
    environment_counts = (
        selected_pairs.groupby("environment_id", observed=True)
        .size()
        .rename("pairs")
        .reset_index()
        .to_dict("records")
    )
    manifest = {
        "schema_version": 1,
        "stage": f"{purpose}_pair_sample_before_homology_clustering",
        "purpose": purpose,
        "final_sample_frozen": purpose == "production",
        "config": _artifact(config_path),
        "input": {
            "window_manifest": _artifact(manifest_path),
            "cluster_input_pairs": _artifact(pair_path),
            "cluster_input_loci": _artifact(locus_path),
            "cluster_input_te_sequences": _artifact(te_fasta_path),
            "cluster_input_windows": _artifact(window_fasta_path),
        },
        "parameters": {
            "seed": str(sampling["seed"]),
            "priority": "SHA256(seed + ':final-pair:' + pair_id)",
            "priority_uses_labels_or_model_results": False,
            "quota_rule": "largest-remainder proportional to capped subfamily availability",
            "configured_production_target_pairs_per_environment": configured_target,
            "effective_target_pairs_per_environment": effective_target,
            "target_override_allowed_only_for_homology_preflight": True,
            "maximum_pairs_per_exact_subfamily": int(sampling["maximum_final_pairs_per_subfamily"]),
            "pair_complete": True,
        },
        "counts": {
            "input_pairs": input_pair_count,
            "pairs_after_gate_decision": len(pairs),
            "selected_pairs": len(selected_pairs),
            "selected_loci": len(selected_loci),
            "pairs_by_environment": environment_counts,
        },
        "artifacts": artifacts,
    }
    retained_environments = sorted(selected_pairs["environment_id"].astype(str).unique())
    manifest["gate_decision"] = (
        {
            "applied": True,
            "evidence": gate_evidence_artifact,
            "expected_evidence_sha256": gate_evidence_sha256,
            "frozen_maximum_root_fraction_strict": float(
                config["gates"]["maximum_fraction_of_environment_observations_in_one_root"]
            ),
            "excluded_environments": sorted(excluded),
            "retained_environments": retained_environments,
            "retained_pilot_gate_violations_for_production_recheck": sorted(
                gate_decision.loc[
                    gate_decision["violates_frozen_gate"] & gate_decision["decision"].eq("retain"),
                    "environment_id",
                ].astype(str)
            ),
            "exclusion_rule": (
                "an explicitly excluded environment must have pilot maximum root "
                "fraction >= the frozen strict threshold"
            ),
            "model_performance_used": False,
        }
        if gate_decision is not None
        else {
            "applied": False,
            "excluded_environments": [],
            "retained_environments": retained_environments,
        }
    )
    _install_text(outputs["manifest"], json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    return outputs


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--window-dir", type=Path, default=DEFAULT_WINDOW_DIR)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--purpose",
        choices=("production", "homology-preflight"),
        default="production",
    )
    parser.add_argument("--target-pairs-per-environment", type=int)
    parser.add_argument("--gate-evidence", type=Path)
    parser.add_argument("--gate-evidence-sha256")
    parser.add_argument("--exclude-environment", action="append", default=[])
    args = parser.parse_args(argv)
    outputs = freeze_cluster_sample(
        args.config,
        args.window_dir,
        args.output_dir,
        purpose=args.purpose,
        target_pairs_per_environment=args.target_pairs_per_environment,
        gate_evidence_path=args.gate_evidence,
        gate_evidence_sha256=args.gate_evidence_sha256,
        exclude_environments=tuple(args.exclude_environment),
    )
    for name, path in outputs.items():
        print(f"{name}: {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
