#!/usr/bin/env python3
"""Construct conservative TE homology roots and leakage-safe data splits.

The root partition is the transitive union of two MMseqs2 connected-component
clusterings: one on oriented TE-copy sequence and one on the corresponding
oriented 512-bp model-input window. Thus, a link in either view prevents a pair
of loci from crossing train, selection, and test partitions.
"""

# ruff: noqa: E402 -- direct-script support requires path setup before imports.

from __future__ import annotations

import argparse
import gzip
import hashlib
import json
import os
import subprocess
import sys
import tempfile
from collections import defaultdict
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path

_PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

import numpy as np
import pandas as pd

from hrms.splits import assert_no_cluster_leakage, split_assignment_hash

DEFAULT_MMSEQS = _PROJECT_ROOT / "tools/te/mmseqs/bin/mmseqs"
DNA_ALPHABET = frozenset("ACGTRYSWKMBDHVN")
SPLIT_NAMES = ("train", "select", "test")
SPLIT_PROPORTIONS = (0.6, 0.2, 0.2)
DOMINANCE_GATE_SPLITS = ("select", "test")


class TEBenchmarkError(ValueError):
    """Raised when a public-TE benchmark invariant is violated."""


@dataclass(frozen=True)
class SplitDiagnostics:
    split_table: pd.DataFrame
    cell_counts: pd.DataFrame
    root_counts: pd.DataFrame
    largest_root_fraction: pd.DataFrame
    objective: float
    feasible_candidates: int


@dataclass(frozen=True)
class MMseqsGraphResult:
    """Bounded-search connected components represented by a spanning forest."""

    spanning_pairs: list[tuple[str, str]]
    job_audit: pd.DataFrame
    commands: tuple[tuple[str, ...], ...]
    plan_hash: str
    checkpoint_dir: Path | None


class UnionFind:
    def __init__(self, values: Iterable[str]):
        self.parent = {value: value for value in values}
        self.rank = {value: 0 for value in self.parent}

    def find(self, value: str) -> str:
        parent = self.parent[value]
        if parent != value:
            self.parent[value] = self.find(parent)
        return self.parent[value]

    def union(self, first: str, second: str) -> bool:
        left, right = self.find(first), self.find(second)
        if left == right:
            return False
        left_rank, right_rank = self.rank[left], self.rank[right]
        if left_rank < right_rank or (left_rank == right_rank and right < left):
            left, right = right, left
            left_rank, right_rank = right_rank, left_rank
        self.parent[right] = left
        if left_rank == right_rank:
            self.rank[left] += 1
        return True


def _open_text(path: Path):
    return (
        gzip.open(path, "rt", encoding="utf-8")
        if path.suffix == ".gz"
        else path.open("r", encoding="utf-8")
    )


def read_table(path: Path) -> pd.DataFrame:
    if path.suffix in {".parquet", ".pq"}:
        return pd.read_parquet(path)
    if path.suffix == ".csv" or path.suffixes[-2:] == [".csv", ".gz"]:
        return pd.read_csv(path)
    if path.suffix in {".tsv", ".txt"} or path.suffixes[-2:] == [".tsv", ".gz"]:
        return pd.read_csv(path, sep="\t")
    raise TEBenchmarkError(f"unsupported table format: {path}")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def write_json(path: Path, payload: dict) -> None:
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def read_fasta(path: Path) -> dict[str, str]:
    """Read a FASTA with unique first-token identifiers."""

    sequences: dict[str, str] = {}
    current: str | None = None
    chunks: list[str] = []
    with _open_text(path) as handle:
        for line_number, raw_line in enumerate(handle, start=1):
            line = raw_line.strip()
            if not line:
                continue
            if line.startswith(">"):
                if current is not None:
                    sequences[current] = "".join(chunks).upper()
                current = line[1:].split(maxsplit=1)[0]
                if not current:
                    raise TEBenchmarkError(f"empty FASTA identifier at {path}:{line_number}")
                if current in sequences:
                    raise TEBenchmarkError(f"duplicate FASTA identifier {current!r} in {path}")
                chunks = []
            else:
                if current is None:
                    raise TEBenchmarkError(
                        f"sequence precedes first header at {path}:{line_number}"
                    )
                chunks.append(line)
    if current is not None:
        if current in sequences:
            raise TEBenchmarkError(f"duplicate FASTA identifier {current!r} in {path}")
        sequences[current] = "".join(chunks).upper()
    if not sequences:
        raise TEBenchmarkError(f"FASTA contains no sequences: {path}")
    return sequences


def map_fasta_ids(sequences: Mapping[str, str], mapping_path: Path | None) -> dict[str, str]:
    """Map external FASTA IDs to observation IDs using a one-to-one table."""

    if mapping_path is None:
        return dict(sequences)
    mapping = read_table(mapping_path)
    required = {"sequence_id", "observation_id"}
    missing = required.difference(mapping.columns)
    if missing:
        raise TEBenchmarkError(f"FASTA map is missing columns: {sorted(missing)}")
    if mapping[list(required)].isna().any().any():
        raise TEBenchmarkError("FASTA map contains null identifiers")
    mapping = mapping[list(required)].astype(str)
    if mapping["sequence_id"].duplicated().any() or mapping["observation_id"].duplicated().any():
        raise TEBenchmarkError("FASTA map must be one-to-one")
    lookup = dict(mapping.itertuples(index=False, name=None))
    if set(lookup) != set(sequences):
        raise TEBenchmarkError("FASTA and mapping identifiers differ")
    return {lookup[sequence_id]: sequence for sequence_id, sequence in sequences.items()}


def validate_loci_and_sequences(
    loci: pd.DataFrame,
    te_sequences: Mapping[str, str],
    window_sequences: Mapping[str, str],
) -> pd.DataFrame:
    """Validate and canonicalize labeled loci plus both oriented sequence views."""

    required = {"observation_id", "pair_id", "environment_id", "label"}
    missing = required.difference(loci.columns)
    if missing:
        raise TEBenchmarkError(f"locus table is missing columns: {sorted(missing)}")
    if loci.empty or loci[list(required)].isna().any().any():
        raise TEBenchmarkError("required locus columns must be nonempty and non-null")
    loci = loci.copy()
    loci["observation_id"] = loci["observation_id"].astype(str)
    loci["pair_id"] = loci["pair_id"].astype(str)
    loci["environment_id"] = loci["environment_id"].astype(str)
    if "candidate_priority_sha256_int" in loci:
        if loci["candidate_priority_sha256_int"].isna().any():
            raise TEBenchmarkError("candidate priority provenance cannot be null")
        loci["candidate_priority_sha256_int"] = loci[
            "candidate_priority_sha256_int"
        ].astype(str)
    if loci["observation_id"].duplicated().any():
        raise TEBenchmarkError("observation identifiers are not unique")
    labels = pd.to_numeric(loci["label"], errors="coerce")
    if labels.isna().any() or not labels.isin([0, 1]).all():
        raise TEBenchmarkError("labels must be binary values in {0, 1}")
    loci["label"] = labels.astype(np.int8)
    pair_cells = loci.groupby("pair_id", observed=True).agg(
        observations=("observation_id", "size"),
        labels=("label", "nunique"),
        environments=("environment_id", "nunique"),
    )
    valid_pairs = (
        pair_cells["observations"].eq(2)
        & pair_cells["labels"].eq(2)
        & pair_cells["environments"].eq(1)
    )
    if not valid_pairs.all():
        raise TEBenchmarkError(
            "every design pair must contain one positive and one negative in one environment"
        )
    if "sample_weight" not in loci:
        loci["sample_weight"] = 1.0
    weights = pd.to_numeric(loci["sample_weight"], errors="coerce")
    if weights.isna().any() or not np.isfinite(weights).all() or (weights <= 0).any():
        raise TEBenchmarkError("sample weights must be finite and strictly positive")
    loci["sample_weight"] = weights.astype(float)
    observations = set(loci["observation_id"])
    if set(te_sequences) != observations or set(window_sequences) != observations:
        raise TEBenchmarkError("both FASTAs must contain exactly one entry for every observation")
    for identifier in sorted(observations):
        for name, sequence, expected in (
            ("TE", te_sequences[identifier], None),
            ("window", window_sequences[identifier], 512),
        ):
            if not sequence:
                raise TEBenchmarkError(f"empty {name} sequence for {identifier}")
            invalid = sorted(set(sequence).difference(DNA_ALPHABET))
            if invalid:
                raise TEBenchmarkError(f"invalid DNA symbols for {identifier}: {invalid}")
            if expected is not None and len(sequence) != expected:
                raise TEBenchmarkError(
                    f"{identifier} has {len(sequence)}-bp {name} sequence; expected {expected}"
                )
    return loci.sort_values("observation_id", kind="stable").reset_index(drop=True)


def _canonical_json(payload: object) -> str:
    return json.dumps(payload, sort_keys=True, separators=(",", ":"))


def _payload_hash(payload: object) -> str:
    return hashlib.sha256(_canonical_json(payload).encode()).hexdigest()


def _sequence_hash(sequences: Mapping[str, str]) -> str:
    digest = hashlib.sha256()
    for identifier in sorted(sequences):
        for value in (identifier, sequences[identifier]):
            encoded = value.encode()
            digest.update(len(encoded).to_bytes(8, "little"))
            digest.update(encoded)
    return digest.hexdigest()


def _write_checkpoint_text(path: Path, text: str) -> None:
    """Atomically install a checkpoint, accepting only byte-identical reuse."""

    path.parent.mkdir(parents=True, exist_ok=True)
    encoded = text.encode()
    if path.exists():
        if path.read_bytes() != encoded:
            raise TEBenchmarkError(f"checkpoint content mismatch: {path}")
        return
    partial = path.with_name(path.name + ".part")
    if partial.exists():
        partial.unlink()
    with partial.open("xb") as handle:
        handle.write(encoded)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(partial, path)


def _write_checkpoint_json(path: Path, payload: dict) -> None:
    _write_checkpoint_text(path, json.dumps(payload, indent=2, sort_keys=True) + "\n")


def _identifier_digest(values: Iterable[str]) -> str:
    digest = hashlib.sha256()
    for value in sorted(values):
        encoded = value.encode()
        digest.update(len(encoded).to_bytes(8, "little"))
        digest.update(encoded)
    return digest.hexdigest()


def _spanning_forest(union: UnionFind, values: Iterable[str]) -> list[tuple[str, str]]:
    members: dict[str, list[str]] = defaultdict(list)
    for value in sorted(values):
        members[union.find(value)].append(value)
    edges: list[tuple[str, str]] = []
    for component in sorted((sorted(group) for group in members.values()), key=lambda x: x[0]):
        edges.extend((component[0], member) for member in component[1:])
    return edges


def _write_safe_fasta(
    path: Path,
    safe_ids: Sequence[str],
    safe_to_observation: Mapping[str, str],
    sequences: Mapping[str, str],
) -> dict[str, object]:
    text = "".join(
        f">{safe_id}\n{sequences[safe_to_observation[safe_id]]}\n" for safe_id in safe_ids
    )
    _write_checkpoint_text(path, text)
    return {
        "path": path.as_posix(),
        "sha256": sha256_file(path),
        "size_bytes": path.stat().st_size,
        "identifier_hash": _identifier_digest(safe_ids),
        "count": len(safe_ids),
        "minimum_length": min(len(sequences[safe_to_observation[value]]) for value in safe_ids),
        "maximum_length": max(len(sequences[safe_to_observation[value]]) for value in safe_ids),
    }


def _parse_job_edges(
    path: Path,
    query_ids: set[str],
    target_ids: set[str],
    *,
    min_identity: float,
    min_coverage: float,
) -> tuple[list[tuple[str, str]], dict[str, int | float | None]]:
    if not path.exists():
        raise TEBenchmarkError(f"MMseqs2 did not create threshold-edge output: {path}")
    local = UnionFind(query_ids | target_ids)
    raw_hits = 0
    nonself_hits = 0
    component_merges = 0
    minimum_identity: float | None = None
    minimum_query_coverage: float | None = None
    minimum_target_coverage: float | None = None
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            fields = line.rstrip("\n").split("\t")
            if len(fields) != 5 or fields[0] not in query_ids or fields[1] not in target_ids:
                raise TEBenchmarkError(f"invalid threshold-edge row at {path}:{line_number}")
            try:
                identity = float(fields[2]) / 100.0
                query_coverage, target_coverage = float(fields[3]), float(fields[4])
            except ValueError as error:
                raise TEBenchmarkError(
                    f"non-numeric threshold-edge row at {path}:{line_number}"
                ) from error
            if not np.isfinite([identity, query_coverage, target_coverage]).all():
                raise TEBenchmarkError(f"non-finite threshold edge at {path}:{line_number}")
            tolerance = 1e-6
            if identity + tolerance < min_identity or min(
                query_coverage, target_coverage
            ) + tolerance < min_coverage:
                raise TEBenchmarkError(
                    f"MMseqs2 emitted an edge below the frozen threshold at {path}:{line_number}"
                )
            raw_hits += 1
            minimum_identity = identity if minimum_identity is None else min(
                minimum_identity, identity
            )
            minimum_query_coverage = (
                query_coverage
                if minimum_query_coverage is None
                else min(minimum_query_coverage, query_coverage)
            )
            minimum_target_coverage = (
                target_coverage
                if minimum_target_coverage is None
                else min(minimum_target_coverage, target_coverage)
            )
            if fields[0] != fields[1]:
                nonself_hits += 1
                component_merges += int(local.union(fields[0], fields[1]))
    return _spanning_forest(local, query_ids | target_ids), {
        "raw_hits": raw_hits,
        "nonself_hits": nonself_hits,
        "component_merges": component_merges,
        "minimum_identity": minimum_identity,
        "minimum_query_coverage": minimum_query_coverage,
        "minimum_target_coverage": minimum_target_coverage,
    }


def _run_sharded_mmseqs(
    sequences: Mapping[str, str],
    checkpoint_dir: Path,
    *,
    persistent: bool,
    view_name: str,
    mmseqs_path: Path,
    min_identity: float,
    min_coverage: float,
    threads: int,
    query_chunk_size: int,
    target_shard_size: int,
    max_job_output_bytes: int,
    max_total_output_bytes: int,
    max_job_hits: int,
    max_total_hits: int,
) -> MMseqsGraphResult:
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    query_dir, target_dir = checkpoint_dir / "queries", checkpoint_dir / "targets"
    jobs_dir = checkpoint_dir / "jobs"
    query_dir.mkdir(exist_ok=True)
    target_dir.mkdir(exist_ok=True)
    jobs_dir.mkdir(exist_ok=True)

    observation_ids = sorted(sequences)
    maximum_sequence_length = max(map(len, sequences.values()))
    safe_to_observation = {
        f"seq_{index:09d}": observation_id
        for index, observation_id in enumerate(observation_ids)
    }
    length_order = sorted(
        safe_to_observation,
        key=lambda safe_id: (len(sequences[safe_to_observation[safe_id]]), safe_id),
    )
    query_chunks = [
        tuple(length_order[index : index + query_chunk_size])
        for index in range(0, len(length_order), query_chunk_size)
    ]
    target_shards = [
        tuple(length_order[index : index + target_shard_size])
        for index in range(0, len(length_order), target_shard_size)
    ]
    query_records = []
    for index, identifiers in enumerate(query_chunks):
        query_records.append(
            _write_safe_fasta(
                query_dir / f"query_{index:05d}.fa",
                identifiers,
                safe_to_observation,
                sequences,
            )
        )
    target_records = []
    for index, identifiers in enumerate(target_shards):
        target_records.append(
            _write_safe_fasta(
                target_dir / f"target_{index:05d}.fa",
                identifiers,
                safe_to_observation,
                sequences,
            )
        )

    jobs = []
    for query_index, query in enumerate(query_records):
        for target_index, target in enumerate(target_records):
            length_compatible = (
                int(target["maximum_length"])
                >= min_coverage * int(query["minimum_length"])
                and int(target["minimum_length"])
                <= int(query["maximum_length"]) / min_coverage
            )
            if length_compatible:
                jobs.append(
                    {
                        "job_id": f"q{query_index:05d}_t{target_index:05d}",
                        "query_index": query_index,
                        "target_index": target_index,
                        "query_identifier_hash": query["identifier_hash"],
                        "target_identifier_hash": target["identifier_hash"],
                        "query_count": query["count"],
                        "target_count": target["count"],
                        "maximum_directed_pairs": int(query["count"]) * int(target["count"]),
                    }
                )
    plan_body = {
        "schema_version": 1,
        "stage": "bounded_sharded_mmseqs_threshold_graph",
        "view_name": view_name,
        "sequence_count": len(sequences),
        "sequence_hash": _sequence_hash(sequences),
        "mmseqs_path": mmseqs_path.as_posix(),
        "mmseqs_sha256": sha256_file(mmseqs_path),
        "parameters": {
            "minimum_identity": min_identity,
            "minimum_bidirectional_coverage": min_coverage,
            "search_type": 3,
            "database_type": 2,
            "alignment_mode": 3,
            "sensitivity": 7.5,
            "query_chunk_size": query_chunk_size,
            "target_shard_size": target_shard_size,
            "maximum_input_sequence_length": maximum_sequence_length,
            "threads": threads,
            "max_job_output_bytes": max_job_output_bytes,
            "max_total_output_bytes": max_total_output_bytes,
            "max_job_hits": max_job_hits,
            "max_total_hits": max_total_hits,
            "length_compatibility_rule": (
                "target_max>=coverage*query_min and "
                "target_min<=query_max/coverage"
            ),
            "max_seqs_rule": "exact target-shard count; never a smaller result cap",
        },
        "query_shards": query_records,
        "target_shards": target_records,
        "jobs": jobs,
    }
    plan_hash = _payload_hash(plan_body)
    plan = {**plan_body, "plan_hash": plan_hash}
    _write_checkpoint_json(checkpoint_dir / "plan.json", plan)

    global_union = UnionFind(safe_to_observation)
    audits: list[dict] = []
    commands: list[tuple[str, ...]] = []
    cumulative_bytes = 0
    cumulative_hits = 0
    for job_number, job in enumerate(jobs, start=1):
        job_id = str(job["job_id"])
        query_index, target_index = int(job["query_index"]), int(job["target_index"])
        query_ids, target_ids = query_chunks[query_index], target_shards[target_index]
        query_path = Path(str(query_records[query_index]["path"]))
        target_path = Path(str(target_records[target_index]["path"]))
        edge_path = jobs_dir / f"{job_id}.spanning.tsv"
        manifest_path = jobs_dir / f"{job_id}.json"
        aborted_path = jobs_dir / f"{job_id}.aborted.json"
        if aborted_path.exists():
            raise TEBenchmarkError(
                f"sharded search previously exceeded a budget; inspect {aborted_path}"
            )
        if manifest_path.exists():
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            if manifest.get("plan_hash") != plan_hash:
                raise TEBenchmarkError(f"job checkpoint plan mismatch: {manifest_path}")
            if sha256_file(edge_path) != manifest["spanning_forest"]["sha256"]:
                raise TEBenchmarkError(f"job spanning-forest hash mismatch: {edge_path}")
            job_pairs = []
            with edge_path.open("r", encoding="utf-8") as handle:
                for line in handle:
                    first, second = line.rstrip("\n").split("\t")
                    job_pairs.append((first, second))
            for first, second in job_pairs:
                global_union.union(first, second)
            audit = dict(manifest["audit"])
            audits.append(audit)
            commands.append(tuple(manifest["command"]))
            cumulative_bytes += int(audit["raw_output_bytes"])
            cumulative_hits += int(audit["raw_hits"])
            print(
                f"[{view_name}] resumed {job_id} ({job_number}/{len(jobs)}): "
                f"{audit['raw_hits']} hits",
                flush=True,
            )
            continue

        raw_path = jobs_dir / f"{job_id}.raw.tsv"
        if raw_path.exists():
            raw_path.unlink()
        with tempfile.TemporaryDirectory(
            prefix=f"{job_id}-", dir=checkpoint_dir
        ) as temporary:
            print(
                f"[{view_name}] searching {job_id} ({job_number}/{len(jobs)}): "
                f"{len(query_ids)} queries x {len(target_ids)} targets",
                flush=True,
            )
            command = [
                str(mmseqs_path),
                "easy-search",
                str(query_path),
                str(target_path),
                str(raw_path),
                str(Path(temporary) / "tmp"),
                "--search-type",
                "3",
                "--dbtype",
                "2",
                "--min-seq-id",
                f"{min_identity:.12g}",
                "-c",
                f"{min_coverage:.12g}",
                "--cov-mode",
                "0",
                "--alignment-mode",
                "3",
                "-s",
                "7.5",
                "--strand",
                "1",
                "--mask",
                "0",
                "--max-seq-len",
                str(maximum_sequence_length),
                "--max-seqs",
                str(len(target_ids)),
                "--threads",
                str(threads),
                "--format-mode",
                "0",
                "--format-output",
                "query,target,pident,qcov,tcov",
                "--remove-tmp-files",
                "1",
                "-v",
                "2",
            ]
            completed = subprocess.run(command, check=False, capture_output=True, text=True)
        normalized_command = tuple(
            "<RAW_OUTPUT>"
            if value == str(raw_path)
            else "<TEMP_DIR>/tmp"
            if value == command[5]
            else value
            for value in command
        )
        if completed.returncode:
            details = (completed.stderr or completed.stdout)[-4000:]
            raise RuntimeError(
                f"MMseqs2 threshold search failed ({completed.returncode}) for {job_id}:\n"
                f"{details}"
            )
        raw_bytes = raw_path.stat().st_size
        raw_hash = sha256_file(raw_path)
        job_pairs, edge_audit = _parse_job_edges(
            raw_path,
            set(query_ids),
            set(target_ids),
            min_identity=min_identity,
            min_coverage=min_coverage,
        )
        projected_bytes = cumulative_bytes + raw_bytes
        projected_hits = cumulative_hits + int(edge_audit["raw_hits"] or 0)
        failure = None
        if raw_bytes > max_job_output_bytes:
            failure = f"job bytes {raw_bytes} exceed {max_job_output_bytes}"
        elif int(edge_audit["raw_hits"] or 0) > max_job_hits:
            failure = f"job hits {edge_audit['raw_hits']} exceed {max_job_hits}"
        elif projected_bytes > max_total_output_bytes:
            failure = f"cumulative bytes {projected_bytes} exceed {max_total_output_bytes}"
        elif projected_hits > max_total_hits:
            failure = f"cumulative hits {projected_hits} exceed {max_total_hits}"
        if failure is not None:
            _write_checkpoint_json(
                aborted_path,
                {
                    "schema_version": 1,
                    "plan_hash": plan_hash,
                    "job_id": job_id,
                    "reason": failure,
                    "raw_output_bytes": raw_bytes,
                    "raw_output_sha256": raw_hash,
                    **edge_audit,
                },
            )
            raw_path.unlink()
            raise TEBenchmarkError(f"bounded MMseqs2 stage aborted: {failure}")
        edge_text = "".join(f"{first}\t{second}\n" for first, second in job_pairs)
        _write_checkpoint_text(edge_path, edge_text)
        audit = {
            "job_id": job_id,
            "query_count": len(query_ids),
            "target_count": len(target_ids),
            "maximum_directed_pairs": len(query_ids) * len(target_ids),
            "raw_output_bytes": raw_bytes,
            "raw_output_sha256": raw_hash,
            "spanning_edges": len(job_pairs),
            "command_json": _canonical_json(normalized_command),
            **edge_audit,
        }
        _write_checkpoint_json(
            manifest_path,
            {
                "schema_version": 1,
                "plan_hash": plan_hash,
                "job_id": job_id,
                "query_identifier_hash": job["query_identifier_hash"],
                "target_identifier_hash": job["target_identifier_hash"],
                "command": normalized_command,
                "audit": audit,
                "spanning_forest": {
                    "path": edge_path.as_posix(),
                    "size_bytes": edge_path.stat().st_size,
                    "sha256": sha256_file(edge_path),
                },
            },
        )
        raw_path.unlink()
        for first, second in job_pairs:
            global_union.union(first, second)
        audits.append(audit)
        commands.append(normalized_command)
        cumulative_bytes = projected_bytes
        cumulative_hits = projected_hits
        print(
            f"[{view_name}] completed {job_id}: {edge_audit['raw_hits']} hits, "
            f"{raw_bytes} bytes, {len(job_pairs)} spanning edges",
            flush=True,
        )

    safe_forest = _spanning_forest(global_union, safe_to_observation)
    forest_path = checkpoint_dir / "component_spanning_forest.tsv"
    _write_checkpoint_text(
        forest_path,
        "".join(f"{first}\t{second}\n" for first, second in safe_forest),
    )
    completion = {
        "schema_version": 1,
        "stage": "bounded_sharded_mmseqs_threshold_graph_complete",
        "plan_hash": plan_hash,
        "completed_jobs": len(jobs),
        "cumulative_raw_hits": cumulative_hits,
        "cumulative_raw_output_bytes": cumulative_bytes,
        "component_count": len(sequences) - len(safe_forest),
        "spanning_forest": {
            "path": forest_path.as_posix(),
            "size_bytes": forest_path.stat().st_size,
            "sha256": sha256_file(forest_path),
        },
    }
    _write_checkpoint_json(checkpoint_dir / "complete.json", completion)
    observation_forest = [
        (safe_to_observation[first], safe_to_observation[second])
        for first, second in safe_forest
    ]
    return MMseqsGraphResult(
        spanning_pairs=observation_forest,
        job_audit=pd.DataFrame(audits),
        commands=tuple(commands),
        plan_hash=plan_hash,
        checkpoint_dir=checkpoint_dir if persistent else None,
    )


def run_mmseqs_connected_components(
    sequences: Mapping[str, str],
    *,
    mmseqs_path: Path = DEFAULT_MMSEQS,
    min_identity: float = 0.8,
    min_coverage: float = 0.8,
    threads: int = 1,
    checkpoint_dir: Path | None = None,
    view_name: str = "sequence",
    query_chunk_size: int = 500,
    target_shard_size: int = 4_000,
    max_job_output_bytes: int = 512 * 1024**2,
    max_total_output_bytes: int = 64 * 1024**3,
    max_job_hits: int = 4_100_000,
    max_total_hits: int = 1_000_000_000,
) -> MMseqsGraphResult:
    """Compute threshold-graph components using bounded, resumable shards.

    Every query chunk is searched against every length-compatible target shard.
    Because bidirectional coverage ``c`` implies ``c L_q <= L_t <= L_q/c``,
    excluded length-shard pairs cannot contain a passing edge. ``--max-seqs``
    equals the complete target-shard size, so it does not truncate candidate
    targets. Raw edges are validated and reduced immediately to a spanning
    forest; immutable per-job checkpoints make an interrupted run resumable.
    """

    if not sequences:
        raise TEBenchmarkError("cannot cluster an empty sequence collection")
    if len(sequences) != len(set(sequences)):
        raise TEBenchmarkError("sequence identifiers are not unique")
    if not 0 < min_identity <= 1 or not 0 < min_coverage <= 1:
        raise TEBenchmarkError("identity and coverage thresholds must lie in (0, 1]")
    if threads < 1 or query_chunk_size < 1 or target_shard_size < 1:
        raise TEBenchmarkError("threads and shard sizes must be positive")
    if any(
        value < 1
        for value in (
            max_job_output_bytes,
            max_total_output_bytes,
            max_job_hits,
            max_total_hits,
        )
    ):
        raise TEBenchmarkError("MMseqs2 output budgets must be positive")
    if not mmseqs_path.is_file():
        raise FileNotFoundError(f"MMseqs2 executable not found: {mmseqs_path}")
    if checkpoint_dir is not None:
        return _run_sharded_mmseqs(
            sequences,
            checkpoint_dir,
            persistent=True,
            view_name=view_name,
            mmseqs_path=mmseqs_path,
            min_identity=min_identity,
            min_coverage=min_coverage,
            threads=threads,
            query_chunk_size=query_chunk_size,
            target_shard_size=target_shard_size,
            max_job_output_bytes=max_job_output_bytes,
            max_total_output_bytes=max_total_output_bytes,
            max_job_hits=max_job_hits,
            max_total_hits=max_total_hits,
        )
    with tempfile.TemporaryDirectory(prefix="hrms-te-mmseqs-sharded-") as temporary:
        return _run_sharded_mmseqs(
            sequences,
            Path(temporary),
            persistent=False,
            view_name=view_name,
            mmseqs_path=mmseqs_path,
            min_identity=min_identity,
            min_coverage=min_coverage,
            threads=threads,
            query_chunk_size=query_chunk_size,
            target_shard_size=target_shard_size,
            max_job_output_bytes=max_job_output_bytes,
            max_total_output_bytes=max_total_output_bytes,
            max_job_hits=max_job_hits,
            max_total_hits=max_total_hits,
        )


def _component_ids(
    observation_ids: Sequence[str],
    pairs: Iterable[tuple[str, str]],
    *,
    prefix: str,
) -> dict[str, str]:
    known = set(observation_ids)
    union = UnionFind(observation_ids)
    for first, second in pairs:
        if first not in known or second not in known:
            raise TEBenchmarkError(
                f"component edge contains unknown observation: {(first, second)}"
            )
        union.union(first, second)
    members: dict[str, list[str]] = defaultdict(list)
    for observation_id in observation_ids:
        members[union.find(observation_id)].append(observation_id)
    output: dict[str, str] = {}
    seen: set[str] = set()
    for component_members in members.values():
        digest = hashlib.sha256()
        for value in sorted(component_members):
            encoded = value.encode()
            digest.update(len(encoded).to_bytes(8, "little"))
            digest.update(encoded)
        component_id = f"{prefix}_{digest.hexdigest()[:24]}"
        if component_id in seen:
            raise RuntimeError("component identifier hash collision")
        seen.add(component_id)
        output.update({value: component_id for value in component_members})
    return output


def union_sequence_components(
    observation_ids: Sequence[str],
    te_pairs: Iterable[tuple[str, str]],
    window_pairs: Iterable[tuple[str, str]],
    design_pairs: Iterable[tuple[str, str]] = (),
) -> pd.DataFrame:
    """Return view/design components and their conservative transitive union."""

    observation_ids = tuple(sorted(observation_ids))
    te_pairs, window_pairs, design_pairs = (
        tuple(te_pairs),
        tuple(window_pairs),
        tuple(design_pairs),
    )
    te_components = _component_ids(observation_ids, te_pairs, prefix="te")
    window_components = _component_ids(observation_ids, window_pairs, prefix="window")
    design_components = _component_ids(observation_ids, design_pairs, prefix="pair")
    roots = _component_ids(
        observation_ids,
        (*te_pairs, *window_pairs, *design_pairs),
        prefix="root",
    )
    return pd.DataFrame(
        {
            "observation_id": observation_ids,
            "te_component_id": [te_components[value] for value in observation_ids],
            "window_component_id": [window_components[value] for value in observation_ids],
            "design_pair_component_id": [
                design_components[value] for value in observation_ids
            ],
            "root_cluster_id": [roots[value] for value in observation_ids],
        }
    )


def _partition_counts(n_roots: int) -> np.ndarray:
    raw = np.asarray(SPLIT_PROPORTIONS) * n_roots
    counts = np.floor(raw).astype(int)
    remainder = n_roots - int(counts.sum())
    counts[np.argsort(-(raw - counts), kind="stable")[:remainder]] += 1
    return counts


def _assignment_score(
    assignment: np.ndarray,
    root_cells: np.ndarray,
    root_environments: np.ndarray,
    root_sizes: np.ndarray,
) -> float:
    proportions = np.asarray(SPLIT_PROPORTIONS)
    score = 0.0
    for matrix, weight in ((root_cells, 4.0), (root_environments, 2.0)):
        expected = proportions[:, None] * matrix.sum(axis=0)[None, :]
        observed = np.stack([matrix[assignment == split].sum(axis=0) for split in range(3)])
        score += weight * float(
            np.mean(np.square((observed - expected) / np.maximum(expected, 1.0)))
        )
    expected_rows = proportions * root_sizes.sum()
    observed_rows = np.asarray(
        [root_sizes[assignment == split].sum() for split in range(3)]
    )
    return score + float(np.mean(np.square((observed_rows - expected_rows) / expected_rows)))


def _largest_root_fractions(loci: pd.DataFrame, *, by_split: bool = False) -> pd.DataFrame:
    group_columns = ["environment_id"]
    if by_split:
        if "split" not in loci:
            raise TEBenchmarkError("split-specific dominance audit requires a split column")
        group_columns.append("split")
    sizes = (
        loci.groupby([*group_columns, "root_cluster_id"], observed=True)
        .size()
        .rename("root_observations")
        .reset_index()
    )
    totals = loci.groupby(group_columns, observed=True).size().rename("group_n")
    sizes = sizes.join(totals, on=group_columns)
    sizes["fraction"] = sizes["root_observations"] / sizes["group_n"]
    return (
        sizes.sort_values(
            [*group_columns, "fraction", "root_cluster_id"],
            ascending=[True] * len(group_columns) + [False, True],
            kind="stable",
        )
        .groupby(group_columns, observed=True)
        .first()
        .reset_index()
    )


def _audit_assignment(
    loci: pd.DataFrame,
    *,
    min_roots_per_environment: int,
    max_root_fraction: float,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    cell_counts = (
        loci.groupby(["environment_id", "split", "label"], observed=True)
        .agg(observations=("observation_id", "size"), roots=("root_cluster_id", "nunique"))
        .reset_index()
    )
    root_counts = (
        loci[["environment_id", "split", "root_cluster_id"]]
        .drop_duplicates()
        .groupby(["environment_id", "split"], observed=True)
        .size()
        .rename("roots")
        .reset_index()
    )
    environments = sorted(loci["environment_id"].unique())
    root_grid = pd.MultiIndex.from_product(
        [environments, SPLIT_NAMES], names=["environment_id", "split"]
    )
    complete_roots = root_counts.set_index(["environment_id", "split"]).reindex(
        root_grid, fill_value=0
    )
    for split in ("select", "test"):
        failing = complete_roots.xs(split, level="split")["roots"] < min_roots_per_environment
        if failing.any():
            raise TEBenchmarkError(
                f"{split} has fewer than {min_roots_per_environment} roots for "
                f"{failing[failing].index.tolist()}"
            )
    cell_grid = pd.MultiIndex.from_product(
        [environments, SPLIT_NAMES, [0, 1]],
        names=["environment_id", "split", "label"],
    )
    complete_cells = cell_counts.set_index(["environment_id", "split", "label"]).reindex(
        cell_grid, fill_value=0
    )
    missing = complete_cells["observations"].eq(0)
    if missing.any():
        raise TEBenchmarkError(
            "every environment/split must contain both labels; missing cells: "
            f"{missing[missing].index.tolist()[:10]}"
        )
    largest = _largest_root_fractions(loci, by_split=True)
    gated_largest = largest.loc[largest["split"].isin(DOMINANCE_GATE_SPLITS)]
    if (gated_largest["fraction"] >= max_root_fraction).any():
        offenders = largest.loc[
            largest["split"].isin(DOMINANCE_GATE_SPLITS)
            & (largest["fraction"] >= max_root_fraction),
            ["environment_id", "split", "root_cluster_id", "fraction"],
        ].to_dict("records")
        raise TEBenchmarkError(
            f"largest root must be < {max_root_fraction:.3f} in every "
            f"environment/split for select or test: "
            f"{offenders}"
        )
    assert_no_cluster_leakage(loci)
    return cell_counts, root_counts, largest


def assign_stratified_root_splits(
    loci: pd.DataFrame,
    *,
    seed: int = 42,
    trials: int = 4096,
    min_roots_per_environment: int = 20,
    max_root_fraction: float = 0.05,
) -> SplitDiagnostics:
    """Search deterministic exact-size root splits balanced by environment and label."""

    required = {"observation_id", "environment_id", "root_cluster_id", "label"}
    missing = required.difference(loci.columns)
    if missing:
        raise TEBenchmarkError(f"split input is missing columns: {sorted(missing)}")
    if trials < 1 or min_roots_per_environment < 1:
        raise TEBenchmarkError("trials and minimum roots must be positive")
    if not 0 < max_root_fraction <= 1:
        raise TEBenchmarkError("max_root_fraction must lie in (0, 1]")
    loci = loci.copy()
    loci["root_cluster_id"] = loci["root_cluster_id"].astype(str)
    loci["environment_id"] = loci["environment_id"].astype(str)
    environments = sorted(loci["environment_id"].unique())
    label_support = loci.groupby("environment_id", observed=True)["label"].nunique()
    if (label_support != 2).any():
        raise TEBenchmarkError(
            f"every environment must contain both labels: {label_support.to_dict()}"
        )
    largest = _largest_root_fractions(loci)
    if (largest["fraction"] >= max_root_fraction).any():
        raise TEBenchmarkError(
            f"largest root must be < {max_root_fraction:.3f} in every environment: "
            f"{largest[largest['fraction'] >= max_root_fraction].to_dict('records')}"
        )

    roots = sorted(loci["root_cluster_id"].unique())
    root_index = {value: index for index, value in enumerate(roots)}
    environment_index = {value: index for index, value in enumerate(environments)}
    root_cells = np.zeros((len(roots), 2 * len(environments)), dtype=np.int64)
    root_environments = np.zeros((len(roots), len(environments)), dtype=np.int8)
    root_sizes = np.zeros(len(roots), dtype=np.int64)
    for root, environment, label in loci[
        ["root_cluster_id", "environment_id", "label"]
    ].itertuples(index=False, name=None):
        root_id, environment_id = root_index[root], environment_index[environment]
        root_cells[root_id, 2 * environment_id + int(label)] += 1
        root_environments[root_id, environment_id] = 1
        root_sizes[root_id] += 1
    required_total = int(np.ceil(min_roots_per_environment / SPLIT_PROPORTIONS[1]))
    roots_per_environment = root_environments.sum(axis=0)
    if np.any(roots_per_environment < required_total):
        failing = {
            environment: int(roots_per_environment[index])
            for environment, index in environment_index.items()
            if roots_per_environment[index] < required_total
        }
        raise TEBenchmarkError(
            f"environments need at least {required_total} roots for 60/20/20: {failing}"
        )

    counts = _partition_counts(len(roots))
    rng = np.random.default_rng(seed)
    best: tuple[float, tuple[int, ...], np.ndarray] | None = None
    feasible = 0
    for _ in range(trials):
        permutation = rng.permutation(len(roots))
        assignment = np.empty(len(roots), dtype=np.int8)
        cursor = 0
        for split, count in enumerate(counts):
            assignment[permutation[cursor : cursor + count]] = split
            cursor += int(count)
        roots_by_environment = np.stack(
            [root_environments[assignment == split].sum(axis=0) for split in range(3)]
        )
        cells = np.stack(
            [root_cells[assignment == split].sum(axis=0) for split in range(3)]
        )
        if np.any(roots_by_environment[1:] < min_roots_per_environment) or np.any(cells == 0):
            continue
        root_environment_sizes = root_cells[:, 0::2] + root_cells[:, 1::2]
        dominance_ok = True
        for split in (SPLIT_NAMES.index(name) for name in DOMINANCE_GATE_SPLITS):
            selected = assignment == split
            totals = root_environment_sizes[selected].sum(axis=0)
            maxima = root_environment_sizes[selected].max(axis=0, initial=0)
            if np.any(maxima / totals >= max_root_fraction):
                dominance_ok = False
                break
        if not dominance_ok:
            continue
        feasible += 1
        score = _assignment_score(assignment, root_cells, root_environments, root_sizes)
        if best is None or score < best[0]:
            best = (score, tuple(int(value) for value in assignment), assignment.copy())
        elif score == best[0]:
            canonical = tuple(int(value) for value in assignment)
            if canonical < best[1]:
                best = (score, canonical, assignment.copy())
    if best is None:
        raise TEBenchmarkError(
            "no feasible stratified split found; inspect root/label support before "
            "increasing trials"
        )
    root_assignment = {
        root: SPLIT_NAMES[int(best[2][index])] for root, index in root_index.items()
    }
    split_table = loci[["observation_id", "root_cluster_id"]].drop_duplicates().copy()
    split_table["split"] = split_table["root_cluster_id"].map(root_assignment)
    split_table["split_seed"] = seed
    split_table["split_hash"] = split_assignment_hash(split_table)
    audited = loci.merge(
        split_table, on=["observation_id", "root_cluster_id"], validate="one_to_one"
    )
    cells, roots_frame, largest = _audit_assignment(
        audited,
        min_roots_per_environment=min_roots_per_environment,
        max_root_fraction=max_root_fraction,
    )
    return SplitDiagnostics(
        split_table=split_table.sort_values("observation_id", kind="stable").reset_index(drop=True),
        cell_counts=cells,
        root_counts=roots_frame,
        largest_root_fraction=largest,
        objective=float(best[0]),
        feasible_candidates=feasible,
    )


def build_clustered_cohort(
    loci_path: Path,
    te_fasta_path: Path,
    window_fasta_path: Path,
    output_dir: Path,
    *,
    te_map_path: Path | None = None,
    window_map_path: Path | None = None,
    mmseqs_path: Path = DEFAULT_MMSEQS,
    min_identity: float = 0.8,
    min_coverage: float = 0.8,
    threads: int = 1,
    seed: int = 42,
    split_trials: int = 4096,
    min_roots_per_environment: int = 20,
    max_root_fraction: float = 0.05,
    query_chunk_size: int = 500,
    target_shard_size: int = 4_000,
    max_job_output_bytes: int = 512 * 1024**2,
    max_total_output_bytes: int = 64 * 1024**3,
    max_job_hits: int = 4_100_000,
    max_total_hits: int = 1_000_000_000,
) -> dict[str, Path]:
    """Cluster both sequence views, union roots, freeze a split, and export it."""

    loci = read_table(loci_path)
    te_sequences = map_fasta_ids(read_fasta(te_fasta_path), te_map_path)
    window_sequences = map_fasta_ids(read_fasta(window_fasta_path), window_map_path)
    loci = validate_loci_and_sequences(loci, te_sequences, window_sequences)
    output_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_root = output_dir / "clustering_checkpoints"
    te_result = run_mmseqs_connected_components(
        te_sequences,
        mmseqs_path=mmseqs_path,
        min_identity=min_identity,
        min_coverage=min_coverage,
        threads=threads,
        checkpoint_dir=checkpoint_root / "te_copy",
        view_name="te_copy",
        query_chunk_size=query_chunk_size,
        target_shard_size=target_shard_size,
        max_job_output_bytes=max_job_output_bytes,
        max_total_output_bytes=max_total_output_bytes,
        max_job_hits=max_job_hits,
        max_total_hits=max_total_hits,
    )
    window_result = run_mmseqs_connected_components(
        window_sequences,
        mmseqs_path=mmseqs_path,
        min_identity=min_identity,
        min_coverage=min_coverage,
        threads=threads,
        checkpoint_dir=checkpoint_root / "window_512",
        view_name="window_512",
        query_chunk_size=query_chunk_size,
        target_shard_size=target_shard_size,
        max_job_output_bytes=max_job_output_bytes,
        max_total_output_bytes=max_total_output_bytes,
        max_job_hits=max_job_hits,
        max_total_hits=max_total_hits,
    )
    design_pairs = [
        tuple(sorted(group["observation_id"].tolist()))
        for _, group in loci.groupby("pair_id", sort=True, observed=True)
    ]
    membership = union_sequence_components(
        loci["observation_id"].tolist(),
        te_result.spanning_pairs,
        window_result.spanning_pairs,
        design_pairs,
    )
    clustered = loci.merge(membership, on="observation_id", validate="one_to_one")
    membership_path = output_dir / "cluster_membership.csv"
    root_summary_path = output_dir / "pre_split_root_summary.csv"
    root_environment_path = output_dir / "pre_split_root_environment_summary.csv"
    _write_checkpoint_text(
        membership_path,
        membership.to_csv(index=False, lineterminator="\n"),
    )
    root_summary = (
        clustered.groupby("root_cluster_id", observed=True)
        .agg(
            observations=("observation_id", "size"),
            design_pairs=("pair_id", "nunique"),
            environments=("environment_id", "nunique"),
            labels=("label", "nunique"),
        )
        .reset_index()
        .sort_values(["observations", "root_cluster_id"], ascending=[False, True])
    )
    _write_checkpoint_text(
        root_summary_path,
        root_summary.to_csv(index=False, lineterminator="\n"),
    )
    root_environment = (
        clustered.groupby(["environment_id", "root_cluster_id"], observed=True)
        .agg(
            observations=("observation_id", "size"),
            design_pairs=("pair_id", "nunique"),
            labels=("label", "nunique"),
        )
        .reset_index()
    )
    environment_totals = (
        clustered.groupby("environment_id", observed=True)
        .size()
        .rename("environment_observations")
    )
    root_environment = root_environment.join(environment_totals, on="environment_id")
    root_environment["environment_fraction"] = (
        root_environment["observations"] / root_environment["environment_observations"]
    )
    root_environment = root_environment.sort_values(
        ["environment_id", "environment_fraction", "root_cluster_id"],
        ascending=[True, False, True],
    )
    _write_checkpoint_text(
        root_environment_path,
        root_environment.to_csv(index=False, lineterminator="\n"),
    )
    diagnostics = assign_stratified_root_splits(
        clustered,
        seed=seed,
        trials=split_trials,
        min_roots_per_environment=min_roots_per_environment,
        max_root_fraction=max_root_fraction,
    )
    cohort = clustered.merge(
        diagnostics.split_table,
        on=["observation_id", "root_cluster_id"],
        validate="one_to_one",
    )
    cohort["te_sequence"] = cohort["observation_id"].map(te_sequences)
    cohort["window_sequence"] = cohort["observation_id"].map(window_sequences)
    assert_no_cluster_leakage(cohort)

    outputs = {
        "cohort": output_dir / "labeled_loci_split.parquet",
        "splits": output_dir / "splits.parquet",
        "membership": membership_path,
        "pre_split_roots": root_summary_path,
        "pre_split_root_environments": root_environment_path,
        "cell_counts": output_dir / "split_cell_counts.csv",
        "root_counts": output_dir / "split_root_counts.csv",
        "largest_roots": output_dir / "largest_root_fraction.csv",
        "te_pairs": output_dir / "te_component_spanning_forest.tsv",
        "window_pairs": output_dir / "window_component_spanning_forest.tsv",
        "te_job_audit": output_dir / "te_mmseqs_job_audit.csv",
        "window_job_audit": output_dir / "window_mmseqs_job_audit.csv",
        "te_graph_plan": checkpoint_root / "te_copy/plan.json",
        "te_graph_complete": checkpoint_root / "te_copy/complete.json",
        "window_graph_plan": checkpoint_root / "window_512/plan.json",
        "window_graph_complete": checkpoint_root / "window_512/complete.json",
        "manifest": output_dir / "cluster_split_manifest.json",
    }
    cohort.to_parquet(outputs["cohort"], index=False)
    diagnostics.split_table.to_parquet(outputs["splits"], index=False)
    diagnostics.cell_counts.to_csv(outputs["cell_counts"], index=False, lineterminator="\n")
    diagnostics.root_counts.to_csv(outputs["root_counts"], index=False, lineterminator="\n")
    diagnostics.largest_root_fraction.to_csv(
        outputs["largest_roots"], index=False, lineterminator="\n"
    )
    for key, pairs in (
        ("te_pairs", te_result.spanning_pairs),
        ("window_pairs", window_result.spanning_pairs),
    ):
        pd.DataFrame(pairs, columns=["representative", "member"]).to_csv(
            outputs[key], sep="\t", index=False, header=False, lineterminator="\n"
        )
    te_result.job_audit.to_csv(outputs["te_job_audit"], index=False, lineterminator="\n")
    window_result.job_audit.to_csv(
        outputs["window_job_audit"], index=False, lineterminator="\n"
    )
    version = subprocess.run(
        [str(mmseqs_path), "version"], check=False, capture_output=True, text=True
    ).stdout.strip()
    artifacts = {
        name: {
            "path": path.as_posix(),
            "size_bytes": path.stat().st_size,
            "sha256": sha256_file(path),
        }
        for name, path in outputs.items()
        if name != "manifest"
    }
    write_json(
        outputs["manifest"],
        {
            "schema_version": 1,
            "inputs": {
                "loci": {"path": loci_path.as_posix(), "sha256": sha256_file(loci_path)},
                "te_fasta": {
                    "path": te_fasta_path.as_posix(),
                    "sha256": sha256_file(te_fasta_path),
                },
                "window_fasta": {
                    "path": window_fasta_path.as_posix(),
                    "sha256": sha256_file(window_fasta_path),
                },
            },
            "clustering": {
                "definition": (
                    "transitive union of audited MMseqs2 nucleotide-search threshold graphs "
                    "on oriented TE-copy and oriented 512-bp input-window sequences plus "
                    "the frozen positive-control design-pair edges"
                ),
                "root_rule": "transitive_union(te_copy,window_512,matched_design_pair)",
                "design_pair_count": len(design_pairs),
                "design_pair_members_never_cross_splits": True,
                "connected_component_implementation": (
                    "deterministic local union-find; equivalent to cluster-mode 1, used because "
                    "MMseqs2 Release 18 easy-cluster rejects nucleotide connected components"
                ),
                "orientation_contract": "both FASTAs are oriented upstream to the TE strand",
                "minimum_sequence_identity": min_identity,
                "minimum_bidirectional_coverage": min_coverage,
                "search_type": 3,
                "database_type": 2,
                "sensitivity": 7.5,
                "mmseqs_version": version,
                "query_chunk_size": query_chunk_size,
                "target_shard_size": target_shard_size,
                "output_budgets": {
                    "maximum_job_bytes": max_job_output_bytes,
                    "maximum_total_bytes": max_total_output_bytes,
                    "maximum_job_hits": max_job_hits,
                    "maximum_total_hits": max_total_hits,
                },
                "te_job_count": len(te_result.commands),
                "window_job_count": len(window_result.commands),
                "te_plan_hash": te_result.plan_hash,
                "window_plan_hash": window_result.plan_hash,
                "te_commands_hash": _payload_hash(te_result.commands),
                "window_commands_hash": _payload_hash(window_result.commands),
                "root_count": int(cohort["root_cluster_id"].nunique()),
            },
            "splitting": {
                "proportions": dict(zip(SPLIT_NAMES, SPLIT_PROPORTIONS, strict=True)),
                "seed": seed,
                "search_trials": split_trials,
                "feasible_candidates": diagnostics.feasible_candidates,
                "objective": diagnostics.objective,
                "minimum_roots_per_environment_in_select_and_test": min_roots_per_environment,
                "maximum_root_fraction_strict": max_root_fraction,
                "root_dominance_gate_splits": list(DOMINANCE_GATE_SPLITS),
                "split_hash": diagnostics.split_table["split_hash"].iloc[0],
                "zero_cross_split_roots": True,
                "both_labels_in_every_environment_split": True,
            },
            "artifacts": artifacts,
        },
    )
    return outputs


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--loci-table", type=Path, required=True)
    parser.add_argument("--te-fasta", type=Path, required=True)
    parser.add_argument("--window-fasta", type=Path, required=True)
    parser.add_argument("--te-map", type=Path)
    parser.add_argument("--window-map", type=Path)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--mmseqs", type=Path, default=DEFAULT_MMSEQS)
    parser.add_argument("--min-identity", type=float, default=0.8)
    parser.add_argument("--min-coverage", type=float, default=0.8)
    parser.add_argument("--threads", type=int, default=1)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--split-trials", type=int, default=4096)
    parser.add_argument("--min-roots-per-environment", type=int, default=20)
    parser.add_argument("--max-root-fraction", type=float, default=0.05)
    parser.add_argument("--query-chunk-size", type=int, default=500)
    parser.add_argument("--target-shard-size", type=int, default=4_000)
    parser.add_argument("--max-job-output-bytes", type=int, default=512 * 1024**2)
    parser.add_argument("--max-total-output-bytes", type=int, default=64 * 1024**3)
    parser.add_argument("--max-job-hits", type=int, default=4_100_000)
    parser.add_argument("--max-total-hits", type=int, default=1_000_000_000)
    args = parser.parse_args(argv)
    outputs = build_clustered_cohort(
        args.loci_table,
        args.te_fasta,
        args.window_fasta,
        args.output_dir,
        te_map_path=args.te_map,
        window_map_path=args.window_map,
        mmseqs_path=args.mmseqs,
        min_identity=args.min_identity,
        min_coverage=args.min_coverage,
        threads=args.threads,
        seed=args.seed,
        split_trials=args.split_trials,
        min_roots_per_environment=args.min_roots_per_environment,
        max_root_fraction=args.max_root_fraction,
        query_chunk_size=args.query_chunk_size,
        target_shard_size=args.target_shard_size,
        max_job_output_bytes=args.max_job_output_bytes,
        max_total_output_bytes=args.max_total_output_bytes,
        max_job_hits=args.max_job_hits,
        max_total_hits=args.max_total_hits,
    )
    for name, path in outputs.items():
        print(f"{name}: {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
