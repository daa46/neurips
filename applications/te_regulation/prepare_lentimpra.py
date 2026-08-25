"""Prepare the public K562 lentiMPRA data as a TE-annotated validation table.

This adapter consumes the processed supplementary workbooks for ENCODE
experiment ENCSR382BVV.  It never consumes raw sequencing reads and it never
invents an element-level significance label: the released K562 tables contain
continuous log2(RNA/DNA) measurements, but no per-element p-value, q-value, or
5%-FDR call.

The derived response measures sequence-intrinsic activity averaged over random
lentiviral integration contexts.  It is not an endogenous-locus causal label.
The output is intentionally blocked from HRMS until a global sequence-homology
``root_cluster_id`` has been assigned.
"""

from __future__ import annotations

import argparse
import bisect
import gzip
import hashlib
import io
import json
import math
import os
import tempfile
import urllib.request
import zipfile
from collections import Counter
from collections.abc import Iterator, Mapping
from contextlib import contextmanager
from dataclasses import dataclass
from fractions import Fraction
from pathlib import Path
from typing import Any, TextIO

import pandas as pd

from hrms.artifacts import sha256_file

DESIGN_COLUMNS = (
    "name",
    "category",
    "chr.hg38",
    "start.hg38",
    "stop.hg38",
    "str.hg38",
    "230nt sequence (15nt 5' adaptor - 200nt element - 15nt 3' adaptor)",
)
ACTIVITY_COLUMNS = (
    "name",
    "replicate 1 [log2(rna/dna)]",
    "replicate 2 [log2(rna/dna)]",
    "replicate 3 [log2(rna/dna)]",
    "mean",
)
ACTIVITY_VALUE_COLUMNS = {
    "activity_replicate_1_log2_rna_dna": "replicate_1",
    "activity_replicate_2_log2_rna_dna": "replicate_2",
    "activity_replicate_3_log2_rna_dna": "replicate_3",
    "activity_mean_log2_rna_dna": "mean",
}
ROOT_CLUSTER_STATUS = "unassigned_until_global_sequence_homology_clustering"
ACTIVITY_CALL_STATUS = "not_materialized_in_processed_tables"


class LentiMPRAError(RuntimeError):
    """Raised when a source or derived lentiMPRA artifact violates the contract."""


@dataclass(frozen=True)
class RepeatHit:
    """One qualifying RepeatMasker record for an assayed interval."""

    chrom: str
    start: int
    end: int
    strand: str
    subfamily: str
    repeat_class: str
    family: str
    repeat_id: str
    overlap_bp: int

    def selection_key(self) -> tuple[Any, ...]:
        """Return the stable ascending key implementing the frozen tie rule."""

        return (
            -self.overlap_bp,
            self.start,
            self.end,
            self.subfamily,
            self.repeat_class,
            self.family,
            self.repeat_id,
        )


@dataclass(frozen=True)
class _ChromosomeIndex:
    starts: tuple[int, ...]
    ends: tuple[int, ...]
    row_ids: tuple[int, ...]
    maximum_length: int


def load_config(path: str | Path) -> dict[str, Any]:
    """Load and validate the frozen lentiMPRA JSON configuration."""

    config = json.loads(Path(path).read_text())
    required = {
        "schema_version",
        "claim_boundary",
        "paper",
        "encode",
        "sources",
        "archive_members",
        "construction",
        "activity_call_fdr05",
    }
    missing = required.difference(config)
    if missing:
        raise LentiMPRAError(f"configuration is missing keys: {sorted(missing)}")
    if config["schema_version"] != 1:
        raise LentiMPRAError("unsupported lentiMPRA configuration schema")
    if config["activity_call_fdr05"]["status"] != ACTIVITY_CALL_STATUS:
        raise LentiMPRAError("the pinned processed source does not support an FDR label")
    if tuple(config["archive_members"]["design"].get("expected_columns", ())) != DESIGN_COLUMNS:
        raise LentiMPRAError("the pinned design-column contract has changed")
    if tuple(config["archive_members"]["activity"].get("expected_columns", ())) != ACTIVITY_COLUMNS:
        raise LentiMPRAError("the pinned activity-column contract has changed")
    fraction = config["construction"][
        "minimum_single_repeat_overlap_fraction_of_assayed_element"
    ]
    if not 0 < float(fraction) <= 1:
        raise LentiMPRAError("RepeatMasker overlap fraction must lie in (0, 1]")
    return config


def _hash_file(path: Path, algorithm: str, *, chunk_size: int = 1 << 20) -> str:
    digest = hashlib.new(algorithm)
    with path.open("rb") as handle:
        while chunk := handle.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()


def verify_pinned_file(path: str | Path, spec: Mapping[str, Any]) -> dict[str, Any]:
    """Verify a file against all byte-count and digest pins present in ``spec``."""

    source = Path(path)
    if not source.is_file():
        raise FileNotFoundError(source)
    size = source.stat().st_size
    expected_size = int(spec["size_bytes"])
    if size != expected_size:
        raise LentiMPRAError(
            f"wrong byte count for {source}: expected {expected_size}, observed {size}"
        )
    observed_sha256 = sha256_file(source)
    if observed_sha256.lower() != str(spec["sha256"]).lower():
        raise LentiMPRAError(
            f"wrong SHA-256 for {source}: expected {spec['sha256']}, observed {observed_sha256}"
        )
    record: dict[str, Any] = {
        "path": source.as_posix(),
        "size_bytes": size,
        "sha256": observed_sha256,
    }
    if "md5" in spec:
        observed_md5 = _hash_file(source, "md5")
        if observed_md5.lower() != str(spec["md5"]).lower():
            raise LentiMPRAError(
                f"wrong MD5 for {source}: expected {spec['md5']}, observed {observed_md5}"
            )
        record["md5"] = observed_md5
    return record


def _download_pinned_source(spec: Mapping[str, Any], path: Path) -> Path:
    """Download a pinned source atomically; reuse only a byte-identical file."""

    if path.exists():
        verify_pinned_file(path, spec)
        return path
    path.parent.mkdir(parents=True, exist_ok=True)
    partial = path.with_name(path.name + ".part")
    if partial.exists():
        raise LentiMPRAError(f"stale partial download exists: {partial}")
    request = urllib.request.Request(
        str(spec["url"]), headers={"User-Agent": "HRMS-lentiMPRA/1"}
    )
    try:
        with urllib.request.urlopen(request) as response, partial.open("xb") as output:
            while chunk := response.read(1 << 20):
                output.write(chunk)
            output.flush()
            os.fsync(output.fileno())
        verify_pinned_file(partial, spec)
        os.replace(partial, path)
    except Exception:
        # Retain partial bytes for an explicit inspection or resumable external download.
        raise
    return path


def download_sources(config: Mapping[str, Any], *, base_dir: str | Path = ".") -> list[Path]:
    """Download or verify the processed archive and public RepeatMasker source."""

    base = Path(base_dir)
    paths = []
    for name in ("processed_tables_archive", "repeatmasker"):
        spec = config["sources"][name]
        paths.append(_download_pinned_source(spec, base / str(spec["path"])))
    return paths


def _copy_and_hash(source: TextIO | Any, destination: Path) -> tuple[int, str, str]:
    sha256 = hashlib.sha256()
    md5 = hashlib.md5()  # noqa: S324 - MD5 is a publisher-provided integrity pin, not security.
    size = 0
    with destination.open("xb") as output:
        while chunk := source.read(1 << 20):
            output.write(chunk)
            sha256.update(chunk)
            md5.update(chunk)
            size += len(chunk)
        output.flush()
        os.fsync(output.fileno())
    return size, sha256.hexdigest(), md5.hexdigest()


def extract_verified_members(
    archive: str | Path, config: Mapping[str, Any], destination: str | Path
) -> dict[str, Path]:
    """Extract only the two pinned K562 workbooks and verify their exact bytes."""

    archive_path = Path(archive)
    destination_path = Path(destination)
    destination_path.mkdir(parents=True, exist_ok=True)
    outputs: dict[str, Path] = {}
    with zipfile.ZipFile(archive_path) as bundle:
        for role in ("design", "activity"):
            spec = config["archive_members"][role]
            member_name = str(spec["path"])
            try:
                info = bundle.getinfo(member_name)
            except KeyError as error:
                raise LentiMPRAError(
                    f"archive lacks pinned {role} member: {member_name}"
                ) from error
            if info.file_size != int(spec["size_bytes"]):
                raise LentiMPRAError(
                    f"wrong uncompressed size for {role}: expected {spec['size_bytes']}, "
                    f"observed {info.file_size}"
                )
            output = destination_path / f"{role}.xlsx"
            if output.exists():
                raise LentiMPRAError(f"refusing to replace extracted member: {output}")
            with bundle.open(info) as source:
                size, observed_sha256, observed_md5 = _copy_and_hash(source, output)
            if (
                size != int(spec["size_bytes"])
                or observed_sha256.lower() != str(spec["sha256"]).lower()
                or observed_md5.lower() != str(spec["md5"]).lower()
            ):
                output.unlink()
                raise LentiMPRAError(f"digest verification failed for extracted {role} workbook")
            outputs[role] = output
    return outputs


def _trim_trailing_empty(values: tuple[Any, ...]) -> tuple[Any, ...]:
    end = len(values)
    while end and values[end - 1] is None:
        end -= 1
    return values[:end]


def _read_xlsx_sheet(
    path: Path,
    *,
    sheet: str,
    header_row_zero_based: int,
    expected_columns: tuple[str, ...],
) -> pd.DataFrame:
    """Stream one exact worksheet through openpyxl's read-only mode."""

    try:
        from openpyxl import load_workbook
    except ImportError as error:  # pragma: no cover - exercised only without the optional extra
        raise LentiMPRAError("install the TE dependencies with `uv sync --extra te`") from error

    workbook = load_workbook(path, read_only=True, data_only=True)
    try:
        if sheet not in workbook.sheetnames:
            raise LentiMPRAError(f"{path} lacks pinned worksheet {sheet!r}")
        worksheet = workbook[sheet]
        rows = worksheet.iter_rows(values_only=True)
        for _ in range(header_row_zero_based):
            next(rows, None)
        header = _trim_trailing_empty(tuple(next(rows, ())))
        if header != expected_columns:
            raise LentiMPRAError(
                f"unexpected columns in {path.name}/{sheet}: expected {expected_columns}, "
                f"observed {header}"
            )
        records = []
        for row in rows:
            values = tuple(row[: len(expected_columns)])
            if all(value is None for value in values):
                continue
            records.append(values)
        return pd.DataFrame.from_records(records, columns=expected_columns)
    finally:
        workbook.close()


def load_processed_workbooks(
    design_path: str | Path,
    activity_path: str | Path,
    config: Mapping[str, Any],
    *,
    verify_row_counts: bool = True,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Load exact K562 design and summary sheets from verified workbooks."""

    design_spec = config["archive_members"]["design"]
    activity_spec = config["archive_members"]["activity"]
    design = _read_xlsx_sheet(
        Path(design_path),
        sheet=str(design_spec["sheet"]),
        header_row_zero_based=int(design_spec["header_row_zero_based"]),
        expected_columns=DESIGN_COLUMNS,
    )
    activity = _read_xlsx_sheet(
        Path(activity_path),
        sheet=str(activity_spec["sheet"]),
        header_row_zero_based=int(activity_spec["header_row_zero_based"]),
        expected_columns=ACTIVITY_COLUMNS,
    )
    if verify_row_counts:
        expected_design = int(design_spec["expected_data_rows"])
        expected_activity = int(activity_spec["expected_data_rows"])
        if len(design) != expected_design or len(activity) != expected_activity:
            raise LentiMPRAError(
                "processed workbook row count drift: "
                f"design {len(design)} != {expected_design} or "
                f"activity {len(activity)} != {expected_activity}"
            )
    return design, activity


def _require_exact_columns(frame: pd.DataFrame, expected: tuple[str, ...], role: str) -> None:
    observed = tuple(str(column) for column in frame.columns)
    if observed != expected:
        raise LentiMPRAError(
            f"unexpected {role} columns: expected {expected}, observed {observed}"
        )


def _normalise_processed_tables(
    design: pd.DataFrame, activity: pd.DataFrame
) -> tuple[pd.DataFrame, dict[str, int]]:
    _require_exact_columns(design, DESIGN_COLUMNS, "design")
    _require_exact_columns(activity, ACTIVITY_COLUMNS, "activity")
    for role, frame in (("design", design), ("activity", activity)):
        if frame["name"].isna().any() or (frame["name"].astype(str).str.len() == 0).any():
            raise LentiMPRAError(f"{role} contains a missing element name")
        if frame["name"].duplicated().any():
            duplicate = frame.loc[frame["name"].duplicated(), "name"].iloc[0]
            raise LentiMPRAError(f"{role} contains duplicate element name {duplicate!r}")

    unknown_activity = sorted(set(activity["name"]) - set(design["name"]))
    if unknown_activity:
        raise LentiMPRAError(
            f"activity table contains names absent from design, first={unknown_activity[0]!r}"
        )
    numeric_activity = activity.copy()
    for column in ACTIVITY_COLUMNS[1:]:
        numeric_activity[column] = pd.to_numeric(numeric_activity[column], errors="coerce")
        if numeric_activity[column].isna().any():
            raise LentiMPRAError(f"activity column {column!r} contains missing/non-numeric values")
        if not numeric_activity[column].map(math.isfinite).all():
            raise LentiMPRAError(f"activity column {column!r} contains non-finite values")

    joined = numeric_activity.merge(design, on="name", how="left", validate="one_to_one")
    starts = pd.to_numeric(joined["start.hg38"], errors="coerce")
    ends = pd.to_numeric(joined["stop.hg38"], errors="coerce")
    chroms = joined["chr.hg38"]
    integral = starts.map(lambda value: pd.isna(value) or float(value).is_integer()) & ends.map(
        lambda value: pd.isna(value) or float(value).is_integer()
    )
    if not integral.all():
        raise LentiMPRAError("design contains a non-integral genomic coordinate")
    valid = chroms.notna() & starts.notna() & ends.notna() & (starts >= 0) & (ends > starts)
    diagnostics = {
        "design_rows": int(len(design)),
        "activity_rows": int(len(activity)),
        "design_rows_without_processed_activity": int(len(design) - len(activity)),
        "activity_rows_with_valid_coordinates": int(valid.sum()),
        "activity_rows_without_valid_coordinates": int((~valid).sum()),
    }
    joined = joined.loc[valid].copy()
    joined["start.hg38"] = starts.loc[valid].astype("int64")
    joined["stop.hg38"] = ends.loc[valid].astype("int64")
    return joined, diagnostics


def _build_interval_indexes(elements: pd.DataFrame) -> dict[str, _ChromosomeIndex]:
    indexes: dict[str, _ChromosomeIndex] = {}
    for chrom, group in elements.groupby("chr.hg38", sort=True):
        ordered = group.sort_values(
            ["start.hg38", "stop.hg38", "_row_id"], kind="mergesort"
        )
        starts = tuple(int(value) for value in ordered["start.hg38"])
        ends = tuple(int(value) for value in ordered["stop.hg38"])
        row_ids = tuple(int(value) for value in ordered["_row_id"])
        maximum_length = max(end - start for start, end in zip(starts, ends, strict=True))
        indexes[str(chrom)] = _ChromosomeIndex(starts, ends, row_ids, maximum_length)
    return indexes


@contextmanager
def _open_repeatmasker(path: Path) -> Iterator[TextIO]:
    if path.suffix == ".gz":
        with gzip.open(path, "rt", encoding="utf-8", newline="") as handle:
            yield handle
    else:
        with path.open("rt", encoding="utf-8", newline="") as handle:
            yield handle


def annotate_repeatmasker(
    elements: pd.DataFrame,
    repeatmasker_path: str | Path,
    *,
    allowed_repeat_classes: set[str],
    minimum_overlap_fraction: float,
) -> tuple[pd.DataFrame, dict[str, int]]:
    """Attach one deterministic qualifying RepeatMasker hit to each element.

    The threshold denominator is the full assayed genomic interval.  One
    RepeatMasker record must meet it by itself; overlaps are never unioned
    across records.
    """

    annotated = elements.reset_index(drop=True).copy()
    annotated["_row_id"] = range(len(annotated))
    indexes = _build_interval_indexes(annotated)
    threshold = Fraction(str(minimum_overlap_fraction))
    best: list[RepeatHit | None] = [None] * len(annotated)
    hit_counts = [0] * len(annotated)
    diagnostics = Counter(
        repeatmasker_records=0,
        allowed_class_records=0,
        overlap_pairs=0,
        qualifying_overlap_pairs=0,
    )

    with _open_repeatmasker(Path(repeatmasker_path)) as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip() or line.startswith("#"):
                continue
            fields = line.rstrip("\n").split("\t")
            if len(fields) < 17:
                raise LentiMPRAError(
                    f"RepeatMasker row {line_number} has {len(fields)} fields; expected >=17"
                )
            diagnostics["repeatmasker_records"] += 1
            repeat_class = fields[11]
            if repeat_class not in allowed_repeat_classes:
                continue
            diagnostics["allowed_class_records"] += 1
            chrom = fields[5]
            index = indexes.get(chrom)
            if index is None:
                continue
            try:
                repeat_start = int(fields[6])
                repeat_end = int(fields[7])
            except ValueError as error:
                raise LentiMPRAError(
                    f"RepeatMasker row {line_number} has non-integral coordinates"
                ) from error
            if repeat_start < 0 or repeat_end <= repeat_start:
                raise LentiMPRAError(f"RepeatMasker row {line_number} has an invalid interval")

            left = bisect.bisect_right(index.starts, repeat_start - index.maximum_length)
            right = bisect.bisect_left(index.starts, repeat_end)
            for position in range(left, right):
                element_start = index.starts[position]
                element_end = index.ends[position]
                overlap = min(element_end, repeat_end) - max(element_start, repeat_start)
                if overlap <= 0:
                    continue
                diagnostics["overlap_pairs"] += 1
                element_length = element_end - element_start
                if overlap * threshold.denominator < element_length * threshold.numerator:
                    continue
                diagnostics["qualifying_overlap_pairs"] += 1
                row_id = index.row_ids[position]
                hit_counts[row_id] += 1
                candidate = RepeatHit(
                    chrom=chrom,
                    start=repeat_start,
                    end=repeat_end,
                    strand=fields[9],
                    subfamily=fields[10],
                    repeat_class=repeat_class,
                    family=fields[12],
                    repeat_id=fields[16],
                    overlap_bp=overlap,
                )
                if best[row_id] is None or candidate.selection_key() < best[row_id].selection_key():
                    best[row_id] = candidate

    selected_ids = [row_id for row_id, hit in enumerate(best) if hit is not None]
    result = annotated.iloc[selected_ids].copy()
    selected_hits = [best[row_id] for row_id in selected_ids]
    # ``selected_ids`` only contains non-null hits; the assertion documents that invariant.
    assert all(hit is not None for hit in selected_hits)
    hits = [hit for hit in selected_hits if hit is not None]
    result["te_class"] = [hit.repeat_class for hit in hits]
    result["te_family"] = [hit.family for hit in hits]
    result["te_subfamily"] = [hit.subfamily for hit in hits]
    result["repeatmasker_chrom"] = [hit.chrom for hit in hits]
    result["repeatmasker_start"] = [hit.start for hit in hits]
    result["repeatmasker_end"] = [hit.end for hit in hits]
    result["repeatmasker_strand"] = [hit.strand for hit in hits]
    result["repeatmasker_id"] = [hit.repeat_id for hit in hits]
    result["repeat_overlap_bp"] = [hit.overlap_bp for hit in hits]
    result["repeat_overlap_fraction_of_assayed_element"] = [
        hit.overlap_bp / (int(end) - int(start))
        for hit, start, end in zip(
            hits, result["start.hg38"], result["stop.hg38"], strict=True
        )
    ]
    result["qualified_repeat_hit_count"] = [hit_counts[row_id] for row_id in selected_ids]
    result = result.drop(columns=["_row_id"])
    diagnostics["assayed_elements_with_qualifying_repeat"] = len(result)
    diagnostics["assayed_elements_without_qualifying_repeat"] = len(elements) - len(result)
    return result, dict(diagnostics)


def _element_sequence(sequence: Any) -> str:
    if pd.isna(sequence):
        return ""
    value = str(sequence)
    return value[15:-15] if len(value) == 230 else ""


def build_te_activity_tables(
    design: pd.DataFrame,
    activity: pd.DataFrame,
    repeatmasker_path: str | Path,
    config: Mapping[str, Any],
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    """Construct element-level and long continuous-activity tables."""

    joined, source_diagnostics = _normalise_processed_tables(design, activity)
    construction = config["construction"]
    annotated, overlap_diagnostics = annotate_repeatmasker(
        joined,
        repeatmasker_path,
        allowed_repeat_classes=set(construction["allowed_repeat_classes"]),
        minimum_overlap_fraction=float(
            construction["minimum_single_repeat_overlap_fraction_of_assayed_element"]
        ),
    )
    sequence_column = DESIGN_COLUMNS[-1]
    elements = pd.DataFrame(
        {
            "observation_id": annotated["name"].astype(str),
            "source_name": "ENCODE_ENCSR382BVV_K562_large_scale_lentiMPRA",
            "source_locus_id": annotated.apply(
                lambda row: f"{row['chr.hg38']}:{row['start.hg38']}-{row['stop.hg38']}", axis=1
            ),
            "assembly": str(construction["assembly"]),
            "chrom": annotated["chr.hg38"].astype(str),
            "start": annotated["start.hg38"].astype("int64"),
            "end": annotated["stop.hg38"].astype("int64"),
            "assayed_element_length_bp": (
                annotated["stop.hg38"] - annotated["start.hg38"]
            ).astype("int64"),
            "assay_strand": annotated["str.hg38"].fillna("").astype(str),
            "design_category": annotated["category"].fillna("").astype(str),
            "sequence_230nt_with_adaptors": annotated[sequence_column].fillna("").astype(str),
            "element_sequence_200nt": annotated[sequence_column].map(_element_sequence),
            "te_class": annotated["te_class"],
            "te_family": annotated["te_family"],
            "te_subfamily": annotated["te_subfamily"],
            "repeatmasker_chrom": annotated["repeatmasker_chrom"],
            "repeatmasker_start": annotated["repeatmasker_start"].astype("int64"),
            "repeatmasker_end": annotated["repeatmasker_end"].astype("int64"),
            "repeatmasker_strand": annotated["repeatmasker_strand"],
            "repeatmasker_id": annotated["repeatmasker_id"].astype(str),
            "repeat_overlap_bp": annotated["repeat_overlap_bp"].astype("int64"),
            "repeat_overlap_fraction_of_assayed_element": annotated[
                "repeat_overlap_fraction_of_assayed_element"
            ],
            "qualified_repeat_hit_count": annotated["qualified_repeat_hit_count"].astype(
                "int64"
            ),
            "activity_replicate_1_log2_rna_dna": annotated[
                "replicate 1 [log2(rna/dna)]"
            ],
            "activity_replicate_2_log2_rna_dna": annotated[
                "replicate 2 [log2(rna/dna)]"
            ],
            "activity_replicate_3_log2_rna_dna": annotated[
                "replicate 3 [log2(rna/dna)]"
            ],
            "activity_mean_log2_rna_dna": annotated["mean"],
            # CSV blank is the serialized nullable value; it is not a negative call.
            "activity_active_fdr05": "",
            "activity_call_status": ACTIVITY_CALL_STATUS,
            "environment_id": annotated["te_family"],
            "root_cluster_id": "",
            "root_cluster_status": ROOT_CLUSTER_STATUS,
            "sample_weight": 1.0,
            "claim_boundary": str(config["claim_boundary"]),
        }
    )
    elements = elements.sort_values(
        ["environment_id", "te_subfamily", "chrom", "start", "end", "observation_id"],
        kind="mergesort",
        ignore_index=True,
    )

    long_identifiers = [
        "observation_id",
        "source_locus_id",
        "assembly",
        "chrom",
        "start",
        "end",
        "assay_strand",
        "design_category",
        "te_class",
        "te_family",
        "te_subfamily",
        "repeat_overlap_bp",
        "repeat_overlap_fraction_of_assayed_element",
        "activity_active_fdr05",
        "activity_call_status",
        "environment_id",
        "root_cluster_id",
        "root_cluster_status",
        "sample_weight",
    ]
    long = elements.melt(
        id_vars=long_identifiers,
        value_vars=list(ACTIVITY_VALUE_COLUMNS),
        var_name="_activity_source_column",
        value_name="activity_log2_rna_dna",
    )
    long["activity_statistic"] = long["_activity_source_column"].map(ACTIVITY_VALUE_COLUMNS)
    long = long.drop(columns=["_activity_source_column"])
    long_prefix = [
        "observation_id",
        "source_locus_id",
        "assembly",
        "chrom",
        "start",
        "end",
        "assay_strand",
        "design_category",
        "te_class",
        "te_family",
        "te_subfamily",
        "repeat_overlap_bp",
        "repeat_overlap_fraction_of_assayed_element",
    ]
    long = long[
        long_prefix
        + [
            "activity_statistic",
            "activity_log2_rna_dna",
            "activity_active_fdr05",
            "activity_call_status",
            "environment_id",
            "root_cluster_id",
            "root_cluster_status",
            "sample_weight",
        ]
    ].sort_values(
        ["environment_id", "te_subfamily", "observation_id", "activity_statistic"],
        kind="mergesort",
        ignore_index=True,
    )

    sequence_lengths = elements["sequence_230nt_with_adaptors"].str.len()
    diagnostics: dict[str, Any] = {
        **source_diagnostics,
        **overlap_diagnostics,
        "te_activity_element_rows": int(len(elements)),
        "te_activity_long_rows": int(len(long)),
        "unique_source_loci": int(elements["source_locus_id"].nunique()),
        "unique_te_families": int(elements["te_family"].nunique()),
        "unique_te_subfamilies": int(elements["te_subfamily"].nunique()),
        "sequence_rows_exactly_230nt": int((sequence_lengths == 230).sum()),
        "sequence_rows_not_230nt": int((sequence_lengths != 230).sum()),
        "materialized_fdr05_calls": 0,
    }
    return elements, long, diagnostics


def _install_immutable(partial: Path, destination: Path) -> None:
    if destination.exists():
        if destination.is_file() and sha256_file(destination) == sha256_file(partial):
            partial.unlink()
            return
        raise LentiMPRAError(f"refusing to replace non-identical artifact: {destination}")
    os.replace(partial, destination)


@contextmanager
def _deterministic_gzip_text_writer(path: Path) -> Iterator[TextIO]:
    path.parent.mkdir(parents=True, exist_ok=True)
    partial = path.with_name(path.name + ".part")
    if partial.exists():
        raise LentiMPRAError(f"stale partial artifact exists: {partial}")
    with partial.open("xb") as raw:
        with (
            gzip.GzipFile(filename="", fileobj=raw, mode="wb", mtime=0) as compressed,
            io.TextIOWrapper(compressed, encoding="utf-8", newline="") as text,
        ):
            yield text
        raw.flush()
        os.fsync(raw.fileno())
    _install_immutable(partial, path)


def write_dataframe_deterministic(frame: pd.DataFrame, path: str | Path) -> Path:
    """Write a stable gzip CSV, accepting only byte-identical repeated output."""

    destination = Path(path)
    with _deterministic_gzip_text_writer(destination) as handle:
        frame.to_csv(handle, index=False, lineterminator="\n", float_format="%.17g", na_rep="")
    return destination


def _artifact_record(path: Path) -> dict[str, Any]:
    return {
        "path": path.as_posix(),
        "size_bytes": path.stat().st_size,
        "sha256": sha256_file(path),
    }


def write_outputs(
    elements: pd.DataFrame,
    long: pd.DataFrame,
    diagnostics: Mapping[str, Any],
    config: Mapping[str, Any],
    output_dir: str | Path,
    *,
    source_records: Mapping[str, Any] | None = None,
) -> dict[str, Path]:
    """Write immutable tables and an auditable preflight contract."""

    output = Path(output_dir)
    elements_path = output / "lentimpra_k562_te_elements.csv.gz"
    long_path = output / "lentimpra_k562_te_activity_long.csv.gz"
    preflight_path = output / "lentimpra_k562_preflight.json"
    write_dataframe_deterministic(elements, elements_path)
    write_dataframe_deterministic(long, long_path)

    environment_counts = []
    for family, group in elements.groupby("environment_id", sort=True):
        environment_counts.append(
            {
                "environment_id": str(family),
                "assayed_element_orientations": int(len(group)),
                "source_loci": int(group["source_locus_id"].nunique()),
                "te_subfamilies": int(group["te_subfamily"].nunique()),
            }
        )
    mean_values = elements["activity_mean_log2_rna_dna"]
    preflight = {
        "schema_version": 1,
        "stage": "processed_k562_lentimpra_te_annotation_preflight",
        "claim_boundary": config["claim_boundary"],
        "source_experiment": config["encode"],
        "source_records": dict(source_records or {}),
        "construction": config["construction"],
        "activity_call_fdr05": config["activity_call_fdr05"],
        "counts": dict(diagnostics),
        "environment_counts": environment_counts,
        "activity_mean_summary": {
            "minimum": float(mean_values.min()) if len(mean_values) else None,
            "median": float(mean_values.median()) if len(mean_values) else None,
            "maximum": float(mean_values.max()) if len(mean_values) else None,
        },
        "output_contract": {
            "elements": {
                "unit": "one assayed element orientation with a qualifying single RepeatMasker hit",
                "columns": list(elements.columns),
                "path": elements_path.as_posix(),
            },
            "activity_long": {
                "unit": "one assayed element orientation by released activity statistic",
                "statistics": sorted(ACTIVITY_VALUE_COLUMNS.values()),
                "columns": list(long.columns),
                "path": long_path.as_posix(),
            },
            "activity_active_fdr05": {
                "type": "nullable boolean",
                "serialized_null": "empty CSV field",
                "all_values_null": True,
            },
        },
        "gates": {
            "ready_for_homology_clustering": True,
            "ready_for_split": False,
            "ready_for_hrms": False,
            "blockers": [
                "Assign root_cluster_id by global sequence-homology clustering before splitting.",
                "Define candidate predictions and a validation loss; reporter activity is a "
                "response, not a model loss.",
                "Do not treat the absent element-level 5%-FDR call as a negative label.",
            ],
        },
        "artifacts": {
            "elements": _artifact_record(elements_path),
            "activity_long": _artifact_record(long_path),
        },
    }
    output.mkdir(parents=True, exist_ok=True)
    partial = preflight_path.with_name(preflight_path.name + ".part")
    if partial.exists():
        raise LentiMPRAError(f"stale partial artifact exists: {partial}")
    partial.write_text(json.dumps(preflight, indent=2, sort_keys=True) + "\n")
    _install_immutable(partial, preflight_path)
    return {"elements": elements_path, "activity_long": long_path, "preflight": preflight_path}


def prepare(
    config_path: str | Path,
    output_dir: str | Path,
    *,
    base_dir: str | Path = ".",
) -> dict[str, Path]:
    """Run verified archive extraction, parsing, TE annotation, and preflight writing."""

    config = load_config(config_path)
    base = Path(base_dir)
    archive_spec = config["sources"]["processed_tables_archive"]
    repeat_spec = config["sources"]["repeatmasker"]
    archive = base / str(archive_spec["path"])
    repeatmasker = base / str(repeat_spec["path"])
    source_records = {
        "processed_tables_archive": verify_pinned_file(archive, archive_spec),
        "repeatmasker": verify_pinned_file(repeatmasker, repeat_spec),
    }
    with tempfile.TemporaryDirectory(prefix="hrms-lentimpra-") as temporary:
        members = extract_verified_members(archive, config, temporary)
        for role in ("design", "activity"):
            member_record = verify_pinned_file(members[role], config["archive_members"][role])
            member_record["path"] = (
                f"{archive.as_posix()}::{config['archive_members'][role]['path']}"
            )
            source_records[f"{role}_workbook"] = member_record
        design, activity = load_processed_workbooks(
            members["design"], members["activity"], config, verify_row_counts=True
        )
        elements, long, diagnostics = build_te_activity_tables(
            design, activity, repeatmasker, config
        )
    return write_outputs(
        elements,
        long,
        diagnostics,
        config,
        output_dir,
        source_records=source_records,
    )


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config", type=Path, default=Path("configs/te/lentimpra_k562.json")
    )
    parser.add_argument("--base-dir", type=Path, default=Path("."))
    subparsers = parser.add_subparsers(dest="command", required=True)
    subparsers.add_parser("download", help="download or verify the two pinned public sources")
    prepare_parser = subparsers.add_parser(
        "prepare", help="build TE-annotated continuous-activity tables and preflight"
    )
    prepare_parser.add_argument(
        "--output-dir", type=Path, default=Path("data/te_lentimpra/processed")
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    config = load_config(args.config)
    if args.command == "download":
        for path in download_sources(config, base_dir=args.base_dir):
            print(path)
        return 0
    outputs = prepare(args.config, args.output_dir, base_dir=args.base_dir)
    for role, path in outputs.items():
        print(f"{role}: {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
