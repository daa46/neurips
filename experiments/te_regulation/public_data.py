"""Deterministic construction utilities for the public TE--cCRE benchmark.

The task label is deliberately narrow: a positive TE copy overlaps an ENCODE
Registry V4 pELS/dELS cCRE by at least the prespecified fraction *of the cCRE*.
A negative TE copy has no Registry V4 cCRE of any class within its buffered
interval.  These are annotation labels, not causal regulatory/non-regulatory
ground truth.
"""

from __future__ import annotations

import bisect
import csv
import gzip
import hashlib
import heapq
import io
import json
import math
import os
import shutil
import subprocess
import urllib.request
from array import array
from collections import Counter, defaultdict
from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import dataclass
from fractions import Fraction
from pathlib import Path
from typing import Any, TextIO

from hrms.artifacts import sha256_file


class PublicDataError(RuntimeError):
    """Raised when a source or derived public-benchmark artifact is invalid."""


@dataclass(frozen=True)
class SourceSpec:
    """Pinned remote input and its expected bytes."""

    name: str
    url: str
    path: Path
    size_bytes: int
    sha256: str


@dataclass(frozen=True)
class EnhancerHit:
    """The deterministically selected enhancer-like cCRE for one TE copy."""

    start: int
    end: int
    ccre_class: str
    overlap_bp: int

    @property
    def overlap_fraction(self) -> float:
        return self.overlap_bp / (self.end - self.start)


def load_config(path: str | Path) -> dict[str, Any]:
    """Load and minimally validate the frozen JSON configuration."""

    source = Path(path)
    config = json.loads(source.read_text())
    required = {
        "schema_version",
        "sources",
        "coordinates",
        "labels",
        "environments",
        "sampling",
        "sequences",
        "clustering",
        "splits",
        "gates",
    }
    missing = required.difference(config)
    if missing:
        raise PublicDataError(f"configuration is missing keys: {sorted(missing)}")
    if config["schema_version"] != 1:
        raise PublicDataError("unsupported public benchmark configuration schema")
    if len(set(config["environments"])) != len(config["environments"]):
        raise PublicDataError("environment names must be unique")
    return config


def source_specs(config: dict[str, Any], *, base_dir: str | Path = ".") -> tuple[SourceSpec, ...]:
    """Return canonical source specifications from a loaded config."""

    base = Path(base_dir)
    specs = []
    for name, value in sorted(config["sources"].items()):
        specs.append(
            SourceSpec(
                name=name,
                url=str(value["url"]),
                path=base / str(value["path"]),
                size_bytes=int(value["size_bytes"]),
                sha256=str(value["sha256"]).lower(),
            )
        )
    return tuple(specs)


def verify_file(path: str | Path, *, size_bytes: int, sha256: str) -> dict[str, Any]:
    """Verify a pinned file and return its content-addressed record."""

    source = Path(path)
    if not source.is_file():
        raise FileNotFoundError(source)
    actual_size = source.stat().st_size
    if actual_size != size_bytes:
        raise PublicDataError(
            f"wrong byte count for {source}: expected {size_bytes}, observed {actual_size}"
        )
    actual_hash = sha256_file(source)
    if actual_hash.lower() != sha256.lower():
        raise PublicDataError(
            f"wrong SHA-256 for {source}: expected {sha256}, observed {actual_hash}"
        )
    return {"path": source.as_posix(), "size_bytes": actual_size, "sha256": actual_hash}


def download_source(spec: SourceSpec, *, chunk_size: int = 1 << 20) -> Path:
    """Stream one source to an atomic ``.part`` file and verify before install.

    Existing valid files are reused. Existing invalid files are never silently
    overwritten, and a stale partial file must be removed explicitly.
    """

    if spec.path.exists():
        verify_file(spec.path, size_bytes=spec.size_bytes, sha256=spec.sha256)
        return spec.path
    spec.path.parent.mkdir(parents=True, exist_ok=True)
    partial = spec.path.with_name(spec.path.name + ".part")
    if partial.exists():
        raise PublicDataError(
            f"partial download already exists: {partial}; inspect or remove it before retrying"
        )

    request = urllib.request.Request(spec.url, headers={"User-Agent": "HRMS-public-data/1"})
    digest = hashlib.sha256()
    observed = 0
    try:
        with urllib.request.urlopen(request) as response, partial.open("xb") as output:
            advertised = response.headers.get("Content-Length")
            if advertised is not None and int(advertised) != spec.size_bytes:
                raise PublicDataError(
                    f"server advertises {advertised} bytes for {spec.name}; "
                    f"configuration pins {spec.size_bytes}"
                )
            while chunk := response.read(chunk_size):
                output.write(chunk)
                digest.update(chunk)
                observed += len(chunk)
            output.flush()
            os.fsync(output.fileno())
        if observed != spec.size_bytes or digest.hexdigest() != spec.sha256:
            raise PublicDataError(
                f"download verification failed for {spec.name}: "
                f"size={observed}, sha256={digest.hexdigest()}"
            )
        os.replace(partial, spec.path)
    except Exception:
        # Retain partial bytes for diagnosis/resumption by an explicit external tool.
        raise
    return spec.path


def download_all(config: dict[str, Any], *, base_dir: str | Path = ".") -> tuple[Path, ...]:
    """Download or verify every pinned public input."""

    return tuple(download_source(spec) for spec in source_specs(config, base_dir=base_dir))


def _open_text(path: Path) -> TextIO:
    if path.suffix == ".gz":
        return gzip.open(path, "rt", encoding="utf-8", newline="")
    return path.open("rt", encoding="utf-8", newline="")


@contextmanager
def _deterministic_gzip_writer(path: Path) -> Iterator[TextIO]:
    """Yield a deterministic gzip text stream, then immutably install it."""

    path.parent.mkdir(parents=True, exist_ok=True)
    partial = path.with_name(path.name + ".part")
    if partial.exists():
        raise PublicDataError(f"stale partial artifact exists: {partial}")
    with partial.open("xb") as raw:
        with (  # noqa: SIM117 - raw must remain open for fsync after gzip's trailer is written
            gzip.GzipFile(filename="", fileobj=raw, mode="wb", mtime=0) as compressed,
            io.TextIOWrapper(compressed, encoding="utf-8", newline="") as text,
        ):
            yield text
        raw.flush()
        os.fsync(raw.fileno())
    _install_immutable(partial, path)


def _install_immutable(partial: Path, destination: Path) -> None:
    """Install a staged artifact, accepting an existing byte-identical copy only."""

    if destination.exists():
        if destination.is_file() and sha256_file(destination) == sha256_file(partial):
            partial.unlink()
            return
        raise PublicDataError(
            f"refusing to replace non-identical immutable artifact: {destination}"
        )
    os.replace(partial, destination)


def write_json_immutable(path: Path, payload: dict[str, Any]) -> None:
    """Write stable JSON without silently changing an existing artifact."""

    path.parent.mkdir(parents=True, exist_ok=True)
    partial = path.with_name(path.name + ".part")
    if partial.exists():
        raise PublicDataError(f"stale partial artifact exists: {partial}")
    partial.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    _install_immutable(partial, path)


def write_csv_immutable(path: Path, fieldnames: list[str], rows: Iterator[dict[str, Any]]) -> None:
    """Write a stable uncompressed CSV artifact."""

    path.parent.mkdir(parents=True, exist_ok=True)
    partial = path.with_name(path.name + ".part")
    if partial.exists():
        raise PublicDataError(f"stale partial artifact exists: {partial}")
    with partial.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)
        handle.flush()
        os.fsync(handle.fileno())
    _install_immutable(partial, path)


def write_text_lines_immutable(path: Path, lines: Iterator[str]) -> None:
    """Write newline-complete text without replacing non-identical bytes."""

    path.parent.mkdir(parents=True, exist_ok=True)
    partial = path.with_name(path.name + ".part")
    if partial.exists():
        raise PublicDataError(f"stale partial artifact exists: {partial}")
    with partial.open("w", encoding="utf-8", newline="") as handle:
        for line in lines:
            handle.write(line.rstrip("\n") + "\n")
        handle.flush()
        os.fsync(handle.fileno())
    _install_immutable(partial, path)


def acquire_stage_lock(output_dir: Path) -> Path:
    """Atomically lock a derived-artifact directory against concurrent writers.

    A failed or interrupted stage intentionally leaves a small stale lock that
    must be inspected and removed explicitly. This prevents a second process
    from pairing artifacts with a manifest produced by a different run.
    """

    output_dir.mkdir(parents=True, exist_ok=True)
    lock = output_dir / ".hrms-public-data.lock"
    try:
        descriptor = os.open(lock, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o644)
    except FileExistsError as error:
        raise PublicDataError(
            f"stage directory is already locked: {lock}; do not run concurrent writers"
        ) from error
    with os.fdopen(descriptor, "w") as handle:
        handle.write("exclusive stage writer; no timestamps or host identifiers recorded\n")
        handle.flush()
        os.fsync(handle.fileno())
    return lock


def verify_recorded_artifact(
    manifest_path: Path,
    artifact_path: Path,
    *,
    artifact_name: str,
) -> None:
    """Require a downstream input to match its upstream immutable manifest."""

    manifest = json.loads(manifest_path.read_text())
    try:
        expected = manifest["artifacts"][artifact_name]
    except KeyError as error:
        raise PublicDataError(
            f"{manifest_path} does not register artifact {artifact_name!r}"
        ) from error
    verify_file(
        artifact_path,
        size_bytes=int(expected["size_bytes"]),
        sha256=str(expected["sha256"]),
    )


def verify_recorded_config(manifest_path: Path, config_path: Path) -> None:
    """Require the current frozen configuration to match an upstream stage."""

    manifest = json.loads(manifest_path.read_text())
    try:
        expected = manifest["config"]
    except KeyError as error:
        raise PublicDataError(f"{manifest_path} does not register its configuration") from error
    verify_file(
        config_path,
        size_bytes=int(expected["size_bytes"]),
        sha256=str(expected["sha256"]),
    )


@dataclass(frozen=True)
class _ChromIntervals:
    starts: array
    ends: array
    prefix_max_ends: array
    class_codes: bytes | None = None

    def overlap_range(self, start: int, end: int) -> range:
        if end <= start:
            return range(0)
        right = bisect.bisect_left(self.starts, end)
        left = bisect.bisect_right(self.prefix_max_ends, start, 0, right)
        return range(left, right)

    def has_overlap(self, start: int, end: int) -> bool:
        return any(self.ends[index] > start for index in self.overlap_range(start, end))


@dataclass(frozen=True)
class CCREIndex:
    all_ccres: dict[str, _ChromIntervals]
    enhancer_ccres: dict[str, _ChromIntervals]
    enhancer_classes: tuple[str, ...]
    record_count: int
    enhancer_count: int

    def overlaps_any(self, chrom: str, start: int, end: int) -> bool:
        intervals = self.all_ccres.get(chrom)
        return intervals is not None and intervals.has_overlap(start, end)

    def best_enhancer_hit(
        self,
        chrom: str,
        start: int,
        end: int,
        *,
        minimum_fraction: Fraction,
    ) -> EnhancerHit | None:
        intervals = self.enhancer_ccres.get(chrom)
        if intervals is None or intervals.class_codes is None:
            return None
        best: tuple[int, int, int, int, int] | None = None
        # tuple stores overlap, length, start, end, class code. Fractional
        # comparisons use cross multiplication to avoid floating-point edges.
        for index in intervals.overlap_range(start, end):
            ccre_start = intervals.starts[index]
            ccre_end = intervals.ends[index]
            if ccre_end <= start:
                continue
            overlap = min(end, ccre_end) - max(start, ccre_start)
            length = ccre_end - ccre_start
            if overlap * minimum_fraction.denominator < length * minimum_fraction.numerator:
                continue
            candidate = (overlap, length, ccre_start, ccre_end, intervals.class_codes[index])
            if best is None:
                best = candidate
                continue
            # Prefer greater overlap fraction, then overlap bp, then the
            # lexicographically earliest genomic/class coordinate.
            left = candidate[0] * best[1]
            right = best[0] * candidate[1]
            if left > right or (
                left == right
                and (candidate[0], -candidate[2], -candidate[3], -candidate[4])
                > (best[0], -best[2], -best[3], -best[4])
            ):
                best = candidate
        if best is None:
            return None
        overlap, _length, ccre_start, ccre_end, class_code = best
        return EnhancerHit(
            start=ccre_start,
            end=ccre_end,
            ccre_class=self.enhancer_classes[class_code],
            overlap_bp=overlap,
        )


def _make_intervals(
    records: list[tuple[int, int, int | None]], *, with_class: bool
) -> _ChromIntervals:
    records.sort(key=lambda value: (value[0], value[1], -1 if value[2] is None else value[2]))
    starts = array("I")
    ends = array("I")
    prefix = array("I")
    class_codes = bytearray() if with_class else None
    running = 0
    for start, end, class_code in records:
        starts.append(start)
        ends.append(end)
        running = max(running, end)
        prefix.append(running)
        if class_codes is not None:
            assert class_code is not None
            class_codes.append(class_code)
    return _ChromIntervals(starts, ends, prefix, bytes(class_codes) if class_codes else None)


def load_ccre_index(
    path: str | Path,
    *,
    chromosomes: set[str],
    enhancer_classes: tuple[str, ...],
) -> CCREIndex:
    """Read the Registry BED once into compact per-chromosome interval arrays."""

    all_records: dict[str, list[tuple[int, int, int | None]]] = defaultdict(list)
    enhancer_records: dict[str, list[tuple[int, int, int | None]]] = defaultdict(list)
    class_index = {name: index for index, name in enumerate(enhancer_classes)}
    record_count = 0
    enhancer_count = 0
    with _open_text(Path(path)) as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip() or line.startswith("#"):
                continue
            fields = line.rstrip("\n").split("\t")
            if len(fields) < 6:
                raise PublicDataError(f"cCRE BED line {line_number} has fewer than six fields")
            chrom, start_text, end_text, _anchor, _accession, ccre_class = fields[:6]
            if chrom not in chromosomes:
                continue
            start, end = int(start_text), int(end_text)
            if start < 0 or end <= start:
                raise PublicDataError(f"invalid cCRE interval at line {line_number}")
            all_records[chrom].append((start, end, None))
            record_count += 1
            if ccre_class in class_index:
                enhancer_records[chrom].append((start, end, class_index[ccre_class]))
                enhancer_count += 1
    if not record_count or not enhancer_count:
        raise PublicDataError("cCRE input produced no canonical enhancer records")
    return CCREIndex(
        all_ccres={
            chrom: _make_intervals(rows, with_class=False) for chrom, rows in all_records.items()
        },
        enhancer_ccres={
            chrom: _make_intervals(rows, with_class=True)
            for chrom, rows in enhancer_records.items()
        },
        enhancer_classes=enhancer_classes,
        record_count=record_count,
        enhancer_count=enhancer_count,
    )


def load_repeat_name_mapping(path: str | Path) -> tuple[dict[str, tuple[str, str]], dict[str, Any]]:
    """Map Repeat Browser ``repName`` to modal UCSC class/family.

    ``rmskOutCurrent`` can contain repeated names and, rarely, inconsistent
    annotations. The modal pair is selected; count then lexical order resolves
    ties deterministically.
    """

    counts: dict[str, Counter[tuple[str, str]]] = defaultdict(Counter)
    rows = 0
    with _open_text(Path(path)) as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip() or line.startswith("#"):
                continue
            fields = line.rstrip("\n").split("\t")
            if len(fields) < 13:
                raise PublicDataError(
                    f"rmskOutCurrent line {line_number} has fewer than thirteen fields"
                )
            counts[fields[10]][(fields[11], fields[12])] += 1
            rows += 1
    mapping: dict[str, tuple[str, str]] = {}
    ambiguous = 0
    for rep_name, alternatives in counts.items():
        if len(alternatives) > 1:
            ambiguous += 1
        mapping[rep_name] = sorted(
            alternatives,
            key=lambda pair: (-alternatives[pair], pair[0], pair[1]),
        )[0]
    return mapping, {
        "table_rows": rows,
        "mapped_repeat_names": len(mapping),
        "repeat_names_with_multiple_class_family_pairs": ambiguous,
        "tie_rule": "highest locus count, then lexical (repClass, repFamily)",
    }


ELIGIBLE_FIELDS = [
    "locus_id",
    "repeatbrowser_line",
    "chrom",
    "start",
    "end",
    "strand",
    "rep_name",
    "rep_class",
    "rep_family",
    "environment_id",
    "label",
    "te_length",
    "ccre_start",
    "ccre_end",
    "ccre_class",
    "ccre_overlap_bp",
    "ccre_overlap_fraction",
]


def _manifest_record(path: Path) -> dict[str, Any]:
    return {
        "path": path.as_posix(),
        "size_bytes": path.stat().st_size,
        "sha256": sha256_file(path),
    }


def run_preflight(
    config_path: str | Path,
    output_dir: str | Path,
    *,
    base_dir: str | Path = ".",
    verify_sources: bool = True,
) -> dict[str, Path]:
    """Derive all eligible labels and emit family/subfamily count preflights."""

    config_path = Path(config_path)
    output_dir = Path(output_dir)
    config = load_config(config_path)
    specs = {spec.name: spec for spec in source_specs(config, base_dir=base_dir)}
    required_sources = {
        "encode_registry_v4",
        "ucsc_repeatbrowser_2020",
        "ucsc_rmsk_out_current",
        "ucsc_hg38_2bit",
    }
    missing = required_sources.difference(specs)
    if missing:
        raise PublicDataError(f"configuration lacks required sources: {sorted(missing)}")

    source_records: dict[str, dict[str, Any]] = {}
    for name, spec in specs.items():
        if verify_sources:
            source_records[name] = verify_file(
                spec.path, size_bytes=spec.size_bytes, sha256=spec.sha256
            )
        else:
            if not spec.path.is_file():
                raise FileNotFoundError(spec.path)
            source_records[name] = _manifest_record(spec.path)
        source_records[name]["url"] = spec.url

    chromosomes = set(config["coordinates"]["canonical_chromosomes"])
    enhancer_classes = tuple(config["labels"]["positive_ccre_classes"])
    ccre_index = load_ccre_index(
        specs["encode_registry_v4"].path,
        chromosomes=chromosomes,
        enhancer_classes=enhancer_classes,
    )
    repeat_mapping, mapping_summary = load_repeat_name_mapping(specs["ucsc_rmsk_out_current"].path)
    environments = tuple(config["environments"])
    environment_set = set(environments)
    minimum_length = int(config["labels"]["minimum_te_length_bp"])
    negative_buffer = int(config["labels"]["negative_no_any_ccre_buffer_bp"])
    minimum_fraction = Fraction(
        str(config["labels"]["positive_minimum_fraction_of_ccre_overlapped_by_one_te"])
    )

    lock = acquire_stage_lock(output_dir)
    eligible_path = output_dir / "eligible_loci.csv.gz"
    family_path = output_dir / "preflight_family_counts.csv"
    subfamily_path = output_dir / "preflight_subfamily_counts.csv"
    manifest_path = output_dir / "preflight_manifest.json"

    family_counts: Counter[tuple[str, int]] = Counter()
    subfamily_counts: Counter[tuple[str, str, int]] = Counter()
    summary: Counter[str] = Counter()
    observed_repeat_names: set[str] = set()
    unmapped_repeat_names: Counter[str] = Counter()

    with _deterministic_gzip_writer(eligible_path) as compressed:
        writer = csv.DictWriter(compressed, fieldnames=ELIGIBLE_FIELDS, lineterminator="\n")
        writer.writeheader()
        with _open_text(specs["ucsc_repeatbrowser_2020"].path) as repeats:
            for line_number, line in enumerate(repeats, 1):
                if not line.strip() or line.startswith("#"):
                    continue
                summary["repeatbrowser_records"] += 1
                fields = line.rstrip("\n").split("\t")
                if len(fields) < 6:
                    raise PublicDataError(
                        f"Repeat Browser BED line {line_number} has fewer than six fields"
                    )
                chrom, start_text, end_text, rep_name, _score, strand = fields[:6]
                if chrom not in chromosomes:
                    summary["excluded_noncanonical_chromosome"] += 1
                    continue
                summary["canonical_records"] += 1
                start, end = int(start_text), int(end_text)
                if start < 0 or end <= start or strand not in {"+", "-"}:
                    raise PublicDataError(f"invalid repeat BED interval at line {line_number}")
                if end - start < minimum_length:
                    summary["excluded_short_te"] += 1
                    continue
                observed_repeat_names.add(rep_name)
                class_family = repeat_mapping.get(rep_name)
                if class_family is None:
                    summary["excluded_unmapped_rep_name"] += 1
                    unmapped_repeat_names[rep_name] += 1
                    continue
                rep_class, rep_family = class_family
                if rep_family not in environment_set:
                    summary["excluded_non_target_family"] += 1
                    continue
                summary["target_family_records"] += 1

                hit = ccre_index.best_enhancer_hit(
                    chrom,
                    start,
                    end,
                    minimum_fraction=minimum_fraction,
                )
                if hit is not None:
                    label = 1
                    summary["positive_records"] += 1
                elif not ccre_index.overlaps_any(
                    chrom,
                    max(0, start - negative_buffer),
                    end + negative_buffer,
                ):
                    label = 0
                    summary["strict_negative_records"] += 1
                else:
                    summary["excluded_ambiguous_or_other_ccre"] += 1
                    continue

                family_counts[(rep_family, label)] += 1
                subfamily_counts[(rep_family, rep_name, label)] += 1
                writer.writerow(
                    {
                        "locus_id": f"te_{line_number:08d}",
                        "repeatbrowser_line": line_number,
                        "chrom": chrom,
                        "start": start,
                        "end": end,
                        "strand": strand,
                        "rep_name": rep_name,
                        "rep_class": rep_class,
                        "rep_family": rep_family,
                        "environment_id": rep_family,
                        "label": label,
                        "te_length": end - start,
                        "ccre_start": "" if hit is None else hit.start,
                        "ccre_end": "" if hit is None else hit.end,
                        "ccre_class": "" if hit is None else hit.ccre_class,
                        "ccre_overlap_bp": "" if hit is None else hit.overlap_bp,
                        "ccre_overlap_fraction": (
                            "" if hit is None else f"{hit.overlap_fraction:.12g}"
                        ),
                    }
                )

    family_rows = []
    for environment in environments:
        positive = family_counts[(environment, 1)]
        negative = family_counts[(environment, 0)]
        family_rows.append(
            {
                "environment_id": environment,
                "positive_loci": positive,
                "strict_negative_loci": negative,
                "matched_pair_capacity_upper_bound": min(positive, negative),
            }
        )
    write_csv_immutable(
        family_path,
        list(family_rows[0]),
        iter(family_rows),
    )

    subfamily_rows = []
    subfamily_keys = sorted({(key[0], key[1]) for key in subfamily_counts})
    for environment, rep_name in subfamily_keys:
        positive = subfamily_counts[(environment, rep_name, 1)]
        negative = subfamily_counts[(environment, rep_name, 0)]
        subfamily_rows.append(
            {
                "environment_id": environment,
                "rep_name": rep_name,
                "positive_loci": positive,
                "strict_negative_loci": negative,
                "matched_pair_capacity": min(positive, negative),
                "eligible_for_exact_subfamily_matching": int(positive > 0 and negative > 0),
            }
        )
    write_csv_immutable(
        subfamily_path,
        list(subfamily_rows[0]),
        iter(subfamily_rows),
    )

    artifacts = {
        "eligible_loci": _manifest_record(eligible_path),
        "family_counts": _manifest_record(family_path),
        "subfamily_counts": _manifest_record(subfamily_path),
    }
    manifest = {
        "schema_version": 1,
        "stage": "label_and_count_preflight",
        "claim_boundary": config["claim_boundary"],
        "config": _manifest_record(config_path),
        "sources": source_records,
        "parameters": {
            "assembly": config["coordinates"]["assembly"],
            "canonical_chromosomes": config["coordinates"]["canonical_chromosomes"],
            "environments": list(environments),
            "minimum_te_length_bp": minimum_length,
            "positive_ccre_classes": list(enhancer_classes),
            "positive_minimum_fraction_of_ccre_overlapped_by_one_te": float(minimum_fraction),
            "negative_no_any_ccre_buffer_bp": negative_buffer,
            "coordinate_convention": "zero-based, half-open",
        },
        "counts": {
            **dict(sorted(summary.items())),
            "canonical_ccres": ccre_index.record_count,
            "canonical_enhancer_like_ccres": ccre_index.enhancer_count,
            "repeat_names_observed_after_length_filter": len(observed_repeat_names),
            "unmapped_repeat_names_after_length_filter": len(unmapped_repeat_names),
            "largest_unmapped_repeat_names": [
                {"rep_name": name, "loci": count}
                for name, count in sorted(
                    unmapped_repeat_names.items(), key=lambda item: (-item[1], item[0])
                )[:20]
            ],
            "repeat_name_mapping": mapping_summary,
        },
        "artifacts": artifacts,
    }
    write_json_immutable(manifest_path, manifest)
    lock.unlink()
    return {
        "eligible_loci": eligible_path,
        "family_counts": family_path,
        "subfamily_counts": subfamily_path,
        "manifest": manifest_path,
    }


def copy_config_snapshot(config_path: Path, destination: Path) -> None:
    """Install an immutable byte-for-byte copy of the frozen config."""

    destination.parent.mkdir(parents=True, exist_ok=True)
    partial = destination.with_name(destination.name + ".part")
    if partial.exists():
        raise PublicDataError(f"stale partial artifact exists: {partial}")
    shutil.copyfile(config_path, partial)
    _install_immutable(partial, destination)


def stable_hash_int(seed: str, *parts: object) -> int:
    """Return a platform-independent 256-bit priority for deterministic sampling."""

    digest = hashlib.sha256()
    digest.update(seed.encode("utf-8"))
    for part in parts:
        digest.update(b"\x1f")
        digest.update(str(part).encode("utf-8"))
    return int.from_bytes(digest.digest(), "big")


def proportional_capped_quotas(
    capacities: dict[str, int], *, total: int, per_group_cap: int
) -> dict[str, int]:
    """Allocate a deterministic proportional quota with an explicit diversity cap."""

    if total < 0 or per_group_cap < 0:
        raise ValueError("quota totals and caps must be non-negative")
    capped = {
        name: min(int(capacity), per_group_cap)
        for name, capacity in capacities.items()
        if capacity > 0 and per_group_cap > 0
    }
    available = sum(capped.values())
    target = min(total, available)
    if target == 0:
        return {name: 0 for name in sorted(capacities)}
    ideals = {name: Fraction(target * capacity, available) for name, capacity in capped.items()}
    quotas = {
        name: min(capped[name], ideals[name].numerator // ideals[name].denominator)
        for name in capped
    }
    remainder = target - sum(quotas.values())
    order = sorted(
        capped,
        key=lambda name: (
            -(ideals[name] - quotas[name]),
            name,
        ),
    )
    for name in order:
        if remainder == 0:
            break
        if quotas[name] < capped[name]:
            quotas[name] += 1
            remainder -= 1
    if remainder:
        raise AssertionError("largest-remainder allocation failed to exhaust its target")
    return {name: quotas.get(name, 0) for name in sorted(capacities)}


def _read_csv(path: Path) -> Iterator[dict[str, str]]:
    with _open_text(path) as handle:
        yield from csv.DictReader(handle)


def build_candidate_pool(
    config_path: str | Path,
    preflight_dir: str | Path,
    output_dir: str | Path,
) -> dict[str, Path]:
    """Hash-sample a large, subfamily-diverse pool without freezing the benchmark.

    Quotas use the exact-subfamily matched capacity from the complete preflight.
    Equal positive and negative quotas are drawn within every retained repName.
    The final 3,000-pair sample is intentionally deferred until homology roots
    and the independent-root gates are available.
    """

    config_path = Path(config_path)
    preflight_dir = Path(preflight_dir)
    output_dir = Path(output_dir)
    config = load_config(config_path)
    eligible_path = preflight_dir / "eligible_loci.csv.gz"
    subfamily_path = preflight_dir / "preflight_subfamily_counts.csv"
    preflight_manifest = preflight_dir / "preflight_manifest.json"
    for path in (eligible_path, subfamily_path, preflight_manifest):
        if not path.is_file():
            raise FileNotFoundError(path)
    verify_recorded_artifact(preflight_manifest, eligible_path, artifact_name="eligible_loci")
    verify_recorded_artifact(preflight_manifest, subfamily_path, artifact_name="subfamily_counts")
    verify_recorded_config(preflight_manifest, config_path)
    lock = acquire_stage_lock(output_dir)

    sampling = config["sampling"]
    seed = str(sampling["seed"])
    target_pairs = int(sampling["target_pairs_per_environment"])
    multiplier = int(sampling["candidate_pool_multiplier"])
    total_per_environment = target_pairs * multiplier
    per_subfamily_cap = int(sampling["maximum_candidate_pairs_per_subfamily"])
    environments = tuple(config["environments"])

    counts: dict[tuple[str, str], tuple[int, int]] = {}
    by_environment: dict[str, dict[str, int]] = defaultdict(dict)
    for row in _read_csv(subfamily_path):
        environment = row["environment_id"]
        rep_name = row["rep_name"]
        positive = int(row["positive_loci"])
        negative = int(row["strict_negative_loci"])
        counts[(environment, rep_name)] = (positive, negative)
        by_environment[environment][rep_name] = min(positive, negative)

    quotas: dict[tuple[str, str], int] = {}
    for environment in environments:
        allocation = proportional_capped_quotas(
            by_environment.get(environment, {}),
            total=total_per_environment,
            per_group_cap=per_subfamily_cap,
        )
        quotas.update(((environment, rep_name), value) for rep_name, value in allocation.items())

    # A max-heap represented with negative priorities retains the smallest
    # content hashes without storing the complete multi-million-row inventory.
    heaps: dict[tuple[str, str, int], list[tuple[int, str, dict[str, str]]]] = defaultdict(list)
    for row in _read_csv(eligible_path):
        environment = row["environment_id"]
        rep_name = row["rep_name"]
        label = int(row["label"])
        quota = quotas.get((environment, rep_name), 0)
        if quota == 0:
            continue
        locus_id = row["locus_id"]
        priority = stable_hash_int(seed + ":candidate-pool", locus_id)
        key = (environment, rep_name, label)
        heap = heaps[key]
        item = (-priority, locus_id, row)
        if len(heap) < quota:
            heapq.heappush(heap, item)
        elif priority < -heap[0][0]:
            heapq.heapreplace(heap, item)

    retained: list[dict[str, str | int]] = []
    selected_counts: Counter[tuple[str, str, int]] = Counter()
    for key in sorted(heaps):
        for negative_priority, _locus_id, row in heaps[key]:
            priority = -negative_priority
            retained.append({**row, "candidate_priority_sha256_int": priority})
            selected_counts[key] += 1
    retained.sort(
        key=lambda row: (
            str(row["environment_id"]),
            str(row["rep_name"]),
            int(row["label"]),
            int(row["candidate_priority_sha256_int"]),
            str(row["locus_id"]),
        )
    )

    pool_path = output_dir / "candidate_pool.csv.gz"
    distribution_path = output_dir / "candidate_pool_subfamily_distribution.csv"
    manifest_path = output_dir / "candidate_pool_manifest.json"
    pool_fields = [*ELIGIBLE_FIELDS, "candidate_priority_sha256_int"]
    with _deterministic_gzip_writer(pool_path) as handle:
        writer = csv.DictWriter(handle, fieldnames=pool_fields, lineterminator="\n")
        writer.writeheader()
        writer.writerows(retained)

    distribution_rows = []
    for environment in environments:
        rep_names = sorted(by_environment.get(environment, {}))
        for rep_name in rep_names:
            positive, negative = counts[(environment, rep_name)]
            quota = quotas.get((environment, rep_name), 0)
            distribution_rows.append(
                {
                    "environment_id": environment,
                    "rep_name": rep_name,
                    "preflight_positive_loci": positive,
                    "preflight_strict_negative_loci": negative,
                    "preflight_exact_match_capacity": min(positive, negative),
                    "candidate_pair_quota": quota,
                    "selected_positive_loci": selected_counts[(environment, rep_name, 1)],
                    "selected_negative_loci": selected_counts[(environment, rep_name, 0)],
                }
            )
    distribution_fields = [
        "environment_id",
        "rep_name",
        "preflight_positive_loci",
        "preflight_strict_negative_loci",
        "preflight_exact_match_capacity",
        "candidate_pair_quota",
        "selected_positive_loci",
        "selected_negative_loci",
    ]
    write_csv_immutable(distribution_path, distribution_fields, iter(distribution_rows))

    selected_by_environment_label = Counter(
        (str(row["environment_id"]), int(row["label"])) for row in retained
    )
    manifest = {
        "schema_version": 1,
        "stage": "candidate_pool_before_homology_gate",
        "final_sample_frozen": False,
        "config": _manifest_record(config_path),
        "inputs": {
            "eligible_loci": _manifest_record(eligible_path),
            "subfamily_counts": _manifest_record(subfamily_path),
            "preflight_manifest": _manifest_record(preflight_manifest),
        },
        "parameters": {
            "seed": seed,
            "target_pairs_per_environment": target_pairs,
            "candidate_pool_multiplier": multiplier,
            "candidate_pairs_per_environment": total_per_environment,
            "maximum_candidate_pairs_per_subfamily": per_subfamily_cap,
            "quota_rule": sampling["candidate_quota_rule"],
            "selection_rule": (
                "smallest SHA-256(seed,locus_id) priorities within environment/repName/label"
            ),
        },
        "counts": {
            "candidate_loci": len(retained),
            "by_environment_and_label": [
                {
                    "environment_id": environment,
                    "label": label,
                    "loci": selected_by_environment_label[(environment, label)],
                }
                for environment in environments
                for label in (0, 1)
            ],
        },
        "artifacts": {
            "candidate_pool": _manifest_record(pool_path),
            "subfamily_distribution": _manifest_record(distribution_path),
        },
    }
    write_json_immutable(manifest_path, manifest)
    lock.unlink()
    return {
        "candidate_pool": pool_path,
        "subfamily_distribution": distribution_path,
        "manifest": manifest_path,
    }


def read_fasta(path: str | Path) -> Iterator[tuple[str, str]]:
    """Yield strict FASTA identifiers and concatenated sequences."""

    identifier: str | None = None
    pieces: list[str] = []
    seen: set[str] = set()
    with _open_text(Path(path)) as handle:
        for line_number, line in enumerate(handle, 1):
            text = line.strip()
            if not text:
                continue
            if text.startswith(">"):
                if identifier is not None:
                    yield identifier, "".join(pieces)
                identifier = text[1:].split()[0]
                if not identifier or identifier in seen:
                    raise PublicDataError(
                        f"missing or duplicate FASTA identifier at line {line_number}"
                    )
                seen.add(identifier)
                pieces = []
            else:
                if identifier is None:
                    raise PublicDataError("FASTA sequence occurs before its first header")
                pieces.append(text)
    if identifier is not None:
        yield identifier, "".join(pieces)


_REVERSE_COMPLEMENT = str.maketrans("ACGTN", "TGCAN")


def normalize_dna(sequence: str) -> str:
    """Uppercase DNA and replace all non-ACGT symbols with N."""

    return "".join(base if base in "ACGTN" else "N" for base in sequence.upper())


def orient_sequence(sequence: str, strand: str) -> str:
    """Orient a genomic sequence 5-prime to 3-prime with respect to its TE."""

    normalized = normalize_dna(sequence)
    if strand == "+":
        return normalized
    if strand == "-":
        return normalized.translate(_REVERSE_COMPLEMENT)[::-1]
    raise PublicDataError(f"invalid TE strand: {strand!r}")


def sequence_quality(sequence: str) -> tuple[float, float]:
    """Return GC/(A+C+G+T) and ambiguous/total fractions."""

    counts = Counter(sequence)
    called = counts["A"] + counts["C"] + counts["G"] + counts["T"]
    if not sequence or called == 0:
        return float("nan"), 1.0
    return (counts["G"] + counts["C"]) / called, (len(sequence) - called) / len(sequence)


def _write_fasta_immutable(path: Path, records: Iterator[tuple[str, str]]) -> None:
    write_text_lines_immutable(
        path,
        (line for identifier, sequence in records for line in (f">{identifier}", sequence)),
    )


def extract_candidate_te_sequences(
    config_path: str | Path,
    pool_dir: str | Path,
    output_dir: str | Path,
    *,
    base_dir: str | Path = ".",
) -> dict[str, Path]:
    """Extract full TE-copy sequence, orient by TE strand, and calculate GC."""

    config_path = Path(config_path)
    pool_dir = Path(pool_dir)
    output_dir = Path(output_dir)
    base = Path(base_dir)
    config = load_config(config_path)
    pool_path = pool_dir / "candidate_pool.csv.gz"
    pool_manifest = pool_dir / "candidate_pool_manifest.json"
    for path in (pool_path, pool_manifest):
        if not path.is_file():
            raise FileNotFoundError(path)
    verify_recorded_artifact(pool_manifest, pool_path, artifact_name="candidate_pool")
    verify_recorded_config(pool_manifest, config_path)

    specs = {spec.name: spec for spec in source_specs(config, base_dir=base)}
    genome = specs["ucsc_hg38_2bit"]
    verify_file(genome.path, size_bytes=genome.size_bytes, sha256=genome.sha256)
    tool = base / str(config["sequences"]["two_bit_to_fa"])
    if not tool.is_file() or not os.access(tool, os.X_OK):
        raise PublicDataError(f"twoBitToFa is absent or not executable: {tool}")

    lock = acquire_stage_lock(output_dir)
    rows = list(_read_csv(pool_path))
    if not rows:
        raise PublicDataError("candidate pool is empty")
    identifiers = {row["locus_id"] for row in rows}
    if len(identifiers) != len(rows):
        raise PublicDataError("candidate-pool locus identifiers are not unique")

    bed_path = output_dir / "candidate_te_intervals.genomic_plus.bed"
    _bed_lines = (
        "\t".join(
            [
                row["chrom"],
                row["start"],
                row["end"],
                row["locus_id"],
                "0",
                "+",
            ]
        )
        for row in rows
    )
    write_text_lines_immutable(bed_path, _bed_lines)

    raw_fasta = output_dir / "candidate_te_sequences.genomic_plus.fa.part"
    if raw_fasta.exists():
        raise PublicDataError(f"stale extraction file exists: {raw_fasta}")
    command = [
        tool.as_posix(),
        genome.path.as_posix(),
        raw_fasta.as_posix(),
        "-noMask",
        f"-bed={bed_path.as_posix()}",
    ]
    subprocess.run(command, check=True)

    raw_sequences = dict(read_fasta(raw_fasta))
    if set(raw_sequences) != identifiers:
        missing = sorted(identifiers.difference(raw_sequences))[:10]
        unexpected = sorted(set(raw_sequences).difference(identifiers))[:10]
        raise PublicDataError(
            f"twoBitToFa identifiers disagree with BED: missing={missing}, unexpected={unexpected}"
        )

    maximum_ambiguous = float(config["sampling"]["maximum_ambiguous_base_fraction"])
    oriented_sequences: dict[str, str] = {}
    annotated_rows: list[dict[str, Any]] = []
    eligible_count = 0
    for row in rows:
        locus_id = row["locus_id"]
        sequence = orient_sequence(raw_sequences[locus_id], row["strand"])
        expected_length = int(row["end"]) - int(row["start"])
        if len(sequence) != expected_length:
            raise PublicDataError(
                f"wrong extracted length for {locus_id}: {len(sequence)} != {expected_length}"
            )
        gc_fraction, ambiguous_fraction = sequence_quality(sequence)
        sequence_eligible = int(
            math.isfinite(gc_fraction) and ambiguous_fraction <= maximum_ambiguous
        )
        eligible_count += sequence_eligible
        oriented_sequences[locus_id] = sequence
        annotated_rows.append(
            {
                **row,
                "te_gc_fraction": "" if not math.isfinite(gc_fraction) else f"{gc_fraction:.12g}",
                "te_ambiguous_fraction": f"{ambiguous_fraction:.12g}",
                "sequence_eligible": sequence_eligible,
            }
        )

    fasta_path = output_dir / "candidate_te_sequences.oriented.fa"
    _write_fasta_immutable(
        fasta_path,
        ((row["locus_id"], oriented_sequences[row["locus_id"]]) for row in annotated_rows),
    )
    annotated_path = output_dir / "candidate_pool_with_gc.csv.gz"
    annotated_fields = [
        *rows[0].keys(),
        "te_gc_fraction",
        "te_ambiguous_fraction",
        "sequence_eligible",
    ]
    with _deterministic_gzip_writer(annotated_path) as handle:
        writer = csv.DictWriter(handle, fieldnames=annotated_fields, lineterminator="\n")
        writer.writeheader()
        writer.writerows(annotated_rows)
    raw_fasta.unlink()

    manifest_path = output_dir / "candidate_sequence_manifest.json"
    manifest = {
        "schema_version": 1,
        "stage": "candidate_full_te_sequence_and_gc",
        "final_sample_frozen": False,
        "config": _manifest_record(config_path),
        "inputs": {
            "candidate_pool": _manifest_record(pool_path),
            "candidate_pool_manifest": _manifest_record(pool_manifest),
            "hg38_2bit": _manifest_record(genome.path),
            "twoBitToFa": _manifest_record(tool),
        },
        "parameters": {
            "extraction_strand": "+ for every BED row",
            "orientation": config["sequences"]["orientation"],
            "no_mask": bool(config["sequences"]["no_mask"]),
            "gc_denominator": config["sampling"]["gc_denominator"],
            "maximum_ambiguous_base_fraction": maximum_ambiguous,
            "command": command,
        },
        "counts": {
            "candidate_loci": len(rows),
            "sequence_eligible_loci": eligible_count,
            "excluded_for_ambiguous_sequence": len(rows) - eligible_count,
        },
        "artifacts": {
            "candidate_te_bed": _manifest_record(bed_path),
            "candidate_te_sequences": _manifest_record(fasta_path),
            "candidate_pool_with_gc": _manifest_record(annotated_path),
        },
    }
    write_json_immutable(manifest_path, manifest)
    lock.unlink()
    return {
        "candidate_pool_with_gc": annotated_path,
        "candidate_te_sequences": fasta_path,
        "candidate_te_bed": bed_path,
        "manifest": manifest_path,
    }


def relative_length_difference(first: int, second: int) -> float:
    """Symmetric relative length difference used by the frozen matcher."""

    return abs(first - second) / max(first, second)


def _feasible_negative_indices(
    positive: dict[str, str],
    negatives: list[dict[str, str]],
    negative_lengths: list[int],
    *,
    maximum_gc_difference: float,
    maximum_length_difference: float,
) -> list[int]:
    positive_length = int(positive["te_length"])
    if maximum_length_difference >= 1:
        low, high = 0, math.inf
    else:
        low = math.ceil(positive_length * (1 - maximum_length_difference))
        high = math.floor(positive_length / (1 - maximum_length_difference))
    left = bisect.bisect_left(negative_lengths, low)
    right = (
        len(negative_lengths) if math.isinf(high) else bisect.bisect_right(negative_lengths, high)
    )
    positive_gc = float(positive["te_gc_fraction"])
    return [
        index
        for index in range(left, right)
        if abs(float(negatives[index]["te_gc_fraction"]) - positive_gc) <= maximum_gc_difference
    ]


def deterministic_greedy_matches(
    positives: list[dict[str, str]],
    negatives: list[dict[str, str]],
    *,
    maximum_gc_difference: float,
    maximum_length_difference: float,
    seed: str,
) -> list[tuple[dict[str, str], dict[str, str], float, float]]:
    """Match without replacement under exact-group and numerical calipers.

    The caller supplies one exact ``(environment, repName, chromosome)`` group.
    Positives with fewer initially feasible controls are processed first. Within
    a positive, scaled squared distance determines the match, followed by a
    content-hash tie break. This algorithm is deterministic but is not claimed
    to solve a globally optimal bipartite assignment.
    """

    ordered_negatives = sorted(
        negatives,
        key=lambda row: (
            int(row["te_length"]),
            float(row["te_gc_fraction"]),
            row["locus_id"],
        ),
    )
    negative_lengths = [int(row["te_length"]) for row in ordered_negatives]
    feasible: dict[str, list[int]] = {
        row["locus_id"]: _feasible_negative_indices(
            row,
            ordered_negatives,
            negative_lengths,
            maximum_gc_difference=maximum_gc_difference,
            maximum_length_difference=maximum_length_difference,
        )
        for row in positives
    }
    ordered_positives = sorted(
        positives,
        key=lambda row: (
            len(feasible[row["locus_id"]]),
            stable_hash_int(seed + ":positive-order", row["locus_id"]),
            row["locus_id"],
        ),
    )
    unused = set(range(len(ordered_negatives)))
    matched = []
    for positive in ordered_positives:
        options = [index for index in feasible[positive["locus_id"]] if index in unused]
        if not options:
            continue
        positive_gc = float(positive["te_gc_fraction"])
        positive_length = int(positive["te_length"])

        def cost(
            index: int,
            positive_gc: float = positive_gc,
            positive_length: int = positive_length,
            positive_id: str = positive["locus_id"],
        ) -> tuple[float, int, str]:
            negative = ordered_negatives[index]
            gc_difference = abs(float(negative["te_gc_fraction"]) - positive_gc)
            length_difference = relative_length_difference(
                int(negative["te_length"]), positive_length
            )
            scaled_gc = gc_difference / maximum_gc_difference if maximum_gc_difference else 0
            scaled_length = (
                length_difference / maximum_length_difference if maximum_length_difference else 0
            )
            return (
                scaled_gc**2 + scaled_length**2,
                stable_hash_int(seed + ":negative-tie", positive_id, negative["locus_id"]),
                negative["locus_id"],
            )

        selected_index = min(options, key=cost)
        unused.remove(selected_index)
        negative = ordered_negatives[selected_index]
        matched.append(
            (
                positive,
                negative,
                abs(float(negative["te_gc_fraction"]) - positive_gc),
                relative_length_difference(int(negative["te_length"]), positive_length),
            )
        )
    return matched


def _model_window(center: int, *, width: int, chromosome_size: int) -> tuple[int, int] | None:
    """Return an exact centered half-open window, or ``None`` at a boundary."""

    start = center - width // 2
    end = start + width
    if start < 0 or end > chromosome_size:
        return None
    return start, end


def _positive_normalized_anchor(positive: dict[str, str]) -> tuple[int, float]:
    ccre_start = int(positive["ccre_start"])
    ccre_end = int(positive["ccre_end"])
    te_start = int(positive["start"])
    te_end = int(positive["end"])
    center = (ccre_start + ccre_end) // 2
    center = min(max(center, te_start), te_end - 1)
    if positive["strand"] == "+":
        normalized = (center - te_start) / (te_end - te_start)
    else:
        normalized = (te_end - center) / (te_end - te_start)
    return center, min(max(normalized, 0.0), 1.0)


def _negative_anchor(negative: dict[str, str], normalized: float) -> int:
    start, end = int(negative["start"]), int(negative["end"])
    offset = math.floor(normalized * (end - start) + 0.5)
    center = start + offset if negative["strand"] == "+" else end - offset
    return min(max(center, start), end - 1)


def build_precluster_matches(
    config_path: str | Path,
    sequence_dir: str | Path,
    output_dir: str | Path,
) -> dict[str, Path]:
    """Create a large exact-subfamily matched set before homology gating."""

    config_path = Path(config_path)
    sequence_dir = Path(sequence_dir)
    output_dir = Path(output_dir)
    config = load_config(config_path)
    annotated_path = sequence_dir / "candidate_pool_with_gc.csv.gz"
    te_fasta_path = sequence_dir / "candidate_te_sequences.oriented.fa"
    sequence_manifest = sequence_dir / "candidate_sequence_manifest.json"
    for path in (annotated_path, te_fasta_path, sequence_manifest):
        if not path.is_file():
            raise FileNotFoundError(path)
    verify_recorded_artifact(
        sequence_manifest, annotated_path, artifact_name="candidate_pool_with_gc"
    )
    verify_recorded_artifact(
        sequence_manifest, te_fasta_path, artifact_name="candidate_te_sequences"
    )
    verify_recorded_config(sequence_manifest, config_path)
    lock = acquire_stage_lock(output_dir)

    rows = list(_read_csv(annotated_path))
    eligible = [row for row in rows if int(row["sequence_eligible"]) == 1]
    groups: dict[tuple[str, str, str], dict[int, list[dict[str, str]]]] = defaultdict(
        lambda: {0: [], 1: []}
    )
    for row in eligible:
        groups[(row["environment_id"], row["rep_name"], row["chrom"])][int(row["label"])].append(
            row
        )

    sampling = config["sampling"]
    seed = str(sampling["seed"])
    maximum_gc = float(sampling["maximum_gc_fraction_difference"])
    maximum_length = float(sampling["maximum_relative_te_length_difference"])
    window_width = int(config["sequences"]["model_window_bp"])
    chromosome_sizes = {
        name: int(size) for name, size in config["coordinates"]["chromosome_sizes"].items()
    }
    if window_width <= 0 or window_width % 2:
        raise PublicDataError("the model-window width must be a positive even integer")

    matches: list[dict[str, Any]] = []
    matched_rows: list[dict[str, Any]] = []
    dropped_boundary: Counter[str] = Counter()
    matched_subfamilies: Counter[tuple[str, str]] = Counter()
    for group_key in sorted(groups):
        environment, rep_name, chrom = group_key
        labels = groups[group_key]
        for positive, negative, gc_difference, length_difference in deterministic_greedy_matches(
            labels[1],
            labels[0],
            maximum_gc_difference=maximum_gc,
            maximum_length_difference=maximum_length,
            seed=seed,
        ):
            positive_center, normalized = _positive_normalized_anchor(positive)
            negative_center = _negative_anchor(negative, normalized)
            positive_window = _model_window(
                positive_center,
                width=window_width,
                chromosome_size=chromosome_sizes[positive["chrom"]],
            )
            negative_window = _model_window(
                negative_center,
                width=window_width,
                chromosome_size=chromosome_sizes[negative["chrom"]],
            )
            if positive_window is None or negative_window is None:
                dropped_boundary[environment] += 1
                continue
            pair_digest = hashlib.sha256(
                f"{seed}\x1f{positive['locus_id']}\x1f{negative['locus_id']}".encode()
            ).hexdigest()
            pair_id = f"pair_{pair_digest[:20]}"
            match_record = {
                "pair_id": pair_id,
                "environment_id": environment,
                "rep_name": rep_name,
                "chrom": chrom,
                "positive_locus_id": positive["locus_id"],
                "negative_locus_id": negative["locus_id"],
                "te_gc_fraction_difference": f"{gc_difference:.12g}",
                "te_relative_length_difference": f"{length_difference:.12g}",
                "positive_normalized_te_anchor": f"{normalized:.12g}",
            }
            matches.append(match_record)
            matched_subfamilies[(environment, rep_name)] += 1
            for row, center, window in (
                (positive, positive_center, positive_window),
                (negative, negative_center, negative_window),
            ):
                matched_rows.append(
                    {
                        **row,
                        "pair_id": pair_id,
                        "match_gc_fraction_difference": f"{gc_difference:.12g}",
                        "match_relative_te_length_difference": f"{length_difference:.12g}",
                        "positive_normalized_te_anchor": f"{normalized:.12g}",
                        "model_window_center": center,
                        "model_window_start": window[0],
                        "model_window_end": window[1],
                    }
                )

    matches.sort(key=lambda row: (row["environment_id"], row["rep_name"], row["pair_id"]))
    matched_rows.sort(
        key=lambda row: (
            row["environment_id"],
            row["rep_name"],
            row["pair_id"],
            -int(row["label"]),
        )
    )
    pair_path = output_dir / "precluster_pairs.csv.gz"
    locus_path = output_dir / "precluster_loci.csv.gz"
    distribution_path = output_dir / "precluster_match_distribution.csv"
    manifest_path = output_dir / "precluster_match_manifest.json"
    pair_fields = (
        list(matches[0])
        if matches
        else [
            "pair_id",
            "environment_id",
            "rep_name",
            "chrom",
            "positive_locus_id",
            "negative_locus_id",
            "te_gc_fraction_difference",
            "te_relative_length_difference",
            "positive_normalized_te_anchor",
        ]
    )
    with _deterministic_gzip_writer(pair_path) as handle:
        writer = csv.DictWriter(handle, fieldnames=pair_fields, lineterminator="\n")
        writer.writeheader()
        writer.writerows(matches)
    locus_fields = (
        list(matched_rows[0])
        if matched_rows
        else [
            *ELIGIBLE_FIELDS,
            "pair_id",
            "model_window_start",
            "model_window_end",
        ]
    )
    with _deterministic_gzip_writer(locus_path) as handle:
        writer = csv.DictWriter(handle, fieldnames=locus_fields, lineterminator="\n")
        writer.writeheader()
        writer.writerows(matched_rows)

    before_counts: Counter[tuple[str, str, int]] = Counter(
        (row["environment_id"], row["rep_name"], int(row["label"])) for row in eligible
    )
    distribution_rows = []
    for environment, rep_name in sorted({(key[0], key[1]) for key in before_counts}):
        distribution_rows.append(
            {
                "environment_id": environment,
                "rep_name": rep_name,
                "sequence_eligible_positive_loci": before_counts[(environment, rep_name, 1)],
                "sequence_eligible_negative_loci": before_counts[(environment, rep_name, 0)],
                "precluster_matched_pairs": matched_subfamilies[(environment, rep_name)],
            }
        )
    distribution_fields = [
        "environment_id",
        "rep_name",
        "sequence_eligible_positive_loci",
        "sequence_eligible_negative_loci",
        "precluster_matched_pairs",
    ]
    write_csv_immutable(distribution_path, distribution_fields, iter(distribution_rows))

    match_counts = Counter(row["environment_id"] for row in matches)
    manifest = {
        "schema_version": 1,
        "stage": "precluster_exact_subfamily_matching",
        "final_sample_frozen": False,
        "config": _manifest_record(config_path),
        "inputs": {
            "candidate_pool_with_gc": _manifest_record(annotated_path),
            "candidate_te_sequences": _manifest_record(te_fasta_path),
            "candidate_sequence_manifest": _manifest_record(sequence_manifest),
        },
        "parameters": {
            "exact_match_fields": sampling["match_exact_fields"],
            "maximum_gc_fraction_difference": maximum_gc,
            "gc_matching_source": (
                "oriented full TE-copy sequence; denominator A+C+G+T with ambiguous bases excluded"
            ),
            "maximum_relative_te_length_difference": maximum_length,
            "matcher": (
                "constraint-first deterministic greedy without replacement; "
                "scaled squared distance then SHA-256 tie break"
            ),
            "model_window_bp": window_width,
            "negative_anchor": config["sequences"]["negative_anchor"],
            "boundary_rule": config["sequences"]["boundary_rule"],
        },
        "counts": {
            "sequence_eligible_loci": len(eligible),
            "precluster_pairs": len(matches),
            "precluster_loci": len(matched_rows),
            "boundary_dropped_pairs_by_environment": dict(sorted(dropped_boundary.items())),
            "pairs_by_environment": [
                {"environment_id": environment, "pairs": match_counts[environment]}
                for environment in config["environments"]
            ],
        },
        "artifacts": {
            "precluster_pairs": _manifest_record(pair_path),
            "precluster_loci": _manifest_record(locus_path),
            "match_distribution": _manifest_record(distribution_path),
        },
    }
    write_json_immutable(manifest_path, manifest)
    lock.unlink()
    return {
        "precluster_pairs": pair_path,
        "precluster_loci": locus_path,
        "match_distribution": distribution_path,
        "manifest": manifest_path,
    }


def extract_precluster_windows(
    config_path: str | Path,
    sequence_dir: str | Path,
    match_dir: str | Path,
    output_dir: str | Path,
    *,
    base_dir: str | Path = ".",
) -> dict[str, Path]:
    """Extract oriented model windows and retain pair-complete sequence records."""

    config_path = Path(config_path)
    sequence_dir = Path(sequence_dir)
    match_dir = Path(match_dir)
    output_dir = Path(output_dir)
    base = Path(base_dir)
    config = load_config(config_path)
    locus_path = match_dir / "precluster_loci.csv.gz"
    pair_path = match_dir / "precluster_pairs.csv.gz"
    match_manifest = match_dir / "precluster_match_manifest.json"
    full_te_fasta = sequence_dir / "candidate_te_sequences.oriented.fa"
    sequence_manifest = sequence_dir / "candidate_sequence_manifest.json"
    for path in (locus_path, pair_path, match_manifest, full_te_fasta, sequence_manifest):
        if not path.is_file():
            raise FileNotFoundError(path)
    verify_recorded_artifact(match_manifest, locus_path, artifact_name="precluster_loci")
    verify_recorded_artifact(match_manifest, pair_path, artifact_name="precluster_pairs")
    verify_recorded_artifact(
        sequence_manifest, full_te_fasta, artifact_name="candidate_te_sequences"
    )
    verify_recorded_config(match_manifest, config_path)
    verify_recorded_config(sequence_manifest, config_path)

    specs = {spec.name: spec for spec in source_specs(config, base_dir=base)}
    genome = specs["ucsc_hg38_2bit"]
    verify_file(genome.path, size_bytes=genome.size_bytes, sha256=genome.sha256)
    tool = base / str(config["sequences"]["two_bit_to_fa"])
    if not tool.is_file() or not os.access(tool, os.X_OK):
        raise PublicDataError(f"twoBitToFa is absent or not executable: {tool}")
    lock = acquire_stage_lock(output_dir)

    loci = list(_read_csv(locus_path))
    pairs = list(_read_csv(pair_path))
    identifiers = {row["locus_id"] for row in loci}
    if len(identifiers) != len(loci):
        raise PublicDataError("a locus appears in more than one precluster pair")
    window_width = int(config["sequences"]["model_window_bp"])
    window_bed = output_dir / "precluster_windows.genomic_plus.bed"
    write_text_lines_immutable(
        window_bed,
        (
            "\t".join(
                [
                    row["chrom"],
                    row["model_window_start"],
                    row["model_window_end"],
                    row["locus_id"],
                    "0",
                    "+",
                ]
            )
            for row in loci
        ),
    )
    raw_fasta = output_dir / "precluster_windows.genomic_plus.fa.part"
    if raw_fasta.exists():
        raise PublicDataError(f"stale extraction file exists: {raw_fasta}")
    command = [
        tool.as_posix(),
        genome.path.as_posix(),
        raw_fasta.as_posix(),
        "-noMask",
        f"-bed={window_bed.as_posix()}",
    ]
    subprocess.run(command, check=True)
    raw_sequences = dict(read_fasta(raw_fasta))
    if set(raw_sequences) != identifiers:
        missing = sorted(identifiers.difference(raw_sequences))[:10]
        unexpected = sorted(set(raw_sequences).difference(identifiers))[:10]
        raise PublicDataError(
            "window FASTA identifiers disagree with BED: "
            f"missing={missing}, unexpected={unexpected}"
        )

    maximum_ambiguous = float(config["sampling"]["maximum_ambiguous_base_fraction"])
    oriented_windows: dict[str, str] = {}
    annotated: dict[str, dict[str, Any]] = {}
    pair_eligible: dict[str, bool] = defaultdict(lambda: True)
    for row in loci:
        identifier = row["locus_id"]
        sequence = orient_sequence(raw_sequences[identifier], row["strand"])
        if len(sequence) != window_width:
            raise PublicDataError(
                f"model window for {identifier} is {len(sequence)} bp, expected {window_width}"
            )
        gc_fraction, ambiguous_fraction = sequence_quality(sequence)
        eligible = math.isfinite(gc_fraction) and ambiguous_fraction <= maximum_ambiguous
        pair_eligible[row["pair_id"]] &= eligible
        oriented_windows[identifier] = sequence
        annotated[identifier] = {
            **row,
            "observation_id": identifier,
            "sample_weight": 1.0,
            "window_gc_fraction": "" if not math.isfinite(gc_fraction) else f"{gc_fraction:.12g}",
            "window_ambiguous_fraction": f"{ambiguous_fraction:.12g}",
        }

    retained_pair_ids = {pair_id for pair_id, eligible in pair_eligible.items() if eligible}
    retained_loci = [row for row in loci if row["pair_id"] in retained_pair_ids]
    retained_locus_ids = {row["locus_id"] for row in retained_loci}
    retained_pairs = [row for row in pairs if row["pair_id"] in retained_pair_ids]
    if len(retained_loci) != 2 * len(retained_pairs):
        raise PublicDataError("pair-complete window filtering invariant failed")

    full_te_sequences = {
        identifier: sequence
        for identifier, sequence in read_fasta(full_te_fasta)
        if identifier in retained_locus_ids
    }
    if set(full_te_sequences) != retained_locus_ids:
        raise PublicDataError("full-TE FASTA is missing a retained matched locus")

    cluster_locus_path = output_dir / "cluster_input_loci.csv.gz"
    cluster_pair_path = output_dir / "cluster_input_pairs.csv.gz"
    cluster_te_fasta = output_dir / "cluster_input_te_sequences.oriented.fa"
    cluster_window_fasta = output_dir / "cluster_input_windows.oriented.fa"
    gc_label_path = output_dir / "window_gc_balance_by_environment_label.csv"
    gc_pair_path = output_dir / "window_gc_paired_differences.csv"
    manifest_path = output_dir / "cluster_input_manifest.json"
    annotated_loci = [annotated[row["locus_id"]] for row in retained_loci]
    locus_fields = list(annotated_loci[0]) if annotated_loci else []
    with _deterministic_gzip_writer(cluster_locus_path) as handle:
        writer = csv.DictWriter(handle, fieldnames=locus_fields, lineterminator="\n")
        writer.writeheader()
        writer.writerows(annotated_loci)
    pair_fields = list(retained_pairs[0]) if retained_pairs else []
    with _deterministic_gzip_writer(cluster_pair_path) as handle:
        writer = csv.DictWriter(handle, fieldnames=pair_fields, lineterminator="\n")
        writer.writeheader()
        writer.writerows(retained_pairs)
    _write_fasta_immutable(
        cluster_te_fasta,
        ((row["locus_id"], full_te_sequences[row["locus_id"]]) for row in retained_loci),
    )
    _write_fasta_immutable(
        cluster_window_fasta,
        ((row["locus_id"], oriented_windows[row["locus_id"]]) for row in retained_loci),
    )

    label_groups: dict[tuple[str, int], list[dict[str, Any]]] = defaultdict(list)
    pair_groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in annotated_loci:
        label_groups[(row["environment_id"], int(row["label"]))].append(row)
        pair_groups[row["pair_id"]].append(row)
    gc_label_rows = []
    for (environment, label), group in sorted(label_groups.items()):
        mean_te_gc = sum(float(row["te_gc_fraction"]) for row in group) / len(group)
        mean_window_gc = sum(float(row["window_gc_fraction"]) for row in group) / len(group)
        mean_te_length = sum(int(row["te_length"]) for row in group) / len(group)
        gc_label_rows.append(
            {
                "environment_id": environment,
                "label": label,
                "loci": len(group),
                "mean_full_te_gc_fraction": f"{mean_te_gc:.12g}",
                "mean_window_gc_fraction": f"{mean_window_gc:.12g}",
                "mean_te_length": f"{mean_te_length:.12g}",
            }
        )
    gc_label_fields = [
        "environment_id",
        "label",
        "loci",
        "mean_full_te_gc_fraction",
        "mean_window_gc_fraction",
        "mean_te_length",
    ]
    write_csv_immutable(gc_label_path, gc_label_fields, iter(gc_label_rows))

    paired_differences: dict[str, list[tuple[float, float]]] = defaultdict(list)
    for pair_id, group in pair_groups.items():
        if len(group) != 2 or {int(row["label"]) for row in group} != {0, 1}:
            raise PublicDataError(f"pair {pair_id} is not one-positive/one-negative")
        positive = next(row for row in group if int(row["label"]) == 1)
        negative = next(row for row in group if int(row["label"]) == 0)
        paired_differences[positive["environment_id"]].append(
            (
                float(positive["te_gc_fraction"]) - float(negative["te_gc_fraction"]),
                float(positive["window_gc_fraction"]) - float(negative["window_gc_fraction"]),
            )
        )

    def empirical_quantile(values: list[float], probability: float) -> float:
        ordered = sorted(values)
        index = max(0, math.ceil(probability * len(ordered)) - 1)
        return ordered[index]

    gc_pair_rows = []
    for environment, differences in sorted(paired_differences.items()):
        te_differences = [value[0] for value in differences]
        window_differences = [value[1] for value in differences]
        mean_te_difference = sum(te_differences) / len(differences)
        mean_absolute_te_difference = sum(map(abs, te_differences)) / len(differences)
        mean_window_difference = sum(window_differences) / len(differences)
        mean_absolute_window_difference = sum(map(abs, window_differences)) / len(differences)
        p95_absolute_window_difference = empirical_quantile(
            [abs(value) for value in window_differences], 0.95
        )
        gc_pair_rows.append(
            {
                "environment_id": environment,
                "pairs": len(differences),
                "mean_full_te_gc_positive_minus_negative": f"{mean_te_difference:.12g}",
                "mean_absolute_full_te_gc_difference": f"{mean_absolute_te_difference:.12g}",
                "mean_window_gc_positive_minus_negative": f"{mean_window_difference:.12g}",
                "mean_absolute_window_gc_difference": (f"{mean_absolute_window_difference:.12g}"),
                "p95_absolute_window_gc_difference": (f"{p95_absolute_window_difference:.12g}"),
            }
        )
    gc_pair_fields = [
        "environment_id",
        "pairs",
        "mean_full_te_gc_positive_minus_negative",
        "mean_absolute_full_te_gc_difference",
        "mean_window_gc_positive_minus_negative",
        "mean_absolute_window_gc_difference",
        "p95_absolute_window_gc_difference",
    ]
    write_csv_immutable(gc_pair_path, gc_pair_fields, iter(gc_pair_rows))
    raw_fasta.unlink()

    manifest = {
        "schema_version": 1,
        "stage": "oriented_window_extraction_before_homology_clustering",
        "final_sample_frozen": False,
        "config": _manifest_record(config_path),
        "inputs": {
            "precluster_loci": _manifest_record(locus_path),
            "precluster_pairs": _manifest_record(pair_path),
            "match_manifest": _manifest_record(match_manifest),
            "candidate_te_sequences": _manifest_record(full_te_fasta),
            "candidate_sequence_manifest": _manifest_record(sequence_manifest),
            "hg38_2bit": _manifest_record(genome.path),
            "twoBitToFa": _manifest_record(tool),
        },
        "parameters": {
            "model_window_bp": window_width,
            "extraction_strand": "+ for every BED row",
            "orientation": config["sequences"]["orientation"],
            "maximum_ambiguous_base_fraction": maximum_ambiguous,
            "pair_filter": "drop both members when either model window fails sequence QC",
            "command": command,
        },
        "counts": {
            "input_pairs": len(pairs),
            "retained_pairs": len(retained_pairs),
            "dropped_pairs_for_window_sequence_qc": len(pairs) - len(retained_pairs),
            "retained_loci": len(retained_loci),
        },
        "artifacts": {
            "cluster_input_loci": _manifest_record(cluster_locus_path),
            "cluster_input_pairs": _manifest_record(cluster_pair_path),
            "cluster_input_te_sequences": _manifest_record(cluster_te_fasta),
            "cluster_input_windows": _manifest_record(cluster_window_fasta),
            "window_bed": _manifest_record(window_bed),
            "window_gc_balance_by_environment_label": _manifest_record(gc_label_path),
            "window_gc_paired_differences": _manifest_record(gc_pair_path),
        },
    }
    write_json_immutable(manifest_path, manifest)
    lock.unlink()
    return {
        "cluster_input_loci": cluster_locus_path,
        "cluster_input_pairs": cluster_pair_path,
        "cluster_input_te_sequences": cluster_te_fasta,
        "cluster_input_windows": cluster_window_fasta,
        "window_bed": window_bed,
        "window_gc_balance_by_environment_label": gc_label_path,
        "window_gc_paired_differences": gc_pair_path,
        "manifest": manifest_path,
    }
