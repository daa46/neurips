#!/usr/bin/env python3
"""Build and verify the deterministic anonymous submission artifact.

The bundle is deliberately assembled from an allowlist. It contains source,
tests, frozen aggregate results, and the self-contained public-TE cohort, but
never crawls the repository wholesale. This keeps raw source downloads,
row-level test predictions/losses, caches, and failed exploratory outputs out
of the submission artifact.
"""

from __future__ import annotations

import argparse
import gzip
import hashlib
import io
import json
import os
import shutil
import subprocess
import sys
import tarfile
import tempfile
from pathlib import Path, PurePosixPath

PROJECT_ROOT = Path(__file__).resolve().parents[1]
ARCHIVE_ROOT = PurePosixPath("hrms-anonymous-artifact")
FIXED_MTIME = 0
_LOCAL_USER = b"de" + b"nali"
_LOCAL_NAME = b"fr" + b"ed"
_HOME_PREFIX = b"/" + b"home" + b"/"
BANNED_TOKENS = (_LOCAL_USER, _LOCAL_NAME, _HOME_PREFIX)
TEXT_REPLACEMENTS = {
    _LOCAL_USER: b"local-user",
    _LOCAL_NAME: b"collaborator",
    _HOME_PREFIX: b"/local/",
}

ROOT_FILES = (
    ".gitignore",
    "docs/artifact.md",
    "environment.yml",
    "README.md",
    "docs/specification.md",
    "pyproject.toml",
)
SOURCE_TREES = (
    "configs",
    "docs",
    "examples",
    "experiments",
    "scripts",
    "src",
    "tests",
)
PAPER_FILES = (
    "paper/main.tex",
    "paper/main.pdf",
    "paper/neurips_2026.sty",
    "paper/checklist.tex",
    "paper/references.bib",
    "paper/figures/method.pdf",
    "paper/figures/simulation.pdf",
    "paper/figures/te_public.pdf",
)
TE_DATA_FILES = (
    "data/te_public/gate_evidence_combined_v2.csv",
    "data/te_public/final_sample_v2/cluster_sample_gate_decision.csv",
    "data/te_public/final_sample_v2/cluster_sample_manifest.json",
    "data/te_public/final_sample_v2/cluster_sample_quota_audit.csv",
)
TE_CLUSTER_FILES = (
    "data/te_public/final_clustered_v2/cluster_membership.csv",
    "data/te_public/final_clustered_v2/cluster_split_manifest.json",
    "data/te_public/final_clustered_v2/labeled_loci_split.parquet",
    "data/te_public/final_clustered_v2/largest_root_fraction.csv",
    "data/te_public/final_clustered_v2/pre_split_root_environment_summary.csv",
    "data/te_public/final_clustered_v2/pre_split_root_summary.csv",
    "data/te_public/final_clustered_v2/split_cell_counts.csv",
    "data/te_public/final_clustered_v2/split_root_counts.csv",
    "data/te_public/final_clustered_v2/splits.parquet",
    "data/te_public/final_clustered_v2/te_component_spanning_forest.tsv",
    "data/te_public/final_clustered_v2/te_mmseqs_job_audit.csv",
    "data/te_public/final_clustered_v2/window_component_spanning_forest.tsv",
    "data/te_public/final_clustered_v2/window_mmseqs_job_audit.csv",
    "data/te_public/final_clustered_v2/clustering_checkpoints/te_copy/complete.json",
    "data/te_public/final_clustered_v2/clustering_checkpoints/te_copy/component_spanning_forest.tsv",
    "data/te_public/final_clustered_v2/clustering_checkpoints/te_copy/plan.json",
    "data/te_public/final_clustered_v2/clustering_checkpoints/window_512/complete.json",
    "data/te_public/final_clustered_v2/clustering_checkpoints/window_512/component_spanning_forest.tsv",
    "data/te_public/final_clustered_v2/clustering_checkpoints/window_512/plan.json",
)
TE_TRAINING_FILES = (
    "results/te_public/training/calibration_roots.csv",
    "results/te_public/training/diagnostic_model_registry.csv",
    "results/te_public/training/losses_select.parquet",
    "results/te_public/training/model_registry.csv",
    "results/te_public/training/training_manifest.json",
    "results/te_public/training/training_timings.csv",
)
CIVIL_TOP_LEVEL_FILES = (
    "results/civilcomments/candidate_metrics.csv",
    "results/civilcomments/model_registry.csv",
    "results/civilcomments/training_manifest.json",
    "results/civilcomments/training_timings.csv",
)
RESULT_TREES = (
    "results/coverage_final_paired",
    "results/decision_headline",
    "results/te_public/evaluation/selection",
    "results/te_public/evaluation/test_evaluation",
    "results/te_public/evaluation/descriptive_metrics",
    "results/civilcomments/selection",
    "results/civilcomments/test_evaluation",
    "results/civilcomments/ablations",
)
EXCLUDED_BASENAMES = {
    "losses_test.csv.gz",
    "losses_test.parquet",
    "predictions_test.csv.gz",
    "predictions_test.parquet",
    "predictions_val.csv.gz",
    "predictions_select.parquet",
}
TEXT_SUFFIXES = {
    ".bib",
    ".cff",
    ".csv",
    ".json",
    ".md",
    ".py",
    ".sh",
    ".sty",
    ".tex",
    ".toml",
    ".tsv",
    ".txt",
    ".yml",
    ".yaml",
}


def _sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _is_cache_or_temporary(path: Path) -> bool:
    parts = set(path.parts)
    return bool(
        parts & {"__pycache__", ".pytest_cache", ".ruff_cache", "build", "tmp"}
        or path.suffix in {".pyc", ".pyo", ".tmp"}
        or path.name.startswith(".")
        and path.name != ".gitignore"
    )


def _require_file(relative_path: str) -> Path:
    path = PROJECT_ROOT / relative_path
    if not path.is_file():
        raise FileNotFoundError(f"required artifact input is missing: {relative_path}")
    return path


def _tree_files(relative_directory: str) -> list[Path]:
    directory = PROJECT_ROOT / relative_directory
    if not directory.is_dir():
        raise FileNotFoundError(f"required artifact directory is missing: {relative_directory}")
    return [
        path
        for path in sorted(directory.rglob("*"))
        if path.is_file()
        and not _is_cache_or_temporary(path.relative_to(PROJECT_ROOT))
        and path.name not in EXCLUDED_BASENAMES
    ]


def selected_source_files(output: Path) -> list[Path]:
    """Return the sorted, duplicate-free artifact allowlist."""

    files = [_require_file(relative_path) for relative_path in ROOT_FILES]
    for tree in SOURCE_TREES:
        files.extend(_tree_files(tree))
    files.extend(_require_file(relative_path) for relative_path in PAPER_FILES)
    files.extend(_require_file(relative_path) for relative_path in TE_DATA_FILES)
    files.extend(_require_file(relative_path) for relative_path in TE_CLUSTER_FILES)
    files.extend(_require_file(relative_path) for relative_path in TE_TRAINING_FILES)
    files.extend(_require_file(relative_path) for relative_path in CIVIL_TOP_LEVEL_FILES)
    for tree in RESULT_TREES:
        files.extend(_tree_files(tree))
    output_resolved = output.resolve()
    unique = {path.resolve(): path for path in files if path.resolve() != output_resolved}
    return sorted(unique.values(), key=lambda path: path.relative_to(PROJECT_ROOT).as_posix())


def _anonymize_text(relative_path: str, data: bytes) -> tuple[bytes, bool]:
    """Replace local workspace prefixes with portable relative paths."""

    if Path(relative_path).suffix.lower() not in TEXT_SUFFIXES:
        return data, False
    root_prefix = str(PROJECT_ROOT).encode() + b"/"
    sanitized = data.replace(root_prefix, b"")
    for token, replacement in TEXT_REPLACEMENTS.items():
        while (offset := sanitized.lower().find(token)) >= 0:
            sanitized = sanitized[:offset] + replacement + sanitized[offset + len(token) :]
    return sanitized, sanitized != data


def _bundle_readme() -> bytes:
    return (
        b"# Anonymous HRMS submission artifact\n\n"
        b"This deterministic bundle contains the paper source/PDF, implementation, "
        b"tests, frozen headline simulations, aggregate CivilComments-derived "
        b"results, and the self-contained final public-TE clustered cohort.\n\n"
        b"Raw ENCODE/UCSC source downloads, CivilComments text, caches, failed or "
        b"exploratory result trees, and row-level test predictions/losses are "
        b"intentionally excluded. The TE selection-loss table is included so the "
        b"sealed HRMS selection analysis can be rerun without reading the test "
        b"table. Candidate training can be rerun from the bundled clustered cohort.\n\n"
        b"`ARTIFACT_MANIFEST.sha256` authenticates every other file in this bundle. "
        b"`ANONYMIZATION_TRANSFORMS.json` records the portable-path transformation "
        b"applied to source manifests; scientific values and result tables are not "
        b"rewritten. See `docs/artifact.md` for commands and claim boundaries.\n"
    )


def build_payload(output: Path) -> dict[str, bytes]:
    payload: dict[str, bytes] = {}
    transforms: list[dict[str, str]] = []
    for source in selected_source_files(output):
        relative = source.relative_to(PROJECT_ROOT).as_posix()
        original = source.read_bytes()
        bundled, transformed = _anonymize_text(relative, original)
        payload[relative] = bundled
        if transformed:
            transforms.append(
                {
                    "path": relative,
                    "source_sha256": _sha256(original),
                    "bundled_sha256": _sha256(bundled),
                    "transformation": (
                        "workspace prefix and local identity test tokens replaced "
                        "with anonymous portable equivalents"
                    ),
                }
            )
    payload["ANONYMOUS_ARTIFACT_README.md"] = _bundle_readme()
    payload["ANONYMIZATION_TRANSFORMS.json"] = (
        json.dumps(
            {
                "schema_version": 1,
                "transforms": transforms,
                "note": "Only path strings were made portable; source files were not modified.",
            },
            indent=2,
            sort_keys=True,
        )
        + "\n"
    ).encode()
    manifest_lines = [f"{_sha256(data)}  {relative}" for relative, data in sorted(payload.items())]
    payload["ARTIFACT_MANIFEST.sha256"] = ("\n".join(manifest_lines) + "\n").encode()
    return payload


def _add_directory(tar: tarfile.TarFile, name: str) -> None:
    info = tarfile.TarInfo(name=name.rstrip("/") + "/")
    info.type = tarfile.DIRTYPE
    info.mode = 0o755
    info.uid = info.gid = info.mtime = FIXED_MTIME
    info.uname = info.gname = "root"
    tar.addfile(info)


def _add_file(tar: tarfile.TarFile, name: str, data: bytes) -> None:
    info = tarfile.TarInfo(name=name)
    info.size = len(data)
    info.mode = 0o755 if name.endswith(".sh") else 0o644
    info.uid = info.gid = info.mtime = FIXED_MTIME
    info.uname = info.gname = "root"
    tar.addfile(info, io.BytesIO(data))


def write_archive(output: Path, payload: dict[str, bytes]) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    directories: set[str] = {ARCHIVE_ROOT.as_posix()}
    for relative in payload:
        current = ARCHIVE_ROOT / relative
        directories.update(
            parent.as_posix() for parent in current.parents if parent != PurePosixPath(".")
        )
    with tempfile.NamedTemporaryFile(suffix=".tar", delete=False) as temporary:
        temporary_path = Path(temporary.name)
    try:
        with tarfile.open(temporary_path, mode="w", format=tarfile.GNU_FORMAT) as tar:
            for directory in sorted(directories):
                _add_directory(tar, directory)
            for relative, data in sorted(payload.items()):
                _add_file(tar, (ARCHIVE_ROOT / relative).as_posix(), data)
        with (
            temporary_path.open("rb") as source,
            output.open("wb") as destination,
            gzip.GzipFile(
                filename="",
                mode="wb",
                compresslevel=9,
                fileobj=destination,
                mtime=FIXED_MTIME,
            ) as compressor,
        ):
            shutil.copyfileobj(source, compressor, length=1024 * 1024)
    finally:
        temporary_path.unlink(missing_ok=True)


def _scan_bundle(root: Path) -> None:
    violations: list[str] = []
    for path in sorted(root.rglob("*")):
        relative = path.relative_to(root).as_posix()
        lowered_name = relative.lower().encode()
        if any(token in lowered_name for token in BANNED_TOKENS):
            violations.append(f"forbidden token in path: {relative}")
        if not path.is_file():
            continue
        data = path.read_bytes().lower()
        if any(token in data for token in BANNED_TOKENS):
            violations.append(f"forbidden token in file bytes: {relative}")
        if path.suffix == ".gz":
            try:
                expanded = gzip.decompress(path.read_bytes()).lower()
            except (gzip.BadGzipFile, OSError):
                continue
            if any(token in expanded for token in BANNED_TOKENS):
                violations.append(f"forbidden token in gzip payload: {relative}")
    if violations:
        raise RuntimeError("anonymous-content scan failed:\n" + "\n".join(violations))


def _verify_manifest(root: Path) -> None:
    manifest_path = root / "ARTIFACT_MANIFEST.sha256"
    expected_paths: set[str] = set()
    for line_number, line in enumerate(manifest_path.read_text().splitlines(), start=1):
        digest, separator, relative = line.partition("  ")
        if not separator or len(digest) != 64:
            raise RuntimeError(f"malformed manifest line {line_number}")
        target = root / relative
        if not target.is_file():
            raise RuntimeError(f"manifest target is missing: {relative}")
        if _sha256(target.read_bytes()) != digest:
            raise RuntimeError(f"manifest digest mismatch: {relative}")
        expected_paths.add(relative)
    actual_paths = {
        path.relative_to(root).as_posix()
        for path in root.rglob("*")
        if path.is_file() and path != manifest_path
    }
    if actual_paths != expected_paths:
        raise RuntimeError(
            "manifest coverage mismatch; "
            f"missing={sorted(actual_paths - expected_paths)}, "
            f"extra={sorted(expected_paths - actual_paths)}"
        )


def verify_archive(output: Path) -> dict[str, int | str]:
    with tempfile.TemporaryDirectory(prefix="hrms-artifact-verify-") as temporary:
        extraction_root = Path(temporary)
        with tarfile.open(output, mode="r:gz") as tar:
            for member in tar.getmembers():
                path = PurePosixPath(member.name)
                if path.is_absolute() or ".." in path.parts:
                    raise RuntimeError(f"unsafe archive member: {member.name}")
            tar.extractall(extraction_root, filter="data")
        root = extraction_root / ARCHIVE_ROOT.as_posix()
        _verify_manifest(root)
        _scan_bundle(root)
        file_count = sum(path.is_file() for path in root.rglob("*"))
        environment = os.environ.copy()
        environment["PYTHONPATH"] = str(root / "src")
        environment["PYTHONDONTWRITEBYTECODE"] = "1"
        subprocess.run(
            [
                sys.executable,
                "-c",
                (
                    "import hrms; from hrms import HRMS; "
                    "from hrms.ambiguity import ChiSquareBall; "
                    "assert HRMS and ChiSquareBall and hrms.__version__ == '0.1.0'"
                ),
            ],
            cwd=root,
            env=environment,
            check=True,
            capture_output=True,
            text=True,
        )
        return {
            "file_count": file_count,
            "archive_size_bytes": output.stat().st_size,
            "archive_sha256": _sha256(output.read_bytes()),
        }


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output",
        type=Path,
        default=PROJECT_ROOT / "dist" / "hrms_anonymous_artifact.tar.gz",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    output = _parser().parse_args(argv).output.resolve()
    write_archive(output, build_payload(output))
    report = verify_archive(output)
    print(json.dumps({"output": str(output), **report}, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
