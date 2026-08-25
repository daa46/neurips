"""Content-addressed manifests for reproducible HRMS artifacts."""

from __future__ import annotations

import hashlib
import json
import platform
from collections.abc import Iterable, Mapping
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import scipy

from . import __version__


def sha256_file(path: str | Path, *, chunk_size: int = 1 << 20) -> str:
    """Return the SHA-256 digest of a file without loading it into memory."""

    source = Path(path)
    digest = hashlib.sha256()
    with source.open("rb") as handle:
        while chunk := handle.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()


def _display_path(path: Path, base_dir: Path) -> str:
    resolved = path.resolve()
    try:
        return resolved.relative_to(base_dir.resolve()).as_posix()
    except ValueError:
        return path.as_posix()


def _file_record(path: Path, base_dir: Path) -> dict[str, str | int]:
    if not path.is_file():
        raise FileNotFoundError(f"artifact does not exist or is not a file: {path}")
    return {
        "path": _display_path(path, base_dir),
        "size_bytes": path.stat().st_size,
        "sha256": sha256_file(path),
    }


def source_tree_record(
    paths: Iterable[str | Path], *, base_dir: str | Path = "."
) -> dict[str, Any]:
    """Return a canonical digest and per-file records for source inputs."""

    base = Path(base_dir)
    records = [_file_record(Path(path), base) for path in paths]
    records.sort(key=lambda record: str(record["path"]))
    digest = hashlib.sha256()
    for record in records:
        digest.update(str(record["path"]).encode())
        digest.update(b"\0")
        digest.update(bytes.fromhex(str(record["sha256"])))
    return {"sha256": digest.hexdigest(), "files": records}


def artifact_manifest(
    *,
    inputs: Mapping[str, str | Path],
    artifacts: Mapping[str, str | Path],
    parameters: Mapping[str, Any] | None = None,
    source_files: Iterable[str | Path] = (),
    source_base_dir: str | Path | None = None,
    base_dir: str | Path = ".",
) -> dict[str, Any]:
    """Build a deterministic manifest for named inputs and generated files.

    Timestamps and machine-specific absolute paths are deliberately omitted.
    Hashes identify the exact bytes; software versions record the execution
    environment relevant to numerical reproduction.
    """

    base = Path(base_dir)
    input_records = {
        name: _file_record(Path(path), base) for name, path in sorted(inputs.items())
    }
    artifact_records = {
        name: _file_record(Path(path), base) for name, path in sorted(artifacts.items())
    }
    manifest = {
        "schema_version": 1,
        "software": {
            "hrms": __version__,
            "python": platform.python_version(),
            "numpy": np.__version__,
            "pandas": pd.__version__,
            "scipy": scipy.__version__,
        },
        "parameters": dict(parameters or {}),
        "inputs": input_records,
        "artifacts": artifact_records,
    }
    source_paths = tuple(source_files)
    if source_paths:
        source_base = base if source_base_dir is None else Path(source_base_dir)
        manifest["source"] = source_tree_record(source_paths, base_dir=source_base)
    return manifest


def write_artifact_manifest(
    output: str | Path,
    *,
    inputs: Mapping[str, str | Path],
    artifacts: Mapping[str, str | Path],
    parameters: Mapping[str, Any] | None = None,
    source_files: Iterable[str | Path] = (),
    source_base_dir: str | Path | None = None,
    base_dir: str | Path = ".",
) -> Path:
    """Write :func:`artifact_manifest` as stable, human-readable JSON."""

    destination = Path(output)
    destination.parent.mkdir(parents=True, exist_ok=True)
    manifest = artifact_manifest(
        inputs=inputs,
        artifacts=artifacts,
        parameters=parameters,
        source_files=source_files,
        source_base_dir=source_base_dir,
        base_dir=base_dir,
    )
    destination.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    return destination
