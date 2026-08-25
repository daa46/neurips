#!/usr/bin/env python3
"""Hash a reproduction directory and its declared external inputs."""

from __future__ import annotations

import argparse
from pathlib import Path

from hrms.artifacts import write_artifact_manifest

_PROJECT_ROOT = Path(__file__).resolve().parents[1]


def _repository_source_files() -> list[Path]:
    files = [
        _PROJECT_ROOT / "docs/specification.md",
        _PROJECT_ROOT / "environment.yml",
        _PROJECT_ROOT / "pyproject.toml",
        _PROJECT_ROOT / "scripts" / "reproduce_paper.sh",
    ]
    for directory in ("experiments", "src/hrms", "scripts", "tests"):
        files.extend(sorted((_PROJECT_ROOT / directory).rglob("*.py")))
    return files


def _named_paths(values: list[str]) -> dict[str, Path]:
    parsed: dict[str, Path] = {}
    for value in values:
        name, separator, raw_path = value.partition("=")
        if not separator or not name or not raw_path:
            raise ValueError(f"expected NAME=PATH, received {value!r}")
        if name in parsed:
            raise ValueError(f"duplicate input name: {name!r}")
        parsed[name] = Path(raw_path)
    return parsed


def build_tree_manifest(
    *,
    root: str | Path,
    output: str | Path,
    inputs: dict[str, Path] | None = None,
) -> Path:
    """Write a manifest covering every file beneath ``root`` except itself."""

    root_path = Path(root)
    output_path = Path(output)
    if not root_path.is_dir():
        raise FileNotFoundError(f"reproduction root is not a directory: {root_path}")
    resolved_output = output_path.resolve()
    artifacts = {
        path.relative_to(root_path).as_posix(): path
        for path in sorted(root_path.rglob("*"))
        if path.is_file() and path.resolve() != resolved_output
    }
    if not artifacts:
        raise ValueError(f"reproduction root contains no artifacts: {root_path}")
    return write_artifact_manifest(
        output_path,
        inputs=inputs or {},
        artifacts=artifacts,
        parameters={"scope": "complete_reproduction_tree"},
        source_files=_repository_source_files(),
        source_base_dir=_PROJECT_ROOT,
        base_dir=root_path,
    )


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--input",
        action="append",
        default=[],
        metavar="NAME=PATH",
        help="external input to hash; may be repeated",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    destination = build_tree_manifest(
        root=args.root,
        output=args.output,
        inputs=_named_paths(args.input),
    )
    print(f"manifest: {destination}")
    return 0


if __name__ == "__main__":  # pragma: no cover - exercised through the CLI
    raise SystemExit(main())
