#!/usr/bin/env python3
"""Validate a frozen wide prediction export and calculate Brier losses."""

# ruff: noqa: E402 -- direct-script support requires path setup before imports.

from __future__ import annotations

import argparse
import sys
from pathlib import Path

_PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

import pandas as pd

from hrms.artifacts import write_artifact_manifest
from hrms.predictions import predictions_to_brier_losses


def _read(path: Path) -> pd.DataFrame:
    if path.suffix.lower() == ".parquet":
        return pd.read_parquet(path)
    if path.suffix.lower() == ".csv":
        return pd.read_csv(path)
    raise ValueError(f"prediction input must be CSV or Parquet: {path}")


def _write(table: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.suffix.lower() == ".parquet":
        table.to_parquet(path, index=False)
    elif path.suffix.lower() == ".csv":
        table.to_csv(path, index=False)
    else:
        raise ValueError(f"loss output must be CSV or Parquet: {path}")


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("predictions", type=Path)
    parser.add_argument("model_registry", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    predictions = _read(args.predictions)
    registry = pd.read_csv(args.model_registry)
    losses = predictions_to_brier_losses(predictions, registry)
    _write(losses, args.output)
    manifest_path = args.output.with_suffix(args.output.suffix + ".manifest.json")
    write_artifact_manifest(
        manifest_path,
        inputs={"predictions": args.predictions, "model_registry": args.model_registry},
        artifacts={"loss_table": args.output},
        parameters={"loss": "brier", "prediction_prefix": "prediction__"},
        source_files=[
            _PROJECT_ROOT / "pyproject.toml",
            _PROJECT_ROOT / "uv.lock",
            *sorted((_PROJECT_ROOT / "hrms").glob("*.py")),
        ],
        source_base_dir=_PROJECT_ROOT,
        base_dir=args.output.parent,
    )
    print(f"loss_table: {args.output}")
    print(f"manifest: {manifest_path}")
    return 0


if __name__ == "__main__":  # pragma: no cover - exercised through the CLI
    raise SystemExit(main())
