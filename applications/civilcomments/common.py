"""Shared constants and deterministic I/O for the CivilComments application."""

from __future__ import annotations

import gzip
import hashlib
import json
from pathlib import Path
from typing import Any

import pandas as pd

SOURCE_URL = (
    "https://huggingface.co/datasets/shlomihod/civil-comments-wilds/resolve/main/"
    "all_data_with_identities.csv"
)
SOURCE_REPOSITORY_COMMIT = "3fbfeca80bad0f3aec37e72fa07eff222b6e752f"
SOURCE_SHA256 = "403e638c83a225d738a937ff98b61fd0631e30f710d57928c7766d413526b77f"
SOURCE_SIZE_BYTES = 272_542_670

IDENTITY_COLUMNS = (
    "male",
    "female",
    "LGBTQ",
    "christian",
    "muslim",
    "other_religions",
    "black",
    "white",
)
ENVIRONMENT_NAMES = {
    "male": "male",
    "female": "female",
    "LGBTQ": "lgbtq",
    "christian": "christian",
    "muslim": "muslim",
    "other_religions": "other_religion",
    "black": "black",
    "white": "white",
}
SPLITS = ("train", "val", "test")


def sha256_file(path: Path, chunk_size: int = 1 << 20) -> str:
    """Hash a file without loading it into memory."""

    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()


def write_json(path: Path, payload: dict[str, Any]) -> None:
    """Write stable human-readable JSON."""

    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")


def write_csv_gzip(frame: pd.DataFrame, path: Path) -> None:
    """Write a deterministic UTF-8 gzip-compressed CSV."""

    path.parent.mkdir(parents=True, exist_ok=True)
    # An empty original filename and a fixed timestamp make the stream
    # content-addressable across machines and output paths.
    with (
        path.open("wb") as raw,
        gzip.GzipFile(filename="", mode="wb", fileobj=raw, mtime=0) as compressed,
    ):
        frame.to_csv(compressed, index=False, lineterminator="\n")


def read_table(path: Path, **kwargs) -> pd.DataFrame:
    """Read CSV/CSV.gz or Parquet from a path."""

    if path.suffix.lower() == ".parquet":
        return pd.read_parquet(path, **kwargs)
    return pd.read_csv(path, **kwargs)
