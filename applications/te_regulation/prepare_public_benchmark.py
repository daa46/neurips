#!/usr/bin/env python3
"""Download, verify, label, and prepare the public TE--cCRE benchmark."""

# ruff: noqa: E402 -- direct-script support requires path setup before imports.

from __future__ import annotations

import argparse
import sys
from pathlib import Path

_PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from applications.te_regulation.public_data import (
    build_candidate_pool,
    build_precluster_matches,
    download_all,
    extract_candidate_te_sequences,
    extract_precluster_windows,
    load_config,
    run_preflight,
)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("configs/te/public_ccre.json"),
        help="frozen public-benchmark JSON configuration",
    )
    parser.add_argument(
        "--base-dir",
        type=Path,
        default=Path("."),
        help="base directory for source paths declared in the configuration",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)
    subparsers.add_parser("download", help="atomically download or verify all pinned inputs")
    preflight = subparsers.add_parser(
        "preflight", help="derive labels and write family/subfamily feasibility counts"
    )
    preflight.add_argument(
        "--output-dir", type=Path, default=Path("data/te_public/preflight")
    )
    preflight.add_argument(
        "--skip-source-verification",
        action="store_true",
        help="development fixtures only: record observed hashes without enforcing pinned bytes",
    )
    pool = subparsers.add_parser(
        "pool", help="construct a subfamily-diverse candidate pool before homology gating"
    )
    pool.add_argument(
        "--preflight-dir", type=Path, default=Path("data/te_public/preflight")
    )
    pool.add_argument("--output-dir", type=Path, default=Path("data/te_public/pool"))
    sequence = subparsers.add_parser(
        "sequence", help="extract oriented full-TE sequences and calculate GC"
    )
    sequence.add_argument("--pool-dir", type=Path, default=Path("data/te_public/pool"))
    sequence.add_argument(
        "--output-dir", type=Path, default=Path("data/te_public/sequences")
    )
    match = subparsers.add_parser(
        "match", help="match positive/control candidates before homology clustering"
    )
    match.add_argument(
        "--sequence-dir", type=Path, default=Path("data/te_public/sequences")
    )
    match.add_argument("--output-dir", type=Path, default=Path("data/te_public/matches"))
    windows = subparsers.add_parser(
        "windows", help="extract oriented model windows and make pair-complete cluster inputs"
    )
    windows.add_argument(
        "--sequence-dir", type=Path, default=Path("data/te_public/sequences")
    )
    windows.add_argument("--match-dir", type=Path, default=Path("data/te_public/matches"))
    windows.add_argument("--output-dir", type=Path, default=Path("data/te_public/windows"))
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    config = load_config(args.config)
    if args.command == "download":
        outputs = download_all(config, base_dir=args.base_dir)
    elif args.command == "preflight":
        outputs = run_preflight(
            args.config,
            args.output_dir,
            base_dir=args.base_dir,
            verify_sources=not args.skip_source_verification,
        ).values()
    elif args.command == "pool":
        outputs = build_candidate_pool(
            args.config,
            args.preflight_dir,
            args.output_dir,
        ).values()
    elif args.command == "sequence":
        outputs = extract_candidate_te_sequences(
            args.config,
            args.pool_dir,
            args.output_dir,
            base_dir=args.base_dir,
        ).values()
    elif args.command == "match":
        outputs = build_precluster_matches(
            args.config,
            args.sequence_dir,
            args.output_dir,
        ).values()
    else:
        outputs = extract_precluster_windows(
            args.config,
            args.sequence_dir,
            args.match_dir,
            args.output_dir,
            base_dir=args.base_dir,
        ).values()
    for output in outputs:
        print(output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
