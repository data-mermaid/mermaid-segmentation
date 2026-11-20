#!/usr/bin/env python3
"""Compare the points of an old point loader with a pinned dataset version.

python scripts/check_dataset_parity.py --data-config configs/data_config.yaml \
    --dataset catlin_seaview --split train --version 20261120-ab12cd3 \
    --manifest-sha256 <sha256> --cache-dir /scratch/cache
"""

from __future__ import annotations

import argparse
from pathlib import Path

import boto3
import yaml

from mermaidseg.data.parity import format_report, run_parity


def main(argv: list[str] | None = None) -> int:
    """Parse arguments, run the parity check, print the report and return 1 on any difference."""
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument("--data-config", type=Path, required=True, help="old data config")
    p.add_argument("--dataset", required=True)
    p.add_argument("--split", required=True)
    p.add_argument("--version", required=True)
    p.add_argument("--manifest-sha256", required=True)
    p.add_argument("--cache-dir", type=Path, required=True)
    p.add_argument("--annotator", action="append", default=None, help="repeat for priority order")
    args = p.parse_args(argv)
    cfg = yaml.safe_load(args.data_config.read_text())
    report = run_parity(
        cfg["data"],
        dataset=args.dataset,
        split=args.split,
        version=args.version,
        manifest_sha256=args.manifest_sha256,
        cache_dir=args.cache_dir,
        client=boto3.client("s3"),
        annotators=args.annotator,
    )
    print(format_report(report))
    return 0 if report.ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
