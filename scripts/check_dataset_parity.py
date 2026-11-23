#!/usr/bin/env python3
"""Compare an old dataset loader (points or dense) with a pinned dataset version.

python scripts/check_dataset_parity.py --data-config configs/data_config.yaml \
    --dataset catlin_seaview --split train --version 20261120-ab12cd3 \
    --manifest-sha256 <sha256> --cache-dir /scratch/cache

What is verified: the manifest against --manifest-sha256, and every table against the
manifest's sha256. For a dense version, every image and mask is also synced and verified by size,
ETag and CRC64NVME; each tile's mask must then equal the old mask pixel for pixel, and its
label_counts rows must equal the mask's pixel counts. A points version syncs no image.

Needs S3 read access through the default boto3 credentials, unless --offline: then no S3 client
is built, and every file must already be staged in --cache-dir, where the same checks run. Tables
sit at <dir>/mermaid-ic-datasets/datasets/<name>/<version>/; dense images and masks sit at
<dir>/<image store bucket>/<prefix><key>. A file that is not staged fails the run.

Native label names are compared lowercased on both sides. With --resize-max-edge N, new point
coordinates are scaled so the longest image edge is N, as in the old loader's space (CoralNet
manifest coordinates are at 2048). Without it they are compared unscaled. Points versions only.

Exits 1 on any difference. Exits with an error when the old split has no points or tiles.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import boto3
import yaml

from mermaidseg.data.parity import format_report, run_parity
from segdata_spec.resize import ResizeSpec


def main(argv: list[str] | None = None) -> int:
    """Parse arguments, run the parity check, print the report and return 1 on any difference."""
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument("--data-config", type=Path, required=True, help="old data config")
    p.add_argument("--dataset", required=True, help="dataset name in the old data config")
    p.add_argument("--split", required=True, help="split to compare")
    p.add_argument("--version", required=True, help="pinned dataset version")
    p.add_argument("--manifest-sha256", required=True, help="sha256 of the version's manifest")
    p.add_argument(
        "--cache-dir", type=Path, required=True, help="directory the version is downloaded into"
    )
    p.add_argument(
        "--offline",
        action="store_true",
        help="verify the version staged in --cache-dir with the same checks; no S3",
    )
    p.add_argument("--annotator", action="append", default=None, help="repeat for priority order")
    p.add_argument(
        "--resize-max-edge",
        type=int,
        default=None,
        help="scale new points to this longest image edge (default filter and quality)",
    )
    args = p.parse_args(argv)
    resize = ResizeSpec(max_edge=args.resize_max_edge) if args.resize_max_edge is not None else None
    cfg = yaml.safe_load(args.data_config.read_text())
    report = run_parity(
        cfg["data"],
        dataset=args.dataset,
        split=args.split,
        version=args.version,
        manifest_sha256=args.manifest_sha256,
        cache_dir=args.cache_dir,
        client=None if args.offline else boto3.client("s3"),
        annotators=args.annotator,
        offline=args.offline,
        resize=resize,
    )
    print(format_report(report))
    return 0 if report.ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
