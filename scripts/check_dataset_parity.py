#!/usr/bin/env python3
"""Compare an old dataset loader (points or dense) with a pinned dataset version.

python scripts/check_dataset_parity.py --data-config configs/data_config.yaml \
    --dataset catlin_seaview --split train --version 20261120-ab12cd3 \
    --manifest-sha256 <sha256> --cache-dir /scratch/cache

What is verified: the manifest against --manifest-sha256, and every table against the
manifest's sha256. For a dense version, every image and mask is also synced and verified by size,
ETag and CRC64NVME; each tile's mask must then equal the old mask pixel for pixel, and its
label_counts rows must equal the mask's pixel counts. A points version syncs no image.

Needs S3 read access through the default boto3 credentials. The old side always loads its
annotations, classes.json and masks through LocalS3Cache, and the new side syncs from S3.

--offline makes only the new side offline: no S3 client is built, and every file must already be
staged in --cache-dir, where the same checks run. Tables sit at
<dir>/mermaid-ic-datasets/datasets/<name>/<version>/; dense images and masks sit at
<dir>/<image store bucket>/<prefix><key>. A file that is not staged fails the run.

To make the old side offline too, pre-fill a local cache (for example with
scripts/prefetch_cache.py) and set MERMAIDSEG_LOCAL_CACHE_DIR to it and MERMAIDSEG_S3_OFFLINE=1.
A cache miss then raises instead of calling S3.

Exits 1 on any difference. Exits with an error when the old split has no points or tiles.
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
        help="no S3 for the new version, staged in --cache-dir; the old side still needs S3 or a cache",
    )
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
        client=None if args.offline else boto3.client("s3"),
        annotators=args.annotator,
        offline=args.offline,
    )
    print(format_report(report))
    return 0 if report.ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
