#!/usr/bin/env python3
"""Download the dataset versions a data config pins into a local cache.

Each manifest is checked against its pin's sha256, and each table against its recorded sha256.
Each image and mask is checked for ETag and size, plus CRC64NVME or MD5 where recorded.
Needs S3 read access through the default boto3 credentials.

The config's ``data:`` section pins each dataset, for example:

    data:
      datasets:
        catlin_seaview:
          version: 20261120-ab12cd3
          manifest_sha256: <sha256>
          split_file: splits/catlin_seaview.csv
          train: {use: [train]}

Run it with:

    python scripts/sync_data.py --data-config configs/my_data_config.yaml --cache-dir /scratch/cache
"""

from __future__ import annotations

import argparse
from pathlib import Path

import boto3
import yaml

from mermaidseg.data.config import sync_plan
from mermaidseg.data.sync import format_summary, run_sync
from mermaidseg.io import REPO_ROOT


def main(argv: list[str] | None = None) -> int:
    """Parse arguments, sync the pinned versions and print one line per dataset."""
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument("--data-config", type=Path, required=True, help="YAML file with a data: section")
    p.add_argument(
        "--cache-dir", type=Path, required=True, help="directory the versions are downloaded into"
    )
    args = p.parse_args(argv)
    cfg = yaml.safe_load(args.data_config.read_text())
    plan = sync_plan(cfg["data"], REPO_ROOT)
    result = run_sync(plan, args.cache_dir, boto3.client("s3"))
    print(format_summary(result))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
