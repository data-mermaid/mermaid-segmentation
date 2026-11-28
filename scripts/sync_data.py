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

With ``--tables-only`` it syncs only the manifests, tables and taxonomy (no image or mask).

Run it with:

    python scripts/sync_data.py --data-config configs/my_data_config.yaml --cache-dir /scratch/cache
"""

from __future__ import annotations

import argparse
from pathlib import Path

import boto3
import yaml
from botocore.config import Config

from mermaidseg.data.config import sync_plan
from mermaidseg.data.sync import format_summary, run_sync, run_table_sync
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
    p.add_argument(
        "--workers",
        type=int,
        default=16,
        help="threads downloading images (default: 16; ignored with --tables-only)",
    )
    p.add_argument(
        "--tables-only",
        action="store_true",
        help="sync only the manifests, tables and taxonomy, with no image or mask",
    )
    args = p.parse_args(argv)
    cfg = yaml.safe_load(args.data_config.read_text())
    plan = sync_plan(cfg["data"], REPO_ROOT)
    # The client's connection pool must hold one connection per worker.
    client = boto3.client("s3", config=Config(max_pool_connections=max(args.workers, 10)))
    if args.tables_only:
        result = run_table_sync(plan, args.cache_dir, client)
    else:
        result = run_sync(plan, args.cache_dir, client, args.workers)
    print(format_summary(result))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
