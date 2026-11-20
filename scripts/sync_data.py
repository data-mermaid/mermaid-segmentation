#!/usr/bin/env python3
"""Download the dataset versions a data config pins into a local cache, verifying every file.

python scripts/sync_data.py --data-config configs/data_config.yaml --cache-dir /scratch/cache
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
    p.add_argument("--data-config", type=Path, required=True)
    p.add_argument("--cache-dir", type=Path, required=True)
    args = p.parse_args(argv)
    cfg = yaml.safe_load(args.data_config.read_text())
    plan = sync_plan(cfg["data"], REPO_ROOT)
    result = run_sync(plan, args.cache_dir, boto3.client("s3"))
    print(format_summary(result))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
