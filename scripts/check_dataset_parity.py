#!/usr/bin/env python3
"""Compare the points of an old point loader with a pinned dataset version.

python scripts/check_dataset_parity.py --data-config configs/data_config.yaml \
    --dataset catlin_seaview --split train --version 20261120-ab12cd3 \
    --manifest-sha256 <sha256> --cache-dir /scratch/cache

Reads points or dense versions, as the manifest's kind says. Needs S3 read access through the
default boto3 credentials; with --no-sync it reads a locally staged version from --cache-dir and
never calls S3. Exits 1 on any difference.
"""

from __future__ import annotations

import argparse
import inspect
import logging
from collections.abc import Callable, Mapping
from pathlib import Path
from typing import Any

import boto3
import yaml
from huggingface_hub import HfApi

from mermaidseg.data.parity import (
    format_dense_report,
    format_report,
    run_dense_parity,
    run_parity,
    sync_tables,
    version_kind,
)
from mermaidseg.datasets.factory import DATASET_CLASSES
from segdata_spec.manifest import Pin

logger = logging.getLogger(__name__)


def hf_head_sha(repo_id: str) -> str | None:
    """Return the head revision of a Hugging Face dataset repo, or ``None`` if unreachable."""
    try:
        return HfApi().dataset_info(repo_id).sha
    except Exception as exc:  # network, auth and local-path failures all leave the sha unknown
        logger.warning("could not read the HF revision of %s: %s", repo_id, exc)
        return None


def hf_repo_id(data_cfg: Mapping[str, Any], dataset: str, split: str) -> str | None:
    """Return the configured ``hf_repo_id`` of a split, else the dataset class default."""
    configured = ((data_cfg.get(dataset) or {}).get(split) or {}).get("hf_repo_id")
    if configured:
        return str(configured)
    parameter = inspect.signature(DATASET_CLASSES[dataset].__init__).parameters.get("hf_repo_id")
    return None if parameter is None else str(parameter.default)


def _revision_probe(
    data_cfg: Mapping[str, Any], dataset: str, split: str
) -> Callable[[], str | None]:
    repo = hf_repo_id(data_cfg, dataset, split)
    return lambda: None if repo is None else hf_head_sha(repo)


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
        "--no-sync", action="store_true", help="read a staged version from --cache-dir; no S3"
    )
    p.add_argument("--annotator", action="append", default=None, help="repeat for priority order")
    args = p.parse_args(argv)
    cfg = yaml.safe_load(args.data_config.read_text())
    pin = Pin(name=args.dataset, version=args.version, manifest_sha256=args.manifest_sha256)
    client = None if args.no_sync else boto3.client("s3")
    if client is not None:
        sync_tables(client, pin, args.cache_dir)
    common: dict[str, Any] = {
        "dataset": args.dataset,
        "split": args.split,
        "version": args.version,
        "manifest_sha256": args.manifest_sha256,
        "cache_dir": args.cache_dir,
        "client": client,
        "sync": not args.no_sync,
    }
    if version_kind(args.cache_dir, pin) == "dense":
        dense_report = run_dense_parity(
            cfg["data"],
            hf_revision_probe=_revision_probe(cfg["data"], args.dataset, args.split),
            **common,
        )
        print(format_dense_report(dense_report))
        return 0 if dense_report.ok else 1
    report = run_parity(cfg["data"], annotators=args.annotator, **common)
    print(format_report(report))
    return 0 if report.ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
