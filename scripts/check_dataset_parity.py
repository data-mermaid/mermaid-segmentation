#!/usr/bin/env python3
"""Compare an old dataset loader (points or dense) with a pinned dataset version.

python scripts/check_dataset_parity.py --data-config configs/data_config.yaml \
    --dataset catlin_seaview --split train --version 20261120-ab12cd3 \
    --manifest-sha256 <sha256> --cache-dir /scratch/cache

What is verified: the manifest against --manifest-sha256, and every table against the
manifest's sha256, also under --no-sync. The manifest's kind picks the comparison. A points
version syncs no image and is compared point by point. A dense version also syncs the images and
masks it compares, each verified by size, ETag and CRC64NVME. A Benthos-shaped old loader is
compared tile by tile: each tile's mask must equal the old mask pixel for pixel, and its
label_counts rows must equal the mask's pixel counts. A Hugging Face old loader is matched image
by image by pixel hash, and its masks, pixel totals and vocabulary are compared.

Needs S3 read access through the default boto3 credentials. The old side always loads its
annotations, classes.json and masks through LocalS3Cache.

--no-sync makes only the new side offline: no S3 client is built, and the version must already
be staged in --cache-dir. Tables sit at <dir>/mermaid-ic-datasets/datasets/<name>/<version>/;
dense images and masks sit at <dir>/<image store bucket>/<prefix><key>. The tables are verified
as a sync verifies them; images and masks are read as staged. A table or file that is not staged
fails the run.

To make the old side offline too, pre-fill a local cache (for example with
scripts/prefetch_cache.py) and set MERMAIDSEG_LOCAL_CACHE_DIR to it and MERMAIDSEG_S3_OFFLINE=1.
A cache miss then raises instead of calling S3.

Native label names of a points version are compared lowercased on both sides. With
--resize-max-edge N, new point coordinates are scaled so the longest image edge is N, as in the
old loader's space (CoralNet manifest coordinates are at 2048). Without it they are compared
unscaled. Points versions only: a dense version with --resize-max-edge is an error.

Exits 1 on any difference. Exits with an error when the old split has no points, tiles or images.
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
from segdata_spec.resize import ResizeSpec

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
        "--no-sync",
        action="store_true",
        help="no S3 for the new version, staged in --cache-dir; the old side still needs S3 or a cache",
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
    kind = version_kind(args.cache_dir, pin)
    if kind == "dense" and resize is not None:
        p.error("--resize-max-edge applies to points versions only")
    if kind == "dense":
        dense_report = run_dense_parity(
            cfg["data"],
            hf_revision_probe=_revision_probe(cfg["data"], args.dataset, args.split),
            **common,
        )
        print(format_dense_report(dense_report))
        return 0 if dense_report.ok else 1
    # The tables were synced above; only the dense path still has images to fetch.
    report = run_parity(
        cfg["data"], annotators=args.annotator, resize=resize, **{**common, "sync": False}
    )
    print(format_report(report))
    return 0 if report.ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
