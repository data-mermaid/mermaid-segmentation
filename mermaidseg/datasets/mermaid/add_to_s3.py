"""Reproducible end-to-end ingestion script for the MERMAID dataset.

Saves a snapshot of the MERMAID annotations file, validates each per-region JPG by 
attempting a full PIL decode, (dropping rows whose image is missing or damaged), uploads only
the valid jpg images to S3, and writes ``manifest.json`` / ``classes.json`` /
``colors.json`` describing the result.

Run::

    python -m mermaidseg.datasets.catlin_seaview.download_raw_data_and_add_to_s3 \
        --bucket dev-datamermaid-sm-sources \
        --prefix external_validation_datasets/catlin_seaview

The raw download step can be skipped by pointing ``--existing-data-dir`` at a
directory that already matches the layout used by ``catlin_seaview.ipynb``::

    <existing_data_dir>/<region>/<quadrat_id>.jpg
    <existing_data_dir>/tabular-data/annotations_<region>.csv
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
import zipfile
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from urllib.parse import urlparse

import boto3
import numpy as np
import pandas as pd
import requests
from PIL import Image, UnidentifiedImageError
from tqdm.auto import tqdm

Image.MAX_IMAGE_PIXELS = None

logger = logging.getLogger(__name__)


CATLIN_BASE_URL = "http://data.qld.edu.au/public/Q1281"
TABULAR_ZIP_URL = f"{CATLIN_BASE_URL}/tabular-data.zip"

REGIONS: list[str] = [
    "ATL",
    "IND_CHA",
    "IND_MDV",
    "PAC_AUS",
    "PAC_IDN_PHL",
    "PAC_SLB",
    "PAC_TLS",
    "PAC_TWN",
    "PAC_USA",
]

ANNOTATIONS_PARQUET_NAME = "catlin_seaview_annotations.parquet"


@dataclass
class IngestionConfig:
    bucket: str
    prefix: str
    workdir: Path
    existing_data_dir: Path | None
    skip_existing_s3: bool
    force_download: bool
    upload_workers: int
    dry_run: bool

    @property
    def annotations_key(self) -> str:
        return f"{self.prefix.rstrip('/')}/{ANNOTATIONS_PARQUET_NAME}"

    @property
    def images_prefix(self) -> str:
        return f"{self.prefix.rstrip('/')}/images"



def build_palette(label_names: list[str], seed: int = 1337) -> dict[str, list[int]]:
    """Build a deterministic RGB palette keyed by label name.

    Mirrors ``build_palette`` in ``catlin_seaview.ipynb``.
    """
    rng = np.random.default_rng(seed)
    colors: dict[str, list[int]] = {"unlabeled": [0, 0, 0]}
    for name in label_names:
        colors[name] = [int(x) for x in rng.integers(0, 256, size=3)]
    return colors





def is_valid_image(path: Path) -> bool:
    """Return True iff the local jpg exists and can be fully decoded by PIL.

    Catches the ``OSError: image file is truncated`` / ``UnidentifiedImageError``
    family of exceptions emitted by PIL when a JPG is corrupt or partially
    written. Forces a full decode via ``img.load()`` rather than ``verify()``
    so truncated bodies are also caught.
    """
    if not path.exists() or path.stat().st_size == 0:
        return False
    try:
        with Image.open(path) as img:
            img.load()
    except (UnidentifiedImageError, OSError, ValueError, SyntaxError):
        return False
    return True


def validate_local_images(
    data_root: Path,
    region_image_ids: dict[str, list[str]],
    workers: int = 16,
) -> tuple[dict[str, list[str]], dict[str, int], dict[str, list[str]]]:
    """Identify damaged or missing local jpgs per region.

    Args:
        data_root: Root directory containing the per-region ``<region>/`` image
            folders.
        region_image_ids: ``{region: [image_id, ...]}`` candidate jpgs to check.
        workers: Number of parallel worker threads (PIL decode is CPU + I/O).

    Returns:
        valid_ids: ``{region: [image_id, ...]}`` jpgs that exist and decode.
        bad_count: ``{region: int}`` count of missing or damaged jpgs.
        bad_ids: ``{region: [image_id, ...]}`` of missing/damaged image_ids.
    """
    valid_ids: dict[str, list[str]] = {r: [] for r in REGIONS}
    bad_count: dict[str, int] = dict.fromkeys(REGIONS, 0)
    bad_ids: dict[str, list[str]] = {r: [] for r in REGIONS}

    tasks: list[tuple[str, str, Path]] = []
    for region, ids in region_image_ids.items():
        for image_id in ids:
            tasks.append((region, image_id, data_root / region / f"{image_id}.jpg"))

    if not tasks:
        return valid_ids, bad_count, bad_ids

    def _check(region: str, image_id: str, path: Path) -> tuple[str, str, bool]:
        return region, image_id, is_valid_image(path)

    with (
        ThreadPoolExecutor(max_workers=workers) as pool,
        tqdm(total=len(tasks), desc="validate jpg", unit="img") as pbar,
    ):
        futures = [pool.submit(_check, r, i, p) for r, i, p in tasks]
        for fut in as_completed(futures):
            region, image_id, ok = fut.result()
            if ok:
                valid_ids[region].append(image_id)
            else:
                bad_count[region] += 1
                bad_ids[region].append(image_id)
            pbar.update(1)

    for region in REGIONS:
        valid_ids[region].sort()
        bad_ids[region].sort()
    return valid_ids, bad_count, bad_ids


def list_existing_image_keys(s3, bucket: str, images_prefix: str) -> set[str]:
    """List every key already present under ``s3://bucket/images_prefix/``."""
    paginator = s3.get_paginator("list_objects_v2")
    keys: set[str] = set()
    for page in paginator.paginate(Bucket=bucket, Prefix=f"{images_prefix.rstrip('/')}/"):
        for obj in page.get("Contents", []):
            keys.add(obj["Key"])
    return keys



def upload_json(
    s3,
    cfg: IngestionConfig,
    relative_key: str,
    payload: dict,
) -> str:
    key = f"{cfg.prefix.rstrip('/')}/{relative_key}"
    uri = f"s3://{cfg.bucket}/{key}"
    body = json.dumps(payload, indent=2).encode("utf-8")
    if cfg.dry_run:
        logger.info("[dry-run] Would upload %s (%d bytes)", uri, len(body))
        return uri
    s3.put_object(Bucket=cfg.bucket, Key=key, Body=body, ContentType="application/json")
    logger.info("Uploaded %s", uri)
    return uri


def build_classes(label_names: list[str]) -> dict[str, int]:
    classes: dict[str, int] = {"unlabeled": 0}
    classes.update({name: i + 1 for i, name in enumerate(label_names)})
    return classes


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Download, transform and upload the Catlin Seaview dataset to S3.",
    )
    parser.add_argument("--bucket", default="dev-datamermaid-sm-sources")
    parser.add_argument("--prefix", default="external_validation_datasets/catlin_seaview")
    parser.add_argument(
        "--workdir",
        type=Path,
        default=Path("./catlin_seaview_raw"),
        help="Local working directory where zips/extracted files live.",
    )
    parser.add_argument(
        "--existing-data-dir",
        type=Path,
        default=None,
        help=(
            "If set, skip the download/unzip step and read raw files directly from "
            "this directory. Expected layout: <dir>/<region>/<quadrat>.jpg and "
            "<dir>/tabular-data/annotations_<region>.csv."
        ),
    )
    parser.add_argument(
        "--no-skip-existing-s3",
        dest="skip_existing_s3",
        action="store_false",
        help="By default, image keys already in S3 are skipped; pass this flag to re-upload.",
    )
    parser.add_argument(
        "--force-download",
        action="store_true",
        help="Re-download zips even if they exist locally.",
    )
    parser.add_argument(
        "--upload-workers",
        type=int,
        default=16,
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Compute everything but skip all S3 writes.",
    )
    parser.add_argument(
        "--log-level",
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    logging.basicConfig(
        level=getattr(logging, args.log_level),
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )

    cfg = IngestionConfig(
        bucket=args.bucket,
        prefix=args.prefix.strip("/"),
        workdir=args.workdir,
        existing_data_dir=args.existing_data_dir,
        skip_existing_s3=args.skip_existing_s3,
        force_download=args.force_download,
        upload_workers=args.upload_workers,
        dry_run=args.dry_run,
    )

    if cfg.existing_data_dir is not None:
        data_root = cfg.existing_data_dir.expanduser().resolve()
        if not data_root.exists():
            raise FileNotFoundError(f"--existing-data-dir does not exist: {data_root}")
        logger.info("Using existing data dir: %s", data_root)
    else:
        data_root = download_raw(cfg.workdir.expanduser().resolve(), force=cfg.force_download)

    df_annotations, region_image_ids = build_annotations(data_root)

    logger.info("Validating local jpgs (PIL decode)...")
    valid_image_ids, _, damaged_ids = validate_local_images(
        data_root, region_image_ids, workers=cfg.upload_workers
    )
    for region, ids in damaged_ids.items():
        if ids:
            sample = ", ".join(ids[:5]) + (", ..." if len(ids) > 5 else "")
            logger.warning(
                "Region %s: %d damaged/missing jpg(s) [%s]",
                region,
                len(ids),
                sample,
            )

    valid_pairs = pd.DataFrame(
        [(region, image_id) for region, ids in valid_image_ids.items() for image_id in ids],
        columns=["region", "image_id"],
    )
    n_before = len(df_annotations)
    df_annotations = df_annotations.merge(
        valid_pairs, on=["region", "image_id"], how="inner"
    ).reset_index(drop=True)
    logger.info(
        "Dropped %d annotation rows referring to damaged/missing images (%d -> %d).",
        n_before - len(df_annotations),
        n_before,
        len(df_annotations),
    )

    label_names = sorted(df_annotations["source_label_name"].unique().tolist())
    classes = build_classes(label_names)
    colors = build_palette(label_names)

    parquet_uri = write_parquet_to_s3(df_annotations, cfg)

    uploaded_per_region = upload_images(cfg, data_root, valid_image_ids)

    s3 = boto3.client("s3")
    classes_uri = upload_json(s3, cfg, "classes.json", classes)
    colors_uri = upload_json(s3, cfg, "colors.json", colors)

    manifest = {
        "dataset_name": "catlin_seaview",
        "source": {
            "paper": (
                "Seaview Survey Photo-quadrat and Image Classification Dataset "
                "(Gonzalez-Rivero et al., 2019)"
            ),
            "raw_data_url": CATLIN_BASE_URL,
        },
        "created_at_utc": datetime.now(UTC).isoformat(timespec="seconds"),
        "s3": {
            "bucket": cfg.bucket,
            "prefix": cfg.prefix,
            "annotations_parquet": cfg.annotations_key,
            "annotations_uri": parquet_uri,
            "images_prefix": cfg.images_prefix,
            "image_key_template": f"{cfg.images_prefix}/{{region}}/{{image_id}}.jpg",
            "classes_json": f"{cfg.prefix}/classes.json",
            "colors_json": f"{cfg.prefix}/colors.json",
        },
        "schema": {
            "parquet_columns": {
                "region": "Catlin Seaview region code (e.g. 'ATL', 'IND_CHA').",
                "image_id": "Quadrat ID (string), filename stem of the per-region jpg.",
                "row": "Annotation pixel row (y, integer).",
                "col": "Annotation pixel column (x, integer).",
                "source_label_name": "Catlin Seaview label name (matches keys in classes.json).",
            },
        },
        "ingestion": {
            "skipped_damaged_or_missing_image_ids": damaged_ids,
            "images_uploaded_this_run_by_region": uploaded_per_region,
        },
    }
    manifest_uri = upload_json(s3, cfg, "manifest.json", manifest)

    print(f"Parquet:  {parquet_uri}")
    print(f"Manifest: {manifest_uri}")
    print(f"Classes:  {classes_uri}")
    print(f"Colors:   {colors_uri}")
    return 0


if __name__ == "__main__":
    sys.exit(main())