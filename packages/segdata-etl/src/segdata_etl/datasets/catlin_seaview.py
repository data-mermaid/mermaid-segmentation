"""Catlin Seaview ingestion: download the raw zips, drop damaged images, publish to S3.

Raw layout (``--data-dir``, or ``--workdir`` after download)::

    <root>/<region>/<quadrat_id>.jpg
    <root>/tabular-data/annotations_<region>.csv
"""

from __future__ import annotations

import argparse
import logging
import zipfile
from pathlib import Path
from typing import Any
from urllib.parse import urlparse

import pandas as pd
import requests
from tqdm.auto import tqdm

from ..common.images import find_undecodable
from ..common.labels import build_classes_and_colors
from ..common.manifest import Source, build_manifest
from ..common.publish import S3Target, list_keys, preflight, publish_dataset, upload_files

logger = logging.getLogger(__name__)

DATASET_NAME = "catlin_seaview"
BASE_URL = "http://data.qld.edu.au/public/Q1281"
TABULAR_ZIP_URL = f"{BASE_URL}/tabular-data.zip"

REGIONS: tuple[str, ...] = (
    "ATL",
    "IND_CHA",
    "IND_MDV",
    "PAC_AUS",
    "PAC_IDN_PHL",
    "PAC_SLB",
    "PAC_TLS",
    "PAC_TWN",
    "PAC_USA",
)

PARQUET_COLUMNS = {
    "region": "Catlin Seaview region code (e.g. 'ATL', 'IND_CHA').",
    "image_id": "Quadrat ID (string), filename stem of the per-region jpg.",
    "row": "Annotation pixel row (y, integer).",
    "col": "Annotation pixel column (x, integer).",
    "source_label_name": "Catlin Seaview label name (matches keys in classes.json).",
}
_REQUIRED_CSV_COLUMNS = ["quadratid", "x", "y", "label_name"]
_SORT_KEYS = ["region", "image_id", "row", "col"]

SOURCE = Source(
    name=DATASET_NAME,
    citation=(
        "Seaview Survey Photo-quadrat and Image Classification Dataset "
        "(Gonzalez-Rivero et al., 2019)"
    ),
    raw_data_url=BASE_URL,
    license="CC BY-NC-SA 4.0",
)


def add_arguments(parser: argparse.ArgumentParser, *, workdir: Path) -> None:
    """Register the Catlin Seaview flags."""
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=None,
        help="Raw tree already on disk; skips the download.",
    )
    parser.add_argument(
        "--workdir",
        type=Path,
        default=workdir,
        help="Where zips are downloaded and extracted when --data-dir is not given.",
    )
    parser.add_argument(
        "--force-download",
        action="store_true",
        help="Re-download and re-extract zips even if they are already on disk.",
    )


def _stream_download(url: str, dest: Path, chunk_size: int = 1 << 20) -> None:
    """Stream ``url`` into ``dest`` through a ``.part`` file."""
    dest.parent.mkdir(parents=True, exist_ok=True)
    tmp = dest.with_suffix(dest.suffix + ".part")
    with requests.get(url, stream=True, timeout=300) as resp:
        resp.raise_for_status()
        total = int(resp.headers.get("Content-Length", 0)) or None
        with (
            tmp.open("wb") as fh,
            tqdm(total=total, unit="B", unit_scale=True, desc=dest.name, leave=False) as pbar,
        ):
            for chunk in resp.iter_content(chunk_size=chunk_size):
                fh.write(chunk)
                pbar.update(len(chunk))
    tmp.replace(dest)


def _download_and_extract(url: str, workdir: Path, *, force: bool) -> None:
    """Download ``url`` into ``<workdir>/zips/`` and extract into ``workdir`` once."""
    zip_name = Path(urlparse(url).path).name
    zip_path = workdir / "zips" / zip_name
    marker = workdir / f".{zip_name}.extracted"
    if force:
        zip_path.unlink(missing_ok=True)
        marker.unlink(missing_ok=True)

    if zip_path.exists():
        logger.info("Zip already on disk: %s", zip_path)
    else:
        logger.info("Downloading %s -> %s", url, zip_path)
        _stream_download(url, zip_path)

    if marker.exists():
        logger.info("Already extracted: %s", zip_name)
        return
    logger.info("Extracting %s -> %s", zip_path, workdir)
    with zipfile.ZipFile(zip_path) as zf:
        zf.extractall(workdir)
    marker.touch()


def _download_raw(workdir: Path, *, force: bool) -> Path:
    """Download and extract the tabular and per-region zips; return ``workdir``."""
    workdir.mkdir(parents=True, exist_ok=True)
    _download_and_extract(TABULAR_ZIP_URL, workdir, force=force)
    for region in REGIONS:
        _download_and_extract(f"{BASE_URL}/annotated-images/{region}.zip", workdir, force=force)
    return workdir


def _clean_region_csv(df: pd.DataFrame, region: str) -> pd.DataFrame:
    """Return the region's annotations as ``[region, image_id, row, col, source_label_name]``."""
    missing = [c for c in _REQUIRED_CSV_COLUMNS if c not in df.columns]
    if missing:
        raise ValueError(f"annotations_{region}.csv is missing required columns: {missing}")
    clean = df.dropna(subset=_REQUIRED_CSV_COLUMNS).copy()
    clean["region"] = region
    clean["image_id"] = clean["quadratid"].astype("Int64").astype(str)
    clean["row"] = pd.to_numeric(clean["y"], errors="coerce").round().astype("Int64")
    clean["col"] = pd.to_numeric(clean["x"], errors="coerce").round().astype("Int64")
    clean["source_label_name"] = clean["label_name"].astype(str)
    clean = clean.dropna(subset=["row", "col"])
    clean["row"] = clean["row"].astype(int)
    clean["col"] = clean["col"].astype(int)
    return clean[["region", "image_id", "row", "col", "source_label_name"]].reset_index(drop=True)


def build_annotations(data_root: Path) -> pd.DataFrame:
    """Read every region CSV under ``<data_root>/tabular-data`` into one annotations frame."""
    tabular_root = data_root / "tabular-data"
    if not tabular_root.is_dir():
        raise FileNotFoundError(
            f"Expected a tabular-data directory at {tabular_root}; pass --data-dir pointing at "
            "a raw tree, or omit it to download."
        )
    frames = []
    for region in REGIONS:
        csv_path = tabular_root / f"annotations_{region}.csv"
        if not csv_path.exists():
            logger.warning("Missing tabular CSV for region %s: %s", region, csv_path)
            continue
        frames.append(_clean_region_csv(pd.read_csv(csv_path), region))
    if not frames:
        raise RuntimeError(
            f"No region annotation CSVs found under {tabular_root}; nothing to ingest."
        )
    return pd.concat(frames, ignore_index=True)


def _find_damaged(
    annotations: pd.DataFrame, data_root: Path, *, workers: int
) -> dict[str, list[str]]:
    """Return ``{region: sorted image ids}`` whose jpg is missing or undecodable."""
    pairs = annotations[["region", "image_id"]].drop_duplicates()
    by_path = {
        data_root / region / f"{image_id}.jpg": (region, image_id)
        for region, image_id in pairs.itertuples(index=False)
    }
    damaged: dict[str, list[str]] = {region: [] for region in REGIONS}
    for path, error in find_undecodable(list(by_path), workers=workers):
        region, image_id = by_path[path]
        logger.warning("Dropping %s/%s.jpg: %s", region, image_id, error)
        damaged[region].append(image_id)
    return {region: sorted(ids) for region, ids in damaged.items()}


def _drop_damaged(annotations: pd.DataFrame, damaged: dict[str, list[str]]) -> pd.DataFrame:
    bad = {(region, image_id) for region, ids in damaged.items() for image_id in ids}
    keep = [
        (region, image_id) not in bad
        for region, image_id in zip(annotations["region"], annotations["image_id"], strict=True)
    ]
    return annotations[keep].reset_index(drop=True)


def _upload_images(
    client: Any,
    target: S3Target,
    annotations: pd.DataFrame,
    data_root: Path,
    *,
    workers: int,
    skip_existing: bool,
) -> dict[str, int]:
    """Upload the images the annotations reference; return counts sent per region."""
    existing = list_keys(client, target.bucket, target.prefix) if skip_existing else set()
    pairs = annotations[["region", "image_id"]].drop_duplicates()
    pending: list[tuple[Path, str]] = []
    uploaded = dict.fromkeys(REGIONS, 0)
    for region, image_id in pairs.itertuples(index=False):
        key = f"{target.prefix}/images/{region}/{image_id}.jpg"
        if key not in existing:
            pending.append((data_root / region / f"{image_id}.jpg", key))
            uploaded[region] += 1
    upload_files(client, target, pending, workers=workers, skip_existing=False)
    return uploaded


def _resolve_data_root(args: argparse.Namespace) -> Path:
    if args.data_dir is not None:
        data_root = args.data_dir.expanduser().resolve()
        if not data_root.is_dir():
            raise FileNotFoundError(f"--data-dir does not exist: {data_root}")
        return data_root
    return _download_raw(args.workdir.expanduser().resolve(), force=args.force_download)


def run(args: argparse.Namespace, client: Any) -> int:
    """Ingest Catlin Seaview into S3; return the process exit code."""
    target = S3Target(args.bucket, args.prefix.strip("/"), args.dry_run)
    preflight(client, target, DATASET_NAME, overwrite=args.overwrite)
    data_root = _resolve_data_root(args)

    annotations = build_annotations(data_root)
    damaged = _find_damaged(annotations, data_root, workers=args.workers)
    n_before = len(annotations)
    annotations = _drop_damaged(annotations, damaged)
    logger.info(
        "Dropped %d of %d annotation rows (damaged/missing images)",
        n_before - len(annotations),
        n_before,
    )

    classes, colors = build_classes_and_colors(annotations["source_label_name"])

    uploaded = _upload_images(
        client,
        target,
        annotations,
        data_root,
        workers=args.workers,
        skip_existing=args.skip_existing_s3,
    )
    manifest = build_manifest(
        SOURCE,
        target,
        image_key_template=f"{target.prefix}/images/{{region}}/{{image_id}}.jpg",
        parquet_columns=PARQUET_COLUMNS,
        ingestion={"damaged_images": damaged, "images_uploaded": uploaded},
    )
    publish_dataset(
        client,
        target,
        dataset_name=DATASET_NAME,
        annotations=annotations,
        sort_by=_SORT_KEYS,
        classes=classes,
        colors=colors,
        manifest=manifest,
        overwrite=args.overwrite,
    )
    return 0
