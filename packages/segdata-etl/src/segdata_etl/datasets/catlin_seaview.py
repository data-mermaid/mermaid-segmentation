"""``segdata build catlin_seaview``: publish the legacy Catlin Seaview prefix as a dataset version."""

from __future__ import annotations

import argparse
import logging
from collections.abc import Mapping
from datetime import date
from pathlib import Path
from typing import Any

import pandas as pd
import pyarrow as pa

from segdata_etl.common.legacy import legacy_dropped, read_legacy, source_info
from segdata_etl.common.store import open_store
from segdata_etl.datasets import external
from segdata_etl.label_maps.curated import load_curated
from segdata_etl.taxonomy import find_repo_root
from segdata_spec.publish import publish_dataset
from segdata_spec.schema import POINTS_SCHEMA

logger = logging.getLogger(__name__)

DATASET = "catlin_seaview"
LICENSE = "CC BY-NC-SA 4.0"


def add_arguments(parser: argparse.ArgumentParser, *, workdir: Path) -> None:
    """Add the shared external-builder options; Catlin needs no others and no ``workdir``."""
    del workdir
    external.add_build_arguments(parser)


def _build_points(annotations: pd.DataFrame, classes: Mapping[str, int]) -> pa.Table:
    """Build ``points.parquet`` from the legacy annotations, one ``default`` annotator row each.

    Raises:
        ValueError: A ``source_label_name`` is not in ``classes.json``.
    """
    unknown = sorted(set(annotations["source_label_name"]) - set(classes))
    if unknown:
        raise ValueError(f"label names not in classes.json: {unknown}")
    return pa.table(
        {
            "image_id": annotations["image_id"].astype(str).tolist(),
            "row": annotations["row"].tolist(),
            "col": annotations["col"].tolist(),
            "label_id": annotations["source_label_name"].map(classes).tolist(),
            "annotator": ["default"] * len(annotations),
        }
    ).cast(POINTS_SCHEMA)


def run(args: argparse.Namespace, client: Any) -> int:
    """Publish a new ``catlin_seaview`` version and print its version and manifest sha256."""
    repo_root = find_repo_root(Path(__file__).resolve().parent)
    dest, bucket = open_store(args.store, source_client=client)
    git = external.checked_git_state(repo_root)
    taxonomy = external.load_taxonomy(
        dest,
        bucket,
        version=args.taxonomy_version,
        manifest_sha256=args.taxonomy_sha256,
        cache_dir=external.taxonomy_cache_dir(args.store, args.cache_dir),
    )
    legacy = read_legacy(client, DATASET, bucket=args.legacy_bucket)
    curated, curated_info = load_curated(DATASET)
    labels = external.build_labels(legacy.classes, curated, taxonomy.benthic_attributes)
    points = _build_points(legacy.annotations, legacy.classes)
    images = external.legacy_images(legacy.objects, external.group_folder_identity)
    annotated = set(zip(legacy.annotations["region"], legacy.annotations["image_id"], strict=True))
    external.require_same_ids(
        {(image.group, image.image_id) for image in images}, annotated, what="catlin images"
    )
    external.require_known_labels(points, labels)
    sniffed = external.sniff_legacy_images(client, images, legacy=legacy, workers=args.workers)
    external.require_points_inside(points, images, sniffed)
    version = external.pick_version(dest, bucket, DATASET, git_sha=git.sha, today=date.today())
    external.require_version_absent(dest, bucket, DATASET, version)
    images_table = external.stage_images(
        client,
        dest,
        images,
        sniffed,
        legacy=legacy,
        dest_bucket=bucket,
        dataset=DATASET,
        workers=args.workers,
    )
    manifest, sha = publish_dataset(
        dest,
        bucket=bucket,
        dataset=DATASET,
        version=version,
        kind="points",
        tables={
            "images.parquet": images_table,
            "points.parquet": points,
            "labels.parquet": labels,
            "stats/label_counts.parquet": external.count_labels(points),
        },
        image_store=external.image_store(bucket, DATASET),
        etl_package="segdata-etl",
        etl_command=external.build_command(DATASET, args),
        source=source_info(legacy, license_override=LICENSE),
        label_mapping=external.label_mapping_record(labels, curated_info, git_sha=git.sha),
        dropped=legacy_dropped(DATASET, legacy.manifest),
        repo_dir=repo_root,
        git=git,
    )
    logger.info("published %s %s", DATASET, manifest.version)
    print(f"version: {manifest.version}\nmanifest_sha256: {sha}")
    return 0
