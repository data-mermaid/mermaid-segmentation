"""``segdata build moorea_labeled_corals``: publish the legacy Moorea prefix as a dataset version."""

from __future__ import annotations

import argparse
import logging
from pathlib import Path
from typing import Any

from segdata_etl.common.legacy import legacy_dropped, read_legacy, source_info
from segdata_etl.common.store import open_store
from segdata_etl.common.version import utc_today
from segdata_etl.datasets import external
from segdata_etl.label_maps.curated import load_curated
from segdata_etl.taxonomy import find_repo_root
from segdata_spec.publish import publish_dataset

logger = logging.getLogger(__name__)

DATASET = "moorea_labeled_corals"


def add_arguments(parser: argparse.ArgumentParser, *, workdir: Path) -> None:
    """Add the shared external-builder options; Moorea needs no others and no ``workdir``."""
    del workdir
    external.add_build_arguments(parser)


def run(args: argparse.Namespace, client: Any) -> int:
    """Publish a new ``moorea_labeled_corals`` version and print its version and manifest sha256.

    Points take the case-merged ``source_label_name``; the licence comes from the legacy manifest.
    """
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
    source = source_info(legacy)
    dropped = legacy_dropped(DATASET, legacy.manifest)
    curated, curated_info = load_curated(DATASET)
    labels = external.build_labels(legacy.classes, curated, taxonomy.benthic_attributes)
    annotations = legacy.annotations
    points = external.build_points(annotations.assign(annotator="default"), legacy.classes)
    images = external.legacy_images(legacy.objects, external.group_folder_identity)
    annotated = set(
        zip(annotations["year"].astype(str), annotations["image_id"].astype(str), strict=True)
    )
    external.require_same_ids(
        {(image.group, image.image_id) for image in images}, annotated, what="moorea images"
    )
    external.require_known_labels(points, labels)
    sniffed = external.sniff_legacy_images(client, images, legacy=legacy, workers=args.workers)
    external.require_points_inside(points, images, sniffed)
    version = external.pick_version(dest, bucket, DATASET, git_sha=git.sha, today=utc_today())
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
        source=source,
        label_mapping=external.label_mapping_record(labels, curated_info, git_sha=git.sha),
        dropped=dropped,
        repo_dir=repo_root,
        git=git,
    )
    logger.info("published %s %s", DATASET, manifest.version)
    print(f"version: {manifest.version}\nmanifest_sha256: {sha}")
    return 0
