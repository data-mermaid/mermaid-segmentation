"""``segdata build catlin_seaview``: stage the legacy Catlin Seaview prefix as a dataset version."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

from segdata_etl.common import versions
from segdata_etl.common.legacy import legacy_dropped, read_legacy, source_info
from segdata_etl.datasets import external
from segdata_etl.label_maps.curated import curated_source, load_curated
from segdata_etl.taxonomy import find_repo_root
from segdata_spec import DATASET_BUCKET
from segdata_spec.git import probe_git, refuse_unpublishable

DATASET = "catlin_seaview"
LICENSE = "CC BY-NC-SA 4.0"


def add_arguments(parser: argparse.ArgumentParser, *, workdir: Path) -> None:
    """Add the shared external-builder options; Catlin needs no others and no ``workdir``."""
    del workdir
    external.add_build_arguments(parser)


def run(args: argparse.Namespace, client: Any) -> int:
    """Stage a new ``catlin_seaview`` version and print its version and manifest sha256.

    Raises:
        PublishRefused: The git tree is dirty or unpushed.
    """
    repo_root = find_repo_root(Path(__file__).resolve().parent)
    git = probe_git(repo_root)
    refuse_unpublishable(git, repo_root)
    taxonomy = external.load_taxonomy(
        client,
        DATASET_BUCKET,
        version=args.taxonomy_version,
        manifest_sha256=args.taxonomy_manifest_sha256,
        cache_dir=args.cache_dir,
    )
    legacy = read_legacy(client, DATASET, bucket=args.legacy_bucket)
    source = source_info(legacy, license_override=LICENSE)
    dropped = legacy_dropped(DATASET, legacy.manifest)
    curated = load_curated(DATASET)
    labels = external.build_labels(legacy.classes, curated, taxonomy.benthic_attributes)
    points = external.build_points(legacy.annotations.assign(annotator="default"), legacy.classes)
    images = external.legacy_images(legacy.objects, external.group_folder_identity)
    annotated = set(zip(legacy.annotations["region"], legacy.annotations["image_id"], strict=True))
    external.require_same_ids(
        {(image.group, image.image_id) for image in images}, annotated, what="catlin images"
    )
    external.require_known_labels(points, labels)
    probe = external.probe_legacy_images(client, images, legacy=legacy, workers=args.workers)
    points = external.drop_unreadable_points(points, probe)
    points, dropped = external.drop_points_outside_and_record(
        points, probe.images, probe.sniffed, dropped
    )
    return external.publish_version(
        dataset=DATASET,
        kind="points",
        tables={
            "points.parquet": points,
            "labels.parquet": labels,
            "stats/label_counts.parquet": versions.count_labels(points),
        },
        source=source,
        label_mapping=external.label_mapping_record(
            labels, curated_source(DATASET), git_sha=git.sha
        ),
        dropped=dropped,
        git=git,
        repo_root=repo_root,
        client=client,
        probe=probe,
        legacy=legacy,
        args=args,
    )
