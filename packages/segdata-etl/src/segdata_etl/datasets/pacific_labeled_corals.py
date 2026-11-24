"""``segdata build pacific_labeled_corals``: publish the legacy Pacific prefix as a dataset version."""

from __future__ import annotations

import argparse
from collections.abc import Sequence
from dataclasses import replace
from pathlib import Path, PurePosixPath
from typing import Any

import pandas as pd

from segdata_etl.common import publishing
from segdata_etl.common.legacy import legacy_dropped, read_legacy, source_info
from segdata_etl.common.store import open_store
from segdata_etl.datasets import external
from segdata_etl.label_maps.curated import load_curated
from segdata_etl.taxonomy import find_repo_root

DATASET = "pacific_labeled_corals"
LICENSE = "CC0 1.0 (Dryad)"
ANNOTATORS = ("archived", "host", *(f"visitor{i}" for i in range(1, 6)))
_IMAGES_FOLDER = "images"
_SHOWN = 5


def add_arguments(parser: argparse.ArgumentParser, *, workdir: Path) -> None:
    """Add the shared external-builder options; Pacific needs no others and no ``workdir``."""
    del workdir
    external.add_build_arguments(parser)


def pacific_identity(key: str) -> tuple[str, str] | None:
    """Identify ``images/<site>/<subset>/<image_id>.<ext>`` as ``(image_id, site)``.

    Returns None for a key off ``images/``.

    Raises:
        ValueError: A key under ``images/`` is not exactly two folders deep.
    """
    parts = PurePosixPath(key).parts
    if parts[0] != _IMAGES_FOLDER:
        return None
    if len(parts) != 4:
        raise ValueError(f"legacy image {key!r} is not at images/<site>/<subset>/<file>")
    return PurePosixPath(parts[3]).stem, parts[1]


def _key_subset(image: external.LegacyImage) -> str:
    return PurePosixPath(image.key).parts[2]


def _long_annotations(annotations: pd.DataFrame) -> pd.DataFrame:
    """Return one row per non-null annotator column per position, ``annotator`` = column name.

    Rows keep the legacy row order, then the ``ANNOTATORS`` order within a position.
    """
    melted = annotations.reset_index(names="_position").melt(
        id_vars=["_position", "image_id", "row", "col"],
        value_vars=list(ANNOTATORS),
        var_name="annotator",
        value_name="source_label_name",
    )
    melted = melted.dropna(subset=["source_label_name"])
    rank = melted["annotator"].map({name: i for i, name in enumerate(ANNOTATORS)})
    order = melted.assign(_rank=rank).sort_values(["_position", "_rank"], kind="stable")
    return order.drop(columns=["_position", "_rank"]).reset_index(drop=True)


def _image_attrs(annotations: pd.DataFrame) -> dict[str, dict[str, Any]]:
    """Return ``{"subset", "raw_imageid"}`` per image id from the legacy annotations.

    Raises:
        ValueError: An image id carries more than one ``(subset, raw_imageid)`` pair.
    """
    pairs = annotations[["image_id", "subset", "raw_imageid"]].drop_duplicates()
    repeated = sorted(set(pairs["image_id"][pairs["image_id"].duplicated()]))
    if repeated:
        raise ValueError(f"images with several (subset, raw_imageid) pairs: {repeated[:5]}")
    return {
        str(image_id): {"subset": str(subset), "raw_imageid": int(raw)}
        for image_id, subset, raw in pairs.itertuples(index=False)
    }


def _require_unique_positions(annotations: pd.DataFrame) -> None:
    """Refuse a ``(image_id, row, col)`` that appears on more than one legacy row.

    Raises:
        ValueError: Some position is duplicated; the message gives the count and the first few.
    """
    duplicated = annotations.loc[
        annotations.duplicated(["image_id", "row", "col"], keep=False), ["image_id", "row", "col"]
    ].drop_duplicates()
    if len(duplicated):
        first = list(duplicated.head(_SHOWN).itertuples(index=False, name=None))
        raise ValueError(f"{len(duplicated)} duplicated positions (image_id, row, col): {first}")


def _annotated_images(annotations: pd.DataFrame) -> set[tuple[str, str, str]]:
    columns = (annotations[c].astype(str) for c in ("site", "subset", "image_id"))
    return set(zip(*columns, strict=True))


def _with_attrs(
    images: Sequence[external.LegacyImage], attrs: dict[str, dict[str, Any]]
) -> list[external.LegacyImage]:
    return [replace(image, attrs=attrs[image.image_id]) for image in images]


def run(args: argparse.Namespace, client: Any) -> int:
    """Publish a new ``pacific_labeled_corals`` version and print its version and manifest sha256.

    Points are long form, one row per annotator label; label counts count every such row.
    """
    repo_root = find_repo_root(Path(__file__).resolve().parent)
    dest, bucket = open_store(args.store, source_client=client)
    git = publishing.checked_git_state(repo_root)
    taxonomy = external.load_taxonomy(
        dest,
        bucket,
        version=args.taxonomy_version,
        manifest_sha256=args.taxonomy_manifest_sha256,
        cache_dir=external.taxonomy_cache_dir(args.store, args.cache_dir),
    )
    legacy = read_legacy(client, DATASET, bucket=args.legacy_bucket)
    source = source_info(legacy, license_override=LICENSE)
    dropped = legacy_dropped(DATASET, legacy.manifest)
    curated, curated_info = load_curated(DATASET)
    labels = external.build_labels(legacy.classes, curated, taxonomy.benthic_attributes)
    points = external.build_points(_long_annotations(legacy.annotations), legacy.classes)
    _require_unique_positions(legacy.annotations)
    images = external.legacy_images(legacy.objects, pacific_identity)
    external.require_same_ids(
        {(image.group, _key_subset(image), image.image_id) for image in images},
        _annotated_images(legacy.annotations),
        what="pacific images",
    )
    images = _with_attrs(images, _image_attrs(legacy.annotations))
    external.require_known_labels(points, labels)
    sniffed = external.sniff_legacy_images(client, images, legacy=legacy, workers=args.workers)
    # Bounds depend only on (image_id, row, col), so a position leaves with all its annotator
    # rows; points_out_of_bounds counts point rows, as in the other builders.
    points, dropped = external.drop_points_outside_and_record(points, images, sniffed, dropped)
    return external.publish_version(
        dataset=DATASET,
        kind="points",
        tables={
            "points.parquet": points,
            "labels.parquet": labels,
            "stats/label_counts.parquet": publishing.count_labels(points),
        },
        source=source,
        label_mapping=external.label_mapping_record(labels, curated_info, git_sha=git.sha),
        dropped=dropped,
        git=git,
        repo_root=repo_root,
        source_client=client,
        dest=dest,
        bucket=bucket,
        images=images,
        sniffed=sniffed,
        legacy=legacy,
        args=args,
    )
