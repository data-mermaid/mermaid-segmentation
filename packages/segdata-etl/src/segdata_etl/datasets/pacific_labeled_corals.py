"""``segdata build pacific_labeled_corals``: stage the ingested Pacific prefix as a dataset version."""

from __future__ import annotations

import argparse
from collections.abc import Sequence
from dataclasses import replace
from pathlib import Path, PurePosixPath
from typing import Any

import pandas as pd

from segdata_etl.common import versions
from segdata_etl.common.ingested import read_ingested, source_info
from segdata_etl.common.stage import publishable_repo
from segdata_etl.datasets import external
from segdata_etl.label_maps.curated import curated_source, load_curated
from segdata_spec import DATASET_BUCKET

DATASET = "pacific_labeled_corals"
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
        raise ValueError(f"ingested image {key!r} is not at images/<site>/<subset>/<file>")
    return PurePosixPath(parts[3]).stem, parts[1]


def _key_subset(image: external.IngestedImage) -> str:
    return PurePosixPath(image.key).parts[2]


def _long_annotations(annotations: pd.DataFrame) -> pd.DataFrame:
    """Return one row per non-null annotator column per position, ``annotator`` = column name.

    Rows keep the ingested row order, then the ``ANNOTATORS`` order within a position.
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
    """Return ``{"subset", "raw_imageid"}`` per image id from the ingested annotations.

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
    """Refuse a ``(image_id, row, col)`` that appears on more than one ingested row.

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
    images: Sequence[external.IngestedImage], attrs: dict[str, dict[str, Any]]
) -> list[external.IngestedImage]:
    return [replace(image, attrs=attrs[image.image_id]) for image in images]


def run(args: argparse.Namespace, client: Any) -> int:
    """Stage a new ``pacific_labeled_corals`` version and print its version and manifest sha256.

    Points are long form, one row per annotator label; label counts count every such row.

    Raises:
        PublishRefused: The git tree is dirty or unpushed.
    """
    repo = publishable_repo()
    taxonomy = external.load_taxonomy(
        client,
        DATASET_BUCKET,
        version=args.taxonomy_version,
        manifest_sha256=args.taxonomy_manifest_sha256,
        cache_dir=args.cache_dir,
    )
    ingested = read_ingested(client, DATASET, bucket=args.ingest_bucket)
    source = source_info(ingested)
    dropped = ingested.manifest.ingestion.dropped
    curated = load_curated(DATASET)
    labels = external.build_labels(ingested.classes, curated, taxonomy.benthic_attributes)
    points = external.build_points(_long_annotations(ingested.annotations), ingested.classes)
    _require_unique_positions(ingested.annotations)
    images = external.ingested_images(ingested.objects, pacific_identity)
    external.require_same_ids(
        {(image.group, _key_subset(image), image.image_id) for image in images},
        _annotated_images(ingested.annotations),
        what="pacific images",
    )
    images = _with_attrs(images, _image_attrs(ingested.annotations))
    external.require_known_labels(points, labels)
    probe = external.probe_ingested_images(client, images, ingested=ingested, workers=args.workers)
    points = external.drop_unreadable_points(points, probe)
    # Bounds depend only on (image_id, row, col), so a position leaves with all its annotator
    # rows; points_out_of_bounds counts point rows, as in the other builders.
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
            labels, curated_source(DATASET), git_sha=repo.git.sha
        ),
        dropped=dropped,
        repo=repo,
        client=client,
        probe=probe,
        ingested=ingested,
        args=args,
    )
