"""``segdata build benthos_yuval``: stage the ingested Benthos tiles and masks as a dense version."""

from __future__ import annotations

import argparse
import io
from collections.abc import Collection, Mapping, Sequence
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, replace
from pathlib import Path, PurePosixPath
from typing import Any

import numpy as np
import pandas as pd
import pyarrow as pa
from botocore.exceptions import ClientError
from PIL import Image

from segdata_etl.common.ingested import IngestedDataset, read_ingested, source_info
from segdata_etl.common.s3_objects import SourceChanged
from segdata_etl.datasets import external
from segdata_etl.label_maps.curated import curated_source, load_curated
from segdata_etl.taxonomy import find_repo_root
from segdata_spec import DATASET_BUCKET
from segdata_spec.git import probe_git, refuse_unpublishable
from segdata_spec.s3 import is_precondition_failed
from segdata_spec.schema import LABEL_COUNTS_SCHEMA

DATASET = "benthos_yuval"
_IMAGES_FOLDER = "images"
_MASKS_FOLDER = "labels"
_UNLABELED = 0


@dataclass(frozen=True)
class _MaskStats:
    """Size of one decoded mask and its pixel count per value, value 0 included."""

    width: int
    height: int
    counts: Mapping[int, int]


def add_arguments(parser: argparse.ArgumentParser, *, workdir: Path) -> None:
    """Add the shared external-builder options; Benthos needs no others and no ``workdir``."""
    del workdir
    external.add_build_arguments(parser)


def benthos_identity(key: str) -> tuple[str, str] | None:
    """Identify ``images/<site>/<y0>_<x0>.png`` as ``("<site>_<y0>_<x0>", site)``.

    Returns None for a key off ``images/``.

    Raises:
        ValueError: A key under ``images/`` is not exactly one folder deep.
    """
    parts = PurePosixPath(key).parts
    if parts[0] != _IMAGES_FOLDER:
        return None
    if len(parts) != 3:
        raise ValueError(f"ingested image {key!r} is not at images/<site>/<file>")
    site = parts[1]
    return f"{site}_{PurePosixPath(parts[2]).stem}", site


def _with_masks(images: Sequence[external.IngestedImage]) -> list[external.IngestedImage]:
    """Point each tile at its ingested mask ``labels/<site>/<file>``."""
    return [
        replace(image, mask_key=f"{_MASKS_FOLDER}/{image.group}/{PurePosixPath(image.key).name}")
        for image in images
    ]


def _tile_ids(annotations: pd.DataFrame) -> pd.Series:
    return annotations["site"].astype(str) + "_" + annotations["image_id"].astype(str)


def _tile_labels(annotations: pd.DataFrame, classes: Mapping[str, int]) -> dict[str, set[int]]:
    """Return the label ids the ingested rows list per tile id, ``unlabeled`` left out.

    Raises:
        ValueError: A ``source_label_name`` is not in ``classes.json``.
    """
    unknown = sorted(set(annotations["source_label_name"]) - set(classes))
    if unknown:
        raise ValueError(f"label names not in classes.json: {unknown}")
    labels: dict[str, set[int]] = {}
    for tile_id, name in zip(_tile_ids(annotations), annotations["source_label_name"], strict=True):
        labels.setdefault(tile_id, set())
        if classes[name] != _UNLABELED:
            labels[tile_id].add(classes[name])
    return labels


def _mask_stats(png: bytes) -> _MaskStats:
    """Decode a mask PNG and count its pixels per value; a 3-channel mask uses channel 0."""
    with Image.open(io.BytesIO(png)) as handle:
        values = np.asarray(handle)
    if values.ndim == 3:
        values = values[..., 0]
    ids, counts = np.unique(values, return_counts=True)
    return _MaskStats(
        width=int(values.shape[1]),
        height=int(values.shape[0]),
        counts=dict(zip(ids.tolist(), counts.tolist(), strict=True)),
    )


def _read_masks(
    client: Any,
    images: Sequence[external.IngestedImage],
    *,
    ingested: IngestedDataset,
    workers: int,
) -> dict[str, bytes]:
    """Download each tile's ingested mask on ``workers`` threads, keyed by image id.

    Raises:
        SourceChanged: A mask's ETag differs from the listing.
    """

    def read(image: external.IngestedImage) -> bytes:
        key = ingested.prefix + str(image.mask_key)
        etag = ingested.objects[str(image.mask_key)].etag
        try:
            data: bytes = client.get_object(Bucket=ingested.bucket, Key=key, IfMatch=f'"{etag}"')[
                "Body"
            ].read()
        except ClientError as exc:
            if is_precondition_failed(exc):
                raise SourceChanged(
                    f"s3://{ingested.bucket}/{key} changed since it was listed"
                ) from exc
            raise
        return data

    with ThreadPoolExecutor(max_workers=workers) as pool:
        masks = list(pool.map(read, images))
    return {image.image_id: mask for image, mask in zip(images, masks, strict=True)}


def _mask_problem(
    stat: _MaskStats, size: tuple[int, int], listed: set[int], label_ids: Collection[int]
) -> str | None:
    """Describe why a mask disagrees with its image size, the labels or the ingested rows."""
    if (stat.width, stat.height) != size:
        return f"mask size {(stat.width, stat.height)} differs from image size {size}"
    present = set(stat.counts) - {_UNLABELED}
    unknown = sorted(present - set(label_ids))
    if unknown:
        return f"mask values not in labels.parquet: {unknown}"
    if present != listed:
        return (
            f"mask labels {sorted(present)} differ from the ingested rows' labels {sorted(listed)}"
        )
    return None


def _require_masks_match(
    images: Sequence[external.IngestedImage],
    sniffed: Mapping[str, tuple[str, int, int]],
    stats: Mapping[str, _MaskStats],
    tile_labels: Mapping[str, set[int]],
    label_ids: Collection[int],
) -> None:
    """Refuse a mask whose size, values or present labels disagree with its tile.

    Raises:
        ValueError: The first such tile, naming its image id and the disagreement.
    """
    for image in images:
        problem = _mask_problem(
            stats[image.image_id],
            sniffed[image.key][1:],
            tile_labels.get(image.image_id, set()),
            label_ids,
        )
        if problem is not None:
            raise ValueError(f"tile {image.image_id}: {problem}")


def _mask_label_counts(stats: Mapping[str, _MaskStats]) -> pa.Table:
    """Build ``stats/label_counts.parquet``: mask pixels per ``(image_id, label_id)``, no 0."""
    rows = [
        {"image_id": image_id, "label_id": label_id, "n": n}
        for image_id in sorted(stats)
        for label_id, n in sorted(stats[image_id].counts.items())
        if label_id != _UNLABELED
    ]
    return pa.Table.from_pylist(rows, schema=LABEL_COUNTS_SCHEMA)


def run(args: argparse.Namespace, client: Any) -> int:
    """Stage a new dense ``benthos_yuval`` version and print its version and manifest sha256.

    Every kept tile's mask is downloaded and checked before anything is staged; the masks are
    staged for upload and the tiles planned as copies.

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
    ingested = read_ingested(client, DATASET, bucket=args.ingest_bucket)
    source = source_info(ingested)
    dropped = ingested.manifest.ingestion.dropped
    curated = load_curated(DATASET)
    labels = external.build_labels(ingested.classes, curated, taxonomy.benthic_attributes)
    images = _with_masks(external.ingested_images(ingested.objects, benthos_identity))
    external.require_same_ids(
        {image.image_id for image in images},
        set(_tile_ids(ingested.annotations)),
        what="benthos tiles",
    )
    external.require_same_ids(
        {key for key in ingested.objects if PurePosixPath(key).parts[0] == _MASKS_FOLDER},
        {str(image.mask_key) for image in images},
        what="benthos masks",
    )
    tile_labels = _tile_labels(ingested.annotations, ingested.classes)
    probe = external.probe_ingested_images(client, images, ingested=ingested, workers=args.workers)
    masks = _read_masks(client, probe.images, ingested=ingested, workers=args.workers)
    stats = {image_id: _mask_stats(mask) for image_id, mask in masks.items()}
    label_ids = labels.column("label_id").to_pylist()
    _require_masks_match(probe.images, probe.sniffed, stats, tile_labels, label_ids)
    return external.publish_version(
        dataset=DATASET,
        kind="dense",
        tables={
            "labels.parquet": labels,
            "stats/label_counts.parquet": _mask_label_counts(stats),
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
        ingested=ingested,
        args=args,
        masks=masks,
    )
