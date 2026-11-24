"""``segdata build benthos_yuval``: publish the legacy Benthos tiles and masks as a dense version."""

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
from PIL import Image

from segdata_etl.common import publishing
from segdata_etl.common.legacy import LegacyDataset, legacy_dropped, read_legacy, source_info
from segdata_etl.common.store import open_store
from segdata_etl.datasets import external
from segdata_etl.label_maps.curated import load_curated
from segdata_etl.taxonomy import find_repo_root
from segdata_spec.schema import LABEL_COUNTS_SCHEMA

DATASET = "benthos_yuval"
LICENSE = "CC0 1.0 (Dryad)"
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
        raise ValueError(f"legacy image {key!r} is not at images/<site>/<file>")
    site = parts[1]
    return f"{site}_{PurePosixPath(parts[2]).stem}", site


def _with_masks(images: Sequence[external.LegacyImage]) -> list[external.LegacyImage]:
    """Point each tile at its legacy mask ``labels/<site>/<file>``."""
    return [
        replace(image, mask_key=f"{_MASKS_FOLDER}/{image.group}/{PurePosixPath(image.key).name}")
        for image in images
    ]


def _tile_ids(annotations: pd.DataFrame) -> pd.Series:
    return annotations["site"].astype(str) + "_" + annotations["image_id"].astype(str)


def _tile_labels(annotations: pd.DataFrame, classes: Mapping[str, int]) -> dict[str, set[int]]:
    """Return the label ids the legacy rows list per tile id, ``unlabeled`` left out.

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


def _read_mask_stats(
    client: Any, images: Sequence[external.LegacyImage], *, legacy: LegacyDataset, workers: int
) -> dict[str, _MaskStats]:
    """Download and decode each tile's legacy mask on ``workers`` threads, keyed by image id."""

    def read(image: external.LegacyImage) -> _MaskStats:
        key = legacy.prefix + str(image.mask_key)
        return _mask_stats(client.get_object(Bucket=legacy.bucket, Key=key)["Body"].read())

    with ThreadPoolExecutor(max_workers=workers) as pool:
        stats = list(pool.map(read, images))
    return {image.image_id: stat for image, stat in zip(images, stats, strict=True)}


def _mask_problem(
    stat: _MaskStats, size: tuple[int, int], listed: set[int], label_ids: Collection[int]
) -> str | None:
    """Describe why a mask disagrees with its image size, the labels or the legacy rows."""
    if (stat.width, stat.height) != size:
        return f"mask size {(stat.width, stat.height)} differs from image size {size}"
    present = set(stat.counts) - {_UNLABELED}
    unknown = sorted(present - set(label_ids))
    if unknown:
        return f"mask values not in labels.parquet: {unknown}"
    if present != listed:
        return f"mask labels {sorted(present)} differ from the legacy rows' labels {sorted(listed)}"
    return None


def _require_masks_match(
    images: Sequence[external.LegacyImage],
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
    """Publish a new dense ``benthos_yuval`` version and print its version and manifest sha256.

    Every mask is downloaded and checked before any image is copied.
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
    images = _with_masks(external.legacy_images(legacy.objects, benthos_identity))
    external.require_same_ids(
        {image.image_id for image in images},
        set(_tile_ids(legacy.annotations)),
        what="benthos tiles",
    )
    external.require_same_ids(
        {key for key in legacy.objects if PurePosixPath(key).parts[0] == _MASKS_FOLDER},
        {str(image.mask_key) for image in images},
        what="benthos masks",
    )
    tile_labels = _tile_labels(legacy.annotations, legacy.classes)
    sniffed = external.sniff_legacy_images(client, images, legacy=legacy, workers=args.workers)
    stats = _read_mask_stats(client, images, legacy=legacy, workers=args.workers)
    label_ids = labels.column("label_id").to_pylist()
    _require_masks_match(images, sniffed, stats, tile_labels, label_ids)
    return external.publish_version(
        dataset=DATASET,
        kind="dense",
        tables={
            "labels.parquet": labels,
            "stats/label_counts.parquet": _mask_label_counts(stats),
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
