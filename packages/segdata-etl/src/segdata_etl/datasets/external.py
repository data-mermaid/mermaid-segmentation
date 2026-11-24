"""Shared steps of the external dataset version builders (Catlin, Moorea, Pacific, Benthos).

Each builder module reads its legacy ``external_validation_datasets/<name>/`` prefix and calls
these helpers to build the uniform tables, copy the images and publish a dataset version.
"""

from __future__ import annotations

import argparse
import json
import logging
import shlex
from collections.abc import Callable, Collection, Iterable, Mapping, Sequence
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from functools import partial
from pathlib import Path, PurePosixPath
from typing import Any

import pandas as pd
import pyarrow as pa
import pyarrow.compute as pc

from segdata_etl.common.cli_flags import _pin_flags
from segdata_etl.common.copy import CopiedObject, CopyItem, copy_images
from segdata_etl.common.legacy import DEFAULT_BUCKET, LegacyDataset, LegacyObject, sniff_image
from segdata_etl.common.publishing import image_prefix
from segdata_etl.common.store import add_store_argument
from segdata_etl.label_maps.curated import CuratedMapInfo
from segdata_spec.manifest import LabelMapping, Pin
from segdata_spec.read import Taxonomy, open_taxonomy
from segdata_spec.schema import IMAGES_SCHEMA, LABELS_SCHEMA, POINTS_SCHEMA
from segdata_spec.sync import sync

logger = logging.getLogger(__name__)

DEFAULT_WORKERS = 16
DEFAULT_CACHE_DIR = Path("outputs") / "segdata-cache"
TAXONOMY_NAME = "mermaid-benthic"
_IMAGES_FOLDER = "images"
_EXTENSIONS = {"image/jpeg": "jpg", "image/png": "png"}


Identify = Callable[[str], tuple[str, str] | None]


@dataclass(frozen=True)
class LegacyImage:
    """One legacy image object: ``key`` is relative to the legacy prefix, ``etag`` unquoted.

    ``mask_key`` is the legacy key of the image's PNG mask, relative to the same prefix;
    ``attrs`` holds dataset-specific extras written to ``images.parquet`` as JSON.
    """

    image_id: str
    group: str
    key: str
    etag: str
    mask_key: str | None = None
    attrs: Mapping[str, Any] | None = None


def add_build_arguments(parser: argparse.ArgumentParser) -> None:
    """Add the options every external builder shares, including ``--store``."""
    add_store_argument(parser)
    _pin_flags(parser, "taxonomy-", "the mermaid-benthic taxonomy")
    parser.add_argument(
        "--cache-dir",
        type=Path,
        default=None,
        help=(
            "Where the taxonomy is synced (default: the store directory, "
            f"or {DEFAULT_CACHE_DIR} with --store s3)"
        ),
    )
    parser.add_argument("--workers", type=int, default=DEFAULT_WORKERS)
    parser.add_argument(
        "--legacy-bucket", default=DEFAULT_BUCKET, help="Bucket holding the legacy prefix"
    )


def taxonomy_cache_dir(store: str | Path, cache_dir: Path | None) -> Path:
    """Return ``cache_dir`` if set, else the store directory, or the default cache for ``s3``."""
    if cache_dir is not None:
        return cache_dir
    if str(store) == "s3":
        return DEFAULT_CACHE_DIR
    return Path(store)


def load_taxonomy(
    client: Any, bucket: str, *, version: str, manifest_sha256: str, cache_dir: Path
) -> Taxonomy:
    """Sync the pinned ``mermaid-benthic`` taxonomy from the store into ``cache_dir`` and open it."""
    pin = Pin(name=TAXONOMY_NAME, version=version, manifest_sha256=manifest_sha256)
    sync(client, {}, cache_dir, bucket=bucket, taxonomy=pin)
    return open_taxonomy(cache_dir, bucket, pin)


def _resolve_target(
    target: str, by_name: Mapping[str, list[tuple[str, str]]]
) -> tuple[str, str] | None:
    """Return the taxonomy ``(id, name)`` of ``target``, or None when no attribute has that name."""
    matches = by_name.get(target.lower(), [])
    if len(matches) > 1:
        raise ValueError(f"curated target {target!r} matches several taxonomy names: {matches}")
    return matches[0] if matches else None


def _attributes_by_name(attributes: pa.Table) -> dict[str, list[tuple[str, str]]]:
    by_name: dict[str, list[tuple[str, str]]] = {}
    for attr_id, name in zip(
        attributes.column("id").to_pylist(), attributes.column("name").to_pylist(), strict=True
    ):
        by_name.setdefault(name.lower(), []).append((attr_id, name))
    return by_name


def build_labels(
    classes: Mapping[str, int], curated: Mapping[str, str], attributes: pa.Table
) -> pa.Table:
    """Build ``labels.parquet`` from ``classes.json``, the curated map and taxonomy attributes.

    Label 0 (``unlabeled``) is left out. A curated target found in the taxonomy, ignoring case,
    takes its canonical name and id; one not found keeps the curated value with a null id.

    Raises:
        ValueError: A curated target matches more than one taxonomy name, ignoring case.
    """
    by_name = _attributes_by_name(attributes)
    rows = []
    for name, label_id in sorted(classes.items(), key=lambda item: item[1]):
        if label_id == 0:
            continue
        target = curated.get(name.lower())
        match = _resolve_target(target, by_name) if target is not None else None
        rows.append(
            {
                "label_id": label_id,
                "name": name,
                "mermaid_benthic_attribute_id": match[0] if match else None,
                "mermaid_benthic_attribute": match[1] if match else target,
                "mermaid_growth_form": None,
                "mapping_source": "curated",
            }
        )
    return pa.Table.from_pylist(rows, schema=LABELS_SCHEMA)


def label_mapping_record(labels: pa.Table, info: CuratedMapInfo, *, git_sha: str) -> LabelMapping:
    """Describe the curated mapping of ``labels`` for the manifest."""
    targets = labels.column("mermaid_benthic_attribute").to_pylist()
    ids = labels.column("mermaid_benthic_attribute_id").to_pylist()
    return LabelMapping(
        source="curated",
        sha256=info.sha256,
        path=info.path,
        git_ref=git_sha,
        n_labels=labels.num_rows,
        n_mapped=sum(target is not None for target in targets),
        n_unresolved=sum(t is not None and i is None for t, i in zip(targets, ids, strict=True)),
    )


def build_points(annotations: pd.DataFrame, classes: Mapping[str, int]) -> pa.Table:
    """Build ``points.parquet`` from ``image_id, row, col, source_label_name, annotator`` rows.

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
            "annotator": annotations["annotator"].astype(str).tolist(),
        }
    ).cast(POINTS_SCHEMA)


def group_folder_identity(key: str) -> tuple[str, str] | None:
    """Identify ``images/<group>/<image_id>.<ext>`` as ``(image_id, group)``; None off ``images/``.

    Raises:
        ValueError: A key under ``images/`` is not exactly one folder deep.
    """
    parts = PurePosixPath(key).parts
    if parts[0] != _IMAGES_FOLDER:
        return None
    if len(parts) != 3:
        raise ValueError(f"legacy image {key!r} is not at images/<group>/<file>")
    return PurePosixPath(parts[2]).stem, parts[1]


def legacy_images(objects: Mapping[str, LegacyObject], identify: Identify) -> list[LegacyImage]:
    """Return the objects ``identify`` names as ``(image_id, group)``, sorted by key.

    ``identify`` takes a key relative to the legacy prefix and returns None to skip it.

    Raises:
        ValueError: Two objects share an image id.
    """
    images: dict[str, LegacyImage] = {}
    for key in sorted(objects):
        identity = identify(key)
        if identity is None:
            continue
        image_id, group = identity
        if image_id in images:
            raise ValueError(f"image id {image_id!r} is both {images[image_id].key} and {key}")
        images[image_id] = LegacyImage(image_id, group, key, objects[key].etag)
    return list(images.values())


def require_known_labels(points: pa.Table, labels: pa.Table) -> None:
    """Refuse points whose ``label_id`` is not in ``labels``, naming the unknown ids.

    Raises:
        ValueError: A point's ``label_id`` is absent from ``labels.label_id``.
    """
    known = pc.is_in(points.column("label_id"), value_set=labels.column("label_id"))
    unknown = points.filter(pc.invert(known)).column("label_id")
    if len(unknown):
        ids = sorted(set(unknown.to_pylist()))
        raise ValueError(f"{len(unknown)} points carry label ids not in labels.parquet: {ids}")


def _sizes_table(
    images: Iterable[LegacyImage], sniffed: Mapping[str, tuple[str, int, int]]
) -> pa.Table:
    rows = [(image.image_id, *sniffed[image.key][1:]) for image in images]
    return pa.table(
        {
            "image_id": pa.array([r[0] for r in rows], pa.string()),
            "width": pa.array([r[1] for r in rows], pa.int64()),
            "height": pa.array([r[2] for r in rows], pa.int64()),
        }
    )


def _bounds_join(
    points: pa.Table,
    images: Iterable[LegacyImage],
    sniffed: Mapping[str, tuple[str, int, int]],
) -> tuple[pa.Table, pa.ChunkedArray]:
    """Join points to their image sizes; return the join in point order and an ``inside`` mask.

    A point is inside when ``0 <= row < height`` and ``0 <= col < width``. The mask is null for a
    point whose image has no sniffed size.
    """
    positioned = points.select(["image_id", "row", "col"]).append_column(
        "_position", pa.array(range(points.num_rows), pa.int64())
    )
    joined = positioned.join(
        _sizes_table(images, sniffed), "image_id", join_type="left outer"
    ).sort_by("_position")
    row, col = joined.column("row"), joined.column("col")
    inside = pc.and_kleene(
        pc.and_kleene(pc.greater_equal(row, 0), pc.less(row, joined.column("height"))),
        pc.and_kleene(pc.greater_equal(col, 0), pc.less(col, joined.column("width"))),
    )
    return joined, inside


def drop_points_outside(
    points: pa.Table,
    images: Iterable[LegacyImage],
    sniffed: Mapping[str, tuple[str, int, int]],
) -> tuple[pa.Table, int]:
    """Return the points inside their image's sniffed size, in order, and the number dropped.

    A point is inside when ``0 <= row < height`` and ``0 <= col < width``. The count is of point
    rows, so a long-form position with several annotator rows counts once per row.

    Raises:
        ValueError: A point's image has no sniffed size.
    """
    joined, inside = _bounds_join(points, images, sniffed)
    unsized = joined.filter(pc.is_null(inside))
    if unsized.num_rows:
        ids = sorted(set(unsized.column("image_id").to_pylist()))[:5]
        raise ValueError(f"{unsized.num_rows} points whose image has no sniffed size: {ids}")
    kept = points.filter(inside)
    return kept, points.num_rows - kept.num_rows


def drop_points_outside_and_record(
    points: pa.Table,
    images: Iterable[LegacyImage],
    sniffed: Mapping[str, tuple[str, int, int]],
    dropped: dict[str, int],
) -> tuple[pa.Table, dict[str, int]]:
    """Drop out-of-bounds points; return the kept points and ``dropped`` with the count merged in.

    ``points_out_of_bounds`` is added only when some point is dropped, and the count is logged.

    Raises:
        ValueError: A point's image has no sniffed size.
    """
    kept, n_out_of_bounds = drop_points_outside(points, images, sniffed)
    if not n_out_of_bounds:
        return kept, dropped
    logger.info("dropped %d points outside their image", n_out_of_bounds)
    return kept, {**dropped, "points_out_of_bounds": n_out_of_bounds}


def require_same_ids(found: Collection[Any], expected: Collection[Any], *, what: str) -> None:
    """Refuse ids in only one of ``found`` and ``expected``, naming up to five of each.

    Raises:
        ValueError: The two id sets differ.
    """
    missing = sorted(set(expected) - set(found))
    extra = sorted(set(found) - set(expected))
    if missing or extra:
        raise ValueError(
            f"{what}: {len(missing)} expected but absent {missing[:5]}, "
            f"{len(extra)} present but unexpected {extra[:5]}"
        )


def sniff_images(
    client: Any, bucket: str, keys: Sequence[str], *, workers: int
) -> dict[str, tuple[str, int, int]]:
    """Return ``(media_type, width, height)`` per key, read from header bytes on ``workers`` threads."""
    with ThreadPoolExecutor(max_workers=workers) as pool:
        results = pool.map(partial(sniff_image, client, bucket), keys)
        return dict(zip(keys, results, strict=True))


def image_key(image: LegacyImage, media_type: str) -> str:
    """Return ``<group>/<image_id>.<ext>``, the key relative to the image store prefix."""
    return f"{image.group}/{image.image_id}.{_EXTENSIONS[media_type]}"


def _mask_dest_key(image: LegacyImage) -> str:
    """Return ``<group>/<image_id>_mask.png``, the mask key relative to the image store prefix."""
    return f"{image.group}/{image.image_id}_mask.png"


def copy_items(
    images: Iterable[LegacyImage],
    sniffed: Mapping[str, tuple[str, int, int]],
    *,
    legacy: LegacyDataset,
    dataset: str,
) -> list[CopyItem]:
    """Plan one copy per image, plus one per mask, from the legacy prefix into the image store."""
    items = []
    for image in images:
        items.append(
            CopyItem(
                source_bucket=legacy.bucket,
                source_key=legacy.prefix + image.key,
                dest_key=image_prefix(dataset) + image_key(image, sniffed[image.key][0]),
            )
        )
        if image.mask_key is not None:
            items.append(
                CopyItem(
                    source_bucket=legacy.bucket,
                    source_key=legacy.prefix + image.mask_key,
                    dest_key=image_prefix(dataset) + _mask_dest_key(image),
                )
            )
    return items


def _mask_columns(
    image: LegacyImage, copied: Mapping[str, CopiedObject], *, dataset: str
) -> dict[str, Any]:
    """Return the ``mask_*`` columns of ``image``'s row, all None when it has no mask."""
    if image.mask_key is None:
        return {"mask_key": None, "mask_etag": None, "mask_size": None, "mask_crc64nvme": None}
    key = _mask_dest_key(image)
    result = copied[image_prefix(dataset) + key]
    return {
        "mask_key": key,
        "mask_etag": result.etag,
        "mask_size": result.size,
        "mask_crc64nvme": result.crc64nvme,
    }


def image_rows(
    images: Iterable[LegacyImage],
    sniffed: Mapping[str, tuple[str, int, int]],
    copied: Mapping[str, CopiedObject],
    *,
    dataset: str,
) -> pa.Table:
    """Build ``images.parquet`` from the legacy images, their header sizes and copy results.

    ``copied`` is keyed by full destination key; the legacy ETag is the ``source_fingerprint``.
    ``attrs`` is written as a JSON object with sorted keys.
    """
    rows = []
    for image in images:
        media_type, width, height = sniffed[image.key]
        key = image_key(image, media_type)
        result = copied[image_prefix(dataset) + key]
        rows.append(
            {
                "image_id": image.image_id,
                "key": key,
                "etag": result.etag,
                "size": result.size,
                "crc64nvme": result.crc64nvme,
                "media_type": media_type,
                "width": width,
                "height": height,
                "group": image.group,
                **_mask_columns(image, copied, dataset=dataset),
                "source_fingerprint": image.etag,
                "source_name": PurePosixPath(image.key).name,
                "attrs": None if image.attrs is None else json.dumps(image.attrs, sort_keys=True),
            }
        )
    return pa.Table.from_pylist(rows, schema=IMAGES_SCHEMA)


def sniff_legacy_images(
    source_client: Any, images: Sequence[LegacyImage], *, legacy: LegacyDataset, workers: int
) -> dict[str, tuple[str, int, int]]:
    """Return ``(media_type, width, height)`` of each legacy image, keyed by its relative key."""
    keys = [legacy.prefix + image.key for image in images]
    by_full_key = sniff_images(source_client, legacy.bucket, keys, workers=workers)
    return {image.key: by_full_key[legacy.prefix + image.key] for image in images}


def stage_images(
    source_client: Any,
    dest: Any,
    images: Sequence[LegacyImage],
    sniffed: Mapping[str, tuple[str, int, int]],
    *,
    legacy: LegacyDataset,
    dest_bucket: str,
    dataset: str,
    workers: int,
) -> pa.Table:
    """Copy each sniffed legacy image and mask into the image store; return ``images.parquet``."""
    items = copy_items(images, sniffed, legacy=legacy, dataset=dataset)
    copied = copy_images(
        dest, items, dest_bucket=dest_bucket, workers=workers, source_client=source_client
    )
    return image_rows(images, sniffed, copied, dataset=dataset)


def build_command(dataset: str, args: argparse.Namespace) -> str:
    """Reconstruct ``segdata build <dataset>`` from the options that shape the version."""
    parts = ["segdata", "build", dataset, "--store", str(args.store)]
    parts += ["--taxonomy-version", args.taxonomy_version]
    parts += ["--taxonomy-manifest-sha256", args.taxonomy_manifest_sha256]
    if args.legacy_bucket != DEFAULT_BUCKET:
        parts += ["--legacy-bucket", args.legacy_bucket]
    return shlex.join(parts)
