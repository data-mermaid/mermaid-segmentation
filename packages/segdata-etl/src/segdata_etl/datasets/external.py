"""Shared steps of the external dataset version builders (Catlin, Moorea, Pacific, Benthos).

Each builder module reads its legacy ``external_validation_datasets/<name>/`` prefix and calls
these helpers to build the uniform tables and stage a dataset version under ``--stage-dir``, with
``copies.jsonl`` (and, for masks, ``uploads.jsonl``) for ``segdata promote`` to run.
"""

from __future__ import annotations

import argparse
import json
import logging
import shlex
from collections.abc import Callable, Collection, Iterable, Mapping, Sequence
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import Any, Literal

import pandas as pd
import pyarrow as pa
import pyarrow.compute as pc

from segdata_etl.common import release
from segdata_etl.common.cli_flags import pin_command_parts, pin_flags
from segdata_etl.common.copy_plan import (
    ImageOutcome,
    ObjectFacts,
    StagedFile,
    copy_plan,
    probe_image,
    stage_bytes,
    write_copies,
    write_uploads,
)
from segdata_etl.common.labels import resolve_taxonomy_name, taxonomy_name_index
from segdata_etl.common.legacy import DEFAULT_BUCKET, LegacyDataset, LegacyObject
from segdata_etl.common.release import image_prefix
from segdata_etl.common.s3_objects import (
    EXTENSIONS,
    SourceChanged,
    head_with_checksum,
    md5_hex,
)
from segdata_etl.common.stage import DEFAULT_CACHE_DIR, next_version
from segdata_etl.label_maps.curated import CuratedSource
from segdata_etl.taxonomy import require_published_taxonomy
from segdata_spec import DATASET_BUCKET
from segdata_spec.git import GitState
from segdata_spec.manifest import LabelMapping, Pin, SourceInfo
from segdata_spec.read import Taxonomy, open_taxonomy
from segdata_spec.s3 import crc64nvme_b64, norm_etag
from segdata_spec.schema import IMAGES_SCHEMA, LABELS_SCHEMA, POINTS_SCHEMA
from segdata_spec.stage import stage_dataset
from segdata_spec.sync import sync

logger = logging.getLogger(__name__)

TAXONOMY_NAME = "mermaid-benthic"
_IMAGES_FOLDER = "images"
_PNG = "image/png"


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


@dataclass(frozen=True)
class LegacyProbe:
    """What probing the legacy images found.

    ``images`` holds the readable images in input order; ``rows`` their ``images.parquet`` rows by
    image id; ``sniffed`` their ``(media_type, width, height)`` by legacy key; ``unreadable`` the
    ids of the images whose header could not be read.
    """

    images: list[LegacyImage]
    rows: dict[str, dict[str, Any]]
    sniffed: dict[str, tuple[str, int, int]]
    unreadable: list[str]


def add_build_arguments(parser: argparse.ArgumentParser) -> None:
    """Add the taxonomy pin, ``--cache-dir`` and ``--legacy-bucket`` every external builder shares."""
    pin_flags(parser, "taxonomy-", "the mermaid-benthic taxonomy")
    parser.add_argument(
        "--cache-dir",
        type=Path,
        default=DEFAULT_CACHE_DIR,
        help="Where the taxonomy is synced (default: %(default)s)",
    )
    parser.add_argument(
        "--legacy-bucket", default=DEFAULT_BUCKET, help="Bucket holding the legacy prefix"
    )


def load_taxonomy(
    client: Any, bucket: str, *, version: str, manifest_sha256: str, cache_dir: Path
) -> Taxonomy:
    """Check the pinned ``mermaid-benthic`` taxonomy is published in ``bucket``, sync it and open it.

    Raises:
        PublishRefused: The pinned taxonomy manifest is not published in ``bucket``.
    """
    pin = Pin(name=TAXONOMY_NAME, version=version, manifest_sha256=manifest_sha256)
    require_published_taxonomy(client, bucket, pin)
    sync(client, {}, cache_dir, bucket=bucket, taxonomy=pin)
    return open_taxonomy(cache_dir, bucket, pin)


def build_labels(classes: Mapping[str, int], curated: pa.Table, attributes: pa.Table) -> pa.Table:
    """Build ``labels.parquet`` from ``classes.json``, the curated map and taxonomy attributes.

    Classes join the curated map on lowercase ``name``. Label 0 (``unlabeled``) is left out. A
    curated target found in the taxonomy, ignoring case, takes its canonical name and id; one not
    found keeps the curated value with a null id.

    Raises:
        AmbiguousTaxonomyName: A curated target matches more than one taxonomy name, ignoring case.
    """
    index = taxonomy_name_index(attributes)
    targets = {
        row["name"].lower(): row["mermaid_benthic_attribute"]
        for row in curated.to_pylist()
        if row["name"] is not None
    }
    rows = []
    for name, label_id in sorted(classes.items(), key=lambda item: item[1]):
        if label_id == 0:
            continue
        target = targets.get(name.lower())
        match = resolve_taxonomy_name(target, index) if target is not None else None
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


def label_mapping_record(labels: pa.Table, info: CuratedSource, *, git_sha: str) -> LabelMapping:
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


def image_key(image: LegacyImage, media_type: str) -> str:
    """Return ``<group>/<image_id>.<ext>``, the key relative to the image store prefix."""
    return f"{image.group}/{image.image_id}.{EXTENSIONS[media_type]}"


def _mask_dest_key(image: LegacyImage) -> str:
    """Return ``<group>/<image_id>_mask.png``, the mask key relative to the image store prefix."""
    return f"{image.group}/{image.image_id}_mask.png"


def _image_row(
    image: LegacyImage, facts: ObjectFacts, header: tuple[str, int, int]
) -> dict[str, Any]:
    """Build the maskless ``images.parquet`` row of ``image``; ``attrs`` as sorted-key JSON."""
    media_type, width, height = header
    return {
        "image_id": image.image_id,
        "key": image_key(image, media_type),
        "etag": facts.etag,
        "size": facts.size,
        "crc64nvme": facts.crc64nvme,
        "media_type": media_type,
        "width": width,
        "height": height,
        "group": image.group,
        "mask_key": None,
        "mask_etag": None,
        "mask_size": None,
        "mask_crc64nvme": None,
        "source_fingerprint": facts.source_etag,
        "source_name": PurePosixPath(image.key).name,
        "attrs": None if image.attrs is None else json.dumps(image.attrs, sort_keys=True),
    }


def _head_listed(client: Any, legacy: LegacyDataset, image: LegacyImage) -> Mapping[str, Any]:
    """HEAD the legacy image with its checksum.

    Raises:
        SourceChanged: The object is gone or its ETag differs from the listing.
    """
    key = legacy.prefix + image.key
    head = head_with_checksum(client, legacy.bucket, key)
    if head is None or norm_etag(head["ETag"]) != image.etag:
        raise SourceChanged(f"s3://{legacy.bucket}/{key} changed since it was listed")
    return head


def _probe_one(client: Any, legacy: LegacyDataset, image: LegacyImage) -> ImageOutcome:
    return probe_image(
        client,
        legacy.bucket,
        image.image_id,
        legacy.prefix + image.key,
        _head_listed(client, legacy, image),
        {},
        lambda facts, header: _image_row(image, facts, header),
    )


def _probe_result(images: Sequence[LegacyImage], outcomes: Sequence[ImageOutcome]) -> LegacyProbe:
    rows = {o.image_id: o.row for o in outcomes if o.row is not None}
    kept = [image for image in images if image.image_id in rows]
    sniffed = {
        image.key: (
            rows[image.image_id]["media_type"],
            rows[image.image_id]["width"],
            rows[image.image_id]["height"],
        )
        for image in kept
    }
    unreadable = [o.image_id for o in outcomes if o.row is None]
    return LegacyProbe(images=kept, rows=rows, sniffed=sniffed, unreadable=unreadable)


def probe_legacy_images(
    client: Any, images: Sequence[LegacyImage], *, legacy: LegacyDataset, workers: int
) -> LegacyProbe:
    """HEAD and sniff every legacy image on ``workers`` threads; an unreadable header drops it.

    Raises:
        SourceChanged: An image changed since the listing or while it was read.
    """
    with ThreadPoolExecutor(max_workers=workers) as pool:
        outcomes = list(pool.map(lambda image: _probe_one(client, legacy, image), images))
    return _probe_result(images, outcomes)


def drop_unreadable_points(points: pa.Table, probe: LegacyProbe) -> pa.Table:
    """Return the points whose image the probe kept, in order."""
    kept = pa.array(list(probe.rows), pa.string())
    return points.filter(pc.is_in(points.column("image_id"), value_set=kept))


def _with_unreadable(dropped: Mapping[str, int], probe: LegacyProbe) -> dict[str, int]:
    """Merge ``image_unreadable_header`` into ``dropped`` when the probe dropped an image."""
    if not probe.unreadable:
        return dict(dropped)
    return {**dropped, "image_unreadable_header": len(probe.unreadable)}


def _mask_columns(image: LegacyImage, mask: bytes) -> dict[str, Any]:
    """Return the ``mask_*`` columns of ``image``'s row for the mask bytes staged for it."""
    return {
        "mask_key": _mask_dest_key(image),
        "mask_etag": md5_hex(mask),
        "mask_size": len(mask),
        "mask_crc64nvme": crc64nvme_b64(mask),
    }


def _stage_masks(
    stage_dir: Path, dataset: str, images: Sequence[LegacyImage], masks: Mapping[str, bytes]
) -> list[StagedFile]:
    """Write each image's mask bytes into the stage; return the uploads that publish them.

    Raises:
        ObjectConflict: A staged mask file exists with other bytes.
    """
    files = []
    for image in images:
        mask = masks[image.image_id]
        key = image_prefix(dataset) + _mask_dest_key(image)
        stage_bytes(stage_dir, key, mask)
        files.append(StagedFile(key, len(mask), crc64nvme_b64(mask), _PNG))
    return files


def _image_rows(probe: LegacyProbe, masks: Mapping[str, bytes] | None) -> list[dict[str, Any]]:
    """Return the kept images' rows in probe order, with mask columns when ``masks`` is given."""
    rows = [probe.rows[image.image_id] for image in probe.images]
    if masks is None:
        return rows
    return [
        {**row, **_mask_columns(image, masks[image.image_id])}
        for image, row in zip(probe.images, rows, strict=True)
    ]


def _legacy_copy_plan(
    rows: Sequence[Mapping[str, Any]], probe: LegacyProbe, *, legacy: LegacyDataset, dataset: str
) -> list[dict[str, Any]]:
    """Plan the copy of each image row from its legacy object into the dataset's image prefix."""
    source_keys = {image.image_id: legacy.prefix + image.key for image in probe.images}
    return copy_plan(
        rows,
        src_bucket=legacy.bucket,
        src_key=lambda row: source_keys[row["image_id"]],
        dst_prefix=image_prefix(dataset),
    )


def build_command(dataset: str, args: argparse.Namespace) -> str:
    """Reconstruct ``segdata build <dataset>`` from the options that shape the version."""
    parts = ["segdata", "build", dataset, "--stage-dir", str(args.stage_dir)]
    parts += pin_command_parts(args, "taxonomy-")
    if args.legacy_bucket != DEFAULT_BUCKET:
        parts += ["--legacy-bucket", args.legacy_bucket]
    return shlex.join(parts)


def publish_version(
    *,
    dataset: str,
    kind: Literal["points", "dense"],
    tables: Mapping[str, pa.Table],
    source: SourceInfo,
    label_mapping: LabelMapping,
    dropped: Mapping[str, int],
    git: GitState,
    repo_root: Path,
    client: Any,
    probe: LegacyProbe,
    legacy: LegacyDataset,
    args: argparse.Namespace,
    masks: Mapping[str, bytes] | None = None,
) -> int:
    """Stage a new version and its plans under ``args.stage_dir``; print its version and sha256.

    ``tables`` holds the dataset-specific tables; ``images.parquet`` is built from the probe, and
    ``image_unreadable_header`` joins ``dropped`` when the probe dropped an image. Each image is
    planned as a copy from the legacy prefix in ``copies.jsonl``. ``masks`` holds each kept
    image's mask bytes by image id; they are staged and listed in ``uploads.jsonl``.

    Raises:
        PublishRefused: The tree is dirty or unpushed, or ``uv.lock`` is missing.
        ObjectConflict: A staged mask file exists with other bytes.
    """
    version = next_version(args.stage_dir, DATASET_BUCKET, dataset, git_sha=git.sha, client=client)
    rows = _image_rows(probe, masks)
    uploads: list[StagedFile] = []
    if masks is not None:
        uploads = _stage_masks(args.stage_dir, dataset, probe.images, masks)
    manifest, sha = stage_dataset(
        args.stage_dir,
        parent_dirs=(args.stage_dir,),
        bucket=DATASET_BUCKET,
        dataset=dataset,
        version=version,
        kind=kind,
        tables={"images.parquet": pa.Table.from_pylist(rows, schema=IMAGES_SCHEMA), **tables},
        image_store=release.image_store(DATASET_BUCKET, dataset),
        etl_package="segdata-etl",
        etl_command=build_command(dataset, args),
        source=source,
        label_mapping=label_mapping,
        dropped=_with_unreadable(dropped, probe),
        repo_dir=repo_root,
        git=git,
    )
    plan = _legacy_copy_plan(rows, probe, legacy=legacy, dataset=dataset)
    write_copies(args.stage_dir, dataset, version, plan)
    if uploads:
        write_uploads(args.stage_dir, dataset, version, uploads)
    logger.info(
        "staged %s %s: %d images, %d copies, %d uploads",
        dataset,
        manifest.version,
        manifest.stats.n_images,
        len(plan),
        len(uploads),
    )
    print(f"version: {manifest.version}\nmanifest_sha256: {sha}")
    return 0
