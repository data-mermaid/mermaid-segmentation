"""``segdata features attach``: publish existing feature vectors as a feature set of a version.

The vectors live outside the dataset bucket. Every source is HEADed, then read and checked to
cover each of its image's points, one ``images.parquet`` group at a time so only that group's
points are held as Python objects, before the first write; only then are they copied write-once
under ``features/<store>/<family>/`` and indexed by ``publish_feature_set``.
"""

from __future__ import annotations

import argparse
import json
import logging
import string
from collections.abc import Callable, Iterable, Mapping, Sequence
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any, TypeVar
from urllib.parse import urlparse

import pyarrow as pa
import pyarrow.compute as pc

from segdata_etl.common.cli_flags import _pin_flags
from segdata_etl.common.featurevector import (
    FEATURE_STORAGE_CLASS,
    NAMED_IDS,
    FeatureVectorsMissing,
    Point,
    VectorFacts,
    points_by_image,
    read_extractor_spec,
    read_vector,
    refusal,
    refuse_invalid,
    store_name,
)
from segdata_etl.common.s3_objects import copy_new, head_with_checksum
from segdata_etl.common.stage import DEFAULT_CACHE_DIR, refuse_unpublishable
from segdata_spec import DATASET_BUCKET
from segdata_spec._s3 import feature_key, feature_store_prefix
from segdata_spec.features import publish_feature_set
from segdata_spec.manifest import Pin
from segdata_spec.publish import GitState, probe_git
from segdata_spec.read import open_tables
from segdata_spec.schema import FEATURE_INDEX_SCHEMA
from segdata_spec.sync import sync

logger = logging.getLogger(__name__)

# The repo root, which holds ``uv.lock`` and the git checkout that attaches the features.
REPO_ROOT = Path(__file__).resolve().parents[4]

_Item = TypeVar("_Item")
_Result = TypeVar("_Result")
Source = tuple[str, str]


def add_arguments(parser: argparse.ArgumentParser) -> None:
    """Add the ``features attach`` flags."""
    parser.add_argument("dataset", help="Published dataset name")
    _pin_flags(parser, "", "the published dataset")
    parser.add_argument("--family", required=True, help="Feature family, a lowercase key segment")
    parser.add_argument(
        "--source-template",
        required=True,
        help="s3://bucket/key template; fields are image_id, key, group and the row's attrs keys",
    )
    parser.add_argument(
        "--extractor-spec",
        type=Path,
        required=True,
        help="JSON file: class_path, crop_size, feature_dim, weights_uri, weights_sha256",
    )
    parser.add_argument(
        "--allow-missing",
        action="store_true",
        help="Leave images whose source vector is absent out of the feature set",
    )
    parser.add_argument("--workers", type=int, default=16)
    parser.add_argument(
        "--cache-dir",
        type=Path,
        default=DEFAULT_CACHE_DIR,
        help="Local cache for the synced version tables (default: %(default)s)",
    )
    parser.add_argument("-v", "--verbose", action="store_true")


def _split_s3_uri(uri: str) -> Source:
    parsed = urlparse(uri)
    if parsed.scheme != "s3" or not parsed.netloc or not parsed.path.strip("/"):
        raise ValueError(f"not an s3:// object URI: {uri!r}")
    return parsed.netloc, parsed.path.lstrip("/")


def template_fields(row: Mapping[str, Any]) -> dict[str, Any]:
    """Return the template fields of an ``images.parquet`` row: its attrs, then image_id, key, group."""
    attrs = json.loads(row["attrs"]) if row["attrs"] else {}
    return {**attrs, "image_id": row["image_id"], "key": row["key"], "group": row["group"]}


def _template_names(template: str) -> set[str]:
    names = set()
    for _, name, _, _ in string.Formatter().parse(template):
        if name is not None:
            names.add(name)
    return names


def _refuse_shared_sources(sources: Mapping[str, Source]) -> None:
    """Refuse a template that resolves two images to the same source object."""
    images_of: dict[Source, list[str]] = {}
    for image_id, source in sources.items():
        images_of.setdefault(source, []).append(image_id)
    shared = sorted(sorted(ids) for ids in images_of.values() if len(ids) > 1)
    if shared:
        raise ValueError(
            f"source template resolves {len(shared)} source(s) for more than one image, e.g. "
            f"the same source for images {shared[0][:NAMED_IDS]}"
        )


def resolve_sources(rows: Iterable[Mapping[str, Any]], template: str) -> dict[str, Source]:
    """Map each image id to the ``(bucket, key)`` the template names for its row.

    Raises:
        ValueError: The template has a positional or dotted field, names a field a row lacks,
            does not yield an ``s3://bucket/key`` URI, or yields the same source for two images.
    """
    names = _template_names(template)
    if any(not name.isidentifier() for name in names):
        raise ValueError(f"source template fields must be plain names: {sorted(names)}")
    sources: dict[str, Source] = {}
    for row in rows:
        fields = template_fields(row)
        unknown = sorted(names - set(fields))
        if unknown:
            raise ValueError(
                f"source template field(s) {unknown} are not in image {row['image_id']!r}; "
                f"known: {sorted(fields)}"
            )
        sources[row["image_id"]] = _split_s3_uri(template.format_map(fields))
    _refuse_shared_sources(sources)
    return sources


def _pool_map(
    fn: Callable[[_Item], _Result], items: Sequence[_Item], workers: int
) -> list[_Result]:
    with ThreadPoolExecutor(max_workers=workers) as pool:
        return list(pool.map(fn, items))


def probe_sources(
    client: Any, sources: Mapping[str, Source], workers: int
) -> tuple[dict[str, str], list[str]]:
    """HEAD every source; return the present ones' ETags by image id and the absent image ids."""
    ids = sorted(sources)
    heads = _pool_map(lambda i: head_with_checksum(client, *sources[i]), ids, workers)
    present = {i: h["ETag"] for i, h in zip(ids, heads, strict=True) if h is not None}
    return present, [i for i, h in zip(ids, heads, strict=True) if h is None]


def points_by_group(points: pa.Table, images: pa.Table) -> dict[str, pa.Table]:
    """Split ``points.parquet`` by the ``group`` of each point's image, as slices of one sorted table."""
    position = pc.index_in(points.column("image_id"), value_set=images.column("image_id"))
    ordered = points.append_column("group", pc.take(images.column("group"), position)).sort_by(
        "group"
    )
    slices: dict[str, pa.Table] = {}
    start = 0
    # value_counts lists values in first-seen order, which on the sorted column is sorted order.
    for entry in pc.value_counts(ordered.column("group")).to_pylist():
        slices[entry["values"]] = ordered.slice(start, entry["counts"])
        start += entry["counts"]
    return slices


def _read_sources(
    client: Any,
    sources: Mapping[str, Source],
    etags: Mapping[str, str],
    ids: Sequence[str],
    points: Mapping[str, Sequence[Point]],
    feature_dim: int,
    workers: int,
) -> dict[str, VectorFacts]:
    facts = _pool_map(
        lambda i: read_vector(client, *sources[i], etags[i], points.get(i, ()), feature_dim),
        ids,
        workers,
    )
    return dict(zip(ids, facts, strict=True))


def verify_sources(
    client: Any,
    sources: Mapping[str, Source],
    etags: Mapping[str, str],
    images: pa.Table,
    points: pa.Table,
    feature_dim: int,
    workers: int,
) -> dict[str, VectorFacts]:
    """Read every present source, group by group, and check it is usable.

    A vector is usable when it is readable, ``feature_dim`` wide and covers its image's points.

    Raises:
        FeatureVectorsInvalid: A vector is refused; raised once, after every group.
        SourceChanged: A source no longer matches its HEAD ETag.
    """
    grouped = points_by_group(points, images)
    ids_by_group: dict[str, list[str]] = {}
    for image_id, group in zip(
        *(images.column(c).to_pylist() for c in ("image_id", "group")), strict=True
    ):
        if image_id in etags:
            ids_by_group.setdefault(group, []).append(image_id)
    by_id: dict[str, VectorFacts] = {}
    for group, ids in sorted(ids_by_group.items()):
        lookup = points_by_image(grouped[group]) if group in grouped else {}
        by_id.update(
            _read_sources(client, sources, etags, sorted(ids), lookup, feature_dim, workers)
        )
    refuse_invalid(by_id, {i: f"s3://{sources[i][0]}/{sources[i][1]}" for i in by_id})
    return by_id


def _copy_feature(
    client: Any, source: Source, src_etag: str, dst_key: str, facts: VectorFacts
) -> dict[str, Any]:
    result = copy_new(
        client,
        src_bucket=source[0],
        src_key=source[1],
        src_etag=src_etag,
        dst_bucket=DATASET_BUCKET,
        dst_key=dst_key,
        expect_etag=facts.etag,
        expect_size=facts.size,
        expect_crc64nvme=facts.crc64nvme,
        storage_class=FEATURE_STORAGE_CLASS,
    )
    return {
        "key": dst_key,
        "etag": result.etag.strip('"'),
        "size": result.size,
        "crc64nvme": result.crc64nvme,
        "n_points": facts.n_points,
    }


def copy_features(
    client: Any,
    sources: Mapping[str, Source],
    etags: Mapping[str, str],
    facts: Mapping[str, VectorFacts],
    store: str,
    family: str,
    workers: int,
) -> pa.Table:
    """Copy every verified vector to its feature key; return the feature index rows.

    Raises:
        ObjectConflict: A destination holds, or a copy landed with, other content.
        SourceChanged: A source no longer matches its HEAD ETag.
    """
    ids = sorted(facts)
    rows = _pool_map(
        lambda i: _copy_feature(
            client, sources[i], etags[i], feature_key(store, family, i), facts[i]
        ),
        ids,
        workers,
    )
    return pa.Table.from_pylist(
        [{"image_id": i, **row} for i, row in zip(ids, rows, strict=True)],
        schema=FEATURE_INDEX_SCHEMA,
    )


def attach(
    client: Any,
    pin: Pin,
    family: str,
    template: str,
    extractor: Mapping[str, Any],
    *,
    cache_dir: Path,
    workers: int,
    allow_missing: bool,
    git: GitState,
    etl_command: str,
) -> str:
    """Copy the existing vectors of a published version and publish them as feature set ``family``.

    Nothing is written to the dataset bucket until every source is found (or ``allow_missing``)
    and covers its image's points.

    Returns:
        The sha256 of the feature-set manifest.

    Raises:
        PinMismatch: The published manifest differs from ``pin``.
        ValueError: The version has no points, ``family`` is not a key segment, or the template
            does not resolve for a row; raised before any source call.
        FeatureVectorsMissing: Sources are absent and ``allow_missing`` is false.
        FeatureVectorsInvalid: A vector is unreadable, of another width than the extractor's
            ``feature_dim``, or lacks one of its image's points.
    """
    sync(client, {pin.name: pin}, cache_dir, bucket=DATASET_BUCKET, images=False)
    tables = open_tables(cache_dir, DATASET_BUCKET, pin)
    if tables.points is None:
        raise ValueError(f"{pin.name} {pin.version} has no points.parquet to check coverage on")
    store = store_name(tables.manifest.image_store.prefix)
    feature_store_prefix(store, family)
    sources = resolve_sources(tables.images.to_pylist(), template)
    etags, missing = probe_sources(client, sources, workers)
    if missing and not allow_missing:
        raise FeatureVectorsMissing(refusal("have no source feature vector", missing))
    facts = verify_sources(
        client, sources, etags, tables.images, tables.points, extractor["feature_dim"], workers
    )
    index = copy_features(client, sources, etags, facts, store, family, workers)
    sha = publish_feature_set(
        client,
        pin,
        family,
        index,
        extractor,
        bucket=DATASET_BUCKET,
        etl_package="segdata-etl",
        etl_command=etl_command,
        repo_dir=REPO_ROOT,
        git=git,
        allow_missing=allow_missing,
    )
    logger.info(
        "attached %s to %s %s (set manifest sha256 %s): %d vectors, %d missing",
        family,
        pin.name,
        pin.version,
        sha,
        index.num_rows,
        len(missing),
    )
    return sha


def _command_line(args: argparse.Namespace) -> str:
    parts = [
        f"segdata features attach {args.dataset}",
        f"--version {args.version}",
        f"--manifest-sha256 {args.manifest_sha256}",
        f"--family {args.family}",
        f"--source-template {args.source_template}",
        f"--extractor-spec {args.extractor_spec}",
    ]
    if args.allow_missing:
        parts.append("--allow-missing")
    return " ".join(parts)


def run(args: argparse.Namespace, client: Any) -> int:
    """Run ``segdata features attach`` from parsed arguments.

    Raises:
        PublishRefused: The git tree is dirty or unpushed; raised before any S3 call.
    """
    extractor = read_extractor_spec(args.extractor_spec)
    git = probe_git(REPO_ROOT)
    refuse_unpublishable(git)
    pin = Pin(name=args.dataset, version=args.version, manifest_sha256=args.manifest_sha256)
    attach(
        client,
        pin,
        args.family,
        args.source_template,
        extractor,
        cache_dir=args.cache_dir,
        workers=args.workers,
        allow_missing=args.allow_missing,
        git=git,
        etl_command=_command_line(args),
    )
    return 0
