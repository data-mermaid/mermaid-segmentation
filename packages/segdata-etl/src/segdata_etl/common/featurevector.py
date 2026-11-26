"""Gather pyspacer ``.featurevector`` files for a dataset version and stage them as a feature set.

A ``.featurevector`` is an npz holding ``meta`` (int64 [3]), ``rows`` and ``cols`` (uint16 [n])
and ``feat`` (float64 [n, dim]); the version's ``points.parquet`` ``row`` and ``col`` compare
directly with ``rows`` and ``cols``.

``gather_features`` decides each image's vector: a file already in the family is adopted, and any
other image's vector comes from a source template outside the dataset bucket. Every vector is
HEADed, then read and checked to cover each of its image's points, one ``images.parquet`` group at
a time so only that group's points are held as Python objects. Nothing is written to S3;
``stage_features`` stages the index, the set manifest and the ``copies.jsonl`` that ``segdata
promote`` runs.
"""

from __future__ import annotations

import dataclasses
import io
import json
import string
import zipfile
from collections.abc import Callable, Collection, Iterable, Mapping, Sequence
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path
from typing import Any, NamedTuple, TypeVar
from urllib.parse import urlparse

import numpy as np
import pyarrow as pa
import pyarrow.compute as pc

from segdata_etl.common.copy_plan import read_full, write_copies
from segdata_etl.common.s3_objects import head_with_checksum, md5_hex
from segdata_spec import DATASET_BUCKET
from segdata_spec.features import EXTRACTOR_IDENTITY_KEYS, feature_file_key
from segdata_spec.git import GitState
from segdata_spec.layout import feature_set_prefix
from segdata_spec.manifest import FeatureSetPin, ImageStore, Pin
from segdata_spec.s3 import crc64nvme_b64, norm_etag
from segdata_spec.schema import FEATURE_INDEX_SCHEMA
from segdata_spec.stage import stage_feature_set

Point = tuple[int, int]
# Image ids named in a refusal message, ahead of the count.
NAMED_IDS = 10
# Feature vectors are small, so they keep the S3 default class.
FEATURE_STORAGE_CLASS = "STANDARD"

_Item = TypeVar("_Item")
_Result = TypeVar("_Result")
Source = tuple[str, str]


class FeatureVectorsMissing(Exception):
    """Feature vectors are absent at their sources and missing ones are not allowed."""


class FeatureVectorsInvalid(Exception):
    """Feature vectors are unreadable, of the wrong width, or lack points of their images."""


@dataclass(frozen=True)
class VectorFacts:
    """A source vector's bytes as a single-part copy will carry them, and why it is refused.

    ``n_points`` is the number of distinct points of the image; ``problem`` is None for a vector
    that is readable, of the expected width and covers all of them.
    """

    etag: str
    size: int
    crc64nvme: str
    n_points: int
    problem: str | None


def vector_points(data: bytes) -> set[Point]:
    """Return the ``(row, col)`` points a ``.featurevector`` holds features for."""
    with np.load(io.BytesIO(data), allow_pickle=False) as npz:
        return set(zip(npz["rows"].tolist(), npz["cols"].tolist(), strict=True))


def _array_problem(
    rows: np.ndarray, cols: np.ndarray, feat: np.ndarray, feature_dim: int
) -> str | None:
    """Describe how the npz arrays break the ``.featurevector`` layout, or return None."""
    if not (np.issubdtype(rows.dtype, np.integer) and np.issubdtype(cols.dtype, np.integer)):
        return f"rows and cols are {rows.dtype} and {cols.dtype}, not integer"
    if not np.issubdtype(feat.dtype, np.floating):
        return f"feat is {feat.dtype}, not float"
    if feat.ndim != 2 or feat.shape[1] != feature_dim:
        width = feat.shape[1] if feat.ndim == 2 else feat.shape
        return f"feature width {width}, expected {feature_dim}"
    if not feat.shape[0] == len(rows) == len(cols):
        return f"feat has {feat.shape[0]} rows for {len(rows)} point(s)"
    return None


def vector_problem(
    data: bytes, points: Collection[Point], feature_dim: int
) -> tuple[int, str | None]:
    """Return the number of distinct ``points`` and why the vector is refused, or None.

    A vector is refused when it is not a readable npz with ``rows``, ``cols`` and ``feat``; when
    ``rows`` and ``cols`` are not integer or ``feat`` is not float; when ``feat`` is not
    ``[len(rows), feature_dim]``; or when it lacks any of ``points``.
    """
    wanted = set(points)
    try:
        with np.load(io.BytesIO(data), allow_pickle=False) as npz:
            rows, cols, feat = npz["rows"], npz["cols"], npz["feat"]
    except (ValueError, zipfile.BadZipFile, KeyError, OSError) as exc:
        return len(wanted), f"unreadable: {exc}"
    problem = _array_problem(rows, cols, feat, feature_dim)
    if problem is not None:
        return len(wanted), problem
    held = set(zip(rows.tolist(), cols.tolist(), strict=True))
    lacking = len(wanted - held)
    if lacking:
        return len(wanted), f"lacks {lacking} of {len(wanted)} point(s)"
    return len(wanted), None


def read_vector(
    client: Any, bucket: str, key: str, etag: str, points: Collection[Point], feature_dim: int
) -> VectorFacts:
    """GET the vector guarded by ``etag``; return the copy's ETag, size, CRC64NVME and verdict.

    Raises:
        SourceChanged: The object no longer matches ``etag``.
    """
    data = read_full(client, bucket, key, etag)
    n_points, problem = vector_problem(data, points, feature_dim)
    return VectorFacts(md5_hex(data), len(data), crc64nvme_b64(data), n_points, problem)


def refuse_invalid(facts: Mapping[str, VectorFacts], keys: Mapping[str, str]) -> None:
    """Raise once for every refused vector, naming the first ones by key with their reasons.

    Raises:
        FeatureVectorsInvalid: A vector has a ``problem``.
    """
    refused = sorted(i for i, f in facts.items() if f.problem is not None)
    if refused:
        named = [f"{keys[i]}: {facts[i].problem}" for i in refused[:NAMED_IDS]]
        raise FeatureVectorsInvalid(f"{len(refused)} feature vector(s) refused: {named}")


def points_by_image(points: pa.Table) -> dict[str, list[Point]]:
    """Group the ``(row, col)`` pairs of a ``points.parquet`` table by image id."""
    grouped: dict[str, list[Point]] = {}
    columns = (points.column(name).to_pylist() for name in ("image_id", "row", "col"))
    for image_id, row, col in zip(*columns, strict=True):
        grouped.setdefault(image_id, []).append((row, col))
    return grouped


def refusal(what: str, image_ids: Collection[str]) -> str:
    """Describe ``image_ids`` as a count and the first ids in sorted order."""
    first = sorted(image_ids)[:NAMED_IDS]
    return f"{len(image_ids)} image(s) {what}: {first}"


def read_extractor_spec(path: Path) -> dict[str, Any]:
    """Read an extractor spec JSON file.

    Raises:
        ValueError: The spec is not a JSON object or lacks an identity key.
    """
    spec = json.loads(path.read_text())
    if not isinstance(spec, dict):
        raise ValueError(f"{path} does not hold a JSON object")
    missing = [key for key in EXTRACTOR_IDENTITY_KEYS if key not in spec]
    if missing:
        raise ValueError(f"{path} lacks extractor identity keys: {missing}")
    return spec


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


class Head(NamedTuple):
    """What a HEAD reports of an existing family file: its raw ETag, size and CRC64NVME."""

    etag: str
    size: int
    crc64nvme: str | None


def _destination_head(client: Any, key: str) -> Head | None:
    head = head_with_checksum(client, DATASET_BUCKET, key)
    if head is None:
        return None
    return Head(head["ETag"], head["ContentLength"], head.get("ChecksumCRC64NVME"))


def probe_destinations(
    client: Any, destinations: Mapping[str, str], workers: int
) -> dict[str, Head]:
    """HEAD every image's family file in the dataset bucket; return the HEADs of those that exist."""
    ids = sorted(destinations)
    heads = _pool_map(lambda i: _destination_head(client, destinations[i]), ids, workers)
    return {i: h for i, h in zip(ids, heads, strict=True) if h is not None}


def _metadata_problem(facts: VectorFacts, head: Head) -> str | None:
    """Describe how an adopted file's HEAD disagrees with its verified bytes, or return None."""
    if head.crc64nvme is None:
        return (
            "lacks a CRC64NVME checksum; writers must put feature files with "
            "ChecksumAlgorithm='CRC64NVME'"
        )
    held = (norm_etag(head.etag), head.size, head.crc64nvme)
    if held == (facts.etag, facts.size, facts.crc64nvme):
        return None
    return (
        f"object holds ETag {held[0]}, size {held[1]} and CRC64NVME {held[2]}, not the "
        f"{facts.etag}, {facts.size} and {facts.crc64nvme} of its bytes"
    )


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
    adopted: Mapping[str, Head],
) -> dict[str, VectorFacts]:
    """Read every present source, group by group, and check it is usable.

    A vector is usable when it is readable, ``feature_dim`` wide and covers its image's points;
    an ``adopted`` family file, keyed by image id to its HEAD, must also match its HEAD's ETag,
    size and CRC64NVME, as a copy must.

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
    for image_id, head in adopted.items():
        facts = by_id[image_id]
        if facts.problem is None:
            by_id[image_id] = dataclasses.replace(facts, problem=_metadata_problem(facts, head))
    refuse_invalid(by_id, {i: f"s3://{sources[i][0]}/{sources[i][1]}" for i in by_id})
    return by_id


def family_destinations(
    image_store: ImageStore, family: str, image_ids: Iterable[str]
) -> dict[str, str]:
    """Map each image id to its family file key in the dataset bucket."""
    return {i: feature_file_key(image_store, DATASET_BUCKET, family, i) for i in image_ids}


def _copy_line(source: Source, src_etag: str, dst_key: str, facts: VectorFacts) -> dict[str, Any]:
    return {
        "src_bucket": source[0],
        "src_key": source[1],
        "src_etag": norm_etag(src_etag),
        "etag": facts.etag,
        "dst_bucket": DATASET_BUCKET,
        "dst_key": dst_key,
        "size": facts.size,
        "crc64nvme": facts.crc64nvme,
    }


def _index_row(image_id: str, key: str, facts: VectorFacts) -> dict[str, Any]:
    return {
        "image_id": image_id,
        "key": key,
        "etag": facts.etag,
        "size": facts.size,
        "crc64nvme": facts.crc64nvme,
        "n_points": facts.n_points,
    }


@dataclass(frozen=True)
class GatheredFeatures:
    """The ``copies.jsonl`` lines and feature index rows of a version, and the image counts."""

    copies: list[dict[str, Any]]
    index: list[dict[str, Any]]
    adopted: int
    missing: int


def gather_features(
    client: Any,
    images: pa.Table,
    points: pa.Table,
    template: str,
    *,
    image_store: ImageStore,
    family: str,
    feature_dim: int,
    workers: int,
    allow_missing: bool,
) -> GatheredFeatures:
    """Decide and verify each image's vector for feature set ``family``; write nothing.

    An image whose family file already exists in the dataset bucket adopts it, and its template
    source is never resolved. Every other image's vector is ``template`` filled from its
    ``images.parquet`` row, copied into the family by promote. Each vector, adopted or not, must be
    readable, ``feature_dim`` wide and cover its image's points; an adopted file must also carry the
    ETag, size and CRC64NVME of its bytes.

    Raises:
        ValueError: The image store or ``family`` is not a valid feature store, or the template
            does not resolve for a row that adopts no file.
        FeatureVectorsMissing: Sources are absent and ``allow_missing`` is false.
        FeatureVectorsInvalid: A vector is refused.
        SourceChanged: A source no longer matches its HEAD ETag.
    """
    rows = images.to_pylist()
    destinations = family_destinations(image_store, family, (r["image_id"] for r in rows))
    adopted = probe_destinations(client, destinations, workers)
    template_sources = resolve_sources([r for r in rows if r["image_id"] not in adopted], template)
    etags, missing = probe_sources(client, template_sources, workers)
    if missing and not allow_missing:
        raise FeatureVectorsMissing(refusal("have no source feature vector", missing))
    sources = {
        **{i: template_sources[i] for i in etags},
        **{i: (DATASET_BUCKET, destinations[i]) for i in adopted},
    }
    all_etags = {**etags, **{i: head.etag for i, head in adopted.items()}}
    facts = verify_sources(
        client, sources, all_etags, images, points, feature_dim, workers, adopted
    )
    return GatheredFeatures(
        copies=[
            _copy_line(template_sources[i], etags[i], destinations[i], facts[i])
            for i in sorted(etags)
        ],
        index=[_index_row(i, destinations[i], facts[i]) for i in sorted(facts)],
        adopted=len(adopted),
        missing=len(missing),
    )


def stage_features(
    stage_dir: Path,
    gathered: GatheredFeatures,
    *,
    dataset_dirs: Sequence[Path],
    dataset: Pin,
    family: str,
    extractor: Mapping[str, Any],
    allow_missing: bool,
    command: str,
    repo_dir: Path,
    git: GitState,
) -> FeatureSetPin:
    """Stage feature set ``family`` of ``dataset`` and the ``copies.jsonl`` of its vectors.

    Raises:
        PublishRefused: The tree is dirty or unpushed, or the set is already staged.
        ManifestNotFound: No directory in ``dataset_dirs`` holds the dataset manifest.
        FeatureIndexInvalid: The index leaves images out without ``allow_missing``.
    """
    pin = stage_feature_set(
        stage_dir,
        dataset_dirs=dataset_dirs,
        bucket=DATASET_BUCKET,
        dataset=dataset,
        family=family,
        index=pa.Table.from_pylist(gathered.index, schema=FEATURE_INDEX_SCHEMA),
        extractor=extractor,
        allow_missing=allow_missing,
        etl_package="segdata-etl",
        etl_command=command,
        repo_dir=repo_dir,
        git=git,
    )
    write_copies(
        stage_dir, feature_set_prefix(dataset.name, dataset.version, family), gathered.copies
    )
    return pin
