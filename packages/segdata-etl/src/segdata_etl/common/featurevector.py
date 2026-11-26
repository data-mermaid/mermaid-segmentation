"""Read pyspacer ``.featurevector`` files and check that they cover a dataset version's points.

A ``.featurevector`` is an npz holding ``meta`` (int64 [3]), ``rows`` and ``cols`` (uint16 [n])
and ``feat`` (float64 [n, dim]); the version's ``points.parquet`` ``row`` and ``col`` compare
directly with ``rows`` and ``cols``.
"""

from __future__ import annotations

import io
import json
import zipfile
from collections.abc import Collection, Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pyarrow as pa

from segdata_etl.common.copy_plan import read_full
from segdata_etl.common.s3_objects import crc64nvme_b64, md5_hex
from segdata_spec.features import EXTRACTOR_IDENTITY_KEYS

Point = tuple[int, int]
# Image ids named in a refusal message, ahead of the count.
NAMED_IDS = 10
# Feature vectors are small, so they keep the S3 default class.
FEATURE_STORAGE_CLASS = "STANDARD"


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
    return _vector_contents(data)[0]


def _vector_contents(data: bytes) -> tuple[set[Point], tuple[int, ...]]:
    with np.load(io.BytesIO(data), allow_pickle=False) as npz:
        rows, cols = npz["rows"].tolist(), npz["cols"].tolist()
        shape = npz["feat"].shape
    return set(zip(rows, cols, strict=True)), shape


def vector_problem(
    data: bytes, points: Collection[Point], feature_dim: int
) -> tuple[int, str | None]:
    """Return the number of distinct ``points`` and why the vector is refused, or None.

    A vector is refused when it is not a readable npz with ``rows``, ``cols`` and ``feat``, when
    ``feat`` is not ``[n, feature_dim]``, or when it lacks any of ``points``.
    """
    wanted = set(points)
    try:
        held, shape = _vector_contents(data)
    except (ValueError, zipfile.BadZipFile, KeyError, OSError) as exc:
        return len(wanted), f"unreadable: {exc}"
    if len(shape) != 2 or shape[1] != feature_dim:
        width = shape[1] if len(shape) == 2 else shape
        return len(wanted), f"feature width {width}, expected {feature_dim}"
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


def extractor_spec_bytes(spec: Mapping[str, Any]) -> bytes:
    """Serialise an extractor spec as indented, key-sorted JSON."""
    return (json.dumps(dict(spec), indent=2, sort_keys=True) + "\n").encode()
