"""Shared helpers for the segdata-spec tests: an in-memory ``FakeS3`` and dataset/taxonomy publishers.

Imported explicitly by test modules. Fixtures live in ``conftest.py``.
"""

from __future__ import annotations

import hashlib
import io
from collections.abc import Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import pyarrow as pa
from botocore.exceptions import ClientError
from PIL import Image

from segdata_spec.features import publish_feature_set
from segdata_spec.git import GitState
from segdata_spec.manifest import (
    EtlInfo,
    FeatureSetPin,
    ImageStore,
    Manifest,
    ParentRef,
    Pin,
    SourceInfo,
)
from segdata_spec.stage import stage_dataset, stage_taxonomy


def _client_error(code: str, status: int, op: str) -> ClientError:
    return ClientError(
        {"Error": {"Code": code, "Message": code}, "ResponseMetadata": {"HTTPStatusCode": status}},
        op,
    )


class _Body:
    def __init__(self, data: bytes):
        self._data = data

    def read(self) -> bytes:
        return self._data


def _as_bytes(body: Any) -> bytes:
    if isinstance(body, str):
        return body.encode("utf-8")
    if isinstance(body, bytes | memoryview):
        return bytes(body)
    raw = body.read()
    return raw.encode("utf-8") if isinstance(raw, str) else raw


@dataclass
class FakeS3:
    """In-memory S3 client surface; ``objects`` maps ``(bucket, key)`` to bytes.

    ``calls`` logs every ``put_object`` and ``get_object`` as ``(op, bucket, key, kwargs)`` where
    ``kwargs`` holds the call's other keyword arguments (``IfNoneMatch``, ``IfMatch``, ...).
    ``body_overrides`` makes ``get_object`` return other bytes than the stored object, as a
    truncated transfer would.
    """

    objects: dict[tuple[str, str], bytes] = field(default_factory=dict)
    etag_overrides: dict[tuple[str, str], str] = field(default_factory=dict)
    calls: list[tuple[str, str, str, dict[str, Any]]] = field(default_factory=list)
    body_overrides: dict[tuple[str, str], bytes] = field(default_factory=dict)

    def put(self, bucket: str, key: str, data: bytes) -> None:
        """Seed an object without logging a call."""
        self.objects[(bucket, key)] = data
        self.etag_overrides.pop((bucket, key), None)

    def set_etag(self, bucket: str, key: str, etag: str) -> None:
        """Give an existing object an explicit ETag, e.g. a multipart-style ``abc-2``."""
        assert (bucket, key) in self.objects
        self.etag_overrides[(bucket, key)] = etag.strip('"')

    def _etag(self, bucket: str, key: str) -> str:
        bare = (
            self.etag_overrides.get((bucket, key))
            or hashlib.md5(  # noqa: S324
                self.objects[(bucket, key)]
            ).hexdigest()
        )
        return f'"{bare}"'

    def put_object(self, *, Bucket: str, Key: str, Body: Any, **kwargs: Any) -> dict[str, Any]:
        self.calls.append(("put", Bucket, Key, dict(kwargs)))
        if kwargs.get("IfNoneMatch") == "*" and (Bucket, Key) in self.objects:
            raise _client_error("PreconditionFailed", 412, "PutObject")
        self.put(Bucket, Key, _as_bytes(Body))
        return {"ETag": self._etag(Bucket, Key)}

    def head_object(self, *, Bucket: str, Key: str) -> dict[str, Any]:
        if (Bucket, Key) not in self.objects:
            raise _client_error("404", 404, "HeadObject")
        return {
            "ContentLength": len(self.objects[(Bucket, Key)]),
            "ETag": self._etag(Bucket, Key),
        }

    def get_object(self, *, Bucket: str, Key: str, **kwargs: Any) -> dict[str, Any]:
        self.calls.append(("get", Bucket, Key, dict(kwargs)))
        if (Bucket, Key) not in self.objects:
            raise _client_error("NoSuchKey", 404, "GetObject")
        etag = self._etag(Bucket, Key)
        if_match = kwargs.get("IfMatch")
        if if_match is not None and if_match.strip('"') != etag.strip('"'):
            raise _client_error("PreconditionFailed", 412, "GetObject")
        data = self.body_overrides.get((Bucket, Key), self.objects[(Bucket, Key)])
        return {"Body": _Body(data), "ContentLength": len(data), "ETag": etag}


BUCKET = "seg-bucket"
IMG_BUCKET = "img-bucket"
IMG_PREFIX = "images/ds/"
VERSION = "20261120-abcdef1"
TAX_VERSION = "20261120-1234567"
CLEAN_GIT = GitState(sha="abcdef1234567", dirty=False, pushed=True)
READY_ETL = EtlInfo(
    package="mermaid-classifier",
    command="extract-features",
    git_sha="0123456789abcdef",
    git_dirty=True,
    uv_lock_sha256="l" * 64,
)
STORE = ImageStore(bucket=IMG_BUCKET, prefix=IMG_PREFIX, owner="segdata")
# An image store inside BUCKET, as a feature set requires; its store name is ``ds``.
FEATURE_STORE = ImageStore(bucket=BUCKET, prefix=IMG_PREFIX, owner="segdata")
SOURCE = SourceInfo(type="test", uri="s3://src/", revision="r1")

# image_id -> (bytes, group); each image's key is "<image_id>.jpg" under IMG_PREFIX.
IMAGES: dict[str, tuple[bytes, str]] = {
    "a": (b"aaaa", "g1"),
    "b": (b"bbbbbbbb", "g1"),
    "c": (b"cc", "g2"),
}
POINTS: list[tuple[str, int, int, int]] = [("a", 1, 1, 5), ("b", 2, 2, 6)]


def _md5(data: bytes) -> str:
    return hashlib.md5(data).hexdigest()  # noqa: S324


def jpeg_bytes(width: int, height: int) -> bytes:
    """Return a real RGB JPEG of the given size, for tests that resize."""
    buf = io.BytesIO()
    Image.new("RGB", (width, height), (200, 100, 50)).save(buf, format="JPEG")
    return buf.getvalue()


def mask_bytes(width: int, height: int, left: int, right: int) -> bytes:
    """Return a mode-L PNG mask whose left half holds label ``left`` and right half ``right``."""
    mask = Image.new("L", (width, height), left)
    mask.paste(right, (width // 2, 0, width, height))
    buf = io.BytesIO()
    mask.save(buf, format="PNG")
    return buf.getvalue()


def _images_table(
    images: dict[str, tuple[bytes, str]],
    etags: dict[str, str],
    masks: dict[str, bytes] | None,
    crcs: dict[str, str],
    mask_crcs: dict[str, str],
    sizes: dict[str, tuple[int, int]],
    media_types: dict[str, str],
    source_names: dict[str, str],
) -> pa.Table:
    ids = list(images)
    n = len(ids)
    if masks is None:
        mask_cols: dict[str, Any] = {
            "mask_key": pa.nulls(n, pa.string()),
            "mask_etag": pa.nulls(n, pa.string()),
            "mask_size": pa.nulls(n, pa.int64()),
        }
    else:
        mask_cols = {
            "mask_key": [f"{i}.png" for i in ids],
            "mask_etag": [_md5(masks[i]) for i in ids],
            "mask_size": [len(masks[i]) for i in ids],
        }
    return pa.table(
        {
            "image_id": ids,
            "key": [f"{i}.jpg" for i in ids],
            "etag": [etags.get(i, _md5(images[i][0])) for i in ids],
            "size": [len(images[i][0]) for i in ids],
            "crc64nvme": [crcs.get(i) for i in ids],
            "media_type": [media_types.get(i, "image/jpeg") for i in ids],
            "width": [sizes.get(i, (4, 4))[0] for i in ids],
            "height": [sizes.get(i, (4, 4))[1] for i in ids],
            "group": [images[i][1] for i in ids],
            **mask_cols,
            "mask_crc64nvme": [mask_crcs.get(i) for i in ids],
            "source_fingerprint": ["fp"] * n,
            "source_name": pa.array([source_names.get(i) for i in ids], pa.string()),
            "attrs": pa.nulls(n, pa.string()),
        }
    )


def _repo_dir(tmp_path: Path) -> Path:
    repo = tmp_path / "repo"
    repo.mkdir(exist_ok=True)
    (repo / "uv.lock").write_bytes(b"lock-bytes")
    return repo


def _upload_staged(client: FakeS3, stage_dir: Path) -> None:
    """Seed ``client`` with every staged object, standing in for ``segdata promote``."""
    root = stage_dir / BUCKET
    for path in sorted(root.rglob("*")):
        if path.is_file():
            client.put(BUCKET, path.relative_to(root).as_posix(), path.read_bytes())


def publish_version(
    client: FakeS3,
    tmp_path: Path,
    *,
    version: str = VERSION,
    images: dict[str, tuple[bytes, str]] = IMAGES,
    points: list[tuple[str, int, int, int]] | None = POINTS,
    masks: dict[str, bytes] | None = None,
    etags: dict[str, str] | None = None,
    crcs: dict[str, str] | None = None,
    mask_crcs: dict[str, str] | None = None,
    sizes: dict[str, tuple[int, int]] | None = None,
    media_types: dict[str, str] | None = None,
    source_names: dict[str, str] | None = None,
    parents: Sequence[ParentRef] = (),
    image_store: ImageStore = STORE,
) -> tuple[Manifest, Pin]:
    """Store the images (and masks) and publish dataset ``ds``; clear the call log.

    ``points=None`` publishes a dense version; ``etags`` gives images explicit ETags;
    ``crcs`` and ``mask_crcs`` record base64 CRC64NVME values by image id; ``sizes`` records
    ``(width, height)`` by image id (default 4x4); ``media_types`` sets the media type by image id
    (default ``image/jpeg``); ``source_names`` sets ``source_name`` by image id (default null);
    ``image_store`` is where the images live (default: ``STORE``, outside ``BUCKET``).
    """
    etags = etags or {}
    for image_id, (data, _) in images.items():
        client.put(image_store.bucket, f"{image_store.prefix}{image_id}.jpg", data)
        if image_id in etags:
            client.set_etag(
                image_store.bucket, f"{image_store.prefix}{image_id}.jpg", etags[image_id]
            )
    for image_id, data in (masks or {}).items():
        client.put(image_store.bucket, f"{image_store.prefix}{image_id}.png", data)
    tables = {
        "images.parquet": _images_table(
            images,
            etags,
            masks,
            crcs or {},
            mask_crcs or {},
            sizes or {},
            media_types or {},
            source_names or {},
        ),
        "labels.parquet": pa.table(
            {
                "label_id": [5, 6],
                "name": ["x", "y"],
                "mermaid_benthic_attribute_id": pa.nulls(2, pa.string()),
                "mermaid_benthic_attribute": pa.nulls(2, pa.string()),
                "mermaid_growth_form": pa.nulls(2, pa.string()),
                "mapping_source": ["curated", "curated"],
            }
        ),
        "stats/label_counts.parquet": pa.table({"image_id": ["a"], "label_id": [5], "n": [1]}),
    }
    if points is not None:
        tables["points.parquet"] = pa.table(
            {
                "image_id": [p[0] for p in points],
                "row": [p[1] for p in points],
                "col": [p[2] for p in points],
                "label_id": [p[3] for p in points],
                "annotator": ["default"] * len(points),
            }
        )
    stage_dir = tmp_path / "fixture-stage"
    manifest, sha = stage_dataset(
        stage_dir,
        parent_dirs=(stage_dir,),
        bucket=BUCKET,
        dataset="ds",
        version=version,
        kind="points" if points is not None else "dense",
        tables=tables,
        image_store=image_store,
        etl_package="segdata-etl",
        etl_command="segdata build ds",
        source=SOURCE,
        parents=parents,
        repo_dir=_repo_dir(tmp_path),
        git=CLEAN_GIT,
    )
    _upload_staged(client, stage_dir)
    client.calls.clear()
    return manifest, Pin(name="ds", version=version, manifest_sha256=sha)


def publish_taxonomy_version(client: FakeS3, tmp_path: Path) -> Pin:
    """Publish taxonomy ``mermaid-benthic`` with a two-level hierarchy; clear the call log."""
    stage_dir = tmp_path / "fixture-stage"
    _, sha = stage_taxonomy(
        stage_dir,
        bucket=BUCKET,
        name="mermaid-benthic",
        version=TAX_VERSION,
        tables={
            "benthic_attributes.parquet": pa.table(
                {"id": ["1", "2"], "name": ["Hard coral", "Acropora"], "parent_id": [None, "1"]}
            ),
            "growth_forms.parquet": pa.table({"id": ["9"], "name": ["Branching"]}),
        },
        etl_package="segdata-etl",
        etl_command="segdata taxonomy",
        source=SOURCE,
        repo_dir=_repo_dir(tmp_path),
        git=CLEAN_GIT,
    )
    _upload_staged(client, stage_dir)
    client.calls.clear()
    return Pin(name="mermaid-benthic", version=TAX_VERSION, manifest_sha256=sha)


FEATURE_FAMILY = "effnet"
# image_id -> the bytes of its feature file under features/ds/effnet/.
FEATURES: dict[str, bytes] = {"a": b"feature-a", "b": b"feature-bb"}
EXTRACTOR: dict[str, Any] = {
    "class_path": "spacer.extract_features.EfficientNetExtractor",
    "crop_size": 224,
    "feature_dim": 1280,
    "weights_sha256": "w" * 64,
    "weights_uri": "s3://weights/a.pt",
}


def feature_key_of(image_id: str) -> str:
    return f"features/ds/{FEATURE_FAMILY}/{image_id}.featurevector"


def publish_feature_set_version(client: FakeS3, tmp_path: Path) -> FeatureSetPin:
    """Publish dataset ``ds`` with images a and b, store their features, publish the set.

    Returns the feature-set pin; clears the call log.
    """
    _, pin = publish_version(
        client,
        tmp_path,
        images={image_id: IMAGES[image_id] for image_id in FEATURES},
        points=POINTS,
        image_store=FEATURE_STORE,
    )
    for image_id, data in FEATURES.items():
        client.put(BUCKET, feature_key_of(image_id), data)
    index = pa.table(
        {
            "image_id": list(FEATURES),
            "key": [feature_key_of(i) for i in FEATURES],
            "etag": [_md5(data) for data in FEATURES.values()],
            "size": [len(data) for data in FEATURES.values()],
            "crc64nvme": pa.nulls(len(FEATURES), pa.string()),
            "n_points": [1] * len(FEATURES),
        }
    )
    feature_pin = publish_feature_set(
        client,
        pin,
        FEATURE_FAMILY,
        index,
        EXTRACTOR,
        bucket=BUCKET,
        etl=READY_ETL,
    )
    client.calls.clear()
    return feature_pin
