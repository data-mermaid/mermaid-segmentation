"""Shared fixtures for the segdata-spec tests.

Provides a hand-rolled in-memory ``FakeS3`` that implements the boto3 client surface the package
touches, including S3 conditional writes and reads, ETags and an ordered call log, plus helpers
that publish small dataset and taxonomy versions into it.
"""

from __future__ import annotations

import hashlib
from collections.abc import Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import pyarrow as pa
import pytest
from botocore.exceptions import ClientError

from segdata_spec.manifest import ImageStore, Manifest, ParentRef, Pin, SourceInfo
from segdata_spec.publish import GitState, publish_dataset, publish_taxonomy


@pytest.fixture(autouse=True)
def _fake_aws_credentials(monkeypatch: pytest.MonkeyPatch) -> None:
    """Give boto3 static credentials so it never queries the EC2 metadata endpoint."""
    monkeypatch.setenv("AWS_ACCESS_KEY_ID", "testing")
    monkeypatch.setenv("AWS_SECRET_ACCESS_KEY", "testing")
    monkeypatch.setenv("AWS_DEFAULT_REGION", "us-east-1")
    monkeypatch.setenv("AWS_EC2_METADATA_DISABLED", "true")
    monkeypatch.delenv("AWS_PROFILE", raising=False)


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


@pytest.fixture
def fake_s3() -> FakeS3:
    return FakeS3()


BUCKET = "seg-bucket"
IMG_BUCKET = "img-bucket"
IMG_PREFIX = "images/ds/"
VERSION = "20261120-abcdef1"
TAX_VERSION = "20261120-1234567"
CLEAN_GIT = GitState(sha="abcdef1234567", dirty=False, pushed=True)
STORE = ImageStore(bucket=IMG_BUCKET, prefix=IMG_PREFIX, owner="segdata")
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


def _images_table(
    images: dict[str, tuple[bytes, str]],
    etags: dict[str, str],
    masks: dict[str, bytes] | None,
    crcs: dict[str, str],
    mask_crcs: dict[str, str],
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
            "media_type": ["image/jpeg"] * n,
            "width": [4] * n,
            "height": [4] * n,
            "group": [images[i][1] for i in ids],
            **mask_cols,
            "mask_crc64nvme": [mask_crcs.get(i) for i in ids],
            "source_fingerprint": ["fp"] * n,
            "attrs": pa.nulls(n, pa.string()),
        }
    )


def _repo_dir(tmp_path: Path) -> Path:
    repo = tmp_path / "repo"
    repo.mkdir(exist_ok=True)
    (repo / "uv.lock").write_bytes(b"lock-bytes")
    return repo


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
    parents: Sequence[ParentRef] = (),
) -> tuple[Manifest, Pin]:
    """Store the images (and masks) and publish dataset ``ds``; clear the call log.

    ``points=None`` publishes a dense version; ``etags`` gives images explicit ETags;
    ``crcs`` and ``mask_crcs`` record base64 CRC64NVME values by image id.
    """
    etags = etags or {}
    for image_id, (data, _) in images.items():
        client.put(IMG_BUCKET, f"{IMG_PREFIX}{image_id}.jpg", data)
        if image_id in etags:
            client.set_etag(IMG_BUCKET, f"{IMG_PREFIX}{image_id}.jpg", etags[image_id])
    for image_id, data in (masks or {}).items():
        client.put(IMG_BUCKET, f"{IMG_PREFIX}{image_id}.png", data)
    tables = {
        "images.parquet": _images_table(images, etags, masks, crcs or {}, mask_crcs or {}),
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
    manifest, sha = publish_dataset(
        client,
        bucket=BUCKET,
        dataset="ds",
        version=version,
        kind="points" if points is not None else "dense",
        tables=tables,
        image_store=STORE,
        etl_package="segdata-etl",
        etl_command="segdata build ds",
        source=SOURCE,
        parents=parents,
        repo_dir=_repo_dir(tmp_path),
        git=CLEAN_GIT,
    )
    client.calls.clear()
    return manifest, Pin(name="ds", version=version, manifest_sha256=sha)


def publish_taxonomy_version(client: FakeS3, tmp_path: Path) -> Pin:
    """Publish taxonomy ``mermaid-benthic`` with a two-level hierarchy; clear the call log."""
    _, sha = publish_taxonomy(
        client,
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
    client.calls.clear()
    return Pin(name="mermaid-benthic", version=TAX_VERSION, manifest_sha256=sha)
