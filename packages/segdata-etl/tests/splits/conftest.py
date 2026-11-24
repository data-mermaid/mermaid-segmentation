"""Shared fixtures for the split builder and leakage tests.

``FixtureCache`` publishes tiny dataset and taxonomy versions through ``segdata_spec.publish`` into
an in-memory store and copies every published object into a local cache laid out as ``sync``
writes it, so ``sync`` then finds every table cached and never calls S3.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import pyarrow as pa
import pytest
from botocore.exceptions import ClientError

from segdata_spec.manifest import ImageStore, ParentRef, Pin, SourceInfo
from segdata_spec.publish import GitState, publish_dataset, publish_taxonomy

BUCKET = "seg-bucket"
CLEAN_GIT = GitState(sha="abcdef1234567", dirty=False, pushed=True)
STORE = ImageStore(bucket="img-bucket", prefix="images/", owner="segdata")
SOURCE = SourceInfo(type="test", uri="s3://src/", revision="r1")


class _Body:
    def __init__(self, data: bytes):
        self._data = data

    def read(self) -> bytes:
        return self._data


@dataclass
class _MemoryS3:
    """The ``put_object`` / ``head_object`` / ``get_object`` surface ``publish`` touches."""

    objects: dict[tuple[str, str], bytes] = field(default_factory=dict)

    def put_object(self, *, Bucket: str, Key: str, Body: bytes, **_: Any) -> dict[str, Any]:
        self.objects[(Bucket, Key)] = bytes(Body)
        return {}

    def head_object(self, *, Bucket: str, Key: str) -> dict[str, Any]:
        if (Bucket, Key) not in self.objects:
            raise ClientError(
                {"Error": {"Code": "404"}, "ResponseMetadata": {"HTTPStatusCode": 404}},
                "HeadObject",
            )
        return {}

    def get_object(self, *, Bucket: str, Key: str, **_: Any) -> dict[str, Any]:
        return {"Body": _Body(self.objects[(Bucket, Key)])}


def images_table(rows: list[tuple[str, str, str | None, int]]) -> pa.Table:
    """Build ``images.parquet`` from ``(image_id, group, crc64nvme, size)`` rows."""
    n = len(rows)
    return pa.table(
        {
            "image_id": [r[0] for r in rows],
            "key": [f"{r[0]}.jpg" for r in rows],
            "etag": ["0" * 32] * n,
            "size": [r[3] for r in rows],
            "crc64nvme": [r[2] for r in rows],
            "media_type": ["image/jpeg"] * n,
            "width": [4] * n,
            "height": [4] * n,
            "group": [r[1] for r in rows],
            "mask_key": pa.nulls(n, pa.string()),
            "mask_etag": pa.nulls(n, pa.string()),
            "mask_size": pa.nulls(n, pa.int64()),
            "mask_crc64nvme": pa.nulls(n, pa.string()),
            "source_fingerprint": ["fp"] * n,
            "source_name": pa.nulls(n, pa.string()),
            "attrs": pa.nulls(n, pa.string()),
        }
    )


def labels_table(rows: list[tuple[int, str | None]]) -> pa.Table:
    """Build ``labels.parquet`` from ``(label_id, mermaid_benthic_attribute)`` rows."""
    n = len(rows)
    return pa.table(
        {
            "label_id": [r[0] for r in rows],
            "name": [f"label{r[0]}" for r in rows],
            "mermaid_benthic_attribute_id": pa.nulls(n, pa.string()),
            "mermaid_benthic_attribute": pa.array([r[1] for r in rows], pa.string()),
            "mermaid_growth_form": pa.nulls(n, pa.string()),
            "mapping_source": ["curated"] * n,
        }
    )


def label_counts_table(rows: list[tuple[str, int, int]]) -> pa.Table:
    """Build ``stats/label_counts.parquet`` from ``(image_id, label_id, n)`` rows."""
    return pa.table(
        {
            "image_id": [r[0] for r in rows],
            "label_id": pa.array([r[1] for r in rows], pa.int32()),
            "n": [r[2] for r in rows],
        }
    )


def benthic_attributes_table(rows: list[tuple[str, str, str | None]]) -> pa.Table:
    """Build ``benthic_attributes.parquet`` from ``(id, name, parent_id)`` rows."""
    return pa.table(
        {
            "id": [r[0] for r in rows],
            "name": [r[1] for r in rows],
            "parent_id": pa.array([r[2] for r in rows], pa.string()),
        }
    )


@dataclass
class FixtureCache:
    """Publishes versions into one shared store and mirrors it into ``cache_dir`` under ``BUCKET``."""

    cache_dir: Path
    repo_dir: Path
    store: _MemoryS3 = field(default_factory=_MemoryS3)

    def _mirror(self) -> None:
        for (bucket, key), data in self.store.objects.items():
            path = self.cache_dir / bucket / key
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(data)

    def dataset(
        self,
        name: str,
        version: str,
        *,
        images: pa.Table,
        labels: pa.Table | None = None,
        label_counts: pa.Table | None = None,
        parents: tuple[ParentRef, ...] = (),
    ) -> Pin:
        _, sha = publish_dataset(
            self.store,
            bucket=BUCKET,
            dataset=name,
            version=version,
            kind="points",
            tables={
                "images.parquet": images,
                "points.parquet": pa.table(
                    {
                        "image_id": pa.array([], pa.string()),
                        "row": pa.array([], pa.int32()),
                        "col": pa.array([], pa.int32()),
                        "label_id": pa.array([], pa.int32()),
                        "annotator": pa.array([], pa.string()),
                    }
                ),
                "labels.parquet": labels if labels is not None else labels_table([]),
                "stats/label_counts.parquet": (
                    label_counts if label_counts is not None else label_counts_table([])
                ),
            },
            image_store=STORE,
            etl_package="segdata-etl",
            etl_command="test",
            source=SOURCE,
            parents=parents,
            repo_dir=self.repo_dir,
            git=CLEAN_GIT,
        )
        self._mirror()
        return Pin(name=name, version=version, manifest_sha256=sha)

    def taxonomy(self, version: str, benthic_attributes: pa.Table) -> Pin:
        _, sha = publish_taxonomy(
            self.store,
            bucket=BUCKET,
            name="mermaid-benthic",
            version=version,
            tables={
                "benthic_attributes.parquet": benthic_attributes,
                "growth_forms.parquet": pa.table({"id": ["9"], "name": ["Branching"]}),
            },
            etl_package="segdata-etl",
            etl_command="test",
            source=SOURCE,
            repo_dir=self.repo_dir,
            git=CLEAN_GIT,
        )
        self._mirror()
        return Pin(name="mermaid-benthic", version=version, manifest_sha256=sha)


@pytest.fixture
def fixture_cache(tmp_path: Path) -> FixtureCache:
    repo = tmp_path / "repo"
    repo.mkdir()
    (repo / "uv.lock").write_bytes(b"lock-bytes")
    return FixtureCache(cache_dir=tmp_path / "cache", repo_dir=repo)


def sha256_of(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()
