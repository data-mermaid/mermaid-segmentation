"""Pydantic models for ``manifest.json`` (datasets) and the taxonomy manifest."""

from __future__ import annotations

import hashlib
from datetime import datetime
from typing import Annotated, Literal

from pydantic import BaseModel, ConfigDict, Field, StringConstraints

# YYYYMMDD-<etl git sha7>, with -N for a second build on the same day.
Version = Annotated[str, StringConstraints(pattern=r"^\d{8}-[0-9a-f]{7}(-\d+)?$")]


class _Model(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)


class Pin(_Model):
    """A data-config reference to one exact version of a dataset or the taxonomy."""

    name: str
    version: Version
    manifest_sha256: str


class ParentRef(_Model):
    """The version a dataset version was built from."""

    dataset: str
    version: Version
    manifest_sha256: str


# Empty, or ending in "/" so ``prefix + key`` concatenates into a valid S3 key.
KeyPrefix = Annotated[str, StringConstraints(pattern=r"^(.*/)?$")]


class ImageStore(_Model):
    """Where a version's image and mask files live."""

    bucket: str
    prefix: KeyPrefix
    owner: Literal["segdata", "external"]


class EtlInfo(_Model):
    """The code state that produced a version."""

    package: str
    command: str
    git_sha: str
    git_dirty: bool
    uv_lock_sha256: str


class SourceInfo(_Model):
    """Where the data came from."""

    type: str
    uri: str
    revision: str
    citation: str | None = None
    license: str | None = None


class LabelMapping(_Model):
    """Provenance of the native-label to MERMAID mapping in ``labels.parquet``."""

    source: Literal["mermaid_api", "curated", "identity", "cleaned_up_rerun_json"]
    sha256: str
    n_labels: int
    n_mapped: int
    fetched_at: datetime | None = None
    git_ref: str | None = None
    path: str | None = None
    blob: str | None = None
    n_unresolved: int | None = None


class FileEntry(_Model):
    """Checksum, size and row count of one published table."""

    sha256: str
    bytes: int
    rows: int


class ExtraFile(_Model):
    """Checksum and size of one non-table file published beside the tables."""

    sha256: str
    size: int


class Stats(_Model):
    """Counts over a version's tables, with every dropped row category."""

    n_images: int
    n_points: int
    n_groups: int
    image_bytes: int
    dropped: dict[str, int]


class DiffFromParent(_Model):
    """Change against the first parent version."""

    images_added: int
    images_removed: int
    images_changed: int
    points_added: int
    points_removed: int


class Manifest(_Model):
    """``manifest.json`` of a dataset version; its sha256 is the version's pin."""

    schema_version: Literal[1]
    dataset: str
    version: Version
    kind: Literal["points", "dense"]
    created_at: datetime
    image_store: ImageStore
    etl: EtlInfo
    source: SourceInfo
    parents: tuple[ParentRef, ...] = ()
    label_mapping: LabelMapping | None = None
    files: dict[str, FileEntry]
    extra_files: dict[str, ExtraFile] = Field(default_factory=dict)
    stats: Stats
    diff_from_parent: DiffFromParent | None = None


class TaxonomyManifest(_Model):
    """``manifest.json`` of a taxonomy version."""

    schema_version: Literal[1]
    name: str
    version: Version
    created_at: datetime
    etl: EtlInfo
    source: SourceInfo
    files: dict[str, FileEntry]


def dump_manifest(manifest: Manifest | TaxonomyManifest) -> bytes:
    """Serialise to JSON in field order, indented, with a trailing newline.

    The same model always yields the same bytes, so the sha256 of the published bytes is stable.
    """
    return (manifest.model_dump_json(indent=2) + "\n").encode("utf-8")


def manifest_sha256(data: bytes) -> str:
    """Return the hex sha256 of the exact manifest bytes."""
    return hashlib.sha256(data).hexdigest()
