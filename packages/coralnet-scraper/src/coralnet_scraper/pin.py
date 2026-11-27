"""A pin to a published ``coralnet-raw`` dataset version, as the image ids it holds per source."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pyarrow as pa
import pyarrow.parquet as pq

from segdata_spec import DATASET_BUCKET
from segdata_spec.layout import cache_path, dataset_prefix
from segdata_spec.manifest import Pin
from segdata_spec.sync import sync_tables

_DATASET = "coralnet-raw"


@dataclass(frozen=True)
class RawPin:
    """A ``coralnet-raw`` version and the image ids it holds for each source.

    Attributes:
        version: Version of the published dataset.
        manifest_sha256: Digest of that version's manifest.
        ids: Image ids per CoralNet source id.
    """

    version: str
    manifest_sha256: str
    ids: Mapping[int, frozenset[int]]

    def ids_for(self, source_id: int) -> frozenset[int]:
        """Return the image ids the pin holds for ``source_id``; empty when it lacks the source."""
        return self.ids.get(source_id, frozenset())


def _group_ids(images: pa.Table) -> dict[int, frozenset[int]]:
    """Return the image ids per source id of a ``coralnet-raw`` images table.

    Raises:
        ValueError: If a ``group`` is not ``s<id>`` or an ``image_id`` is not an integer.
    """
    grouped: dict[int, set[int]] = {}
    for image_id, group in zip(
        images.column("image_id").to_pylist(), images.column("group").to_pylist(), strict=True
    ):
        if not (group.startswith("s") and group[1:].isdigit()):
            raise ValueError(f"unparseable group {group!r}; expected 's<source id>'")
        if not image_id.isdigit():
            raise ValueError(f"unparseable image_id {image_id!r}; expected an integer")
        grouped.setdefault(int(group[1:]), set()).add(int(image_id))
    return {sid: frozenset(ids) for sid, ids in grouped.items()}


def load_raw_pin(s3: Any, version: str, manifest_sha256: str, cache_dir: Path) -> RawPin:
    """Sync the tables of the pinned ``coralnet-raw`` version and read its image ids.

    Reads only the ``image_id`` and ``group`` columns of ``images.parquet``; the other
    tables stay on disk.

    Args:
        s3: S3 client.
        version: Version of the published dataset.
        manifest_sha256: Digest of that version's manifest.
        cache_dir: Local cache directory of the synced tables.

    Raises:
        ValueError: If a group or image id of the version's images table is unparseable.
    """
    pin = Pin(name=_DATASET, version=version, manifest_sha256=manifest_sha256)
    sync_tables(s3, {pin.name: pin}, cache_dir, bucket=DATASET_BUCKET)
    images_key = f"{dataset_prefix(pin.name, pin.version)}images.parquet"
    images = pq.read_table(
        cache_path(cache_dir, DATASET_BUCKET, images_key), columns=["image_id", "group"]
    )
    return RawPin(version, manifest_sha256, _group_ids(images))
