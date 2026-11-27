"""The per-version sync markers: one file per sync, unioned by ``open_version``."""

from __future__ import annotations

import json
import uuid
from collections.abc import Collection, Iterable
from dataclasses import dataclass
from pathlib import Path

from segdata_spec.errors import CorruptSyncMarker
from segdata_spec.resize import ResizeSpec

_ALL = "all"


@dataclass(frozen=True)
class SyncMarker:
    """The resize spec applied, the groups synced (None means all) and the images skipped.

    ``skipped`` holds the ids of images whose original Pillow could not decode, so the resized
    cache has no output for them. A marker written before this field existed reads as empty.
    """

    resize: ResizeSpec | None
    groups: frozenset[str] | None
    skipped: frozenset[str] = frozenset()


def merge_groups(
    existing: frozenset[str] | None, new: Collection[str] | None
) -> frozenset[str] | None:
    """Union two group selections; None (all groups) absorbs everything."""
    if existing is None or new is None:
        return None
    return existing | frozenset(new)


def dump_marker(marker: SyncMarker) -> bytes:
    payload = {
        "resize": None if marker.resize is None else marker.resize.model_dump(),
        "groups": _ALL if marker.groups is None else sorted(marker.groups),
        "skipped": sorted(marker.skipped),
    }
    return json.dumps(payload, sort_keys=True, indent=2).encode()


def dump_feature_marker(feature_set_sha256: str, image_ids: Collection[str] | None) -> bytes:
    """Serialise a feature-set sync: the set's sha256 and the ids synced (``"all"`` for None)."""
    payload = {
        "feature_set_sha256": feature_set_sha256,
        "image_ids": _ALL if image_ids is None else sorted(image_ids),
    }
    return json.dumps(payload, sort_keys=True, indent=2).encode()


def new_marker_path(directory: Path) -> Path:
    """Return a fresh marker file path in ``directory``; every sync writes its own."""
    return directory / f"{uuid.uuid4().hex}.json"


def marker_files(directory: Path) -> list[Path]:
    """Return the sorted marker files of ``directory``; empty when it does not exist."""
    return sorted(directory.glob("*.json")) if directory.is_dir() else []


def load_marker(path: Path) -> SyncMarker:
    try:
        payload = json.loads(path.read_bytes())
        groups = payload["groups"]
        return SyncMarker(
            resize=None if payload["resize"] is None else ResizeSpec(**payload["resize"]),
            groups=None if groups == _ALL else frozenset(groups),
            skipped=frozenset(payload.get("skipped", ())),
        )
    except (OSError, ValueError, KeyError, TypeError) as exc:
        raise CorruptSyncMarker(f"sync marker {path} is unreadable: {exc!r}") from exc


def union_markers(markers: Iterable[SyncMarker]) -> SyncMarker:
    """Union the markers of one resize spec: the groups (None absorbs) and the skipped ids."""
    items = list(markers)
    groups: frozenset[str] | None = frozenset()
    skipped: frozenset[str] = frozenset()
    for marker in items:
        groups = merge_groups(groups, marker.groups)
        skipped |= marker.skipped
    return SyncMarker(items[0].resize, groups, skipped)


def load_synced(directory: Path) -> SyncMarker:
    """Load and union every marker file in ``directory``; it must hold at least one."""
    return union_markers(load_marker(path) for path in marker_files(directory))
