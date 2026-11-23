"""Version names for published dataset and taxonomy versions: ``YYYYMMDD-<sha7>[-N]``."""

from __future__ import annotations

from collections.abc import Iterable
from datetime import UTC, date, datetime
from typing import Any

from pydantic import TypeAdapter, ValidationError

from segdata_spec.manifest import Version

_VERSION = TypeAdapter(Version)


def utc_today() -> date:
    """Return today's date in UTC, the date a version name carries."""
    return datetime.now(UTC).date()


def next_version(existing: Iterable[str], *, git_sha: str, today: date) -> str:
    """Return ``<today>-<sha7>``, or that with the lowest free ``-2``, ``-3``, ... suffix."""
    base = f"{today:%Y%m%d}-{git_sha[:7]}"
    taken = set(existing)
    if base not in taken:
        return base
    suffix = 2
    while f"{base}-{suffix}" in taken:
        suffix += 1
    return f"{base}-{suffix}"


def _is_version(name: str) -> bool:
    try:
        _VERSION.validate_python(name)
    except ValidationError:
        return False
    return True


def list_versions(client: Any, bucket: str, prefix: str) -> list[str]:
    """Return the sorted version folders directly under ``prefix`` (``datasets/<name>/``).

    Folders whose name is not a valid ``Version`` are skipped.
    """
    names: list[str] = []
    paginator = client.get_paginator("list_objects_v2")
    for page in paginator.paginate(Bucket=bucket, Prefix=prefix, Delimiter="/"):
        for entry in page.get("CommonPrefixes", []):
            names.append(entry["Prefix"][len(prefix) :].rstrip("/"))
    return sorted(name for name in names if _is_version(name))
