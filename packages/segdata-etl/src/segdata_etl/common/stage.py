"""Local staging of uniform-format dataset versions: where they live and what they are named.

``segdata_spec.stage`` writes a version to disk; ``segdata promote`` uploads the staged tree write-once.
"""

from __future__ import annotations

from collections.abc import Callable
from datetime import UTC, date, datetime
from functools import partial
from pathlib import Path
from typing import Any

from segdata_spec.layout import cache_path, dataset_prefix

DEFAULT_CACHE_DIR = Path("outputs") / "segdata-cache"

_SHA_LENGTH = 7


def version_taken(client: Any, bucket: str, prefix: str) -> bool:
    """Return whether any object exists under the version-folder ``prefix`` in ``bucket``."""
    response = client.list_objects_v2(Bucket=bucket, Prefix=prefix, MaxKeys=1)
    return bool(response.get("Contents"))


def next_free_version(
    stage_root: Path | None,
    bucket: str,
    prefix_for: Callable[[str], str],
    *,
    git_sha: str,
    today: date | None = None,
    client: Any | None = None,
) -> str:
    """Return ``YYYYMMDD-<sha7>``, with ``-2``, ``-3``, ... appended until the version is free.

    ``prefix_for(version)`` is the version's folder prefix, e.g. ``datasets/<ds>/<version>/``. A version is
    taken when its staged dir exists under ``stage_root`` (skipped when ``None``) or, given ``client``, when
    any object exists under its prefix in ``bucket``, which includes a half-promoted version. ``today``
    defaults to the current UTC date.
    """
    today = today or datetime.now(UTC).date()
    base = f"{today:%Y%m%d}-{git_sha[:_SHA_LENGTH]}"

    def taken(candidate: str) -> bool:
        prefix = prefix_for(candidate)
        if stage_root is not None and cache_path(stage_root, bucket, prefix).exists():
            return True
        return client is not None and version_taken(client, bucket, prefix)

    candidate, counter = base, 1
    while taken(candidate):
        counter += 1
        candidate = f"{base}-{counter}"
    return candidate


def next_version(
    stage_root: Path,
    bucket: str,
    dataset: str,
    *,
    git_sha: str,
    today: date | None = None,
    client: Any | None = None,
) -> str:
    """Return the first free version of ``datasets/<dataset>/``; see ``next_free_version``."""
    return next_free_version(
        stage_root,
        bucket,
        partial(dataset_prefix, dataset),
        git_sha=git_sha,
        today=today,
        client=client,
    )
