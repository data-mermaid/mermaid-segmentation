"""Sync the dataset versions a data config pins into the local cache."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from mermaidseg.data.config import DATASET_BUCKET, SyncPlan
from segdata_spec.splits import coverage_checker
from segdata_spec.sync import SyncResult, sync


def run_sync(
    plan: SyncPlan, cache_dir: Path, client: Any, max_workers: int | None = None
) -> SyncResult:
    """Download and verify the plan's pinned versions and selected images into ``cache_dir``.

    ``client`` needs ``max_pool_connections`` of at least ``max_workers``; ``None`` takes
    ``segdata_spec.sync.sync``'s default.
    """
    return sync(
        client,
        plan.pins,
        cache_dir,
        bucket=DATASET_BUCKET,
        selection=plan.selection,
        on_missing_images=plan.on_missing_images,
        check_split=coverage_checker(plan.split_files),
        taxonomy=plan.taxonomy,
        resize=plan.resize,
        max_workers=max_workers,
    )


def format_summary(result: SyncResult) -> str:
    """One line per dataset (name, version, resize id, images dropped), then the taxonomy version.

    The resize id is ``original`` for a dataset cached without resizing.
    """
    lines = [
        f"{name}: version {r.version}, resize {r.resize.id if r.resize else 'original'}, "
        f"images_dropped {r.images_dropped}"
        for name, r in result.datasets.items()
    ]
    if result.taxonomy is not None:
        lines.append(f"taxonomy: version {result.taxonomy.version}")
    return "\n".join(lines)
