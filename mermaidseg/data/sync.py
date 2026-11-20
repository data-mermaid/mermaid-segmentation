"""Sync the dataset versions a data config pins into the local cache."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from mermaidseg.data.config import DATASET_BUCKET, SyncPlan
from segdata_spec.splits import coverage_checker
from segdata_spec.sync import SyncResult, sync


def run_sync(plan: SyncPlan, cache_dir: Path, client: Any) -> SyncResult:
    """Download and verify the plan's pinned versions and selected images into ``cache_dir``."""
    return sync(
        client,
        plan.pins,
        cache_dir,
        bucket=DATASET_BUCKET,
        selection=plan.selection,
        on_missing_images=plan.on_missing_images,
        check_split=coverage_checker(plan.split_files),
        taxonomy=plan.taxonomy,
    )


def format_summary(result: SyncResult) -> str:
    """One line per dataset (name, version, images dropped), then the taxonomy version if any."""
    lines = [
        f"{name}: version {r.version}, images_dropped {r.images_dropped}"
        for name, r in result.datasets.items()
    ]
    if result.taxonomy is not None:
        lines.append(f"taxonomy: version {result.taxonomy.version}")
    return "\n".join(lines)
