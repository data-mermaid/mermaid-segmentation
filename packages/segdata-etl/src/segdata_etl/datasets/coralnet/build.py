"""``segdata build coralnet``: build the CoralNet dataset parquets.

Subcommands run one stage each (``build-annotations``, ``build-images``) or both in order
(``all``). Outputs are written under ``<output-dir>/<version>/`` and uploaded to
``etl-outputs/coralnet/<version>/`` with ``--upload-to-s3``.
"""

from __future__ import annotations

import argparse
import json
import logging
from dataclasses import dataclass
from pathlib import Path

import pandas as pd

from coralnet_scraper.config import validate_workers
from segdata_spec.s3 import exists

from ...common.jpeg_header import STATUSES_WITH_DIMENSIONS
from ...common.s3_objects import make_s3_client
from .annotations import build_annotations, list_source_folders
from .config import (
    DEFAULT_OUTPUT_S3_PREFIX,
    get_bucket,
    get_prefix,
    get_workers,
)
from .images import build_images
from .io import compute_version_tag, upload_to_s3, write_parquet_deterministic
from .markers import pinned_source_ids
from .schemas import (
    ANNOTATIONS_PRIMARY_KEY,
    ANNOTATIONS_SCHEMA,
    IMAGES_PRIMARY_KEY,
    IMAGES_SCHEMA,
)

logger = logging.getLogger("segdata_etl.datasets.coralnet")


@dataclass(frozen=True)
class _Run:
    """Settings shared by every stage of one invocation."""

    version: str
    out_dir: Path
    workers: int
    upload: bool
    # True when --overwrite was passed, which allows replacing existing S3 objects.
    overwrite: bool
    bucket: str
    prefix: str

    def path(self, kind: str, suffix: str = "parquet") -> Path:
        """Return the local path of this run's ``coralnet_<kind>_<version>.<suffix>``."""
        return self.out_dir / f"coralnet_{kind}_{self.version}.{suffix}"


def _resolve_workers(flag: int | None) -> int:
    """Return ``--workers`` when given, else the environment's worker count."""
    if flag is None:
        return get_workers()
    try:
        return validate_workers(flag)
    except ValueError:
        raise ValueError(f"--workers must be >= 1, got {flag}") from None


def _resolve_run(args: argparse.Namespace) -> _Run:
    """Build the run settings from the common flags and the environment."""
    version = compute_version_tag(args.version_tag)
    return _Run(
        version=version,
        out_dir=Path(args.output_dir) / version,
        workers=_resolve_workers(args.workers),
        upload=args.upload_to_s3,
        overwrite=args.overwrite,
        bucket=get_bucket(),
        prefix=get_prefix(),
    )


def _maybe_upload(run: _Run, local_path: Path) -> None:
    """Upload ``local_path`` to ``etl-outputs/coralnet/<version>/`` when ``--upload-to-s3``.

    Raises:
        RuntimeError: If the target key exists and ``--overwrite`` was not passed.
    """
    if not run.upload:
        return
    key = f"{DEFAULT_OUTPUT_S3_PREFIX}/{run.version}/{local_path.name}"
    client = make_s3_client(workers=run.workers)
    if not run.overwrite and exists(client, run.bucket, key):
        raise RuntimeError(
            f"s3://{run.bucket}/{key} already exists; pass --overwrite to replace it"
        )
    target = f"s3://{run.bucket}/{key}"
    logger.info("Uploading %s -> %s", local_path, target)
    upload_to_s3(client, local_path, target)


def _summarize_exclusions(exclusions: dict[str, list[int]]) -> list[str]:
    """Return the exclusion summary lines, or an empty list when no source was excluded.

    The first line is the number of excluded sources. One line follows per kind, kinds sorted,
    listing that kind's source ids sorted, e.g. ``labelset_missing: [3, 4]``.
    """
    if not exclusions:
        return []
    n_excluded = sum(len(ids) for ids in exclusions.values())
    lines = [f"{n_excluded} source(s) excluded"]
    lines += [f"{kind}: {sorted(exclusions[kind])}" for kind in sorted(exclusions)]
    return lines


def _write_id2name(id2name: dict[str, str], out_path: Path) -> None:
    """Write ``id2name`` as JSON with sorted keys, so equal maps give equal bytes."""
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(id2name, sort_keys=True, indent=2) + "\n")


def _step_build_annotations(run: _Run, source_ids: list[int] | None) -> int:
    """Write the annotations parquet and the id2name JSON.

    Returns:
        1 when a source was excluded or no source produced annotation rows, else 0.
    """
    if source_ids is None:
        source_ids = list_source_folders(
            make_s3_client(workers=run.workers), run.bucket, run.prefix
        )
    build = build_annotations(source_ids, bucket=run.bucket, prefix=run.prefix, workers=run.workers)
    out_path = run.path("annotations")
    write_parquet_deterministic(
        build.annotations, out_path, ANNOTATIONS_SCHEMA, ANNOTATIONS_PRIMARY_KEY
    )
    logger.info("Wrote annotations parquet: %s (%d rows)", out_path, len(build.annotations))
    json_path = run.path("id2name", suffix="json")
    _write_id2name(build.id2name, json_path)
    logger.info("Wrote id2name: %s (%d labels)", json_path, len(build.id2name))
    _maybe_upload(run, out_path)
    _maybe_upload(run, json_path)
    summary = _summarize_exclusions(build.exclusions)
    for line in summary:
        logger.error("%s", line)
    if build.annotations.empty:
        logger.error("No annotation rows built from s3://%s/%s/", run.bucket, run.prefix)
        return 1
    return 1 if summary else 0


def _log_header_statuses(images_df: pd.DataFrame, pinned_sources: frozenset[int]) -> bool:
    """Log the image count per ``header_status``; return whether every image is accounted for.

    An image is accounted for when it has dimensions, or is ``not_found`` in a pinned source,
    whose images may live only in the pinned parent version. Those are logged as their own count.
    """
    counts = images_df["header_status"].astype(str).value_counts().sort_index()
    for status, count in counts.items():
        logger.info("header_status %s: %d", status, count)
    status = images_df["header_status"].astype(str)
    no_dimensions = ~status.isin(STATUSES_WITH_DIMENSIONS)
    expected = (status == "not_found") & images_df["source_id"].isin(pinned_sources)
    if expected.any():
        logger.info("not_found in pinned sources: %d", int(expected.sum()))
    return not (no_dimensions & ~expected).any()


def _step_build_images(run: _Run, annotations_path: Path) -> int:
    """Write the images parquet; return 1 when any image has no dimensions, else 0.

    Those images have statuses such as ``not_found`` or ``corrupt``. A ``not_found`` image in a
    pinned source (see ``pinned_source_ids``) is expected and does not count.

    Raises:
        RuntimeError: A source's completion marker is unreadable.
    """
    annotations_df = pd.read_parquet(annotations_path)
    images_df = build_images(
        annotations_df, bucket=run.bucket, prefix=run.prefix, workers=run.workers
    )
    out_path = run.path("images")
    write_parquet_deterministic(images_df, out_path, IMAGES_SCHEMA, IMAGES_PRIMARY_KEY)
    logger.info("Wrote images parquet: %s (%d rows)", out_path, len(images_df))
    _maybe_upload(run, out_path)
    pinned = pinned_source_ids(
        make_s3_client(workers=run.workers),
        run.bucket,
        run.prefix,
        sorted(int(sid) for sid in images_df["source_id"].unique()),
        workers=run.workers,
    )
    return 0 if _log_header_statuses(images_df, pinned) else 1


def _cmd_build_annotations(args: argparse.Namespace) -> int:
    return _step_build_annotations(_resolve_run(args), args.source_ids)


def _cmd_build_images(args: argparse.Namespace) -> int:
    return _step_build_images(_resolve_run(args), args.annotations)


def _cmd_all(args: argparse.Namespace) -> int:
    """Run both stages in order with one version tag; return 1 if any stage returned 1."""
    run = _resolve_run(args)
    annotations_path = run.path("annotations")
    return_codes = [
        _step_build_annotations(run, args.source_ids),
        _step_build_images(run, annotations_path),
    ]
    return max(return_codes)
