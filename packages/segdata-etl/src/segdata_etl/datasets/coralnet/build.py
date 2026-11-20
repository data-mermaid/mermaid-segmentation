"""``segdata build coralnet``: build the CoralNet dataset parquets.

Subcommands run one stage each (``build-annotations``, ``build-images``, ``resize``,
``build-manifest``) or all four in order (``all``). Outputs are written under
``<output-dir>/<version>/`` and uploaded to ``etl-outputs/coralnet/<version>/`` with
``--upload-to-s3``. ``resize`` always writes to S3, since the resized images are its output.
"""

from __future__ import annotations

import argparse
import json
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import ibis
import pandas as pd
from botocore.exceptions import ClientError

from coralnet_scraper.config import validate_workers

from ...common.jpeg_header import STATUSES_WITH_DIMENSIONS
from .annotations import build_annotations, list_source_folders
from .config import (
    DEFAULT_OUTPUT_S3_PREFIX,
    get_bucket,
    get_prefix,
    get_resized_prefix,
    get_workers,
)
from .images import build_images
from .io import compute_version_tag, make_s3_client, upload_to_s3, write_parquet_deterministic
from .manifest import build_training_manifest, resized_existence_table
from .resize import (
    list_resized_keys,
    make_resize_s3_client,
    resize_and_upload_all_images,
    scan_for_missing_resized_images,
)
from .schemas import (
    ANNOTATIONS_PRIMARY_KEY,
    ANNOTATIONS_SCHEMA,
    IMAGES_PRIMARY_KEY,
    IMAGES_SCHEMA,
)

logger = logging.getLogger("segdata_etl.datasets.coralnet")

_MISSING_KEY_CODES = ("404", "NoSuchKey", "NotFound")
_RESIZED_IBIS_SCHEMA = ibis.schema({"source_id": "int32", "image_id": "string", "status": "string"})


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
    resized_prefix: str

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
        resized_prefix=get_resized_prefix(),
    )


def _s3_object_exists(client: Any, bucket: str, key: str) -> bool:
    """Return whether ``key`` exists in ``bucket``; other S3 errors propagate."""
    try:
        client.head_object(Bucket=bucket, Key=key)
    except ClientError as e:
        if e.response["Error"]["Code"] in _MISSING_KEY_CODES:
            return False
        raise
    return True


def _maybe_upload(run: _Run, local_path: Path) -> None:
    """Upload ``local_path`` to ``etl-outputs/coralnet/<version>/`` when ``--upload-to-s3``.

    Raises:
        RuntimeError: If the target key exists and ``--overwrite`` was not passed.
    """
    if not run.upload:
        return
    key = f"{DEFAULT_OUTPUT_S3_PREFIX}/{run.version}/{local_path.name}"
    client = make_s3_client()
    if not run.overwrite and _s3_object_exists(client, run.bucket, key):
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
        source_ids = list_source_folders(make_s3_client(), run.bucket, run.prefix)
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


def _log_header_statuses(images_df: pd.DataFrame) -> bool:
    """Log the image count per ``header_status``; return whether every image has dimensions."""
    counts = images_df["header_status"].astype(str).value_counts().sort_index()
    for status, count in counts.items():
        logger.info("header_status %s: %d", status, count)
    return set(counts.index) <= STATUSES_WITH_DIMENSIONS


def _step_build_images(run: _Run, annotations_path: Path) -> int:
    """Write the images parquet; return 1 when any image has no dimensions.

    Those images (statuses such as ``not_found`` or ``corrupt``) are dropped by the manifest.
    """
    annotations_df = pd.read_parquet(annotations_path)
    images_df = build_images(
        annotations_df, bucket=run.bucket, prefix=run.prefix, workers=run.workers
    )
    out_path = run.path("images")
    write_parquet_deterministic(images_df, out_path, IMAGES_SCHEMA, IMAGES_PRIMARY_KEY)
    logger.info("Wrote images parquet: %s (%d rows)", out_path, len(images_df))
    _maybe_upload(run, out_path)
    return 0 if _log_header_statuses(images_df) else 1


def _step_resize(run: _Run, images_path: Path) -> int:
    """Resize and upload the missing images; return 1 on any failed or corrupted image."""
    images_df = pd.read_parquet(images_path)
    # One client shared by every worker thread, with a pool sized so workers never wait on it.
    s3_client = make_resize_s3_client(max_pool_connections=run.workers + 4)
    df_todo = scan_for_missing_resized_images(
        images_df, bucket=run.bucket, output_prefix=run.resized_prefix, s3_client=s3_client
    )
    logger.info("Scan complete: %d images to resize", len(df_todo))
    if df_todo.empty:
        logger.info("Nothing to do: every resized image already exists.")
        return 0
    _resized, corrupted, failed = resize_and_upload_all_images(
        df_todo, run.bucket, workers=run.workers, s3_client=s3_client
    )
    return 1 if corrupted or failed else 0


def _log_manifest_quality(annotations: pd.DataFrame, manifest: pd.DataFrame) -> None:
    """Log the manifest's row and image counts, and how many the build dropped."""
    images = manifest.drop_duplicates(["source_id", "image_id"])
    n_resized = int(images["uses_resized_image"].sum())
    annotated_images = annotations.drop_duplicates(["source_id", "image_id"])
    logger.info("=== Training manifest ===")
    logger.info(
        "Annotation rows:        %s  (dropped %s)",
        f"{len(manifest):,}",
        f"{len(annotations) - len(manifest):,}",
    )
    logger.info(
        "Distinct images:        %s  (dropped %s)",
        f"{len(images):,}",
        f"{len(annotated_images) - len(images):,}",
    )
    logger.info("  using RESIZED key:    %s", f"{n_resized:,}")
    logger.info("  using ORIGINAL key:   %s", f"{len(images) - n_resized:,}")


def _step_build_manifest(run: _Run, annotations_path: Path, images_path: Path) -> int:
    """Write the training manifest from the local parquets and the resized objects on S3."""
    annotations = pd.read_parquet(annotations_path)
    images = pd.read_parquet(images_path)
    logger.info(
        "Images: needs_resize=%s sub_threshold=%s null_dims=%s",
        f"{int(images['needs_resize'].sum()):,}",
        f"{int((~images['needs_resize']).sum()):,}",
        f"{int((images['width'].isna() | images['height'].isna()).sum()):,}",
    )
    resized = resized_existence_table(
        list_resized_keys(make_s3_client(), run.bucket, run.resized_prefix)
    )
    logger.info("Resized objects on S3: %s", f"{len(resized):,}")
    # Explicit schemas keep the join keys typed when an input is empty.
    manifest = build_training_manifest(
        annotations=ibis.memtable(annotations, schema=ibis.Schema.from_pyarrow(ANNOTATIONS_SCHEMA)),
        images=ibis.memtable(images, schema=ibis.Schema.from_pyarrow(IMAGES_SCHEMA)),
        resized=ibis.memtable(resized, schema=_RESIZED_IBIS_SCHEMA),
        output_prefix=run.resized_prefix,
    ).to_pandas()
    _log_manifest_quality(annotations, manifest)
    out_path = run.path("training_manifest")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    manifest.to_parquet(out_path, index=False)
    logger.info("Wrote training manifest: %s (%d rows)", out_path, len(manifest))
    _maybe_upload(run, out_path)
    return 0


def _cmd_build_annotations(args: argparse.Namespace) -> int:
    return _step_build_annotations(_resolve_run(args), args.source_ids)


def _cmd_build_images(args: argparse.Namespace) -> int:
    return _step_build_images(_resolve_run(args), args.annotations)


def _cmd_resize(args: argparse.Namespace) -> int:
    return _step_resize(_resolve_run(args), args.images)


def _cmd_build_manifest(args: argparse.Namespace) -> int:
    return _step_build_manifest(_resolve_run(args), args.annotations, args.images)


def _cmd_all(args: argparse.Namespace) -> int:
    """Run every stage in order with one version tag; return 1 if any stage returned 1."""
    run = _resolve_run(args)
    annotations_path, images_path = run.path("annotations"), run.path("images")
    return_codes = [
        _step_build_annotations(run, args.source_ids),
        _step_build_images(run, annotations_path),
        _step_resize(run, images_path),
        _step_build_manifest(run, annotations_path, images_path),
    ]
    return max(return_codes)
