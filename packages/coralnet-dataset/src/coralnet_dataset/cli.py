"""``coralnet-dataset`` console script: build the CoralNet dataset parquets.

Subcommands run one stage each (``build-annotations``, ``build-images``, ``resize``,
``build-manifest``) or all four in order (``all``). Outputs are written under
``<output-dir>/<version>/`` and uploaded to ``etl-outputs/coralnet/<version>/`` with
``--upload-to-s3``. ``resize`` always writes to S3, since the resized images are its output.
"""

from __future__ import annotations

import argparse
import logging
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import ibis
import pandas as pd
from botocore.exceptions import ClientError

from coralnet_scraper.config import validate_workers

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
from .jpeg_header import STATUSES_WITH_DIMENSIONS
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

logger = logging.getLogger("coralnet_dataset")

_DEFAULT_OUTPUT_DIR = Path("outputs/coralnet-dataset")
_MISSING_KEY_CODES = ("404", "NoSuchKey", "NotFound")
_RESIZED_IBIS_SCHEMA = ibis.schema({"source_id": "int32", "image_id": "string", "status": "string"})


@dataclass(frozen=True)
class _Run:
    """Settings shared by every stage of one invocation."""

    version: str
    out_dir: Path
    workers: int
    upload: bool
    # True when --version-tag was passed, which allows overwriting that tag's uploads.
    explicit_version: bool
    bucket: str
    prefix: str
    resized_prefix: str

    def path(self, kind: str) -> Path:
        """Return the local path of this run's ``coralnet_<kind>_<version>.parquet``."""
        return self.out_dir / f"coralnet_{kind}_{self.version}.parquet"


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
        explicit_version=args.version_tag is not None,
        bucket=get_bucket(),
        prefix=get_prefix(),
        resized_prefix=get_resized_prefix(),
    )


def _parse_source_ids(raw: str) -> list[int]:
    """Parse ``--source-ids 1,2,3`` into a list of ints.

    Raises:
        argparse.ArgumentTypeError: If no source id is given.
    """
    source_ids = [int(x.strip()) for x in raw.split(",") if x.strip()]
    if not source_ids:
        raise argparse.ArgumentTypeError("expected at least one source id")
    return source_ids


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
        RuntimeError: If the target key exists and ``--version-tag`` was not passed.
    """
    if not run.upload:
        return
    key = f"{DEFAULT_OUTPUT_S3_PREFIX}/{run.version}/{local_path.name}"
    client = make_s3_client()
    if not run.explicit_version and _s3_object_exists(client, run.bucket, key):
        raise RuntimeError(
            f"s3://{run.bucket}/{key} already exists; pass --version-tag {run.version} to "
            "overwrite it"
        )
    target = f"s3://{run.bucket}/{key}"
    logger.info("Uploading %s -> %s", local_path, target)
    upload_to_s3(client, local_path, target)


def _step_build_annotations(run: _Run, source_ids: list[int] | None) -> int:
    """Write the annotations parquet; return 1 when no source produced annotation rows."""
    if source_ids is None:
        source_ids = list_source_folders(make_s3_client(), run.bucket, run.prefix)
    annotations_df = build_annotations(
        source_ids, bucket=run.bucket, prefix=run.prefix, workers=run.workers
    )
    out_path = run.path("annotations")
    write_parquet_deterministic(
        annotations_df, out_path, ANNOTATIONS_SCHEMA, ANNOTATIONS_PRIMARY_KEY
    )
    logger.info("Wrote annotations parquet: %s (%d rows)", out_path, len(annotations_df))
    _maybe_upload(run, out_path)
    if annotations_df.empty:
        logger.error("No annotation rows built from s3://%s/%s/", run.bucket, run.prefix)
        return 1
    return 0


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


def _add_common_flags(p: argparse.ArgumentParser) -> None:
    p.add_argument(
        "--version-tag", default=None, help="Override version tag (default: YYYYMMDD_<sha>)"
    )
    p.add_argument(
        "--output-dir",
        type=Path,
        default=_DEFAULT_OUTPUT_DIR,
        help="Local output directory; files go under <output-dir>/<version>/",
    )
    p.add_argument(
        "--upload-to-s3",
        action="store_true",
        help="Upload outputs to etl-outputs/coralnet/<version>/ after writing them locally",
    )
    p.add_argument(
        "--workers",
        type=int,
        default=None,
        help="Concurrent S3 workers (default: MERMAID_CORALNET_ETL_WORKERS, else 16)",
    )
    p.add_argument("-v", "--verbose", action="store_true", help="Verbose logging")


def _add_source_ids_flag(p: argparse.ArgumentParser) -> None:
    p.add_argument(
        "--source-ids",
        type=_parse_source_ids,
        default=None,
        help="Comma-separated source ids (default: every s<id>/ folder under the prefix)",
    )


def build_parser() -> argparse.ArgumentParser:
    """Build the ``coralnet-dataset`` argument parser."""
    parser = argparse.ArgumentParser(
        prog="coralnet-dataset",
        description="Build the CoralNet annotations, images and training-manifest parquets.",
    )
    sub = parser.add_subparsers(dest="command", required=True)

    p_anns = sub.add_parser("build-annotations", help="Merge per-source CSVs into one parquet")
    _add_common_flags(p_anns)
    _add_source_ids_flag(p_anns)
    p_anns.set_defaults(func=_cmd_build_annotations)

    p_imgs = sub.add_parser("build-images", help="Scan JPEG headers into an images parquet")
    _add_common_flags(p_imgs)
    p_imgs.add_argument("--annotations", type=Path, required=True, help="Annotations parquet")
    p_imgs.set_defaults(func=_cmd_build_images)

    p_resize = sub.add_parser("resize", help="Resize and upload images missing on S3")
    _add_common_flags(p_resize)
    p_resize.add_argument("--images", type=Path, required=True, help="Images parquet")
    p_resize.set_defaults(func=_cmd_resize)

    p_manifest = sub.add_parser("build-manifest", help="Build the training manifest parquet")
    _add_common_flags(p_manifest)
    p_manifest.add_argument("--annotations", type=Path, required=True, help="Annotations parquet")
    p_manifest.add_argument("--images", type=Path, required=True, help="Images parquet")
    p_manifest.set_defaults(func=_cmd_build_manifest)

    p_all = sub.add_parser("all", help="Run every stage in order with one version tag")
    _add_common_flags(p_all)
    _add_source_ids_flag(p_all)
    p_all.set_defaults(func=_cmd_all)

    return parser


def _configure_logging(verbose: bool) -> None:
    logging.basicConfig(
        level=logging.DEBUG if verbose else logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
        stream=sys.stdout,
    )


def main(argv: list[str] | None = None) -> int:
    """Run the ``coralnet-dataset`` console script.

    Returns:
        The exit code: 0 on success, 1 when a stage reported failures.
    """
    args = build_parser().parse_args(argv)
    _configure_logging(args.verbose)
    return int(args.func(args))


if __name__ == "__main__":
    raise SystemExit(main())
