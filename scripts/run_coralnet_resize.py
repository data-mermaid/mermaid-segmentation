"""Resize all CoralNet images flagged needs_resize in an ETL images parquet."""

from __future__ import annotations

import argparse
import logging
import os
from pathlib import Path

import boto3
import pandas as pd

from mermaidseg.datasets.coralnet.preprocessing.resize import (
    DEFAULT_CHECKPOINT_EVERY,
    build_todo_from_checkpoint,
    read_checkpoint,
    resize_and_upload_all_images,
    scan_for_missing_resized_images,
)

logger = logging.getLogger(__name__)

DEFAULT_CHECKPOINT_S3_KEY = "etl-outputs/coralnet/{run}/resize_checkpoint_{run}.parquet"


def default_worker_count() -> int:
    """I/O-bound default: scale with vCPU, cap for connection sanity."""
    env = os.getenv("MERMAID_CORALNET_RESIZE_WORKERS")
    if env:
        return max(1, int(env))
    cpu = os.cpu_count() or 4
    return min(64, max(32, cpu * 4))


def sync_checkpoint_to_s3(local_path: Path, bucket: str, s3_key: str) -> None:
    boto3.client("s3").upload_file(str(local_path), bucket, s3_key)
    logger.info("Checkpoint synced to s3://%s/%s", bucket, s3_key)


def sync_checkpoint_from_s3(local_path: Path, bucket: str, s3_key: str) -> bool:
    s3 = boto3.client("s3")
    try:
        s3.head_object(Bucket=bucket, Key=s3_key)
    except Exception:
        return False
    local_path.parent.mkdir(parents=True, exist_ok=True)
    s3.download_file(bucket, s3_key, str(local_path))
    logger.info("Checkpoint downloaded from s3://%s/%s", bucket, s3_key)
    return True


def log_checkpoint_summary(checkpoint_path: Path) -> None:
    if not checkpoint_path.exists():
        return
    df = read_checkpoint(checkpoint_path)
    counts = df["status"].value_counts().to_dict()
    logger.info(
        "Checkpoint %s: completed=%s pending=%s failed=%s",
        checkpoint_path,
        counts.get("completed", 0),
        counts.get("pending", 0),
        counts.get("failed", 0),
    )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bucket", default="dev-datamermaid-sm-sources")
    parser.add_argument("--run", default="20260526_807b611")
    parser.add_argument(
        "--images-uri",
        default=None,
        help="Override S3/local parquet URI (default: etl-outputs/coralnet/{run}/...)",
    )
    parser.add_argument("--output-prefix", default="dev/images")
    parser.add_argument("--threshold", type=int, default=2048)
    parser.add_argument(
        "--workers-scan",
        type=int,
        default=None,
        help="Scan thread count (default: MERMAID_CORALNET_RESIZE_WORKERS or auto)",
    )
    parser.add_argument(
        "--workers-resize",
        type=int,
        default=None,
        help="Resize thread count (default: MERMAID_CORALNET_RESIZE_WORKERS or auto)",
    )
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=Path("outputs/resize_full_20260526_807b611.parquet"),
    )
    parser.add_argument(
        "--checkpoint-s3-key",
        default=None,
        help="S3 key for checkpoint sync (default: etl-outputs/coralnet/{run}/resize_checkpoint_{run}.parquet)",
    )
    parser.add_argument(
        "--pull-checkpoint",
        action="store_true",
        help="Download checkpoint from S3 before running (resume on SageMaker)",
    )
    parser.add_argument(
        "--push-checkpoint",
        action="store_true",
        help="Upload checkpoint to S3 after running",
    )
    parser.add_argument(
        "--skip-scan",
        action="store_true",
        help="Resume: build todo from pending checkpoint rows (skip S3 head scan)",
    )
    parser.add_argument(
        "--checkpoint-every",
        type=int,
        default=DEFAULT_CHECKPOINT_EVERY,
        help="Rows between checkpoint flushes (default: %(default)s)",
    )
    args = parser.parse_args()

    workers_default = default_worker_count()
    workers_scan = args.workers_scan or workers_default
    workers_resize = args.workers_resize or workers_default
    checkpoint_s3_key = args.checkpoint_s3_key or DEFAULT_CHECKPOINT_S3_KEY.format(run=args.run)

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
    )

    if args.pull_checkpoint:
        sync_checkpoint_from_s3(args.checkpoint, args.bucket, checkpoint_s3_key)
    log_checkpoint_summary(args.checkpoint)

    if args.images_uri:
        images_uri = args.images_uri
    else:
        images_uri = (
            f"s3://{args.bucket}/etl-outputs/coralnet/{args.run}/coralnet_images_{args.run}.parquet"
        )

    logger.info("Loading %s", images_uri)
    # Size the s3fs/botocore connection pool to the worker count so the parquet load and any
    # follow-on s3fs traffic don't churn through the default 10-connection pool.
    storage_options = None
    if images_uri.startswith("s3://"):
        storage_options = {
            "config_kwargs": {"max_pool_connections": max(workers_scan, workers_resize) + 4}
        }
    df = pd.read_parquet(images_uri, storage_options=storage_options)
    df_work = df.loc[df["needs_resize"]].copy()
    logger.info("needs_resize=True: %s / %s", f"{len(df_work):,}", f"{len(df):,}")
    logger.info(
        "Workers scan=%s resize=%s (shared client pool=%s)",
        workers_scan,
        workers_resize,
        workers_resize + 4,
    )

    if args.skip_scan:
        if not args.checkpoint.exists():
            logger.error("--skip-scan requires an existing checkpoint at %s", args.checkpoint)
            return 1
        df_todo = build_todo_from_checkpoint(
            df_images=df_work,
            checkpoint_path=args.checkpoint,
            output_prefix=args.output_prefix,
            threshold=args.threshold,
        )
        logger.info("Skip-scan resume: %s pending images to resize", f"{len(df_todo):,}")
    else:
        df_todo = scan_for_missing_resized_images(
            df_images=df_work,
            bucket=args.bucket,
            output_prefix=args.output_prefix,
            threshold=args.threshold,
            workers=workers_scan,
            s3_client=None,
        )
        logger.info("Scan complete: %s images to resize", f"{len(df_todo):,}")

    if df_todo.empty:
        logger.info("Nothing to do.")
        if args.push_checkpoint and args.checkpoint.exists():
            sync_checkpoint_to_s3(args.checkpoint, args.bucket, checkpoint_s3_key)
        return 0

    args.checkpoint.parent.mkdir(parents=True, exist_ok=True)
    num_resized, num_skipped, num_failed, num_corrupted = resize_and_upload_all_images(
        df_todo=df_todo,
        bucket=args.bucket,
        output_prefix=args.output_prefix,
        checkpoint_path=args.checkpoint,
        threshold=args.threshold,
        workers=workers_resize,
        s3_client=None,
        checkpoint_every=args.checkpoint_every,
    )
    log_checkpoint_summary(args.checkpoint)
    logger.info(
        "Done: resized=%s skipped=%s corrupted=%s failed=%s checkpoint=%s",
        num_resized,
        num_skipped,
        num_corrupted,
        num_failed,
        args.checkpoint,
    )

    if args.push_checkpoint:
        sync_checkpoint_to_s3(args.checkpoint, args.bucket, checkpoint_s3_key)

    return 1 if num_failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
