#!/usr/bin/env python3
"""CLI for CoralNet image resizing preprocessing.

Usage:
    python scripts/preprocess_coralnet_images.py resize \\
        --images-parquet s3://... \\
        --bucket dev-datamermaid-sm-sources \\
        --output-prefix etl-outputs/coralnet \\
        --threshold 2048 \\
        --workers 16

This script orchestrates: scan → resize/upload → build manifest → upload to S3.
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path
from tempfile import mkdtemp

import boto3
import pandas as pd

from mermaidseg.datasets.coralnet.preprocessing import (
    build_manifest,
    resize_and_upload_all_images,
    scan_for_missing_resized_images,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


def resize_command(args: argparse.Namespace) -> None:
    """Resize CoralNet images exceeding threshold, store on S3 with manifest."""
    temp_dir = args.temp_dir
    if temp_dir is None:
        temp_dir = mkdtemp(prefix="coralnet-resize-")

    logger.info("=== CoralNet Image Resizing Preprocessing ===")
    logger.info("Images parquet: %s", args.images_parquet)
    logger.info("Output: s3://%s/%s", args.bucket, args.output_prefix)
    logger.info("Threshold: %d", args.threshold)
    logger.info("Workers: %d", args.workers)
    logger.info("Temp dir: %s", temp_dir)

    # Load images parquet
    logger.info("Loading images parquet...")
    df_images = pd.read_parquet(args.images_parquet)
    logger.info("Loaded %d images", len(df_images))

    # Create S3 client
    s3_client = boto3.client("s3")

    # Phase 1: Scan
    logger.info("Scanning for images needing resize...")
    df_todo = scan_for_missing_resized_images(
        df_images=df_images,
        bucket=args.bucket,
        output_prefix=args.output_prefix,
        threshold=args.threshold,
        workers=args.workers,
        s3_client=s3_client,
    )
    logger.info("Scan complete: %d images need resizing", len(df_todo))

    if len(df_todo) == 0:
        logger.info("No images to resize. Done!")
        return

    # Phase 2: Resize
    checkpoint_path = Path(temp_dir) / "checkpoint.parquet"
    logger.info("Resizing and uploading images...")
    num_resized, num_skipped, num_failed = resize_and_upload_all_images(
        df_todo=df_todo,
        bucket=args.bucket,
        output_prefix=args.output_prefix,
        checkpoint_path=checkpoint_path,
        threshold=args.threshold,
        workers=args.workers,
        checkpoint_every=args.checkpoint_every,
        s3_client=s3_client,
    )

    # Build and upload manifest
    logger.info("Building manifest...")
    df_checkpoint = pd.read_parquet(checkpoint_path)
    df_manifest = build_manifest(
        df_images=df_images,
        df_checkpoint=df_checkpoint,
        output_prefix=args.output_prefix,
        threshold=args.threshold,
    )

    manifest_s3_key = f"{args.output_prefix}/resized/{args.threshold}/manifest.parquet"
    logger.info("Uploading manifest: s3://%s/%s", args.bucket, manifest_s3_key)
    manifest_bytes = df_manifest.to_parquet(index=False)
    s3_client.put_object(
        Bucket=args.bucket,
        Key=manifest_s3_key,
        Body=manifest_bytes,
        ContentType="application/octet-stream",
    )

    logger.info("=== Complete ===")
    logger.info("Resized: %d", num_resized)
    logger.info("Skipped (already on S3): %d", num_skipped)
    logger.info("Failed: %d", num_failed)
    logger.info("Manifest: s3://%s/%s", args.bucket, manifest_s3_key)


def main() -> None:
    """Parse arguments and execute command."""
    epilog_text = (
        "Examples:\n"
        "  python scripts/preprocess_coralnet_images.py resize \\\n"
        "    --images-parquet s3://bucket/etl-outputs/coralnet/images.parquet \\\n"
        "    --bucket dev-datamermaid-sm-sources \\\n"
        "    --output-prefix etl-outputs/coralnet \\\n"
        "    --threshold 2048 \\\n"
        "    --workers 16"
    )
    parser = argparse.ArgumentParser(
        description="CoralNet image resizing preprocessing CLI",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=epilog_text,
    )

    subparsers = parser.add_subparsers(dest="command", help="Subcommand to run")

    # Resize subcommand
    resize_parser = subparsers.add_parser(
        "resize", help="Resize images exceeding threshold and upload to S3"
    )
    resize_parser.add_argument(
        "--images-parquet",
        required=True,
        type=str,
        help="S3 path to images parquet from ETL",
    )
    resize_parser.add_argument(
        "--bucket",
        required=True,
        type=str,
        help="S3 bucket name",
    )
    resize_parser.add_argument(
        "--output-prefix",
        required=True,
        type=str,
        help="S3 prefix for resized images and manifest",
    )
    resize_parser.add_argument(
        "--threshold",
        default=2048,
        type=int,
        help="Resize target for longest edge (pixels)",
    )
    resize_parser.add_argument(
        "--workers",
        default=16,
        type=int,
        help="ThreadPoolExecutor concurrency for Phase 2",
    )
    resize_parser.add_argument(
        "--checkpoint-every",
        default=500,
        type=int,
        help="Flush checkpoint after N images",
    )
    resize_parser.add_argument(
        "--temp-dir",
        default=None,
        type=str,
        help="Local checkpoint storage directory",
    )
    resize_parser.set_defaults(func=resize_command)

    args = parser.parse_args()

    if args.command is None:
        parser.print_help()
        sys.exit(1)

    args.func(args)


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        logger.error("Fatal error: %s", e, exc_info=True)
        sys.exit(1)
