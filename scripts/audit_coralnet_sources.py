#!/usr/bin/env python
"""Audit CoralNet source folders in S3.

Lists all source folders, checks file structure, counts images and annotations,
and writes results to a parquet file.

Usage:
    python scripts/audit_coralnet_sources.py
    python scripts/audit_coralnet_sources.py --dry-run
    python scripts/audit_coralnet_sources.py --bucket my-bucket --prefix my-prefix
"""

from __future__ import annotations

import argparse
import logging
import sys
from datetime import UTC, datetime
from pathlib import Path

import boto3
import ibis
import pandas as pd
from tqdm import tqdm

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger(__name__)


def list_source_folders(s3_client, bucket: str, prefix: str) -> list[int]:
    """List all source folders directly from the target bucket."""
    full_prefix = f"{prefix}/" if not prefix.endswith("/") else prefix
    paginator = s3_client.get_paginator("list_objects_v2")

    source_ids = []
    for page in paginator.paginate(Bucket=bucket, Prefix=full_prefix, Delimiter="/"):
        for common_prefix in page.get("CommonPrefixes", []):
            folder_name = common_prefix["Prefix"].replace(full_prefix, "").rstrip("/")
            if folder_name.startswith("s") and folder_name[1:].isdigit():
                source_ids.append(int(folder_name[1:]))

    return sorted(source_ids)


def check_prefix_exists(s3_client, bucket: str, prefix: str) -> bool:
    """Check if a prefix (folder) has any objects."""
    response = s3_client.list_objects_v2(Bucket=bucket, Prefix=prefix, MaxKeys=1)
    return response.get("KeyCount", 0) > 0


def count_images_in_s3(s3_client, bucket: str, images_prefix: str) -> int:
    """Count all objects in an images/ folder using pagination."""
    paginator = s3_client.get_paginator("list_objects_v2")
    count = 0
    for page in paginator.paginate(Bucket=bucket, Prefix=images_prefix):
        count += page.get("KeyCount", 0)
    return count


def read_csv_with_ibis(con, s3_uri: str) -> ibis.Table | None:
    """Read a CSV from S3 using ibis.

    Returns None if read fails.
    """
    try:
        return con.read_csv(s3_uri)
    except Exception:
        return None


def audit_source(s3_client, con, bucket: str, source_id: int, prefix: str, audit_ts: datetime) -> dict:
    """Audit a single CoralNet source folder.

    Uses ibis for CSV reads.
    """
    source_prefix = f"{prefix}/s{source_id}/"
    errors = []

    record = {
        "source_id": source_id,
        "audit_timestamp": audit_ts,
        "has_images_folder": False,
        "has_annotations_csv": False,
        "has_image_list_csv": False,
        "has_labelset_csv": False,
        "has_metadata_csv": False,
        "n_images_s3": 0,
        "n_images_csv": 0,
        "n_annotations": 0,
        "n_unique_images_annotated": 0,
        "n_labels": None,
        "n_metadata_rows": None,
        "annotations_empty": False,
        "image_list_empty": False,
        "labelset_empty": False,
        "metadata_empty": False,
        "is_complete": False,
        "image_count_match": False,
        "errors": [],
    }

    try:
        images_prefix = f"{source_prefix}images/"
        record["has_images_folder"] = check_prefix_exists(s3_client, bucket, images_prefix)
        if record["has_images_folder"]:
            record["n_images_s3"] = count_images_in_s3(s3_client, bucket, images_prefix)
    except Exception as e:
        errors.append(f"Error checking images folder: {e}")

    try:
        annotations_uri = f"s3://{bucket}/{source_prefix}annotations.csv"
        table = read_csv_with_ibis(con, annotations_uri)
        if table is not None:
            record["has_annotations_csv"] = True
            count = table.count().execute()
            record["n_annotations"] = count
            record["annotations_empty"] = count == 0
            if count > 0:
                cols = table.columns
                if "Name" in cols:
                    record["n_unique_images_annotated"] = table.Name.nunique().execute()
                elif "image" in cols:
                    record["n_unique_images_annotated"] = table.image.nunique().execute()
    except Exception as e:
        errors.append(f"Error reading annotations.csv: {e}")

    try:
        image_list_uri = f"s3://{bucket}/{source_prefix}image_list.csv"
        table = read_csv_with_ibis(con, image_list_uri)
        if table is not None:
            record["has_image_list_csv"] = True
            count = table.count().execute()
            record["n_images_csv"] = count
            record["image_list_empty"] = count == 0
    except Exception as e:
        errors.append(f"Error reading image_list.csv: {e}")

    try:
        labelset_uri = f"s3://{bucket}/{source_prefix}labelset.csv"
        table = read_csv_with_ibis(con, labelset_uri)
        if table is not None:
            record["has_labelset_csv"] = True
            count = table.count().execute()
            record["n_labels"] = count
            record["labelset_empty"] = count == 0
    except Exception as e:
        errors.append(f"Error reading labelset.csv: {e}")

    try:
        metadata_uri = f"s3://{bucket}/{source_prefix}metadata.csv"
        table = read_csv_with_ibis(con, metadata_uri)
        if table is not None:
            record["has_metadata_csv"] = True
            count = table.count().execute()
            record["n_metadata_rows"] = count
            record["metadata_empty"] = count == 0
    except Exception as e:
        errors.append(f"Error reading metadata.csv: {e}")

    record["is_complete"] = record["has_images_folder"] and record["has_annotations_csv"] and record["has_image_list_csv"]
    record["image_count_match"] = record["n_images_s3"] == record["n_images_csv"]
    record["errors"] = errors

    return record


def run_audit(
    bucket: str,
    prefix: str,
    output_path: str,
    dry_run: bool = False,
) -> pd.DataFrame:
    """Run the full audit and write results to parquet."""
    s3 = boto3.client("s3")

    logger.info("Initializing ibis/duckdb connection with S3 support")
    con = ibis.duckdb.connect()
    con.raw_sql("INSTALL httpfs; LOAD httpfs;")
    con.raw_sql("CREATE OR REPLACE SECRET s3 (TYPE S3, PROVIDER CREDENTIAL_CHAIN)")

    logger.info("Listing source folders from s3://%s/%s", bucket, prefix)
    source_ids = list_source_folders(s3, bucket, prefix)
    logger.info("Found %d source folders to audit", len(source_ids))

    if not source_ids:
        logger.warning("No source folders found. Exiting.")
        return pd.DataFrame()

    audit_timestamp = datetime.now(UTC)
    results = []

    for source_id in tqdm(source_ids, desc="Auditing sources"):
        record = audit_source(s3, con, bucket, source_id, prefix, audit_timestamp)
        results.append(record)
        if record["errors"]:
            for error in record["errors"]:
                logger.warning("Source %d: %s", source_id, error)

    audit_df = pd.DataFrame(results)
    audit_df["errors"] = audit_df["errors"].apply(lambda x: x if x else None)

    complete_count = audit_df["is_complete"].sum()
    logger.info(
        "Audit complete: %d/%d sources are complete (%.1f%%)",
        complete_count,
        len(audit_df),
        complete_count / len(audit_df) * 100,
    )
    logger.info("Total images (S3): %d", audit_df["n_images_s3"].sum())
    logger.info("Total annotations: %d", audit_df["n_annotations"].sum())

    if dry_run:
        logger.info("Dry run - skipping write to %s", output_path)
    else:
        logger.info("Writing audit results to %s", output_path)
        audit_table = ibis.memtable(audit_df)
        con.to_parquet(audit_table, output_path)
        logger.info("Successfully wrote audit results")

    return audit_df


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Audit CoralNet source folders in S3",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--bucket",
        default="dev-datamermaid-sm-sources",
        help="S3 bucket containing CoralNet sources",
    )
    parser.add_argument(
        "--prefix",
        default="coralnet-public-images",
        help="S3 prefix for CoralNet source folders",
    )
    parser.add_argument(
        "--output-path",
        default=None,
        help="S3 path for output parquet (default: s3://BUCKET/dev/coralnet_source_audit_YYYYMMDD.parquet)",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Run audit but skip writing to S3",
    )
    parser.add_argument(
        "--log-file",
        type=Path,
        default=None,
        help="Optional log file path",
    )

    args = parser.parse_args()

    if args.log_file:
        file_handler = logging.FileHandler(args.log_file)
        file_handler.setFormatter(logging.Formatter("%(asctime)s - %(levelname)s - %(message)s"))
        logger.addHandler(file_handler)

    if args.output_path is None:
        today = datetime.now(UTC).strftime("%Y%m%d")
        args.output_path = f"s3://{args.bucket}/dev/coralnet_source_audit_{today}.parquet"

    try:
        run_audit(
            bucket=args.bucket,
            prefix=args.prefix,
            output_path=args.output_path,
            dry_run=args.dry_run,
        )
        return 0
    except Exception as e:
        logger.exception("Audit failed: %s", e)
        return 1


if __name__ == "__main__":
    sys.exit(main())
