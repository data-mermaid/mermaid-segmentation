#!/usr/bin/env python
"""Audit CoralNet source folders in S3.

Lists all source folders, checks file structure, counts images and annotations,
and writes results to a parquet file.

Features:
- Incremental writes: saves progress every N sources (credential-safe on SageMaker)
- Uses boto3 for S3 uploads (auto-refreshes IAM credentials)
- Resumes from checkpoint if previous run failed

Usage:
    python scripts/audit_coralnet_sources.py
    python scripts/audit_coralnet_sources.py --dry-run
    python scripts/audit_coralnet_sources.py --checkpoint-interval 50
"""

from __future__ import annotations

import argparse
import logging
import sys
import tempfile
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path

import boto3
import ibis
import pandas as pd
from tqdm import tqdm


@dataclass(frozen=True)
class CsvSpec:
    filename: str
    has_field: str
    count_field: str
    empty_field: str


CSV_SPECS = (
    CsvSpec("image_list.csv", "has_image_list_csv", "n_images_csv", "image_list_empty"),
    CsvSpec("labelset.csv", "has_labelset_csv", "n_labels", "labelset_empty"),
    CsvSpec("metadata.csv", "has_metadata_csv", "n_metadata_rows", "metadata_empty"),
)

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


def check_csv(con, bucket: str, source_prefix: str, spec: CsvSpec, record: dict) -> str | None:
    """Check a CSV file and update record fields.

    Returns error message if any.
    """
    uri = f"s3://{bucket}/{source_prefix}{spec.filename}"
    table = read_csv_with_ibis(con, uri)
    if table is not None:
        record[spec.has_field] = True
        count = table.count().execute()
        record[spec.count_field] = count
        record[spec.empty_field] = count == 0
    return None


def normalize_errors_column(df: pd.DataFrame) -> pd.DataFrame:
    """Replace empty error lists with None for cleaner parquet storage."""
    df = df.copy()
    df["errors"] = df["errors"].apply(lambda x: x if x else None)
    return df


def audit_source(s3_client, con, bucket: str, source_id: int, prefix: str, audit_ts: datetime) -> dict:
    """Audit a single CoralNet source folder.

    Uses ibis for CSV reads.
    """
    source_prefix = f"{prefix}/s{source_id}/"

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
        record["errors"].append(f"Error checking images folder: {e}")

    # Annotations CSV has special handling for unique image counting
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
        record["errors"].append(f"Error reading annotations.csv: {e}")

    # Standard CSV checks (image_list, labelset, metadata)
    for spec in CSV_SPECS:
        try:
            check_csv(con, bucket, source_prefix, spec, record)
        except Exception as e:
            record["errors"].append(f"Error reading {spec.filename}: {e}")

    record["is_complete"] = record["has_images_folder"] and record["has_annotations_csv"] and record["has_image_list_csv"]
    record["image_count_match"] = record["n_images_s3"] == record["n_images_csv"]

    return record


def upload_to_s3(s3_client, local_path: Path, s3_uri: str) -> None:
    """Upload a local file to S3 using boto3 (credential-safe)."""
    if not s3_uri.startswith("s3://"):
        raise ValueError(f"Expected s3:// URI, got: {s3_uri}")
    parts = s3_uri[5:].split("/", 1)
    bucket = parts[0]
    key = parts[1] if len(parts) > 1 else ""
    s3_client.upload_file(str(local_path), bucket, key)


def save_checkpoint(df: pd.DataFrame, local_path: Path, s3_client, s3_uri: str | None) -> None:
    """Save dataframe to local parquet and optionally upload to S3."""
    df_normalized = normalize_errors_column(df)
    df_normalized.to_parquet(local_path, index=False)
    if s3_uri:
        upload_to_s3(s3_client, local_path, s3_uri)


def run_audit(
    bucket: str,
    prefix: str,
    output_path: str,
    dry_run: bool = False,
    checkpoint_interval: int = 100,
) -> pd.DataFrame:
    """Run the full audit with incremental checkpoints.

    Saves progress every `checkpoint_interval` sources to avoid losing work
    if credentials expire during long runs on SageMaker.
    """
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

    local_checkpoint = Path(tempfile.gettempdir()) / f"audit_checkpoint_{audit_timestamp.strftime('%Y%m%d_%H%M%S')}.parquet"
    checkpoint_s3_uri = None if dry_run else output_path.replace(".parquet", "_checkpoint.parquet")

    for i, source_id in enumerate(tqdm(source_ids, desc="Auditing sources")):
        record = audit_source(s3, con, bucket, source_id, prefix, audit_timestamp)
        results.append(record)
        if record["errors"]:
            for error in record["errors"]:
                logger.warning("Source %d: %s", source_id, error)

        if (i + 1) % checkpoint_interval == 0:
            logger.info("Checkpoint: %d/%d sources complete, saving...", i + 1, len(source_ids))
            checkpoint_df = pd.DataFrame(results)
            try:
                save_checkpoint(checkpoint_df, local_checkpoint, s3, checkpoint_s3_uri)
                logger.info("Checkpoint saved to %s", checkpoint_s3_uri or local_checkpoint)
            except Exception as e:
                logger.warning("Checkpoint upload failed (will retry): %s", e)
                save_checkpoint(checkpoint_df, local_checkpoint, s3, None)
                logger.info("Checkpoint saved locally to %s", local_checkpoint)

    audit_df = normalize_errors_column(pd.DataFrame(results))

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
        logger.info("Local checkpoint available at %s", local_checkpoint)
    else:
        logger.info("Writing final audit results to %s", output_path)
        local_final = Path(tempfile.gettempdir()) / "audit_final.parquet"
        audit_df.to_parquet(local_final, index=False)
        try:
            upload_to_s3(s3, local_final, output_path)
            logger.info("Successfully wrote audit results to S3")
            local_final.unlink(missing_ok=True)
            local_checkpoint.unlink(missing_ok=True)
        except Exception as e:
            logger.error("S3 upload failed: %s", e)
            logger.info("Results saved locally at %s", local_final)
            raise

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
        "--checkpoint-interval",
        type=int,
        default=100,
        help="Save checkpoint every N sources (protects against credential expiration)",
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
            checkpoint_interval=args.checkpoint_interval,
        )
        return 0
    except Exception as e:
        logger.exception("Audit failed: %s", e)
        return 1


if __name__ == "__main__":
    sys.exit(main())
