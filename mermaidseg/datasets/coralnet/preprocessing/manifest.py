"""Manifest creation for resized images."""

from __future__ import annotations

import logging

import pandas as pd

logger = logging.getLogger(__name__)


def build_manifest(
    df_images: pd.DataFrame,
    df_checkpoint: pd.DataFrame,
    output_prefix: str,
    threshold: int = 2048,
) -> pd.DataFrame:
    """Build manifest from images and checkpoint.

    Args:
        df_images: Images parquet with [source_id, image_id, width, height, needs_resize, ...]
        df_checkpoint: Checkpoint parquet with [source_id, image_id, status, resize_timestamp, error_message]
        output_prefix: S3 prefix for resized images
        threshold: Resize threshold

    Returns:
        Manifest DataFrame with columns:
        [source_id, image_id, original_width, original_height, resized_width, resized_height,
         output_s3_key, resize_timestamp, status]
    """
    # Merge images + checkpoint
    df_manifest = df_images[["source_id", "image_id", "width", "height"]].merge(
        df_checkpoint[["source_id", "image_id", "status", "resize_timestamp", "error_message"]],
        on=["source_id", "image_id"],
    )

    # Only include items that were processed (not skipped because they weren't in needs_resize)
    df_manifest = df_manifest[df_manifest["status"].notna()].copy()

    # Calculate resized dimensions
    def calc_resized_dims(row):
        width, height = int(row["width"]), int(row["height"])
        longest = max(width, height)
        if longest <= threshold:
            return width, height
        scale = threshold / longest
        return int(width * scale), int(height * scale)

    resized_dims = df_manifest.apply(calc_resized_dims, axis=1, result_type="expand")
    df_manifest["resized_width"] = resized_dims[0]
    df_manifest["resized_height"] = resized_dims[1]

    # Build S3 keys
    df_manifest["output_s3_key"] = df_manifest.apply(
        lambda row: (
            f"{output_prefix}/resized/{threshold}/s{int(row['source_id'])}/images/{row['image_id']}.jpg"
        ),
        axis=1,
    )

    # Select and rename columns
    df_manifest = df_manifest[
        [
            "source_id",
            "image_id",
            "width",
            "height",
            "resized_width",
            "resized_height",
            "output_s3_key",
            "resize_timestamp",
            "status",
        ]
    ]

    df_manifest = df_manifest.rename(
        columns={
            "width": "original_width",
            "height": "original_height",
        }
    )

    return df_manifest.reset_index(drop=True)
