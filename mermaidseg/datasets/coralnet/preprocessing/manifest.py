"""Manifest creation for resized images."""

from __future__ import annotations

import logging
from collections.abc import Sequence

import ibis

logger = logging.getLogger(__name__)


def combine_checkpoints(checkpoints: Sequence[ibis.Table]) -> ibis.Table:
    """Merge resize checkpoints from successive runs into one status table.

    The row with the most recent ``resize_timestamp`` wins on (source_id, image_id) conflicts, so a
    rerun that reprocessed previously skipped/failed items overrides the earlier status. Rows with
    no timestamp (failed/skipped) lose to any timestamped row; among themselves, the later
    checkpoint in the input sequence wins. All checkpoints must share the same schema.
    """
    if not checkpoints:
        raise ValueError("combine_checkpoints requires at least one checkpoint")
    tagged = [t.mutate(_run_order=ibis.literal(i)) for i, t in enumerate(checkpoints)]
    combined = tagged[0] if len(tagged) == 1 else ibis.union(*tagged)
    latest_first = ibis.window(
        group_by=["source_id", "image_id"],
        order_by=[
            combined.resize_timestamp.notnull().desc(),  # noqa: PD004 — ibis API, not pandas
            combined.resize_timestamp.desc(),
            combined["_run_order"].desc(),
        ],
    )
    return (
        combined.mutate(_rank=ibis.row_number().over(latest_first))
        .filter(ibis._["_rank"] == 0)
        .drop("_rank", "_run_order")
        .order_by(["source_id", "image_id"])
    )


def build_manifest(
    images: ibis.Table,
    checkpoint: ibis.Table,
    output_prefix: str,
    threshold: int = 2048,
) -> ibis.Table:
    """Build manifest from images and checkpoint.

    Args:
        images: Images table with [source_id, image_id, width, height, needs_resize, ...]
        checkpoint: Checkpoint table with [source_id, image_id, status, resize_timestamp, error_message]
        output_prefix: S3 prefix for resized images
        threshold: Resize threshold

    Returns:
        Manifest table with columns:
        [source_id, image_id, original_width, original_height, resized_width, resized_height,
         output_s3_key, resize_timestamp, status]
    """
    joined = images.select("source_id", "image_id", "width", "height").inner_join(
        checkpoint.select("source_id", "image_id", "status", "resize_timestamp", "error_message"),
        ["source_id", "image_id"],
    )

    # Only include items that were processed (not skipped because they weren't in needs_resize)
    joined = joined.filter(joined.status.notnull())  # noqa: PD004 — ibis API, not pandas

    # Scale the longest edge down to the threshold, truncating like the resize worker's int()
    longest = ibis.greatest(joined.width, joined.height)
    scale = ibis.literal(threshold).cast("float64") / longest.cast("float64")
    resized_width = ibis.ifelse(
        longest > threshold,
        (joined.width.cast("float64") * scale).floor().cast("int64"),
        joined.width,
    )
    resized_height = ibis.ifelse(
        longest > threshold,
        (joined.height.cast("float64") * scale).floor().cast("int64"),
        joined.height,
    )

    # Key layout must match resize._resized_s3_key_for
    output_s3_key = (
        ibis.literal(f"{output_prefix}/resized/s")
        + joined.source_id.cast("string")
        + "/images/"
        + joined.image_id
        + ".jpg"
    )

    return (
        joined.mutate(
            resized_width=resized_width,
            resized_height=resized_height,
            output_s3_key=output_s3_key,
        )
        .select(
            "source_id",
            "image_id",
            original_width=ibis._.width,
            original_height=ibis._.height,
            resized_width=ibis._.resized_width,
            resized_height=ibis._.resized_height,
            output_s3_key=ibis._.output_s3_key,
            resize_timestamp=ibis._.resize_timestamp,
            status=ibis._.status,
        )
        .order_by(["source_id", "image_id"])
    )
