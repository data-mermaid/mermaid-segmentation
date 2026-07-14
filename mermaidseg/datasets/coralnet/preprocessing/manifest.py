"""Manifest creation for resized images."""

from __future__ import annotations

import logging
from collections.abc import Sequence

import ibis

logger = logging.getLogger(__name__)


def combine_checkpoints(checkpoints: Sequence[ibis.Table]) -> ibis.Table:
    """Merge resize checkpoints from successive runs into one status table.

    The row with the most recent ``resize_timestamp`` wins on (source_id, image_id)
    conflicts, so a rerun that reprocessed previously skipped/failed items overrides the
    earlier status. Rows with no timestamp (failed/skipped) lose to any timestamped row;
    among themselves, the later checkpoint in the input sequence wins. All checkpoints
    must share the same schema.
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


def build_training_manifest(
    annotations: ibis.Table,
    images: ibis.Table,
    checkpoint: ibis.Table,
    output_prefix: str,
    threshold: int = 2048,
) -> ibis.Table:
    """Build the annotation-level training parquet consumed by ``CoralNetDataset``.

    Resolves the S3 key of the image to actually load (resized when the image was resized,
    original otherwise) and rescales each point annotation's ``row``/``col`` to the loaded
    image's dimensions, so the dataset needs no runtime scaling. Images that ``needs_resize``
    but whose resize did not complete are excluded entirely (their annotations are dropped).

    Args:
        annotations: Annotation rows with [source_id, image_id, row, col, coralnet_id].
        images: Image metadata with [source_id, image_id, s3_key, width, height, needs_resize, ...].
        checkpoint: Resize checkpoint/manifest with [source_id, image_id, status, ...].
        output_prefix: S3 prefix under which resized images live (e.g. ``"dev/images"``).
        threshold: Resize threshold (longest edge in pixels) used when the corpus was resized.

    Returns:
        Annotation-level table with columns:
        [source_id, image_id, row, col, coralnet_id, source_label_name, image_s3_key,
         load_width, load_height, uses_resized_image]. ``row``/``col`` are already scaled
        to (load_height, load_width).
    """
    joined = images.left_join(checkpoint, ["source_id", "image_id"])

    # Treat a resize as usable only when the image needed resizing AND its checkpoint row
    # completed; images absent from the checkpoint get a null status (-> not completed).
    use_resized = joined.needs_resize & (joined.status.fill_null("") == "completed")  # noqa: PD004 — ibis API

    resized_key = (
        ibis.literal(f"{output_prefix}/resized/s")
        + joined.source_id.cast("string")
        + "/images/"
        + joined.image_id
        + ".jpg"
    )

    # Scale the longest edge to the threshold, flooring like the resize worker's int().
    longest = ibis.greatest(joined.width, joined.height)
    scale = ibis.literal(threshold).cast("float64") / longest.cast("float64")
    load_width = ibis.ifelse(
        use_resized, (joined.width.cast("float64") * scale).floor().cast("int64"), joined.width
    )
    load_height = ibis.ifelse(
        use_resized, (joined.height.cast("float64") * scale).floor().cast("int64"), joined.height
    )

    image_manifest = (
        joined.mutate(
            image_s3_key=ibis.ifelse(use_resized, resized_key, joined.s3_key),
            load_width=load_width,
            load_height=load_height,
            uses_resized_image=use_resized,
        )
        # Keep sub-threshold or completed-resize images, and only those with known original
        # dimensions — null dims (e.g. header_status="not_found") can't be scaled or validated,
        # and DuckDB's greatest(NULL, 0) would silently collapse their coords to (0, 0).
        .filter(
            ((~joined.needs_resize) | use_resized)  # noqa: PD004 — ibis API
            & joined.width.notnull()
            & joined.height.notnull()
            & (joined.width > 0)
            & (joined.height > 0)
        )
        .select(
            "source_id",
            "image_id",
            "image_s3_key",
            "load_width",
            "load_height",
            "uses_resized_image",
            orig_width=ibis._.width,
            orig_height=ibis._.height,
        )
    )

    # Inner join drops annotations of excluded images.
    out = annotations.inner_join(image_manifest, ["source_id", "image_id"])
    scale_x = out.load_width.cast("float64") / out.orig_width.cast("float64")
    scale_y = out.load_height.cast("float64") / out.orig_height.cast("float64")
    col_scaled = ibis.least(
        (out.col.cast("float64") * scale_x).floor().cast("int64"), out.load_width - 1
    )
    row_scaled = ibis.least(
        (out.row.cast("float64") * scale_y).floor().cast("int64"), out.load_height - 1
    )

    # row/col and load dims are bounded by the resize threshold (<= 2048), so int16 stores them
    # losslessly at a quarter of int64 — they dominate the file size at annotation scale.
    return (
        out.mutate(
            row=ibis.greatest(row_scaled, ibis.literal(0)).cast("int16"),
            col=ibis.greatest(col_scaled, ibis.literal(0)).cast("int16"),
            load_width=out.load_width.cast("int16"),
            load_height=out.load_height.cast("int16"),
            source_label_name=out.coralnet_id.cast("string"),
        )
        .select(
            "source_id",
            "image_id",
            "row",
            "col",
            "coralnet_id",
            "source_label_name",
            "image_s3_key",
            "load_width",
            "load_height",
            "uses_resized_image",
        )
        .order_by(["source_id", "image_id", "row", "col"])
    )
