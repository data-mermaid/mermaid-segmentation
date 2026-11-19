"""Training-manifest creation from the annotations, images and resized-object listing."""

from __future__ import annotations

import re
from collections.abc import Iterable

import ibis
import pandas as pd

from .config import RESIZE_THRESHOLD

# Matches resize._resized_s3_key_for: <prefix>/resized/s<source_id>/images/<image_id>.jpg
_RESIZED_KEY_RE = re.compile(r"/resized/s(\d+)/images/(.+)\.jpg$")


def resized_existence_table(keys: Iterable[str]) -> pd.DataFrame:
    """Return a ``(source_id, image_id, status="completed")`` row per resized object key.

    Keys that do not match the resized layout are ignored.

    Args:
        keys: Object keys under ``<prefix>/resized/``, as from :func:`resize.list_resized_keys`.

    Returns:
        DataFrame with columns [source_id, image_id, status].
    """
    rows = []
    for key in keys:
        match = _RESIZED_KEY_RE.search(key)
        if match:
            rows.append((int(match.group(1)), match.group(2)))
    df = pd.DataFrame(rows, columns=["source_id", "image_id"])
    df["status"] = "completed"
    return df


def build_training_manifest(
    annotations: ibis.Table,
    images: ibis.Table,
    resized: ibis.Table,
    output_prefix: str,
) -> ibis.Table:
    """Build the annotation-level training parquet consumed by ``CoralNetDataset``.

    Resolves the S3 key of the image to actually load (resized when the image was resized,
    original otherwise) and rescales each point annotation's ``row``/``col`` to the loaded
    image's dimensions, so the dataset needs no runtime scaling. Images that ``needs_resize``
    but have no resized object are excluded entirely (their annotations are dropped).

    Args:
        annotations: Annotation rows with [source_id, image_id, row, col, coralnet_id].
        images: Image metadata with [source_id, image_id, s3_key, width, height, needs_resize, ...].
        resized: Resized objects on S3 with [source_id, image_id, status], as built by
            :func:`resized_existence_table`.
        output_prefix: S3 prefix under which resized images live (e.g. ``"dev/images"``).

    Returns:
        Annotation-level table with columns:
        [source_id, image_id, row, col, coralnet_id, source_label_name, image_s3_key,
         load_width, load_height, uses_resized_image]. ``row``/``col`` are already scaled
        to (load_height, load_width).
    """
    joined = images.left_join(resized, ["source_id", "image_id"])

    # A resize is usable only when the image needed resizing AND has a completed resized row;
    # images absent from ``resized`` get a null status (-> not completed).
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
    scale = ibis.literal(RESIZE_THRESHOLD).cast("float64") / longest.cast("float64")
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
            & joined.width.notnull()  # noqa: PD004 — ibis API
            & joined.height.notnull()  # noqa: PD004 — ibis API
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
