"""Resize specs and the pure image, mask and point transforms that apply them.

The image path decodes only JPEG and PNG and clamps a zero edge to one pixel. It must not
import ``segdata_etl``.
"""

from __future__ import annotations

import hashlib
import io
import json
import math
from typing import Literal

import pyarrow as pa
import pyarrow.compute as pc
from PIL import Image
from pydantic import BaseModel, ConfigDict, Field

_FILTERS: dict[str, Image.Resampling] = {
    "lanczos": Image.Resampling.LANCZOS,
    "bicubic": Image.Resampling.BICUBIC,
    "bilinear": Image.Resampling.BILINEAR,
}


class ResizeSpec(BaseModel):
    """How images are downscaled: longest-edge cap, resampling filter and JPEG quality."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    max_edge: int = Field(gt=0)
    filter: Literal["lanczos", "bicubic", "bilinear"] = "lanczos"
    jpeg_quality: int = Field(default=95, ge=1, le=95)

    @property
    def id(self) -> str:
        """First 12 hex chars of the sha256 of the canonical JSON of the three fields."""
        canonical = json.dumps(self.model_dump(), sort_keys=True, separators=(",", ":"))
        return hashlib.sha256(canonical.encode()).hexdigest()[:12]


def scaled_size(width: int, height: int, spec: ResizeSpec) -> tuple[int, int]:
    """Return ``(width, height)`` scaled so the longest edge is at most ``spec.max_edge``.

    The scale is ``max_edge / longest`` multiplied into each edge and truncated, in that float
    order, which gives 2047 for a longest edge of 2215 at ``max_edge=2048``; the ETL computes the
    same way, so the sizes stay byte-for-byte aligned. Each edge is at least one pixel; this
    differs from the ETL only where the ETL would produce a zero edge and fail.
    """
    longest = max(width, height)
    if longest <= spec.max_edge:
        return width, height
    scale = spec.max_edge / longest
    return max(1, int(width * scale)), max(1, int(height * scale))


def _draft_to_max_edge(img: Image.Image, max_edge: int) -> None:
    """Ask the JPEG decoder for a reduced scale that keeps the longest edge >= ``max_edge``."""
    if img.format != "JPEG":
        return
    width, height = img.size
    longest = max(width, height)
    if longest <= max_edge:
        return
    scale = max_edge / longest
    img.draft(img.mode, (math.ceil(width * scale), math.ceil(height * scale)))


def resize_image(data: bytes, spec: ResizeSpec) -> bytes:
    """Return ``data`` downscaled per ``spec`` as JPEG, or unchanged when no re-encode is needed.

    A JPEG in RGB or L mode within ``spec.max_edge`` passes through. Any other image is
    re-encoded as JPEG, converted to RGB when its mode is not RGB or L. An image above twice Pillow's
    ``Image.MAX_IMAGE_PIXELS`` raises ``DecompressionBombError``; one between the limit and
    twice it only warns. Only JPEG and PNG input is decoded; any other format raises
    ``PIL.UnidentifiedImageError``.
    """
    img: Image.Image = Image.open(io.BytesIO(data), formats=("JPEG", "PNG"))
    original_size = img.size  # header size, read before a JPEG draft shrinks it
    _draft_to_max_edge(img, spec.max_edge)
    img.load()

    needs_resize = max(original_size) > spec.max_edge
    if not needs_resize and img.mode in ("RGB", "L") and img.format == "JPEG":
        return data

    if needs_resize:
        img = img.resize(scaled_size(*original_size, spec), _FILTERS[spec.filter])
    if img.mode not in ("RGB", "L"):
        img = img.convert("RGB")

    output = io.BytesIO()
    img.save(output, format="JPEG", quality=spec.jpeg_quality)
    return output.getvalue()


def resize_mask(data: bytes, size: tuple[int, int]) -> bytes:
    """Return the mask PNG resized to ``size`` with nearest-neighbour, keeping its mode.

    Only PNG input is decoded; any other format raises ``PIL.UnidentifiedImageError``.
    """
    mask = Image.open(io.BytesIO(data), formats=("PNG",))
    if mask.size == size:
        return data
    output = io.BytesIO()
    mask.resize(size, Image.Resampling.NEAREST).save(output, format="PNG")
    return output.getvalue()


def _scaled_sizes(images: pa.Table, spec: ResizeSpec) -> tuple[list[int], list[int]]:
    """Return the scaled ``(widths, heights)`` of ``images``, in row order."""
    sizes = [
        scaled_size(w, h, spec)
        for w, h in zip(images["width"].to_pylist(), images["height"].to_pylist(), strict=True)
    ]
    return [w for w, _ in sizes], [h for _, h in sizes]


def scale_images(images: pa.Table, spec: ResizeSpec) -> pa.Table:
    """Return ``images`` with ``width`` and ``height`` scaled per ``spec`` and JPEG media types.

    Other columns, the schema and the field types are preserved.
    """
    widths, heights = _scaled_sizes(images, spec)
    replacements = {
        "width": widths,
        "height": heights,
        "media_type": ["image/jpeg"] * images.num_rows,
    }
    for name, values in replacements.items():
        field = images.schema.field(name)
        index = images.schema.get_field_index(name)
        images = images.set_column(index, field, pa.array(values, field.type))
    return images


def scale_points(points: pa.Table, images: pa.Table, spec: ResizeSpec) -> pa.Table:
    """Return ``points`` with ``row`` and ``col`` moved into the resized image's pixel grid.

    ``images`` supplies ``image_id``, ``width`` and ``height`` of the originals. Each coordinate
    becomes ``floor(coord * (new / orig))`` clamped to ``[0, new - 1]``, with the ratio a float64,
    as the ETL manifest computes it. Row order and column types are preserved.

    Raises:
        ValueError: A point references an ``image_id`` absent from ``images``.
    """
    widths, heights = _scaled_sizes(images, spec)
    index = pc.index_in(points["image_id"], value_set=images["image_id"])
    if index.null_count:
        raise ValueError("points reference image_ids missing from images")

    def _axis(column: str, orig_name: str, new_values: list[int]) -> pa.ChunkedArray:
        orig = pc.take(images[orig_name].cast(pa.float64()), index)
        new = pc.take(pa.array(new_values, pa.float64()), index)
        scaled = pc.floor(pc.multiply(points[column].cast(pa.float64()), pc.divide(new, orig)))
        clamped = pc.min_element_wise(pc.max_element_wise(scaled, 0.0), pc.subtract(new, 1.0))
        return clamped.cast(points.schema.field(column).type)

    out = points
    for column, orig_name, new_values in (("row", "height", heights), ("col", "width", widths)):
        out = out.set_column(
            out.schema.get_field_index(column),
            points.schema.field(column),
            _axis(column, orig_name, new_values),
        )
    return out
