"""Annotation overrides applied to the raw points.

Overrides were authored against a trainer manifest in resized pixel space (longest edge
``load_edge``); the dataset stores points in original pixel space, so overrides on images that
get resized are mapped back.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import pandas as pd
import pyarrow as pa

from segdata_spec.resize import ResizeSpec, scale_points, scaled_size

_OVERRIDE_COLUMNS = ["image_id", "row", "col", "coralnet_id"]


@dataclass(frozen=True)
class OverrideStats:
    """Counts of what ``apply_overrides`` did."""

    images_overridden: int
    points_exact: int
    points_inverted: int
    points_unscaled: int
    override_images_not_in_raw: int


def _forward(coord: int, orig: int, new: int) -> int:
    """Resized coordinate of ``coord``: the floor of ``coord * (new / orig)`` clamped to the grid."""
    return min(max(math.floor(coord * (new / orig)), 0), new - 1)


def _invert(resized: int, orig: int, new: int) -> int:
    """Smallest original coordinate in ``[0, orig - 1]`` that scales forward to ``resized``.

    ``ceil(resized / factor)`` can miss by one on float error, so the result is nudged both ways.
    """
    factor = new / orig
    coord = min(max(math.ceil(resized / factor), 0), orig - 1)
    while coord > 0 and _forward(coord - 1, orig, new) >= resized:
        coord -= 1
    while coord < orig - 1 and _forward(coord, orig, new) < resized:
        coord += 1
    return coord


def _exact_originals(points: pd.DataFrame, sizes: pd.DataFrame, edge: int) -> pd.DataFrame:
    """Return each resized ``(image_id, row, col, label_id)`` with its smallest original ``(row, col)``.

    ``sizes`` holds ``image_id``, ``width`` and ``height`` of the originals; ``points`` must only
    reference those images. Output columns are ``image_id``, ``s_row``, ``s_col``, ``label_id``,
    ``o_row``, ``o_col``.
    """
    scaled = scale_points(
        pa.Table.from_pandas(points[["image_id", "row", "col"]], preserve_index=False),
        pa.Table.from_pandas(sizes[["image_id", "width", "height"]], preserve_index=False),
        ResizeSpec(max_edge=edge),
    ).to_pandas()
    table = pd.DataFrame(
        {
            "image_id": points["image_id"].to_numpy(),
            "s_row": scaled["row"].to_numpy(),
            "s_col": scaled["col"].to_numpy(),
            "label_id": points["label_id"].to_numpy(),
            "o_row": points["row"].to_numpy(),
            "o_col": points["col"].to_numpy(),
        }
    )
    key = ["image_id", "s_row", "s_col", "label_id"]
    return table.sort_values(["o_row", "o_col"]).drop_duplicates(key, keep="first")


def _map_to_original(
    overrides: pd.DataFrame, points: pd.DataFrame, sizes: pd.DataFrame, edge: int
) -> tuple[pd.DataFrame, int, int]:
    """Map resized-space override rows to original space.

    ``overrides`` has ``image_id``, ``row``, ``col``, ``label_id`` and ``sizes`` has the originals'
    ``width`` and ``height`` per ``image_id``. Returns the mapped rows with ``row`` and ``col`` in
    original space, the exact-match count and the inverted count.
    """
    if overrides.empty:
        return overrides, 0, 0
    raw = points[points["image_id"].isin(set(overrides["image_id"]))]
    exact = _exact_originals(raw, sizes, edge) if len(raw) else None
    merged = overrides.rename(columns={"row": "s_row", "col": "s_col"})
    if exact is None:
        merged = merged.assign(o_row=pd.NA, o_col=pd.NA)
    else:
        merged = merged.merge(exact, on=["image_id", "s_row", "s_col", "label_id"], how="left")
    matched = merged["o_row"].notna()
    merged = merged.merge(sizes, on="image_id")
    rows: list[int] = []
    cols: list[int] = []
    for rec in merged.itertuples(index=False):
        if pd.notna(rec.o_row):
            rows.append(int(rec.o_row))
            cols.append(int(rec.o_col))
            continue
        new_w, new_h = scaled_size(rec.width, rec.height, ResizeSpec(max_edge=edge))
        rows.append(_invert(int(rec.s_row), int(rec.height), new_h))
        cols.append(_invert(int(rec.s_col), int(rec.width), new_w))
    mapped = merged[["image_id", "label_id"]].assign(row=rows, col=cols)
    return mapped, int(matched.sum()), int((~matched).sum())


def apply_overrides(
    points: pd.DataFrame,
    images: pd.DataFrame,
    overrides: pd.DataFrame,
    *,
    load_edge: int = 2048,
) -> tuple[pd.DataFrame, OverrideStats]:
    """Replace every point of each overridden image with that image's override points.

    ``overrides`` (``image_id``, ``row``, ``col``, ``coralnet_id``) are in the space of images
    resized to ``load_edge``; ``images`` (``image_id``, ``width``, ``height``) are the originals.
    Override points on images larger than ``load_edge`` are mapped back to original coordinates:
    a raw point that scales forward to the same position with the same label supplies its exact
    original coordinates (smallest ``(row, col)`` on ties); otherwise the smallest original
    coordinate that scales to the override position is used. Overrides for images absent from
    ``images`` are dropped. Returned points have the columns and dtypes of ``points``.
    """
    ov = overrides[_OVERRIDE_COLUMNS].copy()
    ov["image_id"] = ov["image_id"].astype(str)
    ov = ov.rename(columns={"coralnet_id": "label_id"})
    sizes = images[["image_id", "width", "height"]].assign(image_id=images["image_id"].astype(str))

    known = ov["image_id"].isin(set(sizes["image_id"]))
    not_in_raw = int(ov.loc[~known, "image_id"].nunique())
    ov = ov[known]
    needs_resize = sizes.assign(big=sizes[["width", "height"]].max(axis=1) > load_edge).set_index(
        "image_id"
    )["big"]
    resized = ov["image_id"].map(needs_resize).astype(bool)

    str_points = points.assign(image_id=points["image_id"].astype(str))
    mapped, n_exact, n_inverted = _map_to_original(
        ov[resized].astype({"row": "int64", "col": "int64", "label_id": "int64"}),
        str_points,
        sizes,
        load_edge,
    )
    unscaled = ov[~resized][["image_id", "label_id", "row", "col"]]
    replacement = pd.concat([mapped, unscaled], ignore_index=True)
    replacement["annotator"] = "default"

    overridden = set(ov["image_id"])
    kept = points[~str_points["image_id"].isin(overridden)]
    replacement = replacement[list(points.columns)].astype(points.dtypes.to_dict())
    out = pd.concat([kept, replacement], ignore_index=True)
    stats = OverrideStats(
        images_overridden=len(overridden),
        points_exact=n_exact,
        points_inverted=n_inverted,
        points_unscaled=len(unscaled),
        override_images_not_in_raw=not_in_raw,
    )
    return out, stats
