"""Restrict point-annotation frames to an ``image_keep`` allowlist.

The allowlist is applied after each dataset's own train filters.
"""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path

import numpy as np
import pandas as pd


def _normalize_value(value: object) -> str:
    """String form of one key. Real numbers drop a trailing ``.0``; strings are kept as written."""
    if isinstance(value, bool | np.bool_):
        return str(value)
    if isinstance(value, int | np.integer):
        return str(int(value))
    if isinstance(value, float | np.floating):
        if not np.isfinite(value) or not np.isclose(value, round(float(value))):
            raise ValueError(f"image key {value!r} is not an integer")
        return str(int(round(float(value))))
    return str(value)


def _normalize_key(series: pd.Series) -> pd.Series:
    """String keys so ``109``, ``\"109\"`` and ``109.0`` name the same image."""
    if series.isna().any():
        raise ValueError("an image key column has missing values")
    return series.map(_normalize_value)


def _row_keys(frame: pd.DataFrame, keys: Sequence[str]) -> pd.Series:
    parts = [_normalize_key(frame[key]) for key in keys]
    key = parts[0]
    for part in parts[1:]:
        key = key.str.cat(part, sep="\0")
    return key


def filter_point_frames(
    df_annotations: pd.DataFrame,
    df_images: pd.DataFrame,
    keep: pd.DataFrame | str | Path,
    dataset: str,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Restrict point-annotation frames to an ``image_keep`` allowlist.

    Every keep key must already be in ``df_images`` (this split, after the dataset's own
    filters). Keys are the allowlist columns other than ``dataset``, and the ``dataset``
    column must be exactly ``dataset``. Images not listed are dropped.
    """
    if not isinstance(keep, pd.DataFrame):
        keep = pd.read_parquet(keep)
    if keep.empty:
        raise ValueError(f"image_keep for {dataset} is empty")
    if "dataset" not in keep.columns:
        raise ValueError("image_keep is missing a dataset column")
    names = set(keep["dataset"].astype(str))
    if names != {dataset}:
        raise ValueError(f"image_keep dataset column is {sorted(names)}, expected {dataset}")
    keys = [column for column in keep.columns if column != "dataset"]
    if not keys:
        raise ValueError("image_keep has no key columns")
    for frame_name, frame in (("annotations", df_annotations), ("images", df_images)):
        missing = [key for key in keys if key not in frame.columns]
        if missing:
            raise ValueError(f"image_keep columns {missing} are not in {dataset} {frame_name}")

    keep_keys = set(_row_keys(keep, keys))
    image_keys = _row_keys(df_images, keys)
    absent = keep_keys - set(image_keys)
    if absent:
        sample = sorted(absent)[:5]
        raise ValueError(
            f"image_keep for {dataset} has {len(absent)} keys that are not in this split, "
            f"e.g. {sample}"
        )
    images = df_images.loc[image_keys.isin(keep_keys)].reset_index(drop=True)
    ann_keys = _row_keys(df_annotations, keys)
    annotations = df_annotations.loc[ann_keys.isin(keep_keys)].reset_index(drop=True)
    if images.empty:
        raise ValueError(f"image_keep for {dataset} matched no images")
    return annotations, images
