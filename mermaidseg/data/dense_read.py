"""Strict readers for dense-version image and label-mask files."""

from __future__ import annotations

from pathlib import Path

import numpy as np
from numpy.typing import NDArray
from PIL import Image


def read_rgb(path: Path) -> NDArray[np.uint8]:
    """Read an image file as an ``(H, W, 3)`` uint8 RGB array."""
    with Image.open(path) as handle:
        return np.asarray(handle.convert("RGB"), dtype=np.uint8)


def read_label_mask(path: Path, num_ids: int) -> NDArray[np.integer]:
    """Read a mask PNG of native label ids as an ``(H, W)`` array of the file's integer dtype.

    Args:
        path: Mask file; a 3-channel file contributes its first channel.
        num_ids: Number of valid ids; ids run ``0..num_ids - 1``.

    Raises:
        ValueError: The mask holds an id of ``num_ids`` or more.
    """
    with Image.open(path) as handle:
        native = np.asarray(handle)
    if native.ndim == 3:
        native = native[..., 0]
    if native.size and int(native.max()) >= num_ids:
        raise ValueError(
            f"mask {path} has label {int(native.max())}, above the largest id {num_ids - 1}"
        )
    return native
