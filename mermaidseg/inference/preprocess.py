"""Deterministic image preprocessing shared by training, evaluation, and serving.

``preprocess`` reproduces the val ``Resize`` + ``Normalize`` albumentations pipeline built by
``mermaidseg.io.build_eval_transform`` with plain OpenCV and NumPy, so serving and training see
the same pixels. The ImageNet constants here are the single definition both pipelines read.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import cast

import cv2
import numpy as np
import torch
from numpy.typing import NDArray

IMAGENET_MEAN: tuple[float, float, float] = (0.485, 0.456, 0.406)
IMAGENET_STD: tuple[float, float, float] = (0.229, 0.224, 0.225)


def resize_rgb(image_rgb_uint8: NDArray[np.uint8], size_hw: Sequence[int]) -> NDArray[np.uint8]:
    """Resize an HxWx3 image to ``size_hw`` with the interpolation albumentations ``Resize`` uses."""
    height, width = int(size_hw[0]), int(size_hw[1])
    if image_rgb_uint8.shape[:2] == (height, width):
        return image_rgb_uint8
    resized = cv2.resize(image_rgb_uint8, (width, height), interpolation=cv2.INTER_LINEAR)
    return cast("NDArray[np.uint8]", resized)


def preprocess(
    image_rgb_uint8: NDArray[np.uint8],
    size_hw: Sequence[int],
    mean: Sequence[float] = IMAGENET_MEAN,
    std: Sequence[float] = IMAGENET_STD,
) -> torch.Tensor:
    """Resize + normalize an HxWx3 uint8 RGB image to a ``(3, h, w)`` float32 tensor.

    Raises:
        ValueError: If the image is not HxWx3.
    """
    if image_rgb_uint8.ndim != 3 or image_rgb_uint8.shape[2] != 3:
        raise ValueError(f"image must be HxWx3 RGB; got shape {image_rgb_uint8.shape}")
    resized = resize_rgb(image_rgb_uint8, size_hw).astype(np.float32) / np.float32(255.0)
    mean_arr = np.asarray(mean, dtype=np.float32)
    std_arr = np.asarray(std, dtype=np.float32)
    normalized = (resized - mean_arr) / std_arr
    return torch.from_numpy(normalized).permute(2, 0, 1).contiguous()
