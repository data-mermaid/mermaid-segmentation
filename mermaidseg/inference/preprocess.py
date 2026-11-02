"""Image preprocessing and tiling for CBM inference, free of albumentations.

``preprocess`` reproduces the training/val ``Resize`` + ``Normalize`` albumentations pipeline
(``configs/data_config*.yaml``) with plain OpenCV and NumPy, so serving and training see the same
pixels.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence

import cv2
import numpy as np
import torch
from numpy.typing import NDArray

IMAGENET_MEAN: tuple[float, float, float] = (0.485, 0.456, 0.406)
IMAGENET_STD: tuple[float, float, float] = (0.229, 0.224, 0.225)
DEFAULT_TILE_OVERLAP = 0.2

TilePredictFn = Callable[[NDArray[np.uint8]], tuple[torch.Tensor | NDArray, torch.Tensor | NDArray]]


def resize_rgb(image_rgb_uint8: NDArray[np.uint8], size_hw: Sequence[int]) -> NDArray[np.uint8]:
    """Resize an HxWx3 image to ``size_hw`` with the interpolation albumentations ``Resize`` uses."""
    height, width = int(size_hw[0]), int(size_hw[1])
    if image_rgb_uint8.shape[:2] == (height, width):
        return image_rgb_uint8
    return cv2.resize(image_rgb_uint8, (width, height), interpolation=cv2.INTER_LINEAR)


def preprocess(
    image_rgb_uint8: NDArray[np.uint8],
    size_hw: Sequence[int],
    mean: Sequence[float] = IMAGENET_MEAN,
    std: Sequence[float] = IMAGENET_STD,
) -> torch.Tensor:
    """Resize + normalize an HxWx3 uint8 RGB image to a ``(3, h, w)`` float32 tensor."""
    if image_rgb_uint8.ndim != 3 or image_rgb_uint8.shape[2] != 3:
        raise ValueError(f"image must be HxWx3 RGB; got shape {image_rgb_uint8.shape}")
    resized = resize_rgb(image_rgb_uint8, size_hw).astype(np.float32) / np.float32(255.0)
    mean_arr = np.asarray(mean, dtype=np.float32)
    std_arr = np.asarray(std, dtype=np.float32)
    normalized = (resized - mean_arr) / std_arr
    return torch.from_numpy(normalized).permute(2, 0, 1).contiguous()


def tile_starts(
    length: int, tile_size: int, min_overlap: float = DEFAULT_TILE_OVERLAP
) -> list[int]:
    """Return tile origin positions along one axis with at least ``min_overlap`` overlap."""
    if length <= tile_size:
        return [0]
    stride = max(1, int(tile_size * (1.0 - min_overlap)))
    last = length - tile_size
    starts = list(range(0, last + 1, stride))
    if starts[-1] != last:
        starts.append(last)
    return starts


def tile_blend_weights(height: int, width: int) -> NDArray[np.float32]:
    """Separable linear distance-to-edge weights for feathering overlapping tiles.

    Each axis weight ramps linearly from the border toward the center (``min(i + 1, size - i)``),
    so two tiles sharing an overlap blend into a single linear ramp, equal at the midpoint.
    """
    wy = np.minimum(np.arange(1, height + 1), np.arange(height, 0, -1)).astype(np.float32)
    wx = np.minimum(np.arange(1, width + 1), np.arange(width, 0, -1)).astype(np.float32)
    return wy[:, None] * wx[None, :]


def _as_numpy(maps: torch.Tensor | NDArray) -> NDArray[np.float32]:
    if isinstance(maps, torch.Tensor):
        maps = maps.detach().cpu().numpy()
    return np.asarray(maps, dtype=np.float32)


def predict_tiled(
    predict_fn: TilePredictFn,
    image_rgb_uint8: NDArray[np.uint8],
    tile_size: Sequence[int],
    *,
    processing_resolution: Sequence[int] | None = None,
    min_overlap: float = DEFAULT_TILE_OVERLAP,
) -> tuple[NDArray[np.float32], NDArray[np.float32]]:
    """Run ``predict_fn`` over overlapping tiles and blend the probabilities.

    ``predict_fn`` maps a ``tile_size`` uint8 RGB tile to ``(class_probs, concept_probs)`` of shape
    ``(C, tile_h, tile_w)``. The image is first resized to ``processing_resolution`` when given.
    Overlaps combine as the ``tile_blend_weights`` ramp-weighted mean of probabilities; the result
    is ``(class_probs, concept_probs)`` at the (processing-resolution) image size.
    """
    image = (
        resize_rgb(image_rgb_uint8, processing_resolution)
        if processing_resolution is not None
        else image_rgb_uint8
    )
    height, width = image.shape[:2]
    tile_height, tile_width = int(tile_size[0]), int(tile_size[1])
    blend = tile_blend_weights(tile_height, tile_width)

    class_acc: NDArray[np.float32] | None = None
    concept_acc: NDArray[np.float32] | None = None
    weight = np.zeros((height, width), dtype=np.float32)

    for y0 in tile_starts(height, tile_height, min_overlap):
        for x0 in tile_starts(width, tile_width, min_overlap):
            patch = image[y0 : y0 + tile_height, x0 : x0 + tile_width]
            valid_h, valid_w = patch.shape[:2]
            tile = np.zeros((tile_height, tile_width, 3), dtype=np.uint8)
            tile[:valid_h, :valid_w] = patch

            tile_class, tile_concept = (_as_numpy(m) for m in predict_fn(tile))
            if class_acc is None or concept_acc is None:
                class_acc = np.zeros((tile_class.shape[0], height, width), dtype=np.float32)
                concept_acc = np.zeros((tile_concept.shape[0], height, width), dtype=np.float32)

            # Zero padding past the image edge is predicted but never accumulated.
            w = blend[:valid_h, :valid_w]
            region = (slice(y0, y0 + valid_h), slice(x0, x0 + valid_w))
            class_acc[(slice(None), *region)] += tile_class[:, :valid_h, :valid_w] * w
            concept_acc[(slice(None), *region)] += tile_concept[:, :valid_h, :valid_w] * w
            weight[region] += w

    assert class_acc is not None and concept_acc is not None
    weight = np.maximum(weight, 1e-6)
    return class_acc / weight, concept_acc / weight
