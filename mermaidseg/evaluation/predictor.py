"""CBM inference helpers for the evaluation suite.

``CBMPredictor`` and the preprocessing constants live in ``mermaidseg.inference``; they are
re-exported here so evaluation imports stay stable. The albumentations helpers below are
research-side only and stay out of ``mermaidseg.inference``.
"""

from __future__ import annotations

import albumentations as A
import numpy as np
import torch
from numpy.typing import NDArray

from mermaidseg.inference.predictor import FEATURE_KINDS, CBMPredictor
from mermaidseg.inference.preprocess import IMAGENET_MEAN, IMAGENET_STD

__all__ = [
    "FEATURE_KINDS",
    "IMAGENET_MEAN",
    "IMAGENET_STD",
    "CBMPredictor",
    "make_eval_transform",
    "preprocess_image",
]


def make_eval_transform(input_size: tuple[int, int]) -> A.Compose:
    """Albumentations transform matching the demo/val pipeline (Resize + Normalize)."""
    height, width = int(input_size[0]), int(input_size[1])
    return A.Compose(
        [
            A.Resize(height=height, width=width, p=1),
            A.Normalize(mean=list(IMAGENET_MEAN), std=list(IMAGENET_STD)),
        ]
    )


def preprocess_image(image_rgb_uint8: NDArray[np.uint8], transform: A.Compose) -> torch.Tensor:
    """Resize + normalize an HxWx3 uint8 RGB image to a (3, h, w) float tensor."""
    normalized = transform(image=image_rgb_uint8)["image"]
    return torch.from_numpy(normalized).permute(2, 0, 1).contiguous().float()
