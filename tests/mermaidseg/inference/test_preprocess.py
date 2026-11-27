"""Preprocessing parity between the albumentations val transform and ``inference.preprocess``."""

from __future__ import annotations

import numpy as np
import pytest
import torch

from mermaidseg.inference.preprocess import IMAGENET_MEAN, IMAGENET_STD, preprocess
from mermaidseg.io import build_eval_transform


@pytest.mark.parametrize(
    ("source_hw", "size_hw"),
    [
        pytest.param((100, 130), (48, 80), id="downscale-rect"),
        pytest.param((30, 40), (64, 64), id="upscale"),
        pytest.param((64, 64), (64, 64), id="exact-size"),
    ],
)
def test_preprocess_matches_the_albumentations_eval_transform(source_hw, size_hw):
    image = np.random.default_rng(0).integers(0, 256, (*source_hw, 3), dtype=np.uint8)
    expected = build_eval_transform(size_hw)(image=image)["image"]
    actual = preprocess(image, size_hw, mean=IMAGENET_MEAN, std=IMAGENET_STD)
    assert actual.dtype == torch.float32
    assert actual.shape == (3, *size_hw)
    np.testing.assert_allclose(actual.numpy(), expected.transpose(2, 0, 1), atol=1e-5)


@pytest.mark.parametrize("shape", [(8, 8), (8, 8, 4)])
def test_preprocess_rejects_a_non_hxwx3_array(shape):
    with pytest.raises(ValueError, match="HxWx3"):
        preprocess(np.zeros(shape, dtype=np.uint8), (8, 8), mean=IMAGENET_MEAN, std=IMAGENET_STD)
