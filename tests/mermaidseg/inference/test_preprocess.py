"""Preprocessing parity between the albumentations val transform and ``inference.preprocess``."""

from __future__ import annotations

import cv2
import numpy as np
import pytest
import torch

from mermaidseg.inference.preprocess import IMAGENET_MEAN, IMAGENET_STD, preprocess
from mermaidseg.io import build_eval_transform


@pytest.mark.parametrize(
    ("source_hw", "size_hw"),
    [
        pytest.param((100, 130), (64, 64), id="downscale-square"),
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


def test_imagenet_constants_hold_the_published_values():
    assert IMAGENET_MEAN == (0.485, 0.456, 0.406)
    assert IMAGENET_STD == (0.229, 0.224, 0.225)


def test_preprocess_requires_mean_and_std():
    image = np.zeros((8, 8, 3), dtype=np.uint8)
    with pytest.raises(TypeError):
        preprocess(image, (8, 8))  # type: ignore[call-arg]


@pytest.mark.parametrize("shape", [(8, 8), (8, 8, 4), (1, 8, 8, 3)])
def test_preprocess_rejects_a_non_hxwx3_array(shape):
    with pytest.raises(ValueError, match="HxWx3"):
        preprocess(np.zeros(shape, dtype=np.uint8), (8, 8), mean=IMAGENET_MEAN, std=IMAGENET_STD)


def test_preprocess_applies_the_given_mean_and_std():
    image = np.random.default_rng(0).integers(0, 256, (100, 130, 3), dtype=np.uint8)
    resized = cv2.resize(image, (64, 64), interpolation=cv2.INTER_LINEAR)
    expected = (resized.astype(np.float32) / 255 - 0.5) / 0.25
    actual = preprocess(image, (64, 64), mean=(0.5, 0.5, 0.5), std=(0.25, 0.25, 0.25))
    np.testing.assert_allclose(actual.numpy(), expected.transpose(2, 0, 1), atol=1e-5)
