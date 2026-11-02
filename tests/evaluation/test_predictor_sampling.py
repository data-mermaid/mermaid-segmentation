"""Tests for exact source-resolution point sampling and score upsampling."""

from __future__ import annotations

import numpy as np
import torch
import torch.nn.functional as F

from mermaidseg.evaluation.predictor import CBMPredictor


def test_sample_at_points_equals_interpolate_and_index():
    torch.manual_seed(0)
    c, h, w = 6, 9, 11
    height, width = 71, 83
    maps = torch.randn(c, h, w)
    up = F.interpolate(maps.unsqueeze(0), size=(height, width), mode="bilinear", align_corners=False)[0]
    rows = np.array([0, 5, 40, 70, height - 1, 33])
    cols = np.array([0, 6, 82, 1, width - 1, 50])
    ref = up[:, rows, cols]
    got = CBMPredictor.sample_at_points(maps, rows, cols, height, width)
    assert torch.allclose(ref, got, atol=1e-5)


def test_sample_at_points_corner_pixels():
    maps = torch.arange(2 * 3 * 4, dtype=torch.float32).reshape(2, 3, 4)
    height, width = 30, 40
    up = F.interpolate(maps.unsqueeze(0), size=(height, width), mode="bilinear", align_corners=False)[0]
    rows = np.array([0, height - 1])
    cols = np.array([0, width - 1])
    ref = up[:, rows, cols]
    got = CBMPredictor.sample_at_points(maps, rows, cols, height, width)
    assert torch.allclose(ref, got, atol=1e-5)


def test_upsample_scores_shape_and_argmax_consistency():
    torch.manual_seed(1)
    scores = torch.randn(5, 8, 8)
    up = CBMPredictor.upsample_scores(scores, 32, 40)
    assert up.shape == (5, 32, 40)
    ref = F.interpolate(scores.unsqueeze(0), size=(32, 40), mode="bilinear", align_corners=False)[0]
    assert torch.allclose(up, ref, atol=1e-5)
