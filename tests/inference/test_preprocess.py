"""preprocess() matches the training val transform; tiling reassembles tiles in place."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
import yaml

from mermaidseg.inference import preprocess
from mermaidseg.inference.preprocess import predict_tiled

REPO_ROOT = Path(__file__).resolve().parents[2]


def _val_transform_from_data_config():
    from mermaidseg.io import ConfigDict, preprocess_data_config

    raw = yaml.safe_load((REPO_ROOT / "configs" / "data_config.yaml").read_text())
    data_cfg = preprocess_data_config(ConfigDict({"data": raw["data"]}))
    return data_cfg.data.coralscapes_v2.val.transform


def test_preprocess_equals_the_albumentations_val_transform():
    transform = _val_transform_from_data_config()
    resize = transform.transforms[0]
    image = np.random.default_rng(4).integers(0, 256, size=(300, 410, 3), dtype=np.uint8)

    expected = torch.from_numpy(transform(image=image)["image"]).permute(2, 0, 1)
    got = preprocess(image, (resize.height, resize.width))

    assert got.dtype == torch.float32
    assert got.shape == (3, 256, 256)
    assert (got - expected).abs().max().item() <= 1e-6


def test_predict_tiled_places_each_tile_where_it_came_from():
    # Every tile reports its own red channel, so overlapping tiles agree and the blended
    # map must equal the red channel of the full image.
    image = np.random.default_rng(5).integers(0, 256, size=(50, 70, 3), dtype=np.uint8)

    def red_channel(tile):
        red = torch.from_numpy(tile[..., 0].astype(np.float32))[None]
        return red, red * 2

    class_probs, concept_probs = predict_tiled(red_channel, image, (32, 32))

    expected = image[..., 0].astype(np.float32)
    assert class_probs.shape == (1, 50, 70)
    np.testing.assert_allclose(class_probs[0], expected, atol=1e-3)
    np.testing.assert_allclose(concept_probs[0], expected * 2, atol=1e-3)


def test_predict_tiled_blends_overlaps_by_distance_to_each_tile_edge():
    # Width 52 with 32-wide tiles and 0.2 overlap gives tiles at x=0 and x=20. The first tile
    # predicts 0, the second 1, so an overlap column c holds wB / (wA + wB) with
    # wA = min(c + 1, 32 - c) and wB = min(c - 19, 52 - c).
    image = np.zeros((32, 52, 3), dtype=np.uint8)
    calls = []

    def constant_per_tile(tile):
        value = torch.full((1, 32, 32), float(len(calls)))
        calls.append(tile)
        return value, value

    class_probs, _ = predict_tiled(constant_per_tile, image, (32, 32))

    columns = [0, 19, 20, 26, 31, 32, 51]
    expected = [0.0, 0.0, 1 / 13, 7 / 13, 12 / 13, 1.0, 1.0]
    assert len(calls) == 2
    np.testing.assert_allclose(class_probs[0, 5, columns], expected, atol=1e-6)
