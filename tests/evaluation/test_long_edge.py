"""Long-edge resize used by the Benthos dense eval."""

from __future__ import annotations

import pytest

from mermaidseg.evaluation.predictor import long_edge_size


def test_square_tiles_stay_square():
    assert long_edge_size(2048, 2048, 336, multiple=16) == (336, 336)
    assert long_edge_size(2048, 2048, 784, multiple=16) == (784, 784)


def test_short_side_keeps_aspect_and_patch_grid():
    # 2048 x 830, long edge 336 -> short side 830 * 336 / 2048 ≈ 136.23 -> 144.
    assert long_edge_size(2048, 830, 336, multiple=16) == (336, 144)
    # 1934 x 2048, long edge 784 -> short side 1934 * 784 / 2048 ≈ 740.32 -> 736.
    assert long_edge_size(1934, 2048, 784, multiple=16) == (736, 784)


def test_long_edge_must_match_patch_grid():
    with pytest.raises(ValueError, match="multiple of 16"):
        long_edge_size(2048, 2048, 300, multiple=16)
