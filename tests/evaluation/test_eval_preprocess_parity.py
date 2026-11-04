"""The evaluation albumentations preprocessing and the serving preprocess() agree pixel for pixel.

``point_datasets`` evaluates through ``make_eval_transform`` / ``preprocess_image`` while
serving runs ``mermaidseg.inference.preprocess``; a drift between them would skew evaluation
against what is deployed.
"""

from __future__ import annotations

import numpy as np
import pytest

from mermaidseg.evaluation.predictor import make_eval_transform, preprocess_image
from mermaidseg.inference import preprocess


@pytest.mark.parametrize(
    ("source_hw", "size_hw"),
    [((300, 410), (64, 64)), ((40, 50), (96, 128)), ((64, 64), (64, 64))],
    ids=["downscale", "upscale_non_square", "same_size"],
)
def test_eval_transform_matches_the_serving_preprocess(source_hw, size_hw):
    # Catches: a different interpolation, mean/std or scaling on either side.
    image = np.random.default_rng(11).integers(0, 256, size=(*source_hw, 3), dtype=np.uint8)

    evaluated = preprocess_image(image, make_eval_transform(size_hw))
    served = preprocess(image, size_hw)

    assert evaluated.shape == served.shape == (3, *size_hw)
    assert evaluated.dtype == served.dtype
    assert (evaluated - served).abs().max().item() <= 1e-5
