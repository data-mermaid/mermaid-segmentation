"""ConceptBottleneckLoss takes its config weights explicitly."""

from __future__ import annotations

import pytest

from mermaidseg.model.loss import ConceptBottleneckLoss


def test_concept_bottleneck_loss_requires_lambda_weight():
    with pytest.raises(TypeError, match="lambda_weight"):
        ConceptBottleneckLoss(
            {}, ignore_index=0, per_pixel_loss_weight=0.5, per_image_loss_weight=0.5
        )
