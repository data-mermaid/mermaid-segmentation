"""Fixtures for the model tests: a MetaModel whose forward outputs are fixed numbers."""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

import pytest
import torch
from torch import nn

from mermaidseg.io import ConfigDict
from mermaidseg.model.meta import MetaModel
from mermaidseg.model.models import ConceptBottleneckOutput
from tests.conftest import CBM_CONCEPT_VALUE2ID, NUM_CLASSES, cbm_concept_lookup


class FixedOutputsModel(nn.Module):
    """Returns the same logits for every image, plus a trainable scalar offset.

    The offset is added to every channel, so softmax terms ignore it. It gives the optimizer
    a parameter and the loss a gradient path without changing the hand-derived loss values.
    """

    def __init__(self, class_logits: torch.Tensor, concept_logits: torch.Tensor) -> None:
        super().__init__()
        self.class_logits = class_logits
        self.concept_logits = concept_logits
        self.offset = nn.Parameter(torch.zeros(()))

    def forward(self, images: torch.Tensor) -> ConceptBottleneckOutput:
        batch = images.shape[0]
        logits = self.class_logits.expand(batch, -1, -1, -1) + self.offset
        concept_logits = self.concept_logits.expand(batch, -1, -1, -1) + self.offset
        return ConceptBottleneckOutput(
            logits=logits, concept_logits=concept_logits, concept_probs=concept_logits.detach()
        )


FixedMetaFactory = Callable[..., MetaModel]


@pytest.fixture()
def fixed_meta_model(minimal_config: ConfigDict) -> FixedMetaFactory:
    """Factory for a MetaModel over ``FixedOutputsModel`` (one pixel row of width 2).

    ``class_logits`` are the per-class logits (default zeros), and ``training`` overrides the
    config's training section.
    """

    def _factory(
        class_logits: list[float] | None = None,
        training: dict[str, Any] | None = None,
        **meta_kwargs: Any,
    ) -> MetaModel:
        logits = torch.tensor(class_logits if class_logits is not None else [0.0] * NUM_CLASSES)
        num_concepts = int(cbm_concept_lookup().shape[1])
        model = FixedOutputsModel(
            logits.reshape(1, -1, 1, 1).expand(1, -1, 1, 2),
            torch.zeros(1, num_concepts, 1, 2),
        )
        cfg = minimal_config
        for key, value in (training or {}).items():
            if isinstance(value, dict):
                cfg.training[key].update(value)
            else:
                cfg.training[key] = value
        meta_kwargs.setdefault("concept_value2id", CBM_CONCEPT_VALUE2ID)
        meta_kwargs.setdefault("source_to_concepts_lookup", cbm_concept_lookup())
        meta_kwargs.setdefault("num_concepts", num_concepts)
        meta_kwargs.setdefault("source_to_target_lookup", torch.arange(NUM_CLASSES))
        return MetaModel(
            run_name="fixed",
            num_classes=NUM_CLASSES,
            model_kwargs=cfg.model,
            training_kwargs=cfg.training,
            device="cpu",
            model=model,
            **meta_kwargs,
        )

    return _factory
