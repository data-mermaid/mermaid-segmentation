"""The TrainingMode seam scaffold (deferred #7): documents the intended per-mode adapter
shape."""

from __future__ import annotations

from types import SimpleNamespace

import torch

from mermaidseg.model.training_mode import TrainingMode


class _StandardModeStub:
    """A minimal standard-mode adapter — how the real extraction is expected to look."""

    def predict(self, seg_outputs):
        return seg_outputs.logits, None

    def predict_and_loss(self, seg_outputs, loss_fn, target_labels, target_concepts):
        loss, components = loss_fn(seg_outputs.logits, target_labels)
        return loss, seg_outputs.logits, None, components


def test_conforming_adapter_satisfies_the_seam():
    assert isinstance(_StandardModeStub(), TrainingMode)


def test_incomplete_adapter_does_not_satisfy_the_seam():
    class _MissingLoss:
        def predict(self, seg_outputs):
            return seg_outputs.logits, None

    assert not isinstance(_MissingLoss(), TrainingMode)


def test_stub_matches_current_standard_mode_behavior():
    """Standard mode: predict returns (logits, None); predict_and_loss delegates to the
    loss fn."""
    logits = torch.zeros(1, 3, 2, 2)
    targets = torch.zeros(1, 2, 2, dtype=torch.long)

    def loss_fn(outputs, target_labels):
        return outputs.sum(), {"classification": 0.0}

    adapter = _StandardModeStub()
    outputs, concept_outputs = adapter.predict(SimpleNamespace(logits=logits))
    assert concept_outputs is None
    assert torch.equal(outputs, logits)

    loss, out, concept_out, components = adapter.predict_and_loss(
        SimpleNamespace(logits=logits), loss_fn, targets, None
    )
    assert concept_out is None
    assert components == {"classification": 0.0}
