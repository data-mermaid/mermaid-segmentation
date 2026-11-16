"""Tests for MetaModel's checkpoint-loading strict/non-strict decision.

Regression coverage for a bug where Logger.save_model_checkpoint's frozen-param
exclusion (excludes_frozen_params=True, the default) produced checkpoints that raised
RuntimeError: Missing key(s) in state_dict on load, since the loader always used
strict=True.
"""

from __future__ import annotations

import torch
from torch import nn

from mermaidseg.model.checkpoint import load_into as _load_checkpoint_into_model


def _frozen_backbone_model() -> nn.Module:
    model = nn.Sequential(nn.Linear(4, 4), nn.Linear(4, 2))
    for param in model[0].parameters():
        param.requires_grad = False
    return model


class TestLoadCheckpointIntoModel:
    def test_excludes_frozen_params_uses_non_strict_load(self):
        model = _frozen_backbone_model()
        full_state_dict = model.state_dict()
        frozen_keys = {"0.weight", "0.bias"}
        slimmed_state_dict = {k: v for k, v in full_state_dict.items() if k not in frozen_keys}
        checkpoint = {
            "model_state_dict": slimmed_state_dict,
            "excludes_frozen_params": True,
        }

        # A freshly-constructed model already has the frozen backbone; only the
        # missing trainable keys should be loaded, without raising.
        fresh_model = _frozen_backbone_model()
        _load_checkpoint_into_model(fresh_model, checkpoint)

        for key, value in slimmed_state_dict.items():
            torch.testing.assert_close(fresh_model.state_dict()[key], value)

    def test_full_state_dict_uses_strict_load(self):
        model = _frozen_backbone_model()
        checkpoint = {
            "model_state_dict": model.state_dict(),
            "excludes_frozen_params": False,
        }
        fresh_model = _frozen_backbone_model()
        _load_checkpoint_into_model(fresh_model, checkpoint)

        for key, value in model.state_dict().items():
            torch.testing.assert_close(fresh_model.state_dict()[key], value)

    def test_missing_keys_still_raise_when_not_marked_as_excluding_frozen_params(self):
        model = _frozen_backbone_model()
        full_state_dict = model.state_dict()
        incomplete_state_dict = {
            k: v for k, v in full_state_dict.items() if k not in {"0.weight", "0.bias"}
        }
        checkpoint = {"model_state_dict": incomplete_state_dict}  # no excludes_frozen_params

        fresh_model = _frozen_backbone_model()
        try:
            _load_checkpoint_into_model(fresh_model, checkpoint)
        except RuntimeError as e:
            assert "Missing key" in str(e)
        else:
            raise AssertionError("Expected strict load_state_dict to raise on missing keys")

    def test_legacy_bare_state_dict_without_wrapper_still_loads(self):
        model = _frozen_backbone_model()
        bare_state_dict = model.state_dict()

        fresh_model = _frozen_backbone_model()
        _load_checkpoint_into_model(fresh_model, bare_state_dict)

        for key, value in bare_state_dict.items():
            torch.testing.assert_close(fresh_model.state_dict()[key], value)
