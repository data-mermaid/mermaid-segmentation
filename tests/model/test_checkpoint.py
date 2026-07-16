"""Round-trip tests for the checkpoint format module (slim / normalize / load)."""

from __future__ import annotations

import torch

from mermaidseg.model.checkpoint import (
    frozen_param_names,
    load_into,
    normalize_state_dict_keys,
    slim_state_dict,
)


class _Tiny(torch.nn.Module):
    """A frozen 'backbone' + a trainable 'head', mirroring the frozen-probe shape."""

    def __init__(self):
        super().__init__()
        self.encoder = torch.nn.Linear(4, 4)
        self.head = torch.nn.Linear(4, 2)
        for param in self.encoder.parameters():
            param.requires_grad_(False)


def test_frozen_param_names_lists_only_backbone():
    assert frozen_param_names(_Tiny()) == {"encoder.weight", "encoder.bias"}


def test_slim_drops_frozen_and_sets_flag():
    sd, excludes = slim_state_dict(_Tiny())
    assert excludes is True
    assert not any(k.startswith("encoder.") for k in sd)
    assert {"head.weight", "head.bias"} <= set(sd)


def test_slim_full_keeps_everything():
    sd, excludes = slim_state_dict(_Tiny(), save_full=True)
    assert excludes is False
    assert any(k.startswith("encoder.") for k in sd)


def test_slim_does_not_move_the_model():
    model = _Tiny()
    slim_state_dict(model)
    assert model.encoder.weight.device.type == "cpu"  # unchanged, no hidden .to() side effect


def test_normalize_reconciles_peft_nesting():
    out = normalize_state_dict_keys(
        {
            "encoder.base_model.model.model.q_proj.weight": 1,
            "encoder.model.cls_token": 2,
            "head.weight": 3,
        }
    )
    assert set(out) == {
        "encoder.base_model.model.q_proj.weight",  # collapsed one nesting level
        "encoder.cls_token",  # encoder.model. -> encoder.
        "head.weight",  # untouched
    }


def test_load_into_roundtrip_slim_restores_head_and_flags_no_unexpected():
    src = _Tiny()
    with torch.no_grad():
        src.head.weight.add_(1.0)  # make the head distinguishable
    sd, excludes = slim_state_dict(src)
    ckpt = {"model_state_dict": sd, "excludes_frozen_params": excludes}

    dst = _Tiny()  # fresh: frozen backbone re-initialised
    result = load_into(dst, ckpt)

    assert result.unexpected_keys == []  # every saved key resolved
    assert set(result.missing_keys) == frozen_param_names(dst)  # only frozen missing
    assert torch.equal(dst.head.weight, src.head.weight)  # head restored
    assert not dst.encoder.weight.requires_grad  # frozen state from construction, not load


def test_load_into_bare_state_dict_is_strict():
    result = load_into(_Tiny(), _Tiny().state_dict())
    assert result.missing_keys == [] and result.unexpected_keys == []
