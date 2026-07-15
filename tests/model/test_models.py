"""Tests for DPTHead's head_norm option and model-kwargs plumbing."""

from types import SimpleNamespace

import pytest
import torch

import mermaidseg.model.models as mm_models
from mermaidseg.model.models import DPTHead, LinearDPTDINOv3

# Tiny-but-real dimensions: DPTNeck at these sizes builds fast on CPU.
TINY_HEAD_KWARGS = {
    "hidden_size": 32,
    "out_channels": 7,
    "token_width": 4,
    "token_height": 4,
    "neck_hidden_sizes": (16, 16, 16, 16),
    "fusion_hidden_size": 32,
}


def test_default_head_norm_is_group():
    head = DPTHead(**TINY_HEAD_KWARGS)
    assert isinstance(head.head[1], torch.nn.GroupNorm)
    assert not any("running_mean" in key for key in head.state_dict())


def test_head_norm_batch_builds_batchnorm():
    head = DPTHead(**TINY_HEAD_KWARGS, head_norm="batch")
    assert isinstance(head.head[1], torch.nn.BatchNorm2d)
    state_keys = set(head.state_dict())
    assert {
        "head.1.running_mean",
        "head.1.running_var",
        "head.1.num_batches_tracked",
    } <= state_keys


def test_head_norm_batch_strict_round_trip():
    torch.manual_seed(0)
    source = DPTHead(**TINY_HEAD_KWARGS, head_norm="batch")
    # Drive the conv head in train mode so BN running stats become non-trivial.
    source.train()
    source.head(torch.randn(2, 32, 8, 8))

    target = DPTHead(**TINY_HEAD_KWARGS, head_norm="batch")
    target.load_state_dict(source.state_dict())  # strict by default

    torch.testing.assert_close(target.head[1].running_mean, source.head[1].running_mean)
    source.eval()
    target.eval()
    x = torch.randn(1, 32, 8, 8)
    torch.testing.assert_close(target.head(x), source.head(x))


def test_bn_state_dict_into_group_head_raises():
    """The pre-head_norm failure mode: unexpected running-stat keys, no missing keys."""
    batch_head = DPTHead(**TINY_HEAD_KWARGS, head_norm="batch")
    group_head = DPTHead(**TINY_HEAD_KWARGS)
    with pytest.raises(RuntimeError, match="running_mean"):
        group_head.load_state_dict(batch_head.state_dict())


def test_invalid_head_norm_raises():
    with pytest.raises(ValueError, match="head_norm"):
        DPTHead(**TINY_HEAD_KWARGS, head_norm="layer")


class _StubEncoder(torch.nn.Module):
    """Construction-only encoder stand-in: config attributes, no forward."""

    def __init__(self):
        super().__init__()
        self.config = SimpleNamespace(
            hidden_size=32,
            patch_size=16,
            num_hidden_layers=4,
            num_register_tokens=4,
        )


def test_head_norm_reaches_dpt_head_and_not_from_pretrained(monkeypatch):
    """head_norm must be consumed by _DPTDINOv3Base, not leak into
    AutoModel.from_pretrained."""
    captured: dict = {}

    def _fake_from_pretrained(name, **kwargs):
        captured.update(kwargs)
        return _StubEncoder()

    monkeypatch.setattr(
        mm_models, "AutoModel", SimpleNamespace(from_pretrained=_fake_from_pretrained)
    )
    model = LinearDPTDINOv3(
        encoder_name="stub",
        num_classes=3,
        input_size=(64, 64),
        neck_hidden_sizes=(16, 16, 16, 16),
        fusion_hidden_size=32,
        head_norm="batch",
    )
    assert isinstance(model.dpt_head.head[1], torch.nn.BatchNorm2d)
    assert set(captured) <= {"token"}
