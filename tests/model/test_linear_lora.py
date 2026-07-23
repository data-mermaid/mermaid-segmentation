"""Smoke tests for LinearDINOv3 LoRA wiring."""

from __future__ import annotations

import torch

import mermaidseg.model.models as models


def test_linear_lora_only_adapters_and_head_trainable(mock_dinov3_encoder, monkeypatch):
    mock_dinov3_encoder(models)

    def fake_wrap(encoder, **kwargs):
        encoder.register_parameter("lora_q", torch.nn.Parameter(torch.ones(2)))
        return encoder

    monkeypatch.setattr(models, "_wrap_encoder_with_lora", fake_wrap)
    model = models.LinearLoRADINOv3(num_classes=3, input_size=(56, 56))
    assert model.use_lora is True
    trainable = {n for n, p in model.named_parameters() if p.requires_grad}
    assert any(n.endswith("lora_q") or "lora_q" in n for n in trainable)
    assert any(n.startswith("head") for n in trainable)
    non_lora_encoder = [
        n for n, p in model.encoder.named_parameters() if p.requires_grad and "lora_" not in n
    ]
    assert non_lora_encoder == []
