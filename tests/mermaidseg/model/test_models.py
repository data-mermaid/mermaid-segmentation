"""ConceptBottleneckDPTLoRADINOv3 takes only the arguments it names."""

from __future__ import annotations

from typing import Any

import pytest
import torch

import mermaidseg.model.models as models_module
from mermaidseg.model.models import ConceptBottleneckDPTLoRADINOv3
from tests.conftest import CBM_CONCEPT_VALUE2ID, CBM_NUM_CONCEPTS, IMAGE_SIZE


def _model_kwargs(**extra: Any) -> dict[str, Any]:
    return {
        "encoder_name": "facebook/dinov3-vits16-pretrain-lvd1689m",
        "num_classes": 4,
        "num_concepts": CBM_NUM_CONCEPTS,
        "input_size": IMAGE_SIZE,
        "concept_value2id": CBM_CONCEPT_VALUE2ID,
        "concept_feature_dim": 8,
        "detach_concepts": True,
        "out_indices": None,
        "neck_hidden_sizes": [8, 8, 8, 8],
        "fusion_hidden_size": 8,
        "reassemble_factors": [4, 2, 1, 0.5],
        "lora_r": 2,
        "lora_alpha": 4,
        "lora_dropout": 0.0,
        "lora_target_modules": ["q_proj"],
        "lora_bias": "none",
        **extra,
    }


def test_unknown_model_kwarg_raises(mock_dinov3_encoder):
    mock_dinov3_encoder(models_module)
    with pytest.raises(TypeError, match="foo"):
        ConceptBottleneckDPTLoRADINOv3(**_model_kwargs(foo=1))


def test_from_pretrained_gets_only_the_encoder_name_and_hf_token(mock_dinov3_encoder, monkeypatch):
    encoder = mock_dinov3_encoder(models_module)
    calls: list[tuple[tuple[Any, ...], dict[str, Any]]] = []

    def _from_pretrained(*args: Any, **kwargs: Any) -> Any:
        calls.append((args, kwargs))
        return encoder

    monkeypatch.setattr(models_module.AutoModel, "from_pretrained", _from_pretrained)
    monkeypatch.setenv("HF_TOKEN", "hf-test-token")

    ConceptBottleneckDPTLoRADINOv3(**_model_kwargs())

    assert calls == [(("facebook/dinov3-vits16-pretrain-lvd1689m",), {"token": "hf-test-token"})]


def test_cpu_bf16_autocast_keeps_logits_in_fp32(mock_dinov3_encoder):
    mock_dinov3_encoder(models_module)
    model = ConceptBottleneckDPTLoRADINOv3(**_model_kwargs())
    images = torch.randn(2, 3, *IMAGE_SIZE)
    with torch.autocast(device_type="cpu", dtype=torch.bfloat16):
        out = model(images)
    assert out.logits.dtype == torch.float32
    assert out.concept_logits.dtype == torch.float32
