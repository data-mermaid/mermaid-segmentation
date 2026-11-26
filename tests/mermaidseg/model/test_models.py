"""ConceptBottleneckDPTLoRADINOv3 encoder loading, trainable parameters and DPTHead grid handling."""

from __future__ import annotations

import math
from typing import Any

import pytest
import torch

import mermaidseg.model.models as models_module
from mermaidseg.dataset_reconciliation.concept_ranks import TAXONOMIC_CONCEPTS
from mermaidseg.model.models import ConceptBottleneckDPTLoRADINOv3, DPTHead
from tests.conftest import CBM_CONCEPT_VALUE2ID, CBM_NUM_CONCEPTS, IMAGE_SIZE


def _model_kwargs(base_encoder: Any, **extra: Any) -> dict[str, Any]:
    return {
        "base_encoder": base_encoder,
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


def test_load_pretrained_encoder_requests_float32_and_hf_token(mock_dinov3_encoder, monkeypatch):
    encoder = mock_dinov3_encoder(models_module)
    calls: list[tuple[tuple[Any, ...], dict[str, Any]]] = []

    def _from_pretrained(*args: Any, **kwargs: Any) -> Any:
        calls.append((args, kwargs))
        return encoder

    monkeypatch.setattr(models_module.AutoModel, "from_pretrained", _from_pretrained)
    monkeypatch.setenv("HF_TOKEN", "hf-test-token")

    assert models_module.load_pretrained_encoder("some/encoder") is encoder
    assert calls == [(("some/encoder",), {"token": "hf-test-token", "dtype": torch.float32})]


def test_model_built_with_an_injected_encoder_never_calls_the_hub(mock_dinov3_encoder, monkeypatch):
    encoder = mock_dinov3_encoder(models_module)

    def _hub_call(*_args: Any, **_kwargs: Any) -> Any:
        raise AssertionError("the model class must not load an encoder from the hub")

    monkeypatch.setattr(models_module.AutoModel, "from_pretrained", _hub_call)

    model = ConceptBottleneckDPTLoRADINOv3(**_model_kwargs(encoder))

    assert model.patch_size == encoder.config.patch_size


def test_cpu_bf16_autocast_keeps_logits_in_fp32(mock_dinov3_encoder):
    encoder = mock_dinov3_encoder(models_module)
    model = ConceptBottleneckDPTLoRADINOv3(**_model_kwargs(encoder))
    images = torch.randn(2, 3, *IMAGE_SIZE)
    with torch.autocast(device_type="cpu", dtype=torch.bfloat16):
        out = model(images)
    assert out.logits.dtype == torch.float32
    assert out.concept_logits.dtype == torch.float32


def test_only_lora_adapters_and_heads_are_trainable(mock_dinov3_encoder):
    encoder = mock_dinov3_encoder(models_module)
    model = ConceptBottleneckDPTLoRADINOv3(**_model_kwargs(encoder))

    names = [name for name, _ in model.named_parameters()]
    lora = {n for n in names if "lora_" in n}
    dpt_head = {n for n in names if n.startswith("dpt_head.")}
    concept_proj = {n for n in names if n.startswith("concept_proj.")}
    concept_classifier = {n for n in names if n.startswith("concept_classifier.")}
    frozen_encoder = {n for n in names if n.startswith("encoder.") and "lora_" not in n}

    for group in (lora, dpt_head, concept_proj, concept_classifier, frozen_encoder):
        assert group
    trainable = {n for n, p in model.named_parameters() if p.requires_grad}
    assert trainable == lora | dpt_head | concept_proj | concept_classifier
    assert not trainable & frozen_encoder


def test_dpt_head_runs_on_nonsquare_grid_when_told_the_shape():
    head = DPTHead(
        hidden_size=16,
        out_channels=4,
        neck_hidden_sizes=(8, 8, 8, 8),
        fusion_hidden_size=8,
        reassemble_factors=(4, 2, 1, 0.5),
    )
    token_height, token_width = 3, 5
    tokens = torch.randn(1, 1 + token_height * token_width, 16)
    hidden = [tokens, tokens, tokens, tokens]

    output = head(hidden, token_height=token_height, token_width=token_width)

    assert output.shape[0] == 1
    assert output.shape[1] == 4
    assert output.shape[-2] != output.shape[-1]


def _small_head(**overrides) -> DPTHead:
    kwargs: dict[str, Any] = {
        "hidden_size": 16,
        "out_channels": 4,
        "neck_hidden_sizes": (8, 8, 8, 8),
        "fusion_hidden_size": 8,
        "reassemble_factors": (4, 2, 1, 0.5),
    }
    kwargs.update(overrides)
    return DPTHead(**kwargs)


def test_dpt_head_drops_the_unused_coarsest_fusion_residual_and_trains_the_rest():
    head = _small_head()
    layers = head.neck.fusion_stage.layers
    assert not hasattr(layers[0], "residual_layer1")
    assert all(layer.residual_layer1 is not None for layer in layers[1:])

    token_height, token_width = 4, 4
    tokens = torch.randn(1, 1 + token_height * token_width, 16)
    output = head([tokens, tokens, tokens, tokens], token_height, token_width)
    output.sum().backward()

    missing = [
        name
        for name, param in head.named_parameters()
        if param.requires_grad and param.grad is None
    ]
    assert missing == []


def test_dpt_head_rejects_a_grid_that_does_not_match_the_tokens():
    head = DPTHead(
        hidden_size=16,
        out_channels=4,
        neck_hidden_sizes=(8,),
        fusion_hidden_size=8,
        reassemble_factors=(1,),
    )
    tokens = torch.randn(1, 1 + 6, 16)
    try:
        head([tokens], token_height=2, token_width=2)
    except ValueError as exc:
        assert "does not match" in str(exc)
    else:
        raise AssertionError("expected a grid/token mismatch")


def test_dpt_head_rejects_the_wrong_number_of_hidden_states():
    head = _small_head()
    tokens = torch.randn(1, 1 + 16, 16)

    with pytest.raises(ValueError, match="expects 4 hidden states, got 3"):
        head([tokens] * 3, token_height=4, token_width=4)


@pytest.mark.parametrize(
    ("images_hw", "expected"),
    [((56, 56), (4, 4)), ((28, 84), (2, 6))],
    ids=["square", "wide"],
)
def test_input_token_grid_divides_each_side_by_the_patch_size(
    mock_dinov3_encoder, images_hw, expected
):
    model = ConceptBottleneckDPTLoRADINOv3(**_model_kwargs(mock_dinov3_encoder(models_module)))

    assert model._input_token_grid(torch.zeros(1, 3, *images_hw)) == expected


@pytest.mark.parametrize("images_hw", [(56, 60), (57, 56)], ids=["width", "height"])
def test_input_token_grid_rejects_a_side_not_divisible_by_the_patch_size(
    mock_dinov3_encoder, images_hw
):
    model = ConceptBottleneckDPTLoRADINOv3(**_model_kwargs(mock_dinov3_encoder(models_module)))

    with pytest.raises(ValueError, match="not divisible by patch size 14"):
        model._input_token_grid(torch.zeros(1, 3, *images_hw))


@pytest.mark.parametrize(
    ("neck_sizes", "factors", "expected"),
    [
        pytest.param([8, 8, 8, 8], [4, 2, 1, 0.5], [1, 2, 3, 4], id="four_stages_of_four_layers"),
        pytest.param([8, 8], [4, 2], [2, 4], id="two_stages"),
    ],
)
def test_default_out_indices_are_evenly_spaced_over_the_encoder_depth(
    mock_dinov3_encoder, neck_sizes, factors, expected
):
    model = ConceptBottleneckDPTLoRADINOv3(
        **_model_kwargs(
            mock_dinov3_encoder(models_module),
            neck_hidden_sizes=neck_sizes,
            reassemble_factors=factors,
        )
    )

    assert model.out_indices == expected


def test_explicit_out_indices_are_kept_and_must_match_the_stage_count(mock_dinov3_encoder):
    encoder = mock_dinov3_encoder(models_module)

    model = ConceptBottleneckDPTLoRADINOv3(**_model_kwargs(encoder, out_indices=[4, 3, 2, 1]))
    assert model.out_indices == [4, 3, 2, 1]

    with pytest.raises(ValueError, match=r"same length \(2 vs 4\)"):
        ConceptBottleneckDPTLoRADINOv3(**_model_kwargs(encoder, out_indices=[1, 3]))


def test_concept_activation_is_a_softmax_per_rank_and_a_sigmoid_on_the_binary_tail(
    mock_dinov3_encoder,
):
    model = ConceptBottleneckDPTLoRADINOv3(**_model_kwargs(mock_dinov3_encoder(models_module)))
    # Rank i scores its two channels (ln(i + 2), 0), so its softmax is ((i + 2) / (i + 3), 1 / (i + 3)).
    ranks = [[math.log(i + 2), 0.0] for i in range(len(TAXONOMIC_CONCEPTS))]
    logits = torch.tensor([value for rank in ranks for value in rank] + [math.log(3.0)])

    activated = model.concept_outputs_activation(logits.reshape(1, -1, 1, 1))

    expected = [p for i in range(6) for p in ((i + 2) / (i + 3), 1 / (i + 3))] + [0.75]
    assert activated.flatten().tolist() == pytest.approx(expected, rel=1e-6)


@pytest.mark.parametrize("detach", [True, False], ids=["detached", "attached"])
def test_detach_concepts_blocks_the_class_loss_from_the_concept_head(mock_dinov3_encoder, detach):
    model = ConceptBottleneckDPTLoRADINOv3(
        **_model_kwargs(mock_dinov3_encoder(models_module), detach_concepts=detach)
    )

    out = model(torch.randn(1, 3, *IMAGE_SIZE))
    out.logits.sum().backward()

    assert model.concept_classifier.weight.grad is not None
    concept_grad = model.concept_proj.weight.grad
    if detach:
        assert concept_grad is None
    else:
        assert concept_grad is not None
        assert concept_grad.abs().sum() > 0
