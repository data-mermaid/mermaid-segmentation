"""DPT head accepts a non-square patch grid when the caller passes it."""

from __future__ import annotations

from typing import Any

import torch

from mermaidseg.model.models import DPTHead


def test_dpt_head_runs_on_nonsquare_grid_when_told_the_shape():
    head = DPTHead(
        hidden_size=16,
        out_channels=4,
        token_width=8,
        token_height=8,
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
        "token_width": 8,
        "token_height": 8,
        "neck_hidden_sizes": (8, 8, 8, 8),
        "fusion_hidden_size": 8,
        "reassemble_factors": (4, 2, 1, 0.5),
    }
    kwargs.update(overrides)
    return DPTHead(**kwargs)


def test_dpt_head_drops_the_unused_coarsest_fusion_residual_and_trains_the_rest():
    head = _small_head()
    layers = head.neck.fusion_stage.layers
    assert getattr(layers[0], "residual_layer1", None) is None
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
        token_width=4,
        token_height=4,
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
