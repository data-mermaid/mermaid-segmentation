"""DPT head accepts a non-square patch grid when the caller passes it."""

from __future__ import annotations

import pytest
import torch

from mermaidseg.model.models import (
    DPTHead,
    align_optimizer_state_dict,
    align_peft_checkpoint_state_dict,
)


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
    kwargs = dict(
        hidden_size=16,
        out_channels=4,
        token_width=8,
        token_height=8,
        neck_hidden_sizes=(8, 8, 8, 8),
        fusion_hidden_size=8,
        reassemble_factors=(4, 2, 1, 0.5),
    )
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


def test_align_peft_drops_the_legacy_unused_fusion_residual():
    head = _small_head()
    state = dict(head.state_dict())
    legacy_key = "neck.fusion_stage.layers.0.residual_layer1.convolution1.weight"
    state[legacy_key] = torch.zeros(8, 8, 3, 3)

    aligned = align_peft_checkpoint_state_dict(state, head)

    assert legacy_key not in aligned
    assert "neck.fusion_stage.layers.0.projection.weight" in aligned
    assert "neck.fusion_stage.layers.1.residual_layer1.convolution1.weight" in aligned
    head.load_state_dict(aligned)


class _FusionResidual(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.convolution1 = torch.nn.Conv2d(2, 2, kernel_size=3, padding=1)
        self.convolution2 = torch.nn.Conv2d(2, 2, kernel_size=3, padding=1)


class _ResumeToy(torch.nn.Module):
    """LoRA param plus a fusion block shaped like the deleted DPT residual."""

    def __init__(self, keep_residual: bool) -> None:
        super().__init__()
        self.encoder_lora_A = torch.nn.Linear(3, 3, bias=False)
        self.fusion_stage = torch.nn.Module()
        layer0 = torch.nn.Module()
        layer0.projection = torch.nn.Conv2d(2, 2, kernel_size=1)
        if keep_residual:
            layer0.residual_layer1 = _FusionResidual()
        layer0.residual_layer2 = _FusionResidual()
        self.fusion_stage.layers = torch.nn.ModuleList([layer0])
        self.head = torch.nn.Conv2d(2, 1, kernel_size=1)


def _adamw_like_training(model: torch.nn.Module) -> torch.optim.AdamW:
    named = [(name, param) for name, param in model.named_parameters() if param.requires_grad]
    lora = [param for name, param in named if "lora_" in name]
    other = [param for name, param in named if "lora_" not in name]
    return torch.optim.AdamW(
        [{"params": lora, "lr": 1e-5}, {"params": other, "lr": 5e-5}],
        weight_decay=0.01,
    )


def _step_distinct_grads(model: torch.nn.Module, optimizer: torch.optim.Optimizer) -> None:
    for index, param in enumerate(model.parameters()):
        param.grad = torch.full_like(param, float(index + 1))
    optimizer.step()


def _moments_by_name(
    model: torch.nn.Module, optimizer: torch.optim.Optimizer
) -> dict[str, dict[str, torch.Tensor]]:
    named = [(name, param) for name, param in model.named_parameters() if param.requires_grad]
    group_names = [
        [name for name, _ in named if "lora_" in name],
        [name for name, _ in named if "lora_" not in name],
    ]
    state = optimizer.state_dict()
    moments: dict[str, dict[str, torch.Tensor]] = {}
    for names, group in zip(group_names, state["param_groups"], strict=True):
        for name, param_id in zip(names, group["params"], strict=True):
            moments[name] = state["state"][param_id]
    return moments


def test_align_optimizer_drops_removed_residual_slots_without_shifting_moments():
    old = _ResumeToy(keep_residual=True)
    old_opt = _adamw_like_training(old)
    _step_distinct_grads(old, old_opt)
    old_moments = _moments_by_name(old, old_opt)

    saved_opt = old_opt.state_dict()
    saved_opt["param_groups"][1]["lr"] = 1.23e-4
    saved_opt["param_groups"][1]["weight_decay"] = 0.02
    reference = dict(old.state_dict())
    reference["fusion_stage.layers.0.residual_layer1.batch_norm1.running_mean"] = torch.zeros(2)
    reference["fusion_stage.layers.0.residual_layer1.batch_norm1.running_var"] = torch.ones(2)
    reference["fusion_stage.layers.0.residual_layer1.batch_norm1.num_batches_tracked"] = (
        torch.tensor(0)
    )

    new = _ResumeToy(keep_residual=False)
    surviving = {
        key: value for key, value in old.state_dict().items() if "residual_layer1" not in key
    }
    new.load_state_dict(surviving)
    new_opt = _adamw_like_training(new)

    aligned, n_dropped = align_optimizer_state_dict(saved_opt, new_opt, new, reference)

    assert n_dropped == 4
    assert len(aligned["param_groups"][1]["params"]) == len(new_opt.param_groups[1]["params"])
    new_opt.load_state_dict(aligned)

    new_moments = _moments_by_name(new, new_opt)
    for name, slot in new_moments.items():
        assert torch.equal(slot["exp_avg"], old_moments[name]["exp_avg"])
        assert torch.equal(slot["exp_avg_sq"], old_moments[name]["exp_avg_sq"])
    dead = old_moments["fusion_stage.layers.0.residual_layer1.convolution1.weight"]["exp_avg"]
    live = new_moments["fusion_stage.layers.0.residual_layer2.convolution1.weight"]["exp_avg"]
    assert not torch.equal(dead, live)
    assert torch.equal(
        new_moments["fusion_stage.layers.0.projection.weight"]["exp_avg"],
        old_moments["fusion_stage.layers.0.projection.weight"]["exp_avg"],
    )
    assert torch.equal(
        new_moments["head.weight"]["exp_avg"],
        old_moments["head.weight"]["exp_avg"],
    )
    assert new_opt.param_groups[0]["lr"] == 1e-5
    assert new_opt.param_groups[1]["lr"] == 1.23e-4
    assert new_opt.param_groups[1]["weight_decay"] == 0.02


def test_align_optimizer_leaves_a_matching_checkpoint_unchanged():
    model = _ResumeToy(keep_residual=False)
    optimizer = _adamw_like_training(model)
    _step_distinct_grads(model, optimizer)
    before = {
        name: slot["exp_avg"].clone() for name, slot in _moments_by_name(model, optimizer).items()
    }
    saved = optimizer.state_dict()

    aligned, n_dropped = align_optimizer_state_dict(saved, optimizer, model, model.state_dict())

    assert n_dropped == 0
    assert aligned is saved
    optimizer.load_state_dict(aligned)
    after = _moments_by_name(model, optimizer)
    for name, moment in before.items():
        assert torch.equal(after[name]["exp_avg"], moment)


def test_align_optimizer_rejects_a_group_the_removed_residual_does_not_explain():
    old = _ResumeToy(keep_residual=True)
    old_opt = _adamw_like_training(old)
    _step_distinct_grads(old, old_opt)
    saved = old_opt.state_dict()
    saved["param_groups"][1]["params"] = saved["param_groups"][1]["params"][:-1]
    new = _ResumeToy(keep_residual=False)
    new_opt = _adamw_like_training(new)

    with pytest.raises(ValueError, match="doesn't match"):
        align_optimizer_state_dict(saved, new_opt, new, old.state_dict())


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
