"""DINOv3 + DPT segmentation models, with optional LoRA adapters and a concept bottleneck."""

import math
import os
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from typing import Any, Literal, cast

import torch
from peft import LoraConfig, get_peft_model
from transformers import AutoModel, DPTConfig, PreTrainedModel
from transformers.modeling_outputs import SemanticSegmenterOutput
from transformers.models.dpt.modeling_dpt import DPTNeck

from mermaidseg.dataset_reconciliation.concepts import TAXONOMIC_CONCEPTS


@dataclass
class ConceptBottleneckOutput(SemanticSegmenterOutput):
    """Segmenter output that also carries pre-activation concept logits.

    ``hidden_states`` holds the activated concept outputs in ``[0, 1]`` (per-group softmax for
    taxonomic ranks, sigmoid for the binary tail) that feed the class head and are consumed at
    inference. ``concept_logits`` holds the raw pre-activation logits, used only by the training
    loss so it can run numerically-stable logit-space losses instead of BCE on probabilities.
    """

    concept_logits: torch.Tensor | None = None


def _wrap_encoder_with_lora(
    encoder: torch.nn.Module,
    lora_r: int,
    lora_alpha: int,
    lora_dropout: float,
    lora_target_modules: Sequence[str],
    lora_bias: str,
) -> torch.nn.Module:
    """Wrap a DINOv3 encoder with PEFT LoRA adapters.

    The pretrained backbone weights are frozen and only the injected low-rank adapter matrices
    remain trainable, which keeps the parameter/optimizer footprint small while still adapting the
    attention projections.
    """
    lora_config = LoraConfig(
        r=lora_r,
        lora_alpha=lora_alpha,
        lora_dropout=lora_dropout,
        target_modules=list(lora_target_modules),
        bias=cast("Literal['none', 'all', 'lora_only']", lora_bias),
    )
    return get_peft_model(cast("PreTrainedModel", encoder), lora_config)


# The coarsest fusion block is called with no skip, so this residual never runs.
# Checkpoints saved before it was deleted still contain its weights and Adam slots.
_UNUSED_FUSION_RESIDUAL = "fusion_stage.layers.0.residual_layer1."
_FUSION_RESIDUAL_INSERT_BEFORE = "fusion_stage.layers.0.residual_layer2."
_REMOVED_NORM_BUFFERS = (".running_mean", ".running_var", ".num_batches_tracked")


def _remap_checkpoint_state_dict(
    state_dict: dict[str, torch.Tensor],
    model: torch.nn.Module,
) -> dict[str, torch.Tensor]:
    """Pick the PEFT key layout that overlaps the live module the most.

    PEFT / transformers version skew can change the number of nested ``.model.``
    segments under ``encoder.base_model``. The returned dict keeps every saved
    tensor, including ones the live module no longer has.
    """
    target_keys = set(model.state_dict().keys())

    def _legacy_encoder_prefix(key: str) -> str:
        if key.startswith("encoder.model."):
            return key.replace("encoder.model.", "encoder.", 1)
        return key

    def _collapse_peft_model(key: str) -> str:
        return key.replace(".base_model.model.model.", ".base_model.model.", 1)

    def _expand_peft_model(key: str) -> str:
        marker = "encoder.base_model.model."
        nested = "encoder.base_model.model.model."
        if key.startswith(marker) and not key.startswith(nested):
            return key.replace(marker, nested, 1)
        return key

    def _remap(fn: Callable[[str], str]) -> dict[str, torch.Tensor]:
        return {fn(k): v for k, v in state_dict.items()}

    variants = (
        state_dict,
        _remap(_legacy_encoder_prefix),
        _remap(lambda k: _collapse_peft_model(_legacy_encoder_prefix(k))),
        _remap(lambda k: _expand_peft_model(_legacy_encoder_prefix(k))),
    )
    return max(variants, key=lambda sd: len(set(sd.keys()) & target_keys))


def align_peft_checkpoint_state_dict(
    state_dict: dict[str, torch.Tensor],
    model: torch.nn.Module,
) -> dict[str, torch.Tensor]:
    """Remap saved checkpoint keys to match ``model.state_dict()``.

    Keys for the unused coarsest-fusion residual
    (``fusion_stage.layers.0.residual_layer1``) are dropped so a checkpoint
    saved before that module was removed still loads.
    """
    best_state_dict = _remap_checkpoint_state_dict(state_dict, model)
    target_keys = set(model.state_dict().keys())
    return {
        key: value
        for key, value in best_state_dict.items()
        if _UNUSED_FUSION_RESIDUAL not in key
        and (key in target_keys or not key.endswith(_REMOVED_NORM_BUFFERS))
    }


def _optimizer_group_names(
    optimizer: torch.optim.Optimizer,
    model: torch.nn.Module,
) -> list[list[str]]:
    """Parameter names in each live optimizer group, in group order."""
    name_of = {id(param): name for name, param in model.named_parameters()}
    groups: list[list[str]] = []
    for group in optimizer.param_groups:
        names: list[str] = []
        for param in group["params"]:
            try:
                names.append(name_of[id(param)])
            except KeyError as exc:
                raise ValueError("Optimizer parameter is not registered on the model") from exc
        groups.append(names)
    return groups


def _removed_residual_param_names(state_dict: dict[str, torch.Tensor]) -> list[str]:
    """Parameter names of the deleted residual, in checkpoint registration order."""
    return [
        key
        for key in state_dict
        if _UNUSED_FUSION_RESIDUAL in key and not key.endswith(_REMOVED_NORM_BUFFERS)
    ]


def _group_size_error(saved: int, live: int, detail: str = "") -> ValueError:
    message = (
        "loaded state dict contains a parameter group that doesn't match "
        f"the size of optimizer's group (saved {saved}, live {live})"
    )
    if detail:
        message = f"{message}: {detail}"
    return ValueError(message)


def align_optimizer_state_dict(
    optimizer_state: dict[str, Any],
    optimizer: torch.optim.Optimizer,
    model: torch.nn.Module,
    checkpoint_state_dict: dict[str, torch.Tensor],
) -> tuple[dict[str, Any], int]:
    """Drop Adam slots for the removed coarsest-fusion residual.

    The residual sat in the middle of a parameter group, between
    ``layers.0.projection`` and ``layers.0.residual_layer2``. Slots that still
    correspond to live parameters stay in that order, and the group's learning
    rate and other hyperparameters are copied through unchanged.

    Returns the state dict for ``optimizer.load_state_dict`` and how many
    parameter slots were removed. A checkpoint that already matches the live
    optimizer is returned unchanged with a count of zero.
    """
    saved_groups = optimizer_state["param_groups"]
    live_groups = optimizer.param_groups
    if len(saved_groups) != len(live_groups):
        raise ValueError(
            "loaded state dict has a different number of parameter groups "
            f"(saved {len(saved_groups)}, live {len(live_groups)})"
        )

    live_names_per_group = _optimizer_group_names(optimizer, model)
    if all(
        len(saved["params"]) == len(live_names)
        for saved, live_names in zip(saved_groups, live_names_per_group, strict=True)
    ):
        return optimizer_state, 0

    removed_names = _removed_residual_param_names(
        _remap_checkpoint_state_dict(checkpoint_state_dict, model)
    )
    new_groups: list[dict[str, Any]] = []
    n_dropped = 0
    for saved_group, live_group, live_names in zip(
        saved_groups, live_groups, live_names_per_group, strict=True
    ):
        saved_ids = list(saved_group["params"])
        if len(saved_ids) == len(live_names):
            new_groups.append(saved_group)
            continue
        if len(saved_ids) < len(live_names):
            raise _group_size_error(len(saved_ids), len(live_names))

        insert_at = next(
            (i for i, name in enumerate(live_names) if _FUSION_RESIDUAL_INSERT_BEFORE in name),
            None,
        )
        if insert_at is None:
            raise _group_size_error(
                len(saved_ids),
                len(live_names),
                "could not find fusion_stage.layers.0.residual_layer2 "
                "to place the removed residual",
            )
        old_names = live_names[:insert_at] + removed_names + live_names[insert_at:]
        if len(old_names) != len(saved_ids):
            raise _group_size_error(
                len(saved_ids),
                len(live_names),
                f"reconstructed {len(old_names)} names after accounting for the removed residual",
            )

        kept_ids: list[int] = []
        for param_id, name in zip(saved_ids, old_names, strict=True):
            if _UNUSED_FUSION_RESIDUAL in name:
                n_dropped += 1
                continue
            kept_ids.append(param_id)
        if len(kept_ids) != len(live_group["params"]):
            raise _group_size_error(len(saved_ids), len(live_group["params"]))

        saved_slots = optimizer_state["state"]
        for param_id, param, name in zip(kept_ids, live_group["params"], live_names, strict=True):
            slot = saved_slots.get(param_id)
            if not isinstance(slot, dict):
                continue
            moment = slot.get("exp_avg")
            if isinstance(moment, torch.Tensor) and tuple(moment.shape) != tuple(param.shape):
                raise _group_size_error(
                    len(saved_ids),
                    len(live_names),
                    f"moment shape {tuple(moment.shape)} does not match {name} "
                    f"shape {tuple(param.shape)}",
                )

        new_group = dict(saved_group)
        new_group["params"] = kept_ids
        param_names = new_group.get("param_names")
        if param_names is not None:
            if len(param_names) == len(old_names):
                new_group["param_names"] = [
                    name
                    for name, old_name in zip(param_names, old_names, strict=True)
                    if _UNUSED_FUSION_RESIDUAL not in old_name
                ]
            else:
                new_group.pop("param_names")
        new_groups.append(new_group)

    kept_id_set = {param_id for group in new_groups for param_id in group["params"]}
    saved_id_set = {param_id for group in saved_groups for param_id in group["params"]}
    filtered_state = {
        key: value
        for key, value in optimizer_state["state"].items()
        if key in kept_id_set or key not in saved_id_set
    }
    return {"state": filtered_state, "param_groups": new_groups}, n_dropped


class DPTHead(torch.nn.Module):
    """Dense Prediction Transformer head over selected ViT hidden states.

    Reassembles a handful of intermediate transformer feature maps into a
    multi-scale image-like pyramid (DPT "reassemble" + RefineNet fusion) and
    applies a small conv head, producing dense ``out_channels`` features at a
    fraction of the input resolution.

    Args:
        hidden_size: Channel dimension of the backbone token embeddings.
        out_channels: Number of output channels produced by the head.
        token_width: Width of the patch-token grid.
        token_height: Height of the patch-token grid.
        neck_hidden_sizes: Per-stage channel sizes for the reassemble stage.
        fusion_hidden_size: Channel width of the RefineNet fusion blocks.
        reassemble_factors: Spatial resampling factor per stage.
        dropout: Dropout applied inside the conv head.
    """

    def __init__(
        self,
        hidden_size: int,
        out_channels: int,
        token_width: int,
        token_height: int,
        neck_hidden_sizes: Sequence[int],
        fusion_hidden_size: int,
        reassemble_factors: Sequence[float],
        dropout: float = 0.1,
    ):
        """Build the reassemble stage, fusion stage, and conv head from the arguments."""
        super().__init__()
        self.token_width = token_width
        self.token_height = token_height
        self.num_stages = len(neck_hidden_sizes)

        dpt_config = DPTConfig(
            hidden_size=hidden_size,
            neck_hidden_sizes=list(neck_hidden_sizes),
            reassemble_factors=list(reassemble_factors),
            fusion_hidden_size=fusion_hidden_size,
            readout_type="project",
            is_hybrid=False,
            neck_ignore_stages=[],
        )
        self.neck = DPTNeck(dpt_config)
        # The coarsest fusion block is called with no skip, so its first residual
        # unit never runs. Leaving those weights trainable makes DDP wait for a
        # gradient that never arrives (huggingface/transformers#30633).
        layer0 = self.neck.fusion_stage.layers[0]
        if getattr(layer0, "residual_layer1", None) is not None:
            del layer0.residual_layer1
        norm_groups = min(32, fusion_hidden_size)
        while fusion_hidden_size % norm_groups != 0:
            norm_groups -= 1
        self.head = torch.nn.Sequential(
            torch.nn.Conv2d(
                fusion_hidden_size, fusion_hidden_size, kernel_size=3, padding=1, bias=False
            ),
            torch.nn.GroupNorm(norm_groups, fusion_hidden_size),
            torch.nn.ReLU(inplace=True),
            torch.nn.Dropout(dropout),
            torch.nn.Conv2d(fusion_hidden_size, out_channels, kernel_size=1),
        )

    def forward(
        self,
        hidden_states: Sequence[torch.Tensor],
        token_height: int | None = None,
        token_width: int | None = None,
    ) -> torch.Tensor:
        """Run the DPT neck + conv head.

        Args:
            hidden_states: Sequence of ``num_stages`` token tensors, each of
                shape ``(B, 1 + token_height * token_width, hidden_size)`` with
                the CLS token at index 0 (register tokens already stripped).
            token_height: Patch rows in ``hidden_states``. Required for a grid
                whose aspect differs from the grid stored at construction.
            token_width: Patch columns in ``hidden_states``.

        Returns:
            torch.Tensor: Dense features of shape ``(B, out_channels, H_f, W_f)``.
        """
        if len(hidden_states) != self.num_stages:
            raise ValueError(
                f"DPTHead expects {self.num_stages} hidden states, got {len(hidden_states)}."
            )
        # Prefer the image's real patch grid. Falling back to the sequence length
        # only works when that length factors at the aspect ratio baked in from
        # ``input_size`` (square, for the configs used here).
        if token_height is None or token_width is None:
            token_height, token_width = self._infer_token_grid(hidden_states[0].shape[1] - 1)
        else:
            token_height = int(token_height)
            token_width = int(token_width)
            num_patch_tokens = hidden_states[0].shape[1] - 1
            if token_height * token_width != num_patch_tokens:
                raise ValueError(
                    f"Token grid {token_height}x{token_width} does not match "
                    f"{num_patch_tokens} patch tokens."
                )
        features = self.neck(hidden_states, token_height, token_width)
        return self.head(features[-1])

    def _infer_token_grid(self, num_patch_tokens: int) -> tuple[int, int]:
        """Resolve the (height, width) patch grid for ``num_patch_tokens``.

        Uses the configured token grid when it matches; otherwise reshapes
        ``num_patch_tokens`` while preserving the configured aspect ratio (which
        handles square and non-square inputs at arbitrary resolutions).
        """
        if num_patch_tokens == self.token_height * self.token_width:
            return self.token_height, self.token_width

        aspect = self.token_width / self.token_height
        token_height = int(round(math.sqrt(num_patch_tokens / aspect)))
        token_width = num_patch_tokens // token_height if token_height else 0
        if token_height * token_width != num_patch_tokens:
            raise ValueError(
                f"Cannot reshape {num_patch_tokens} patch tokens into a grid with "
                f"aspect ratio {aspect:.4f} (configured grid "
                f"{self.token_height}x{self.token_width})."
            )
        return token_height, token_width


class _DPTDINOv3Base(torch.nn.Module):
    """Shared backbone/head wiring for the DPT DINOv3 models.

    The encoder is adapted with PEFT LoRA adapters.
    """

    def __init__(
        self,
        *,
        encoder_name: str,
        input_size: tuple[int, int],
        out_indices: Sequence[int] | None,
        neck_hidden_sizes: Sequence[int],
        fusion_hidden_size: int,
        reassemble_factors: Sequence[float],
        head_out_channels: int,
        lora_r: int,
        lora_alpha: int,
        lora_dropout: float,
        lora_target_modules: Sequence[str],
        lora_bias: str,
    ):
        super().__init__()
        base_encoder = AutoModel.from_pretrained(encoder_name, token=os.environ.get("HF_TOKEN"))

        config = base_encoder.config
        hidden_size = config.hidden_size
        patch_size = config.patch_size
        num_register_tokens = getattr(config, "num_register_tokens", 4)
        self.num_prefix_tokens = 1 + num_register_tokens  # CLS + register tokens
        num_hidden_layers = config.num_hidden_layers

        if out_indices is None:
            # 4 evenly spaced blocks across the transformer depth.
            out_indices = [
                round((i + 1) * num_hidden_layers / len(neck_hidden_sizes))
                for i in range(len(neck_hidden_sizes))
            ]
        if len(out_indices) != len(neck_hidden_sizes):
            raise ValueError(
                "out_indices and neck_hidden_sizes must have the same length "
                f"({len(out_indices)} vs {len(neck_hidden_sizes)})."
            )
        # hidden_states has length num_hidden_layers + 1 (index 0 == embeddings).
        self.out_indices = list(out_indices)

        self.encoder = _wrap_encoder_with_lora(
            base_encoder,
            lora_r=lora_r,
            lora_alpha=lora_alpha,
            lora_dropout=lora_dropout,
            lora_target_modules=lora_target_modules,
            lora_bias=lora_bias,
        )

        self.patch_size = patch_size
        self.token_width = input_size[1] // patch_size
        self.token_height = input_size[0] // patch_size

        self.dpt_head = DPTHead(
            hidden_size=hidden_size,
            out_channels=head_out_channels,
            token_width=self.token_width,
            token_height=self.token_height,
            neck_hidden_sizes=neck_hidden_sizes,
            fusion_hidden_size=fusion_hidden_size,
            reassemble_factors=reassemble_factors,
        )

    def _input_token_grid(self, images: torch.Tensor) -> tuple[int, int]:
        """Patch grid of an input batch. Both sides must be divisible by ``patch_size``."""
        height, width = int(images.shape[-2]), int(images.shape[-1])
        patch = int(self.patch_size)
        if height % patch or width % patch:
            raise ValueError(f"Input {(height, width)} is not divisible by patch size {patch}.")
        return height // patch, width // patch

    def _encode(self, x: torch.Tensor) -> list[torch.Tensor]:
        """Run the encoder and return the selected hidden states (CLS + patches).

        The encoder pass keeps gradients so the LoRA adapters can learn.
        """
        outputs = self.encoder(x, output_hidden_states=True)

        hidden_states = outputs.hidden_states
        selected = []
        for idx in self.out_indices:
            hs = hidden_states[idx]
            # Keep CLS (index 0), drop register tokens, keep patch tokens.
            selected.append(torch.cat([hs[:, :1], hs[:, self.num_prefix_tokens :]], dim=1))
        return selected

    def freeze_encoder(self) -> None:
        """Freeze the backbone.

        The pretrained weights are frozen but the low-rank adapters stay
        trainable (so gradients still flow through the encoder).
        """
        for name, param in self.encoder.named_parameters():
            param.requires_grad = "lora_" in name


class _ConceptBottleneckDPTDINOv3(_DPTDINOv3Base):
    """DINOv3 + DPT head feeding a concept bottleneck before classification."""

    def __init__(
        self,
        *,
        num_classes: int,
        num_concepts: int,
        concept_value2id: dict[str, dict[str, list[int]]] | None,
        concept_feature_dim: int,
        detach_concepts: bool,
        encoder_name: str,
        input_size: tuple[int, int],
        out_indices: Sequence[int] | None,
        neck_hidden_sizes: Sequence[int],
        fusion_hidden_size: int,
        reassemble_factors: Sequence[float],
        lora_r: int,
        lora_alpha: int,
        lora_dropout: float,
        lora_target_modules: Sequence[str],
        lora_bias: str,
    ):
        super().__init__(
            head_out_channels=concept_feature_dim,
            encoder_name=encoder_name,
            input_size=input_size,
            out_indices=out_indices,
            neck_hidden_sizes=neck_hidden_sizes,
            fusion_hidden_size=fusion_hidden_size,
            reassemble_factors=reassemble_factors,
            lora_r=lora_r,
            lora_alpha=lora_alpha,
            lora_dropout=lora_dropout,
            lora_target_modules=lora_target_modules,
            lora_bias=lora_bias,
        )
        self.concept_value2id = concept_value2id
        self.detach_concepts = detach_concepts
        self.concept_proj = torch.nn.Conv2d(concept_feature_dim, num_concepts, kernel_size=1)
        self.concept_classifier = torch.nn.Conv2d(num_concepts, num_classes, kernel_size=1)

    def forward(self, x: torch.Tensor) -> ConceptBottleneckOutput:
        """Run encoder + DPT head, the concept bottleneck, and the class classifier."""
        hidden_states = self._encode(x)
        token_height, token_width = self._input_token_grid(x)
        concept_features = self.dpt_head(hidden_states, token_height, token_width)
        concept_features = torch.nn.functional.interpolate(
            concept_features, size=x.shape[-2:], mode="bilinear", align_corners=False
        )
        if concept_features.device.type == "cuda":
            with torch.autocast(device_type="cuda", enabled=False):
                logits, concept_outputs, concept_logits = self._forward_concept_tail(
                    concept_features
                )
        else:
            logits, concept_outputs, concept_logits = self._forward_concept_tail(concept_features)
        return ConceptBottleneckOutput(
            loss=None,
            logits=cast("torch.FloatTensor", logits),
            hidden_states=cast("tuple[torch.FloatTensor, ...]", concept_outputs),
            concept_logits=concept_logits,
        )

    def _forward_concept_tail(
        self, concept_features: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Run concept projection, activation, and class projection in fp32."""
        concept_logits = self.concept_proj(concept_features.float())
        concept_outputs = self.concept_outputs_activation(concept_logits)
        classifier_input = concept_outputs.detach() if self.detach_concepts else concept_outputs
        logits = self.concept_classifier(classifier_input.float())
        return logits, concept_outputs, concept_logits

    def concept_outputs_activation(self, concept_outputs: torch.Tensor) -> torch.Tensor:
        """Apply per-rank softmax to taxonomic channels and sigmoid to the binary tail."""
        offset = 0
        activated_parts = []

        for concept in TAXONOMIC_CONCEPTS:
            # concept_value2id is None only for a model built without concept metadata.
            concept_values = self.concept_value2id[concept]  # type: ignore[index]
            order_concept_length = len(list(concept_values.values())[0])
            chunk = concept_outputs[:, offset : offset + order_concept_length, ...]
            activated_parts.append(torch.softmax(chunk.float(), dim=1))
            offset += order_concept_length

        activated_parts.append(torch.sigmoid(concept_outputs[:, offset:, ...].float()))
        return torch.cat(activated_parts, dim=1)


class ConceptBottleneckDPTLoRADINOv3(_ConceptBottleneckDPTDINOv3):
    """DINOv3 encoder (LoRA-adapted) with a DPT head feeding a concept bottleneck."""

    def __init__(
        self,
        encoder_name: str,
        num_classes: int,
        num_concepts: int,
        input_size: tuple[int, int],
        concept_value2id: dict[str, dict[str, list[int]]] | None,
        concept_feature_dim: int,
        detach_concepts: bool,
        out_indices: Sequence[int] | None,
        neck_hidden_sizes: Sequence[int],
        fusion_hidden_size: int,
        reassemble_factors: Sequence[float],
        lora_r: int,
        lora_alpha: int,
        lora_dropout: float,
        lora_target_modules: Sequence[str],
        lora_bias: str,
    ):
        """Build the concept-bottleneck model with a LoRA-adapted DINOv3 encoder."""
        super().__init__(
            num_classes=num_classes,
            num_concepts=num_concepts,
            concept_value2id=concept_value2id,
            concept_feature_dim=concept_feature_dim,
            detach_concepts=detach_concepts,
            encoder_name=encoder_name,
            input_size=input_size,
            out_indices=out_indices,
            neck_hidden_sizes=neck_hidden_sizes,
            fusion_hidden_size=fusion_hidden_size,
            reassemble_factors=reassemble_factors,
            lora_r=lora_r,
            lora_alpha=lora_alpha,
            lora_dropout=lora_dropout,
            lora_target_modules=lora_target_modules,
            lora_bias=lora_bias,
        )


MODEL_CLASSES: dict[str, type[ConceptBottleneckDPTLoRADINOv3]] = {
    "ConceptBottleneckDPTLoRADINOv3": ConceptBottleneckDPTLoRADINOv3,
}
