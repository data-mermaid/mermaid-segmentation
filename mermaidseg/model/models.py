"""DINOv3 + DPT segmentation models, with optional LoRA adapters and a concept bottleneck."""

import math
import os
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Literal, cast

import torch
from peft import LoraConfig, get_peft_model
from transformers import AutoModel, DPTConfig, PreTrainedModel
from transformers.modeling_outputs import SemanticSegmenterOutput
from transformers.models.dpt.modeling_dpt import DPTNeck

from mermaidseg.dataset_reconciliation.concept_ranks import TAXONOMIC_CONCEPTS


@dataclass
class ConceptBottleneckOutput(SemanticSegmenterOutput):
    """Segmenter output that also carries pre-activation concept logits.

    ``hidden_states`` holds the activated concept outputs in ``[0, 1]`` (per-group softmax for
    taxonomic ranks, sigmoid for the binary tail) that feed the class head and are consumed at
    inference. ``concept_logits`` holds the raw pre-activation logits, used only by the training
    loss so it can run numerically-stable logit-space losses instead of BCE on probabilities.
    """

    concept_logits: torch.Tensor | None = None


def load_pretrained_encoder(encoder_name: str) -> PreTrainedModel:
    """Load the pretrained base encoder from the HF hub in float32, the one place the hub is called."""
    return AutoModel.from_pretrained(
        encoder_name, token=os.environ.get("HF_TOKEN"), dtype=torch.float32
    )


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

    concept_proj: torch.nn.Conv2d

    @property
    def dtype(self) -> torch.dtype:
        """Dtype of the concept head parameters, which the concept bottleneck runs in.

        This is fp32 for training.
        """
        return self.concept_proj.weight.dtype

    def __init__(
        self,
        *,
        base_encoder: PreTrainedModel,
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
        # The pretrained encoder weights stay frozen; only the LoRA adapters train.
        for name, param in self.encoder.named_parameters():
            param.requires_grad = "lora_" in name

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


class _ConceptBottleneckDPTDINOv3(_DPTDINOv3Base):
    """DINOv3 + DPT head feeding a concept bottleneck before classification."""

    def __init__(
        self,
        *,
        num_classes: int,
        num_concepts: int,
        concept_value2id: dict[str, dict[str, list[int]]],
        concept_feature_dim: int,
        detach_concepts: bool,
        base_encoder: PreTrainedModel,
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
            base_encoder=base_encoder,
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
        with torch.autocast(device_type=concept_features.device.type, enabled=False):
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
        """Run concept projection, activation, and class projection in ``self.dtype``."""
        concept_logits = self.concept_proj(concept_features.to(self.dtype))
        concept_outputs = self.concept_outputs_activation(concept_logits)
        classifier_input = concept_outputs.detach() if self.detach_concepts else concept_outputs
        logits = self.concept_classifier(classifier_input.to(self.dtype))
        return logits, concept_outputs, concept_logits

    def concept_outputs_activation(self, concept_outputs: torch.Tensor) -> torch.Tensor:
        """Apply per-rank softmax to taxonomic channels and sigmoid to the binary tail."""
        offset = 0
        activated_parts = []

        for concept in TAXONOMIC_CONCEPTS:
            concept_values = self.concept_value2id[concept]
            order_concept_length = len(list(concept_values.values())[0])
            chunk = concept_outputs[:, offset : offset + order_concept_length, ...]
            activated_parts.append(torch.softmax(chunk.to(self.dtype), dim=1))
            offset += order_concept_length

        activated_parts.append(torch.sigmoid(concept_outputs[:, offset:, ...].to(self.dtype)))
        return torch.cat(activated_parts, dim=1)


class ConceptBottleneckDPTLoRADINOv3(_ConceptBottleneckDPTDINOv3):
    """DINOv3 encoder (LoRA-adapted) with a DPT head feeding a concept bottleneck."""

    def __init__(
        self,
        *,
        base_encoder: PreTrainedModel,
        num_classes: int,
        num_concepts: int,
        input_size: tuple[int, int],
        concept_value2id: dict[str, dict[str, list[int]]],
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
            base_encoder=base_encoder,
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
