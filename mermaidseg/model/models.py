"""DINOv3 + DPT segmentation models, with optional LoRA adapters and a concept bottleneck."""

import os
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Literal, cast

import torch
from peft import LoraConfig, get_peft_model
from transformers import AutoModel, DPTConfig, PreTrainedModel
from transformers.models.dpt.modeling_dpt import DPTNeck

from mermaidseg.dataset_reconciliation.concept_ranks import binary_offset, rank_slices


@dataclass
class ConceptBottleneckOutput:
    """Model output: class logits plus the concept bottleneck's pre- and post-activation maps.

    ``concept_probs`` holds the activated concept outputs in ``[0, 1]`` (per-group softmax for
    taxonomic ranks, sigmoid for the binary tail) that feed the class head and are consumed at
    inference. ``concept_logits`` holds the raw pre-activation logits, used only by the training
    loss so it can run numerically-stable logit-space losses instead of BCE on probabilities.
    """

    logits: torch.Tensor
    concept_logits: torch.Tensor
    concept_probs: torch.Tensor


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
        neck_hidden_sizes: Per-stage channel sizes for the reassemble stage.
        fusion_hidden_size: Channel width of the RefineNet fusion blocks.
        reassemble_factors: Spatial resampling factor per stage.
        dropout: Dropout applied inside the conv head.
    """

    def __init__(
        self,
        hidden_size: int,
        out_channels: int,
        neck_hidden_sizes: Sequence[int],
        fusion_hidden_size: int,
        reassemble_factors: Sequence[float],
        dropout: float = 0.1,
    ):
        """Build the reassemble stage, fusion stage, and conv head from the arguments."""
        super().__init__()

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
        self, hidden_states: Sequence[torch.Tensor], token_height: int, token_width: int
    ) -> torch.Tensor:
        """Run the DPT neck + conv head.

        Args:
            hidden_states: One token tensor per neck stage, each of
                shape ``(B, 1 + token_height * token_width, hidden_size)`` with
                the CLS token at index 0 (register tokens already stripped).
            token_height: Patch rows in ``hidden_states``.
            token_width: Patch columns in ``hidden_states``.

        Returns:
            torch.Tensor: Dense features of shape ``(B, out_channels, H_f, W_f)``.
        """
        features = self.neck(hidden_states, token_height, token_width)
        return self.head(features[-1])


class ConceptBottleneckDPTLoRADINOv3(torch.nn.Module):
    """DINOv3 encoder (LoRA-adapted) with a DPT head feeding a concept bottleneck.

    ``input_size`` is accepted because it is part of the bundle contract (``model.json``); the
    forward pass derives the patch grid from each input batch instead.
    """

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
        super().__init__()
        config = base_encoder.config
        self.num_prefix_tokens = 1 + config.num_register_tokens  # CLS + register tokens
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

        self.patch_size = config.patch_size
        self.dpt_head = DPTHead(
            hidden_size=config.hidden_size,
            out_channels=concept_feature_dim,
            neck_hidden_sizes=neck_hidden_sizes,
            fusion_hidden_size=fusion_hidden_size,
            reassemble_factors=reassemble_factors,
        )
        self.concept_value2id = concept_value2id
        self.detach_concepts = detach_concepts
        self.concept_proj = torch.nn.Conv2d(concept_feature_dim, num_concepts, kernel_size=1)
        self.concept_classifier = torch.nn.Conv2d(num_concepts, num_classes, kernel_size=1)

    def _input_token_grid(self, images: torch.Tensor) -> tuple[int, int]:
        """Patch grid of an input batch. Both sides must be divisible by ``patch_size``."""
        height, width = int(images.shape[-2]), int(images.shape[-1])
        patch = int(self.patch_size)
        if height % patch or width % patch:
            raise ValueError(f"Input {(height, width)} is not divisible by patch size {patch}.")
        return height // patch, width // patch

    def forward(self, x: torch.Tensor) -> ConceptBottleneckOutput:
        """Run encoder + DPT head, the concept bottleneck, and the class classifier."""
        token_height, token_width = self._input_token_grid(x)

        # The encoder pass keeps gradients so the LoRA adapters can learn.
        encoder_outputs = self.encoder(x, output_hidden_states=True)
        selected = []
        for idx in self.out_indices:
            hs = encoder_outputs.hidden_states[idx]
            # Keep CLS (index 0), drop register tokens, keep patch tokens.
            selected.append(torch.cat([hs[:, :1], hs[:, self.num_prefix_tokens :]], dim=1))

        concept_features = self.dpt_head(selected, token_height, token_width)
        concept_features = torch.nn.functional.interpolate(
            concept_features, size=x.shape[-2:], mode="bilinear", align_corners=False
        )
        with torch.autocast(device_type=concept_features.device.type, enabled=False):
            concept_logits = self.concept_proj(concept_features.to(self.dtype))
            concept_probs = self.concept_outputs_activation(concept_logits)
            classifier_input = concept_probs.detach() if self.detach_concepts else concept_probs
            logits = self.concept_classifier(classifier_input.to(self.dtype))
        return ConceptBottleneckOutput(
            logits=logits, concept_logits=concept_logits, concept_probs=concept_probs
        )

    def concept_outputs_activation(self, concept_outputs: torch.Tensor) -> torch.Tensor:
        """Apply per-rank softmax to taxonomic channels and sigmoid to the binary tail."""
        activated_parts = [
            torch.softmax(concept_outputs[:, offset : offset + width, ...].to(self.dtype), dim=1)
            for _, offset, width in rank_slices(self.concept_value2id)
        ]
        tail = binary_offset(self.concept_value2id)
        activated_parts.append(torch.sigmoid(concept_outputs[:, tail:, ...].to(self.dtype)))
        return torch.cat(activated_parts, dim=1)


MODEL_CLASSES: dict[str, type[ConceptBottleneckDPTLoRADINOv3]] = {
    "ConceptBottleneckDPTLoRADINOv3": ConceptBottleneckDPTLoRADINOv3,
}
