import os
from collections.abc import Sequence
from typing import Any

import torch
from peft import LoraConfig, get_peft_model
from transformers import AutoModel, DPTConfig
from transformers.modeling_outputs import SemanticSegmenterOutput
from transformers.models.dpt.modeling_dpt import DPTNeck

from mermaidseg.dataset_reconciliation.concepts import TAXONOMIC_CONCEPTS

DEFAULT_LORA_TARGET_MODULES = ("q_proj", "k_proj", "v_proj", "o_proj")


class LinearClassifier(torch.nn.Module):
    """A linear classifier module that performs pixel-wise classification on reshaped embeddings.

    This module takes input embeddings, reshapes them to a 2D spatial format, and applies
    a 1x1 convolution to perform classification. It's commonly used as a classification
    head for segmentation tasks.
    Args:
        in_channels (int): Number of input channels in the embeddings.
        tokenW (int, optional): Width of the token/patch grid. Defaults to 32.
        tokenH (int, optional): Height of the token/patch grid. Defaults to 32.
        num_labels (int, optional): Number of output classes/labels. Defaults to 1.
    Attributes:
        in_channels (int): Number of input channels.
        width (int): Width dimension for reshaping.
        height (int): Height dimension for reshaping.
        classifier (torch.nn.Conv2d): 1x1 convolution layer for classification.
    Forward:
        Args:
            embeddings (torch.Tensor): Input embeddings tensor to be classified.
        Returns:
            torch.Tensor: Classification output with shape (batch_size, num_labels, tokenH, tokenW).
    """

    def __init__(
        self,
        in_channels: int,
        token_width: int = 32,
        token_height: int = 32,
        num_labels: int = 1,
    ):
        super().__init__()

        self.in_channels = in_channels
        self.width = token_width
        self.height = token_height
        self.classifier = torch.nn.Conv2d(in_channels, num_labels, (1, 1))

    def forward(self, embeddings: torch.Tensor) -> torch.Tensor:
        """Reshape token embeddings and apply the 1×1 classification conv.

        Args:
            embeddings (torch.Tensor): Flattened patch embeddings with shape
                (B, token_height * token_width, in_channels).
        Returns:
            torch.Tensor: Per-pixel logits with shape (B, num_labels, token_height, token_width).
        """
        embeddings = embeddings.reshape(-1, self.height, self.width, self.in_channels)
        embeddings = embeddings.permute(0, 3, 1, 2)

        return self.classifier(embeddings)


class ConceptHead(torch.nn.Module):
    """A concept classification head module that performs pixel-wise classification on reshaped
    embeddings.

    This module takes input embeddings, reshapes them to a 2D spatial format, and
    applies a 1x1 convolution to perform classification. It's commonly used as a classification
    head for segmentation tasks.
    Args:
        in_channels (int): Number of input channels in the embeddings.
        token_width (int): Width of the token/patch grid.
        token_height (int): Height of the token/patch grid.
        num_concepts (int): Number of output concepts/classes.
    Attributes:
        in_channels (int): Number of input channels.
        width (int): Width dimension for reshaping.
        height (int): Height dimension for reshaping.
        classifier (torch.nn.Conv2d): 1x1 convolution layer for classification.
    """

    def __init__(self, in_channels: int, token_width: int, token_height: int, num_concepts: int):
        super().__init__()
        self.in_channels = in_channels
        self.width = token_width
        self.height = token_height
        self.concept_classifier = torch.nn.Conv2d(in_channels, num_concepts, kernel_size=1)

    def forward(self, embeddings: torch.Tensor) -> torch.Tensor:
        """Reshape token embeddings and apply the 1×1 concept classification conv.

        Args:
            embeddings (torch.Tensor): Flattened patch embeddings with shape
                (B, token_height * token_width, in_channels).
        Returns:
            torch.Tensor: Per-pixel concept logits with shape (B, num_concepts, token_height, token_width).
        """
        embeddings = embeddings.reshape(-1, self.height, self.width, self.in_channels)
        embeddings = embeddings.permute(0, 3, 1, 2)

        return self.concept_classifier(embeddings)


class LinearDINOv3(torch.nn.Module):
    """DINOv3 encoder with a linear segmentation head.

    Encodes input images with a frozen-or-trainable DINOv3 backbone and
    classifies each spatial token with a 1×1 conv, then bilinearly upsamples
    back to input resolution.

    Attributes:
        encoder: DINOv3 model loaded via `AutoModel.from_pretrained`.
        head (LinearClassifier): Per-token classification head.
        token_width (int): Width of the patch-token grid.
        token_height (int): Height of the patch-token grid.
    """

    def __init__(
        self,
        encoder_name: str = "facebook/dinov3-vitb16-pretrain-lvd1689m",
        num_classes: int = 2,
        input_size: tuple[int, int] = (512, 512),
        **kwargs: Any,
    ):
        """Initialize encoder and linear segmentation head.

        Args:
            encoder_name (str): HuggingFace model ID for the DINOv3 encoder.
                Defaults to "facebook/dinov3-vitb16-pretrain-lvd1689m".
            num_classes (int): Number of segmentation output classes. Defaults to 2.
            input_size (tuple[int, int]): Expected (height, width) of input images.
                Determines the token grid size. Defaults to (512, 512).
            **kwargs: Forwarded to `AutoModel.from_pretrained` (e.g., `token`).
        """
        super().__init__()

        token = kwargs.pop("token", None) or os.environ.get("HF_TOKEN")
        self.encoder = AutoModel.from_pretrained(encoder_name, token=token, **kwargs)
        hidden_size = self.encoder.config.hidden_size
        patch_size = self.encoder.config.patch_size
        self.token_width = input_size[1] // patch_size
        self.token_height = input_size[0] // patch_size
        self.head = LinearClassifier(hidden_size, self.token_width, self.token_height, num_classes)

    def forward(self, x: torch.Tensor, labels=None, **kwargs: Any) -> SemanticSegmenterOutput:
        """Run encoder + linear head and upsample to input resolution.

        Args:
            x (torch.Tensor): Input image tensor with shape (B, C, H, W).
            labels: Unused; accepted for API compatibility.
        Returns:
            SemanticSegmenterOutput: `.logits` has shape (B, num_classes, H, W).
        """
        outputs = self.encoder(x, **kwargs)
        # Skip the 5 DINOv3 prefix tokens (CLS + 4 register tokens)
        patch_embeddings = outputs.last_hidden_state[:, 5:, :]

        # convert to logits and upsample to the size of the pixel values
        logits = self.head(patch_embeddings)
        logits = torch.nn.functional.interpolate(
            logits, size=x.shape[-2:], mode="bilinear", align_corners=False
        )

        loss = None
        # if labels is not None:
        #     logits = torch.nn.functional.interpolate(
        #         logits, size=labels.shape[-2:], mode="bilinear", align_corners=False
        #     )
        #     loss_fct = torch.nn.CrossEntropyLoss(ignore_index=0)
        #     loss = loss_fct(logits.squeeze(), labels.squeeze())

        return SemanticSegmenterOutput(loss=loss, logits=logits)

    def freeze_encoder(self) -> None:
        """Freezes the encoder layers of the model by setting the `requires_grad` attribute of their
        parameters to `False`.

        This prevents the encoder layers from being updated during training, effectively making them
        static while allowing other parts of the model to be trained.
        """
        for param in self.encoder.parameters():
            param.requires_grad = False

    def unfreeze_encoder(self) -> None:
        """Unfreezes the encoder layers of the model by setting the `requires_grad` attribute of all
        parameters in the encoder to True.

        This allows these layers to be trainable during the training process.
        """
        for param in self.encoder.parameters():
            param.requires_grad = True

"""
class ConceptBottleneckDINOv3(torch.nn.Module):

    def __init__(
        self,
        encoder_name: str = "facebook/dinov3-vitb16-pretrain-lvd1689m",
        num_classes: int = 2,
        num_concepts: int = 2,
        input_size: tuple[int, int] = (512, 512),
        concept_value2id: dict[str, dict[str, int]] | None = None,
        **kwargs: Any,
    ):
        super().__init__()

        token = kwargs.pop("token", None) or os.environ.get("HF_TOKEN")
        self.encoder = AutoModel.from_pretrained(encoder_name, token=token, **kwargs)
        self.concept_value2id = concept_value2id
        hidden_size = self.encoder.config.hidden_size
        patch_size = self.encoder.config.patch_size
        self.token_width = input_size[1] // patch_size
        self.token_height = input_size[0] // patch_size
        self.concept_head = LinearClassifier(
            hidden_size, self.token_width, self.token_height, num_concepts
        )
        self.concept_classifier = torch.nn.Conv2d(num_concepts, num_classes, kernel_size=1)

    def forward(self, x: torch.Tensor, labels=None, **kwargs: Any) -> SemanticSegmenterOutput:
        outputs = self.encoder(x, **kwargs)
        # Skip the 5 DINOv3 prefix tokens (CLS + 4 register tokens)
        patch_embeddings = outputs.last_hidden_state[:, 5:, :]

        # convert to logits and upsample to the size of the pixel values
        concept_logits = self.concept_head(patch_embeddings)
        concept_logits = torch.nn.functional.interpolate(
            concept_logits, size=x.shape[-2:], mode="bilinear", align_corners=False
        )
        concept_outputs = self.concept_outputs_activation(concept_logits)

        logits = self.concept_classifier(concept_outputs)

        logits = torch.nn.functional.interpolate(
            logits, size=x.shape[-2:], mode="bilinear", align_corners=False
        )
        loss = None

        # if labels is not None:
        #     logits = torch.nn.functional.interpolate(
        #         logits, size=labels.shape[-2:], mode="bilinear", align_corners=False
        #     )
        #     loss_fct = torch.nn.CrossEntropyLoss(ignore_index=0)
        #     loss = loss_fct(logits.squeeze(), labels.squeeze())

        return SemanticSegmenterOutput(loss=loss, logits=logits, hidden_states=concept_outputs)

    def freeze_encoder(self) -> None:
        for param in self.encoder.parameters():
            param.requires_grad = False

    def unfreeze_encoder(self) -> None:

        for param in self.encoder.parameters():
            param.requires_grad = True

    def concept_outputs_activation(self, concept_outputs: torch.Tensor) -> torch.Tensor:
        offset = 0
        activated_parts = []

        for concept in TAXONOMIC_CONCEPTS:
            concept_values = self.concept_value2id[concept]
            order_concept_length = len(list(concept_values.values())[0])
            chunk = concept_outputs[:, offset : offset + order_concept_length, ...]
            activated_parts.append(torch.softmax(chunk, dim=1))
            offset += order_concept_length

        activated_parts.append(torch.sigmoid(concept_outputs[:, offset:, ...]))
        return torch.cat(activated_parts, dim=1)
"""        
class ConceptTokenHead(torch.nn.Module):
    """Apply three 1×1 convs (+ LeakyReLU) on patch tokens at token resolution."""

    def __init__(
        self,
        in_channels: int,
        token_width: int,
        token_height: int,
        out_channels: int = 256,
    ):
        super().__init__()
        self.in_channels = in_channels
        self.width = token_width
        self.height = token_height
        self.out_channels = out_channels

        self.conv1 = torch.nn.Conv2d(in_channels, in_channels, kernel_size=1)
        self.conv2 = torch.nn.Conv2d(in_channels, in_channels, kernel_size=1)
        self.conv3 = torch.nn.Conv2d(in_channels, out_channels, kernel_size=1)
        self.act = torch.nn.LeakyReLU(inplace=True)

    def forward(self, embeddings: torch.Tensor) -> torch.Tensor:
        """Reshape tokens to (B, C, H, W), apply conv stack, return token-grid features."""
        x = embeddings.reshape(-1, self.height, self.width, self.in_channels)
        x = x.permute(0, 3, 1, 2)  # (B, hidden_size, token_H, token_W)

        x = self.act(self.conv1(x))
        x = self.act(self.conv2(x))
        x = self.act(self.conv3(x))
        return x


class ConceptBottleneckDINOv3(torch.nn.Module):
    """DINOv3 encoder with a concept bottleneck for segmentation."""

    def __init__(
        self,
        encoder_name: str = "facebook/dinov3-vitb16-pretrain-lvd1689m",
        num_classes: int = 2,
        num_concepts: int = 2,
        input_size: tuple[int, int] = (512, 512),
        concept_value2id: dict[str, dict[str, int]] | None = None,
        concept_feature_dim: int = 256,
        **kwargs: Any,
    ):
        super().__init__()
        self.frozen_encoder = False
        token = kwargs.pop("token", None) or os.environ.get("HF_TOKEN")
        self.encoder = AutoModel.from_pretrained(encoder_name, token=token, **kwargs)
        self.concept_value2id = concept_value2id

        hidden_size = self.encoder.config.hidden_size
        patch_size = self.encoder.config.patch_size
        self.patch_size = patch_size
        self.token_width = input_size[1] // patch_size
        self.token_height = input_size[0] // patch_size

        self.concept_head = ConceptTokenHead(
            hidden_size,
            self.token_width,
            self.token_height,
            out_channels=concept_feature_dim,
        )
        # Projects 256-dim pixel features back to concept logits for the existing bottleneck API
        self.concept_proj = torch.nn.Conv2d(concept_feature_dim, num_concepts, kernel_size=1)
        self.concept_classifier = torch.nn.Conv2d(num_concepts, num_classes, kernel_size=1)

    def forward(self, x: torch.Tensor, labels=None, **kwargs: Any) -> SemanticSegmenterOutput:
        if self.frozen_encoder:
            with torch.no_grad():
                outputs = self.encoder(x, **kwargs)
        else:
            outputs = self.encoder(x, **kwargs)

        # Skip the 5 DINOv3 prefix tokens (CLS + 4 register tokens)
        patch_embeddings = outputs.last_hidden_state[:, 5:, :]

        # 3× 1×1 convs on token grid → (B, 256, token_H, token_W)
        concept_features = self.concept_head(patch_embeddings)

        # Upscale token grid to pixel resolution (e.g. 32×32 → 512×512 when patch_size=16)
        concept_features = torch.nn.functional.interpolate(
            concept_features,
            scale_factor=self.patch_size,
            mode="bilinear",
            align_corners=False,
        )
        concept_logits = self.concept_proj(concept_features)
        concept_outputs = self.concept_outputs_activation(concept_logits)

        logits = self.concept_classifier(concept_outputs)
        logits = torch.nn.functional.interpolate(
            logits, size=x.shape[-2:], mode="bilinear", align_corners=False
        )

        loss = None
        return SemanticSegmenterOutput(loss=loss, logits=logits, hidden_states=concept_outputs)

    def freeze_encoder(self) -> None:
        self.frozen_encoder = True
        for param in self.encoder.parameters():
            param.requires_grad = False

    def unfreeze_encoder(self) -> None:
        self.frozen_encoder = False
        for param in self.encoder.parameters():
            param.requires_grad = True

    def concept_outputs_activation(self, concept_outputs: torch.Tensor) -> torch.Tensor:
        offset = 0
        activated_parts = []

        for concept in TAXONOMIC_CONCEPTS:
            concept_values = self.concept_value2id[concept]
            order_concept_length = len(list(concept_values.values())[0])
            chunk = concept_outputs[:, offset : offset + order_concept_length, ...]
            activated_parts.append(torch.softmax(chunk, dim=1))
            offset += order_concept_length

        activated_parts.append(torch.sigmoid(concept_outputs[:, offset:, ...]))
        return torch.cat(activated_parts, dim=1)


def _wrap_encoder_with_lora(
    encoder: torch.nn.Module,
    lora_r: int,
    lora_alpha: int,
    lora_dropout: float,
    lora_target_modules: Sequence[str],
    lora_bias: str,
) -> torch.nn.Module:
    """Wrap a DINOv3 encoder with PEFT LoRA adapters.

    The pretrained backbone weights are frozen and only the injected low-rank
    adapter matrices remain trainable, which keeps the parameter/optimizer
    footprint small while still adapting the attention projections.
    """
    lora_config = LoraConfig(
        r=lora_r,
        lora_alpha=lora_alpha,
        lora_dropout=lora_dropout,
        target_modules=list(lora_target_modules),
        bias=lora_bias,
    )
    return get_peft_model(encoder, lora_config)


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
        neck_hidden_sizes: Sequence[int] = (256, 512, 1024, 1024),
        fusion_hidden_size: int = 256,
        reassemble_factors: Sequence[float] = (4, 2, 1, 0.5),
        dropout: float = 0.1,
    ):
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
        self.head = torch.nn.Sequential(
            torch.nn.Conv2d(
                fusion_hidden_size, fusion_hidden_size, kernel_size=3, padding=1, bias=False
            ),
            torch.nn.BatchNorm2d(fusion_hidden_size),
            torch.nn.ReLU(inplace=True),
            torch.nn.Dropout(dropout),
            torch.nn.Conv2d(fusion_hidden_size, out_channels, kernel_size=1),
        )

    def forward(self, hidden_states: Sequence[torch.Tensor]) -> torch.Tensor:
        """Run the DPT neck + conv head.

        Args:
            hidden_states: Sequence of ``num_stages`` token tensors, each of
                shape ``(B, 1 + token_height * token_width, hidden_size)`` with
                the CLS token at index 0 (register tokens already stripped).
        Returns:
            torch.Tensor: Dense features of shape ``(B, out_channels, H_f, W_f)``.
        """
        if len(hidden_states) != self.num_stages:
            raise ValueError(
                f"DPTHead expects {self.num_stages} hidden states, got {len(hidden_states)}."
            )
        features = self.neck(hidden_states, self.token_height, self.token_width)
        return self.head(features[-1])


class _DPTDINOv3Base(torch.nn.Module):
    """Shared backbone/head wiring for the DPT DINOv3 models.

    The encoder can either be adapted with PEFT LoRA (``use_lora=True``) or
    used as a plain frozen backbone (``use_lora=False``), in which case only
    the DPT head (and any downstream concept layers) is trained.
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
        use_lora: bool = True,
        lora_r: int = 16,
        lora_alpha: int = 32,
        lora_dropout: float = 0.05,
        lora_target_modules: Sequence[str] = DEFAULT_LORA_TARGET_MODULES,
        lora_bias: str = "none",
        **kwargs: Any,
    ):
        super().__init__()
        token = kwargs.pop("token", None) or os.environ.get("HF_TOKEN")
        base_encoder = AutoModel.from_pretrained(encoder_name, token=token, **kwargs)

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

        self.use_lora = use_lora
        if use_lora:
            self.encoder = _wrap_encoder_with_lora(
                base_encoder,
                lora_r=lora_r,
                lora_alpha=lora_alpha,
                lora_dropout=lora_dropout,
                lora_target_modules=lora_target_modules,
                lora_bias=lora_bias,
            )
        else:
            self.encoder = base_encoder
        # Only meaningful for the non-LoRA path: gates the no-grad encoder pass.
        self._encoder_frozen = False

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

        # No-LoRA models default to a frozen backbone (head-only training).
        if not use_lora:
            self.freeze_encoder()

    def _encode(self, x: torch.Tensor, **kwargs: Any) -> list[torch.Tensor]:
        """Run the encoder and return the selected hidden states (CLS + patches).

        When the (non-LoRA) backbone is frozen the encoder pass runs under
        ``torch.no_grad`` to save memory; the LoRA path always keeps gradients
        so the adapters can learn.
        """
        if self._encoder_frozen:
            with torch.no_grad():
                outputs = self.encoder(x, output_hidden_states=True, **kwargs)
        else:
            outputs = self.encoder(x, output_hidden_states=True, **kwargs)

        hidden_states = outputs.hidden_states
        selected = []
        for idx in self.out_indices:
            hs = hidden_states[idx]
            # Keep CLS (index 0), drop register tokens, keep patch tokens.
            selected.append(torch.cat([hs[:, :1], hs[:, self.num_prefix_tokens :]], dim=1))
        return selected

    def freeze_encoder(self) -> None:
        """Freeze the backbone.

        With LoRA the pretrained weights are frozen but the low-rank adapters
        stay trainable (so gradients still flow through the encoder). Without
        LoRA the whole encoder is frozen and its forward pass runs under
        ``torch.no_grad``.
        """
        if self.use_lora:
            for name, param in self.encoder.named_parameters():
                param.requires_grad = "lora_" in name
        else:
            for param in self.encoder.parameters():
                param.requires_grad = False
            self._encoder_frozen = True

    def unfreeze_encoder(self) -> None:
        """Unfreeze the full encoder (base weights + LoRA adapters, if any)."""
        for param in self.encoder.parameters():
            param.requires_grad = True
        self._encoder_frozen = False


class _LinearDPTDINOv3(_DPTDINOv3Base):
    """DINOv3 + DPT head producing class logits directly."""

    def forward(self, x: torch.Tensor, labels=None, **kwargs: Any) -> SemanticSegmenterOutput:
        """Run encoder + DPT head and upsample logits to input resolution."""
        hidden_states = self._encode(x, **kwargs)
        logits = self.dpt_head(hidden_states)
        logits = torch.nn.functional.interpolate(
            logits, size=x.shape[-2:], mode="bilinear", align_corners=False
        )
        return SemanticSegmenterOutput(loss=None, logits=logits)


class _ConceptBottleneckDPTDINOv3(_DPTDINOv3Base):
    """DINOv3 + DPT head feeding a concept bottleneck before classification."""

    def __init__(
        self,
        *,
        num_classes: int,
        num_concepts: int,
        concept_value2id: dict[str, dict[str, int]] | None,
        concept_feature_dim: int,
        **kwargs: Any,
    ):
        super().__init__(head_out_channels=concept_feature_dim, **kwargs)
        self.concept_value2id = concept_value2id
        self.concept_proj = torch.nn.Conv2d(concept_feature_dim, num_concepts, kernel_size=1)
        self.concept_classifier = torch.nn.Conv2d(num_concepts, num_classes, kernel_size=1)

    def forward(self, x: torch.Tensor, labels=None, **kwargs: Any) -> SemanticSegmenterOutput:
        """Run encoder + DPT head, the concept bottleneck, and the class classifier."""
        hidden_states = self._encode(x, **kwargs)
        concept_features = self.dpt_head(hidden_states)
        concept_features = torch.nn.functional.interpolate(
            concept_features, size=x.shape[-2:], mode="bilinear", align_corners=False
        )
        concept_logits = self.concept_proj(concept_features)
        concept_outputs = self.concept_outputs_activation(concept_logits)

        logits = self.concept_classifier(concept_outputs)
        return SemanticSegmenterOutput(loss=None, logits=logits, hidden_states=concept_outputs)

    def concept_outputs_activation(self, concept_outputs: torch.Tensor) -> torch.Tensor:
        offset = 0
        activated_parts = []

        for concept in TAXONOMIC_CONCEPTS:
            concept_values = self.concept_value2id[concept]
            order_concept_length = len(list(concept_values.values())[0])
            chunk = concept_outputs[:, offset : offset + order_concept_length, ...]
            activated_parts.append(torch.softmax(chunk, dim=1))
            offset += order_concept_length

        activated_parts.append(torch.sigmoid(concept_outputs[:, offset:, ...]))
        return torch.cat(activated_parts, dim=1)


class LinearDPTLoRADINOv3(_LinearDPTDINOv3):
    """DINOv3 encoder (LoRA-adapted) with a DPT segmentation head."""

    def __init__(
        self,
        encoder_name: str = "facebook/dinov3-vitl16-pretrain-lvd1689m",
        num_classes: int = 2,
        input_size: tuple[int, int] = (512, 512),
        out_indices: Sequence[int] | None = None,
        neck_hidden_sizes: Sequence[int] = (256, 512, 1024, 1024),
        fusion_hidden_size: int = 256,
        reassemble_factors: Sequence[float] = (4, 2, 1, 0.5),
        lora_r: int = 16,
        lora_alpha: int = 32,
        lora_dropout: float = 0.05,
        lora_target_modules: Sequence[str] = DEFAULT_LORA_TARGET_MODULES,
        lora_bias: str = "none",
        **kwargs: Any,
    ):
        super().__init__(
            encoder_name=encoder_name,
            input_size=input_size,
            out_indices=out_indices,
            neck_hidden_sizes=neck_hidden_sizes,
            fusion_hidden_size=fusion_hidden_size,
            reassemble_factors=reassemble_factors,
            head_out_channels=num_classes,
            use_lora=True,
            lora_r=lora_r,
            lora_alpha=lora_alpha,
            lora_dropout=lora_dropout,
            lora_target_modules=lora_target_modules,
            lora_bias=lora_bias,
            **kwargs,
        )


class LinearDPTDINOv3(_LinearDPTDINOv3):
    """DINOv3 encoder (frozen, no LoRA) with a DPT segmentation head."""

    def __init__(
        self,
        encoder_name: str = "facebook/dinov3-vitl16-pretrain-lvd1689m",
        num_classes: int = 2,
        input_size: tuple[int, int] = (512, 512),
        out_indices: Sequence[int] | None = None,
        neck_hidden_sizes: Sequence[int] = (256, 512, 1024, 1024),
        fusion_hidden_size: int = 256,
        reassemble_factors: Sequence[float] = (4, 2, 1, 0.5),
        **kwargs: Any,
    ):
        super().__init__(
            encoder_name=encoder_name,
            input_size=input_size,
            out_indices=out_indices,
            neck_hidden_sizes=neck_hidden_sizes,
            fusion_hidden_size=fusion_hidden_size,
            reassemble_factors=reassemble_factors,
            head_out_channels=num_classes,
            use_lora=False,
            **kwargs,
        )


class ConceptBottleneckDPTLoRADINOv3(_ConceptBottleneckDPTDINOv3):
    """DINOv3 encoder (LoRA-adapted) with a DPT head feeding a concept bottleneck."""

    def __init__(
        self,
        encoder_name: str = "facebook/dinov3-vitl16-pretrain-lvd1689m",
        num_classes: int = 2,
        num_concepts: int = 2,
        input_size: tuple[int, int] = (512, 512),
        concept_value2id: dict[str, dict[str, int]] | None = None,
        concept_feature_dim: int = 256,
        out_indices: Sequence[int] | None = None,
        neck_hidden_sizes: Sequence[int] = (256, 512, 1024, 1024),
        fusion_hidden_size: int = 256,
        reassemble_factors: Sequence[float] = (4, 2, 1, 0.5),
        lora_r: int = 16,
        lora_alpha: int = 32,
        lora_dropout: float = 0.05,
        lora_target_modules: Sequence[str] = DEFAULT_LORA_TARGET_MODULES,
        lora_bias: str = "none",
        **kwargs: Any,
    ):
        super().__init__(
            num_classes=num_classes,
            num_concepts=num_concepts,
            concept_value2id=concept_value2id,
            concept_feature_dim=concept_feature_dim,
            encoder_name=encoder_name,
            input_size=input_size,
            out_indices=out_indices,
            neck_hidden_sizes=neck_hidden_sizes,
            fusion_hidden_size=fusion_hidden_size,
            reassemble_factors=reassemble_factors,
            use_lora=True,
            lora_r=lora_r,
            lora_alpha=lora_alpha,
            lora_dropout=lora_dropout,
            lora_target_modules=lora_target_modules,
            lora_bias=lora_bias,
            **kwargs,
        )


class ConceptBottleneckDPTDINOv3(_ConceptBottleneckDPTDINOv3):
    """DINOv3 encoder (frozen, no LoRA) with a DPT head feeding a concept bottleneck."""

    def __init__(
        self,
        encoder_name: str = "facebook/dinov3-vitl16-pretrain-lvd1689m",
        num_classes: int = 2,
        num_concepts: int = 2,
        input_size: tuple[int, int] = (512, 512),
        concept_value2id: dict[str, dict[str, int]] | None = None,
        concept_feature_dim: int = 256,
        out_indices: Sequence[int] | None = None,
        neck_hidden_sizes: Sequence[int] = (256, 512, 1024, 1024),
        fusion_hidden_size: int = 256,
        reassemble_factors: Sequence[float] = (4, 2, 1, 0.5),
        **kwargs: Any,
    ):
        super().__init__(
            num_classes=num_classes,
            num_concepts=num_concepts,
            concept_value2id=concept_value2id,
            concept_feature_dim=concept_feature_dim,
            encoder_name=encoder_name,
            input_size=input_size,
            out_indices=out_indices,
            neck_hidden_sizes=neck_hidden_sizes,
            fusion_hidden_size=fusion_hidden_size,
            reassemble_factors=reassemble_factors,
            use_lora=False,
            **kwargs,
        )