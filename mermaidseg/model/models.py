import os
from typing import Any

import torch
from transformers import AutoModel
from transformers.modeling_outputs import SemanticSegmenterOutput

from mermaidseg.dataset_reconciliation.concepts import TAXONOMIC_CONCEPTS


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


class ConceptBottleneckDINOv3(torch.nn.Module):
    """DINOv3 encoder with a concept bottleneck for segmentation.

    Adds an intermediate concept prediction head between the DINOv3 backbone and the
    final segmentation classifier. The concept logits are supervised jointly with the
    segmentation task and are returned via `SemanticSegmenterOutput.hidden_states`.

    Attributes:
        encoder: DINOv3 backbone loaded via `AutoModel.from_pretrained`.
        concept_head (LinearClassifier): Per-token concept prediction head.
        concept_classifier (torch.nn.Conv2d): 1×1 conv mapping concepts → classes.
        token_width (int): Width of the patch-token grid.
        token_height (int): Height of the patch-token grid.
        concept_value2id (dict): Optional mapping of concept names to ID mappings for each taxonomic rank.
    """

    def __init__(
        self,
        encoder_name: str = "facebook/dinov3-vitb16-pretrain-lvd1689m",
        num_classes: int = 2,
        num_concepts: int = 2,
        input_size: tuple[int, int] = (512, 512),
        concept_value2id: dict[str, dict[str, int]] | None = None,
        **kwargs: Any,
    ):
        """Initialize encoder, concept head, and segmentation classifier.

        Args:
            encoder_name (str): HuggingFace model ID for the DINOv3 encoder.
                Defaults to "facebook/dinov3-vitb16-pretrain-lvd1689m".
            num_classes (int): Number of segmentation output classes. Defaults to 2.
            num_concepts (int): Number of intermediate concept dimensions. Defaults to 2.
            input_size (tuple[int, int]): Expected (height, width) of input images.
                Determines the token grid size. Defaults to (512, 512).
            **kwargs: Forwarded to `AutoModel.from_pretrained` (e.g., `token`).
        """
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
        """Run encoder → concept head → segmentation classifier.

        Args:
            x (torch.Tensor): Input image tensor with shape (B, C, H, W).
            labels: Unused; accepted for API compatibility.
        Returns:
            SemanticSegmenterOutput: `.logits` has shape (B, num_classes, H, W);
                `.hidden_states` contains concept logits with shape (B, num_concepts, H, W).
        """
        outputs = self.encoder(x, **kwargs)
        # Skip the 5 DINOv3 prefix tokens (CLS + 4 register tokens)
        patch_embeddings = outputs.last_hidden_state[:, 5:, :]

        # convert to logits and upsample to the size of the pixel values
        concept_logits = self.concept_head(patch_embeddings)
        concept_logits = torch.nn.functional.interpolate(
            concept_logits, size=x.shape[-2:], mode="bilinear", align_corners=False
        )
        concept_outputs = self._concept_outputs_activation(concept_logits)

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

    def _concept_outputs_activation(self, concept_outputs: torch.Tensor) -> torch.Tensor:
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
