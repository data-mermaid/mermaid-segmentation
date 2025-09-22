"""
title: mermaidseg.model.models
abstract: Module that contains segmentation model network architectures.
author: Viktor Domazetoski
date: 21-09-2025

Classes:
    Segformer(torch.nn.Module)
        __init__
        forward
        freeze_encoder
        unfreeze_encoder
"""

from typing import Any

import torch
from transformers import SegformerForSemanticSegmentation


class Segformer(torch.nn.Module):
    """
    Wrapper around the Segformer segmentation model.
    This class provides an interface to initialize, train, and use a Segformer model for semantic segmentation tasks.
    It allows customization of the encoder and the number of output classes, and includes methods to freeze or unfreeze
    the encoder layers.
        model (SegformerForSemanticSegmentation): The initialized Segformer model for semantic segmentation.
    Methods:
        __init__(encoder_name: str = "nvidia/mit-b2", num_classes: int = 2, **kwargs: Any):
            Initializes the Segformer model with a specified encoder and number of output classes.
        forward(x: torch.Tensor) -> torch.Tensor:
            Performs a forward pass through the model.
        freeze_encoder():
            Freezes the encoder layers to prevent them from being updated during training.
        unfreeze_encoder():
            Unfreezes the encoder layers to allow them to be updated during training.
    """

    def __init__(
        self, encoder_name: str = "nvidia/mit-b2", num_classes: int = 2, **kwargs: Any
    ):
        """
        Initializes the semantic segmentation model with a specified encoder and number of classes.
        Args:
            encoder_name (str): The name of the HF encoder to use for the model. Defaults to "nvidia/mit-b2".
            num_classes (int): The number of output classes for semantic segmentation. Defaults to 2.
            **kwargs (Any): Additional keyword arguments to pass to the model initialization.
        Attributes:
            model (SegformerForSemanticSegmentation): The initialized semantic segmentation model.
        Note:
            The encoder is frozen by default using the `freeze_encoder` method.
        """
        super().__init__()
        # super(Segformer, self).__init__()

        # self.model = SegformerForSemanticSegmentation.from_pretrained(  # type: ignore
        #     encoder_name, id2label={i: i for i in range(1, num_classes)}, **kwargs
        # )
        self.model = SegformerForSemanticSegmentation.from_pretrained(
            encoder_name,
            id2label={i: i for i in range(0, num_classes + 1)}, # do we need a plus one here 
            semantic_loss_ignore_index=0,
            ignore_mismatched_sizes=True,
            **kwargs
        )
        # self.freeze_encoder()  # Freeze the backbone - should this be done by default

    def forward(self, x: torch.Tensor, **kwargs: Any) -> torch.Tensor:
        """
        Perform a forward pass through the model.
        Args:
            x (torch.Tensor): Input tensor to the model.
        Returns:
            torch.Tensor: Output tensor after passing through the model.
        """

        return self.model(x, **kwargs)

    def freeze_encoder(self) -> None:
        """
        Freezes the encoder layers of the model by setting the `requires_grad`
        attribute of their parameters to `False`. This prevents the encoder
        layers from being updated during training, effectively making them
        static while allowing other parts of the model to be trained.
        """

        for param in self.model.segformer.encoder.pixel_level_module.parameters():  # type: ignore
            param.requires_grad = False

    def unfreeze_encoder(self) -> None:
        """
        Unfreezes the encoder layers of the model by setting the `requires_grad`
        attribute of all parameters in the encoder's feature extraction layers to True.
        This allows these layers to be trainable during the training process.
        """

        for param in self.model.segformer.encoder.pixel_level_module.parameters():  # type: ignore
            param.requires_grad = True


# class DPTDino(torch.nn.Module):
#     """
#     Wrapper around the DPT semantic segmentation model with a DINO backbone.
#     It allows customization of the encoder and the number of output classes, and includes methods to freeze or unfreeze
#     the encoder layers.
#         model (DPTForSemanticSegmentation): The initialized DPT model for semantic segmentation.
#     Methods:
#         __init__(encoder_name: str = "facebook/dinov2-base", num_classes: int = 2, **kwargs: Any):
#             Initializes the Segformer model with a specified encoder and number of output classes.
#         forward(x: torch.Tensor) -> torch.Tensor:
#             Performs a forward pass through the model.
#     """

#     def __init__(
#         self, encoder_name: str = "facebook/dinov2-base", num_classes: int = 2, **kwargs: Any
#     ):
#         super().__init__()
#         dinov2_config = Dinov2Config(reshape_hidden_states=True).from_pretrained(encoder_name)
#         dinov2_backbone = Dinov2Model.from_pretrained(encoder_name, reshape_hidden_states=True)
#         self.backbone = dinov2_backbone
#         if encoder_name == "facebook/dinov2-base":
#             self.indices = (2, 5, 8, 11)
#         elif encoder_name == "facebook/dinov2-giant":
#             self.indices = (9, 19, 29, 39)

#         config = DPTConfig(
#             num_labels=num_classes,
#             ignore_index=0,
#             semantic_loss_ignore_index=0,
#             is_hybrid=False,
#             backbone_out_indices=self.indices,
#             backbone_config=dinov2_config
#         )

#         # Load the DPT segmentation model
#         dpt_model = DPTForSemanticSegmentation(config)

#         self.neck = dpt_model.neck
#         self.head = dpt_model.head

#     def forward(self, pixel_values, labels=None):
#         features = self.backbone(pixel_values, output_hidden_states = True)
#         features = [features.hidden_states[i] for i in self.indices]

#         h, w = pixel_values.shape[-2:]
#         if h != w: # In a non square image
#             if(h%14 != 0 or w %14 !=0):
#                 raise ValueError("Height and width must be divisible by the patch size (14).")

#             patch_height = h//14
#             patch_width = w//14
#             features = self.neck(features, patch_height=patch_height, patch_width=patch_width)
#         else:
#             features = self.neck(features)

#         logits = self.head(features)
#         logits = torch.nn.functional.interpolate(logits,
#                                                  size=pixel_values.shape[-2:],
#                                                  mode="bilinear",
#                                                  align_corners=False)

#         loss = None
#         if labels is not None:
#             logits = torch.nn.functional.interpolate(logits,
#                                                      size=labels.shape[-2:],
#                                                      mode="bilinear",
#                                                      align_corners=False)
#             # important: we're going to use 0 here as ignore index instead of the default -100
#             # as we don't want the model to learn to predict background
#             loss_fct = torch.nn.CrossEntropyLoss(ignore_index=0)
#             loss = loss_fct(logits.squeeze(), labels.squeeze())

#         return SemanticSegmenterOutput(
#             loss=loss,
#             logits=logits,
#         )

#         return SemanticSegmenterOutput(
#             loss=loss,
#             logits=logits,
#         )
