"""
title: mermaidseg.model.models
abstract: Module that contains segmentation model network architectures.
author: Viktor Domazetoski
date: 21-10-2025

Classes:
    Segformer(torch.nn.Module)
        __init__
        forward
        freeze_encoder
        unfreeze_encoder
    Dinov2ForSemanticSegmentation(torch.nn.Module)
        __init__
        forward
        freeze_encoder
        unfreeze_encoder
    LinearClassifier(torch.nn.Module)
        __init__
        forward
"""

import os
from typing import Any

import torch
from transformers import AutoModel, Dinov2Model, SegformerForSemanticSegmentation
from transformers.modeling_outputs import SemanticSegmenterOutput


class LinearClassifier(torch.nn.Module):
    """
    A linear classifier module that performs pixel-wise classification on reshaped embeddings.
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
        """
        Run the model's forward pass on a batch of flattened embeddings.
        Parameters
        ----------
        embeddings : torch.Tensor
            Input tensor containing embeddings that can be reshaped to
            (-1, self.height, self.width, self.in_channels). The first
            dimension is inferred (batch size). Typical input shapes include
            (batch_size, height * width * in_channels) or any shape that
            can be viewed as a flattened per-sample embedding vector.
        Returns
        -------
        torch.Tensor
            The output of self.classifier applied to the reshaped and
            channel-permuted embeddings. The exact shape depends on
            self.classifier, but for a per-pixel segmentation head this is
            typically (batch_size, num_classes, height, width).
        """

        embeddings = embeddings.reshape(-1, self.height, self.width, self.in_channels)
        embeddings = embeddings.permute(0, 3, 1, 2)

        return self.classifier(embeddings)


class ConceptHead(torch.nn.Module):
    """
    A concept classification head module that performs pixel-wise classification on reshaped embeddings.
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
        """
        Run the model's forward pass on a batch of flattened embeddings.
        Parameters
        ----------
        embeddings : torch.Tensor
            Input tensor containing embeddings that can be reshaped to
            (-1, self.height, self.width, self.in_channels). The first
            dimension is inferred (batch size). Typical input shapes include
            (batch_size, height * width * in_channels) or any shape that
            can be viewed as a flattened per-sample embedding vector.
        Returns
        -------
        torch.Tensor
            The output of self.classifier applied to the reshaped and
            channel-permuted embeddings. The exact shape depends on
            self.classifier, but for a per-pixel segmentation head this is
            typically (batch_size, num_classes, height, width).
        """
        embeddings = embeddings.reshape(-1, self.height, self.width, self.in_channels)
        embeddings = embeddings.permute(0, 3, 1, 2)

        return self.concept_classifier(embeddings)


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

    def __init__(self, encoder_name: str = "nvidia/mit-b2", num_classes: int = 2, **kwargs: Any):
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
            id2label={i: i for i in range(0, num_classes)},  # do we need a plus one here
            semantic_loss_ignore_index=0,
            ignore_mismatched_sizes=True,
            **kwargs,
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


# self.model = Dinov2ForSemanticSegmentation.from_pretrained("facebook/dinov2-base",
#                                                                        id2label={i:i for i in range(0, N_classes)},
#                                                                        num_labels=N_classes)


class LinearDINOv2(torch.nn.Module):
    """
    Wrapper around the DINOv2 model for semantic segmentation.
    This class provides an interface to initialize, train, and use a DINOv2 model for semantic segmentation tasks.
    It allows customization of the encoder and the number of output classes, and includes methods to freeze or unfreeze
    the encoder layers.
        encoder (Dinov2Model): The DINOv2 encoder model.
        head (LinearClassifier): The classification head for segmentation.
    Methods:
        __init__(encoder_name: str = "facebook/dinov2-base", num_classes: int = 2, **kwargs: Any):
            Initializes the DINOv2 model with a specified encoder and number of output classes.
        forward(x: torch.Tensor) -> torch.Tensor:
            Performs a forward pass through the model.
        freeze_encoder():
            Freezes the encoder layers to prevent them from being updated during training.
        unfreeze_encoder():
            Unfreezes the encoder layers to allow them to be updated during training.
    """

    def __init__(
        self,
        encoder_name: str = "facebook/dinov2-base",
        num_classes: int = 2,
        input_size: tuple[int, int] = (518, 518),
        **kwargs: Any,
    ):
        """
        Initializes the semantic segmentation model with a specified encoder and number of classes.
        Args:
            encoder_name (str): The name of the HF encoder to use for the model. Defaults to "facebook/dinov2-base".
            num_classes (int): The number of output classes for semantic segmentation. Defaults to 2.
            input_size (tuple[int, int]): The size of the input images (height, width). Defaults to (518, 518).
            **kwargs (Any): Additional keyword arguments to pass to the model initialization.
        Attributes:
            encoder (Dinov2Model): The initialized DINOv2 encoder model.
            head (LinearClassifier): The classification head for segmentation.
        Note:
            The encoder is frozen by default using the `freeze_encoder` method.
        """
        super().__init__()

        self.encoder = Dinov2Model.from_pretrained(encoder_name, **kwargs)
        # Assuming hidden_size=768 for dinov2-base, adjust as needed
        hidden_size = self.encoder.config.hidden_size
        patch_size = self.encoder.config.patch_size
        tokenW = input_size[1] // patch_size
        tokenH = input_size[0] // patch_size

        self.head = LinearClassifier(
            hidden_size, tokenW, tokenH, num_classes
        )  # The tokenW and tokenH are calculated based on input_size and patch_size

        # self.freeze_encoder()  # Freeze the backbone - should this be done by default

    def forward(self, x: torch.Tensor, labels=None, **kwargs: Any) -> torch.Tensor:
        """
        Perform a forward pass through the model.
        Args:
            x (torch.Tensor): Input tensor to the model.
            labels (torch.Tensor, optional): Ground truth labels for loss computation.
        Returns:
            torch.Tensor: Output tensor after passing through the model.
        """
        # use frozen features
        outputs = self.encoder(x, **kwargs)
        # get the patch embeddings - so we exclude the CLS token
        patch_embeddings = outputs.last_hidden_state[:, 1:, :]

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
        """
        Freezes the encoder layers of the model by setting the `requires_grad`
        attribute of their parameters to `False`. This prevents the encoder
        layers from being updated during training, effectively making them
        static while allowing other parts of the model to be trained.
        """
        for param in self.encoder.parameters():
            param.requires_grad = False

    def unfreeze_encoder(self) -> None:
        """
        Unfreezes the encoder layers of the model by setting the `requires_grad`
        attribute of all parameters in the encoder to True.
        This allows these layers to be trainable during the training process.
        """
        for param in self.encoder.parameters():
            param.requires_grad = True


class LinearDINOv3(torch.nn.Module):
    """
    Wrapper around the DINOv3 model for semantic segmentation.
    This class provides an interface to initialize, train, and use a DINOv3 model for semantic segmentation tasks.
    It allows customization of the encoder and the number of output classes, and includes methods to freeze or unfreeze
    the encoder layers.
    To be able to use the model one needs to authenticate with HuggingFace using the command:
    `huggingface-cli login` and have applied for access to the DINOv3 models.
        encoder (Dinov3Model): The DINOv3 encoder model.
        head (LinearClassifier): The classification head for segmentation.
    Methods:
        __init__(encoder_name: str = "facebook/dinov3-vits16-pretrain-lvd1689m", num_classes: int = 2, **kwargs: Any):
            Initializes the DINOv3 model with a specified encoder and number of output classes.
        forward(x: torch.Tensor) -> torch.Tensor:
            Performs a forward pass through the model.
        freeze_encoder():
            Freezes the encoder layers to prevent them from being updated during training.
        unfreeze_encoder():
            Unfreezes the encoder layers to allow them to be updated during training.
    """

    def __init__(
        self,
        encoder_name: str = "facebook/dinov3-vits16-pretrain-lvd1689m",
        num_classes: int = 2,
        input_size: tuple[int, int] = (512, 512),
        **kwargs: Any,
    ):
        """
        Initializes the semantic segmentation model with a specified encoder and number of classes.
        Args:
            encoder_name (str): The name of the HF encoder to use for the model. Defaults to "facebook/dinov3-vits16-pretrain-lvd1689m".
            num_classes (int): The number of output classes for semantic segmentation. Defaults to 2.
            input_size (tuple[int, int]): The size of the input images (height, width). Defaults to (512, 512).
            **kwargs (Any): Additional keyword arguments to pass to the model initialization.
        Attributes:
            encoder (Dinov2Model): The initialized DINOv2 encoder model.
            head (LinearClassifier): The classification head for segmentation.
        Note:
            The encoder is frozen by default using the `freeze_encoder` method.
        """
        super().__init__()

        token = kwargs.pop("token", None) or os.environ.get("HF_TOKEN")
        self.encoder = AutoModel.from_pretrained(encoder_name, token=token, **kwargs)
        # Assuming hidden_size=768 for dinov2-base, adjust as needed
        hidden_size = self.encoder.config.hidden_size
        patch_size = self.encoder.config.patch_size
        self.token_width = input_size[1] // patch_size
        self.token_height = input_size[0] // patch_size
        self.head = LinearClassifier(
            hidden_size, self.token_width, self.token_height, num_classes
        )  # The tokenW and tokenH are calculated based on input_size and patch_size

        # self.freeze_encoder()  # Freeze the backbone - should this be done by default

    def forward(self, x: torch.Tensor, labels=None, **kwargs: Any) -> torch.Tensor:
        """
        Perform a forward pass through the model.
        Args:
            x (torch.Tensor): Input tensor to the model.
            labels (torch.Tensor, optional): Ground truth labels for loss computation.
        Returns:
            torch.Tensor: Output tensor after passing through the model.
        """
        # use frozen features
        outputs = self.encoder(x, **kwargs)
        # get the patch embeddings - so we exclude the CLS token
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
        """
        Freezes the encoder layers of the model by setting the `requires_grad`
        attribute of their parameters to `False`. This prevents the encoder
        layers from being updated during training, effectively making them
        static while allowing other parts of the model to be trained.
        """
        for param in self.encoder.parameters():
            param.requires_grad = False

    def unfreeze_encoder(self) -> None:
        """
        Unfreezes the encoder layers of the model by setting the `requires_grad`
        attribute of all parameters in the encoder to True.
        This allows these layers to be trainable during the training process.
        """
        for param in self.encoder.parameters():
            param.requires_grad = True


class ConceptBottleneckDINOv3(torch.nn.Module):
    """
    Wrapper around the DINOv3 model for semantic segmentation.
    This class provides an interface to initialize, train, and use a DINOv3 model for semantic segmentation tasks.
    It allows customization of the encoder and the number of output classes, and includes methods to freeze or unfreeze
    the encoder layers.
    To be able to use the model one needs to authenticate with HuggingFace using the command:
    `huggingface-cli login` and have applied for access to the DINOv3 models.
        encoder (Dinov3Model): The DINOv3 encoder model.
        head (LinearClassifier): The classification head for segmentation.
    Methods:
        __init__(encoder_name: str = "facebook/dinov3-vits16-pretrain-lvd1689m", num_classes: int = 2, **kwargs: Any):
            Initializes the DINOv3 model with a specified encoder and number of output classes.
        forward(x: torch.Tensor) -> torch.Tensor:
            Performs a forward pass through the model.
        freeze_encoder():
            Freezes the encoder layers to prevent them from being updated during training.
        unfreeze_encoder():
            Unfreezes the encoder layers to allow them to be updated during training.
    """

    def __init__(
        self,
        encoder_name: str = "facebook/dinov3-vits16-pretrain-lvd1689m",
        num_classes: int = 2,
        num_concepts: int = 2,
        input_size: tuple[int, int] = (512, 512),
        **kwargs: Any,
    ):
        """
        Initializes the semantic segmentation model with a specified encoder and number of classes.
        Args:
            encoder_name (str): The name of the HF encoder to use for the model. Defaults to "facebook/dinov3-vits16-pretrain-lvd1689m".
            num_classes (int): The number of output classes for semantic segmentation. Defaults to 2.
            input_size (tuple[int, int]): The size of the input images (height, width). Defaults to (512, 512).
            **kwargs (Any): Additional keyword arguments to pass to the model initialization.
        Attributes:
            encoder (Dinov2Model): The initialized DINOv2 encoder model.
            head (LinearClassifier): The classification head for segmentation.
        Note:
            The encoder is frozen by default using the `freeze_encoder` method.
        """
        super().__init__()

        token = kwargs.pop("token", None) or os.environ.get("HF_TOKEN")
        self.encoder = AutoModel.from_pretrained(encoder_name, token=token, **kwargs)
        # Assuming hidden_size=768 for dinov2-base, adjust as needed
        hidden_size = self.encoder.config.hidden_size
        patch_size = self.encoder.config.patch_size
        self.token_width = input_size[1] // patch_size
        self.token_height = input_size[0] // patch_size
        self.concept_head = LinearClassifier(
            hidden_size, self.token_width, self.token_height, num_concepts
        )  # The tokenW and tokenH are calculated based on input_size and patch_size
        self.concept_classifier = torch.nn.Conv2d(num_concepts, num_classes, kernel_size=1)
        # self.freeze_encoder()  # Freeze the backbone - should this be done by default

    def forward(self, x: torch.Tensor, labels=None, **kwargs: Any) -> torch.Tensor:
        """
        Perform a forward pass through the model.
        Args:
            x (torch.Tensor): Input tensor to the model.
            labels (torch.Tensor, optional): Ground truth labels for loss computation.
        Returns:
            torch.Tensor: Output tensor after passing through the model.
        """
        # use frozen features
        outputs = self.encoder(x, **kwargs)
        # get the patch embeddings - so we exclude the CLS token
        patch_embeddings = outputs.last_hidden_state[:, 5:, :]

        # convert to logits and upsample to the size of the pixel values
        concept_logits = self.concept_head(patch_embeddings)
        concept_logits = torch.nn.functional.interpolate(
            concept_logits, size=x.shape[-2:], mode="bilinear", align_corners=False
        )
        logits = self.concept_classifier(concept_logits)

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

        return SemanticSegmenterOutput(loss=loss, logits=logits, hidden_states=concept_logits)

    def freeze_encoder(self) -> None:
        """
        Freezes the encoder layers of the model by setting the `requires_grad`
        attribute of their parameters to `False`. This prevents the encoder
        layers from being updated during training, effectively making them
        static while allowing other parts of the model to be trained.
        """
        for param in self.encoder.parameters():
            param.requires_grad = False

    def unfreeze_encoder(self) -> None:
        """
        Unfreezes the encoder layers of the model by setting the `requires_grad`
        attribute of all parameters in the encoder to True.
        This allows these layers to be trainable during the training process.
        """
        for param in self.encoder.parameters():
            param.requires_grad = True
