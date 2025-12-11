"""
title: mermaidseg.model.loss
abstract: Module that contains loss functions and classes.
author: Viktor Domazetoski
date: 15-09-2025

Classes:
    CrossEntropyLoss(torch.nn.CrossEntropyLoss)
"""

from typing import Any

import torch


class CrossEntropyLoss(torch.nn.CrossEntropyLoss):
    """
    CrossEntropyLoss is a wrapper of `torch.nn.CrossEntropyLoss` that allows for additional customization.
    Attributes:
        ignore_index (int): Specifies a target value that is ignored and does not contribute to the input gradient.
            This is useful for masking certain values in the target tensor. Defaults to -1.
        kwargs: Additional keyword arguments that are passed to the base `torch.nn.CrossEntropyLoss` class.
    """

    def __init__(self, ignore_index: int = -1, **kwargs: Any) -> None:
        super().__init__(ignore_index=ignore_index, **kwargs)


class BCEWithLogitsLoss(torch.nn.BCEWithLogitsLoss):
    """
    CrossEntropyLoss is a wrapper of `torch.nn.CrossEntropyLoss` that allows for additional customization.
    Attributes:
        ignore_index (int): Specifies a target value that is ignored and does not contribute to the input gradient.
            This is useful for masking certain values in the target tensor. Defaults to -1.
        kwargs: Additional keyword arguments that are passed to the base `torch.nn.CrossEntropyLoss` class.
    """

    def __init__(self, reduction: str = "none", **kwargs: Any) -> None:
        super().__init__(reduction=reduction, **kwargs)

    def forward(
        self,
        concept_outputs: torch.Tensor,
        concept_labels: torch.Tensor,
        labels: torch.Tensor,
    ) -> torch.Tensor:
        """
        Computes the total loss as the sum of the classification loss and a weighted concept loss.
        Args:
            outputs (torch.Tensor): The model's output logits for classification.
            labels (torch.Tensor): The ground truth labels for classification.
            concept_outputs (torch.Tensor): The model's output logits for concept prediction.
            concept_labels (torch.Tensor): The ground truth labels for concept prediction.
        Returns:
            torch.Tensor: The computed total loss.
        """
        per_element_loss = super().forward(concept_outputs, concept_labels)
        label_mask = (labels > 0).unsqueeze(1).to(self.device)

        masked_loss = per_element_loss * label_mask
        denom = label_mask.sum() * concept_outputs.shape[1] + 1e-8
        loss = masked_loss.sum() / denom

        return loss


class ConceptBottleneckLoss(torch.nn.Module):
    """
    ConceptBottleneckLoss combines a classification loss with a concept prediction loss.
    It computes the total loss as the sum of the classification loss and a weighted concept loss.
    Attributes:
        class_loss (torch.nn.Module): The loss function used for classification. Defaults to
            `torch.nn.CrossEntropyLoss`.
        concept_loss (torch.nn.Module): The loss function used for concept prediction. Defaults to
            `torch.nn.BCEWithLogitsLoss`.
        ignore_index (int): Specifies a target value that is ignored and does not contribute to the input gradient
            for the classification loss. Defaults to -1.
        lambda_weight (float): The weight applied to the concept loss when computing the total loss. Defaults to 1.0.
        kwargs: Additional keyword arguments that are passed to the loss functions.
    """

    def __init__(
        self,
        class_loss: torch.nn.Module = torch.nn.CrossEntropyLoss,
        concept_loss: torch.nn.Module = torch.nn.BCEWithLogitsLoss,
        ignore_index: int = -1,
        lambda_weight: float = 1.0,
        **kwargs: Any,
    ) -> None:
        super().__init__()
        self.class_loss = class_loss(ignore_index=ignore_index, **kwargs)
        self.concept_loss = concept_loss(reduction="none", **kwargs)
        self.lambda_weight = lambda_weight

    def forward(
        self,
        outputs: torch.Tensor,
        labels: torch.Tensor,
        concept_outputs: torch.Tensor,
        concept_labels: torch.Tensor,
    ) -> torch.Tensor:
        """
        Computes the total loss as the sum of the classification loss and a weighted concept loss.
        Args:
            outputs (torch.Tensor): The model's output logits for classification.
            labels (torch.Tensor): The ground truth labels for classification.
            concept_outputs (torch.Tensor): The model's output logits for concept prediction.
            concept_labels (torch.Tensor): The ground truth labels for concept prediction.
        Returns:
            torch.Tensor: The computed total loss.
        """
        class_loss_value = self.class_loss(outputs, labels)

        label_mask = (labels > 0).unsqueeze(1).to(self.device)
        per_element_loss = self.concept_loss(concept_outputs, concept_labels)

        masked_loss = per_element_loss * label_mask
        denom = label_mask.sum() * outputs.shape[1] + 1e-8
        concept_loss_value = masked_loss.sum() / denom

        total_loss = class_loss_value + self.lambda_weight * concept_loss_value
        return total_loss
