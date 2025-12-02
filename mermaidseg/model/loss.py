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
