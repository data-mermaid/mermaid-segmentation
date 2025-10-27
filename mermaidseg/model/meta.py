"""
title: mermaidseg.model.meta
abstract: Module that contains the MetaModel class which is the main ML model wrapper.
author: Viktor Domazetoski
date: 27-07-2025

Classes:
    MetaModel
"""

from typing import Any, Dict, Optional, Union

import albumentations as A
import mermaidseg.model.loss
import mermaidseg.model.models
import torch
import torch.nn.functional as F
import transformers
from mermaidseg.io import ConfigDict
from numpy.typing import NDArray
from torch.utils.data import DataLoader
from tqdm import tqdm


class MetaModel:
    """
    MetaModel is a class designed to handle the initialization, training, validation,
    and prediction processes for machine learning models. It supports various tasks
    such as classification, semantic segmentation, and object detection, and integrates
    with PyTorch and Hugging Face Transformers.
    Attributes:
        run_name (str): Name of the current run or experiment.
        task (str): The type of task (e.g., "classification", "semantic_segmentation", "object_detection").
        model_name (str): Name of the model architecture.
        num_classes (int): Number of output classes for the task.
        device (Union[str, torch.device]): Device to run the model on (e.g., "cuda" or "cpu").
        model_kwargs (ConfigDict): Configuration dictionary for model-specific parameters.
        training_kwargs (ConfigDict): Configuration dictionary for training parameters.
        use_amp (bool): Whether to use Automatic Mixed Precision (AMP) for training.
        scaler (torch.GradScaler): Gradient scaler for AMP.
        model (Union[torch.nn.Module, transformers.PreTrainedModel]): The initialized model.
        loss (Optional[torch.nn.Module]): The loss function used for training.
        optimizer (torch.optim.Optimizer): The optimizer used for training.
        scheduler (Optional[torch.optim.lr_scheduler.LRScheduler]): The learning rate scheduler.
    Methods:
        __init__(run_name, task, num_classes, model_kwargs, lora_kwargs=None, device="cuda",
                 model_checkpoint=None, training_kwargs=ConfigDict):
            Initializes the MetaModel with the specified parameters, loads the model,
            and prepares it for training or inference.
        batch_predict(inputs: Union[torch.Tensor, Dict[str, torch.Tensor]]) -> torch.Tensor:
            Perform batch prediction using the model. Takes input data in the form of a tensor
            or a dictionary of tensors, processes it on the appropriate device, and returns
            the model's output.
        batch_predict_loss(batch: Union[tuple[torch.Tensor, torch.Tensor], Dict[str, torch.Tensor]])
            -> tuple[torch.Tensor, torch.Tensor]:
            Returns the model's output predictions and the computed loss.
        train_epoch(train_loader: DataLoader[torch.Tensor]) -> float:
            Returns the average loss over all batches in the epoch.
        validation_epoch(val_loader: DataLoader[torch.Tensor]) -> float:
            Returns the average loss over all batches in the epoch.
        predict(image: Union[torch.Tensor, NDArray[Any]], transform: Optional[A.BasicTransform] = None,
                proba: Optional[str] = None, id2label: Optional[Dict[int, str]] = None)
                -> Union[NDArray[Any], str, int]:
            Makes predictions on the given input image. Optionally applies a transformation,
            maps class IDs to labels, and applies a probability function (e.g., Softmax or Sigmoid).
            Returns the predicted output.
    """

    run_name: str
    model_name: str
    num_classes: int
    device: Union[str, torch.device]
    model_kwargs: ConfigDict
    training_kwargs: ConfigDict
    model: Union[torch.nn.Module, transformers.PreTrainedModel]
    loss: Optional[torch.nn.Module]
    optimizer: torch.optim.Optimizer
    scheduler: Optional[torch.optim.lr_scheduler.LRScheduler]

    def __init__(
        self,
        run_name: str,
        num_classes: int,
        model_kwargs: ConfigDict,
        device: Union[str, torch.device] = "cuda",
        model_checkpoint: Optional[str] = None,
        training_kwargs: ConfigDict = ConfigDict(
            {
                "epochs": 50,
                "optimizer": {
                    "type": "AdamW",
                    "lr": 0.001,
                    "weight_decay": 0.01,
                },
            }
        ),
    ):
        self.run_name = run_name
        self.num_classes = num_classes
        self.device = device

        self.model_name = model_kwargs.pop("name", None)
        self.model_kwargs = model_kwargs
        self.model_checkpoint = model_checkpoint

        self.training_kwargs = training_kwargs

        self.model = getattr(mermaidseg.model.models, self.model_name)(
            num_classes=self.num_classes, **model_kwargs
        )

        ## Load model checkpoint if necessary
        if model_checkpoint:
            checkpoint = torch.load(model_checkpoint)
            if "model_state_dict" in checkpoint:
                self.model.load_state_dict(checkpoint["model_state_dict"])
            else:
                self.model.load_state_dict(checkpoint)

        # Move to device
        self.model = self.model.to(device)

        ## Load training procedure: optimizer, scheduler, loss
        if "loss" in training_kwargs:
            loss = training_kwargs.loss.pop("type", None)
            self.loss = getattr(mermaidseg.model.loss, loss)(
                params=self.model.parameters(), **training_kwargs.optimizer
            )
        # else:
        #     self.loss = None  # Often the case for HF models where the loss is already included in the model

        optimizer = training_kwargs.optimizer.pop("type", None)
        self.optimizer = getattr(torch.optim, optimizer)(
            params=self.model.parameters(), **training_kwargs.optimizer
        )

        if "scheduler" in training_kwargs:
            scheduler = training_kwargs.scheduler.pop("type", None)
            self.scheduler = getattr(torch.optim.lr_scheduler, scheduler)(
                self.optimizer, **training_kwargs.scheduler
            )

    def batch_predict(
        self,
        inputs: torch.Tensor,
        target_dim: Optional[tuple[int, int]] = None,
    ) -> torch.Tensor:
        """
        Perform batch prediction using the model.
        This method takes input data in the form of a tensor or a dictionary of tensors,
        processes it on the appropriate device, and returns the model's output.
        Args:
            inputs (Union[torch.Tensor, Dict[str, torch.Tensor]]):
                The input data for prediction. It can be a single tensor or a dictionary
                where keys are input names and values are tensors.
            target_dim (Optional[tuple[int, int]], optional): The target dimensions for resizing the model's output.
                If not provided, the dimensions of the input tensor are used. Defaults to None.
        Returns:
            torch.Tensor: The output tensor from the model. If the model's output contains
            attributes like `logits` or `out`, those are extracted and returned.
        """

        inputs = inputs.to(self.device).float()
        if target_dim is None:
            target_dim = (inputs.size(-2), inputs.size(-1))
        outputs = self.model(inputs)
        if hasattr(outputs, "logits") and not hasattr(self, "loss"):
            outputs = outputs.logits

        outputs = F.interpolate(
            outputs,
            size=target_dim,
            mode="bilinear",
            align_corners=False,
        )

        assert isinstance(outputs, torch.Tensor)
        return outputs

    def batch_predict_loss(
        self,
        batch: Union[tuple[torch.Tensor, torch.Tensor], Dict[str, torch.Tensor]],
        target_dim: Optional[tuple[int, int]] = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Perform batch prediction using the model and compute the loss if applicable.
        Args:
            batch (Union[tuple[torch.Tensor, torch.Tensor], Dict[str, torch.Tensor]]):
                The input batch for prediction. It can either be:
                - A tuple containing input tensors and corresponding labels.
                - A dictionary where keys are input names and values are input tensors.
            target_dim (Optional[tuple[int, int]], optional): The target dimensions for resizing the model's output.
                If not provided, the dimensions of the label tensor are used. Defaults to None.
        Returns:
            Tuple[torch.Tensor, Optional[torch.Tensor]]:
                - The model's output predictions.
                - The computed loss if available, otherwise None.
        """

        loss = None
        inputs, labels = batch

        if target_dim is None:
            target_dim = (inputs.size(-2), inputs.size(-1))

        inputs = inputs.to(self.device).float()
        labels = labels.long().to(self.device)

        if hasattr(self, "loss") and self.loss is not None:
            outputs = self.model(inputs)
            loss = self.loss(outputs, labels)
        else:
            outputs = self.model(inputs, labels=labels)
            loss = outputs.loss
            outputs = outputs.logits

        outputs = F.interpolate(
            outputs,
            size=target_dim,
            mode="bilinear",
            align_corners=False,
        )

        assert loss is not None, "Loss is not computed for the given batch."
        assert isinstance(outputs, torch.Tensor)

        return outputs, loss

    def train_epoch(
        self,
        train_loader: DataLoader[
            Union[tuple[torch.Tensor, torch.Tensor], Dict[str, torch.Tensor]]
        ],
    ) -> float:
        """
        Trains the model for one epoch using the provided data loader.
        Args:
            train_loader (DataLoader[Union[tuple[torch.Tensor, torch.Tensor], Dict[str, torch.Tensor]]]):
            A DataLoader object that provides batches of training data.
        Returns:
            float: The average loss over all batches in the epoch.
        Notes:
            - This method uses mixed precision training if `self.use_amp` is enabled.
            - Gradients are scaled using `self.scaler` to prevent underflow during mixed precision training.
            - The optimizer is stepped and updated after each batch.
        """

        running_loss = 0.0

        for data in tqdm(train_loader):
            __, loss = self.batch_predict_loss(data)
            assert isinstance(loss, torch.Tensor), "Loss must be a torch.Tensor"
            loss.backward()  # Double check
            self.optimizer.step()
            running_loss += loss.item()
            self.optimizer.zero_grad()

        last_loss = running_loss / len(train_loader)
        return last_loss

    @torch.no_grad()
    def validation_epoch(
        self,
        val_loader: DataLoader[
            Union[tuple[torch.Tensor, torch.Tensor], Dict[str, torch.Tensor]]
        ],
    ) -> float:
        """
        Calculates the validation loss of the model for one epoch using the provided data loader.
        Args:
            val_loader (DataLoader[Union[tuple[torch.Tensor, torch.Tensor], Dict[str, torch.Tensor]]]):
            A DataLoader object that provides batches of validation data.
        Returns:
            float: The average loss over all batches in the epoch.
        """

        running_loss = 0.0

        for data in tqdm(val_loader):
            __, loss = self.batch_predict_loss(data)
            assert isinstance(loss, torch.Tensor), "Loss must be a torch.Tensor"
            running_loss += loss.item()

        last_loss = running_loss / len(val_loader)
        return last_loss

    @torch.no_grad()  # type:ignore
    def predict(
        self,
        image: Union[torch.Tensor, NDArray[Any]],
        transform: Optional[A.BasicTransform] = None,
    ) -> Union[NDArray[Any]]:
        """
        Predicts the output for a given input image using the model.
        Args:
            image (Union[torch.Tensor, NDArray[Any]]): The input image as a PyTorch tensor or a NumPy array.
            transform (Optional[A.BasicTransform], optional): An optional transformation to apply to the image.
                Defaults to None.
        Returns:
            Union[NDArray[Any]]: The predicted output.
        """
        if transform:
            image = transform(image=image)["image"]
        inputs = torch.tensor(image).unsqueeze(0)

        pred = self.batch_predict(inputs)
        pred = pred.argmax(dim=1).cpu().numpy()[0]
        return pred
