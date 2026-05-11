import time
from typing import Any

import albumentations as A
import numpy as np
import pandas as pd
import torch
import transformers

# from mermaidseg.model.eval import Evaluator
from numpy.typing import NDArray
from torch.utils.data import DataLoader
from tqdm import tqdm

import mermaidseg.model.loss
import mermaidseg.model.models
from mermaidseg.datasets.concepts import (
    labels_to_concepts,
    postprocess_predicted_concepts,
)
from mermaidseg.io import ConfigDict


class MetaModel:
    """Wrapper for training and inference of segmentation models.

    Handles model initialisation, optimizer/scheduler/loss wiring, and the
    train/validation loop primitives. Supports three training modes:

    - ``"standard"``: standard cross-entropy segmentation.
    - ``"concept"``: outputs are concept logits mapped back to class predictions.
    - ``"concept-bottleneck"``: joint segmentation + concept supervision.

    Attributes:
        run_name (str): Identifier for the current run/experiment.
        model_name (str): Name of the model architecture (looked up in `mermaidseg.model.models`).
        num_classes (int): Number of segmentation output classes.
        device (str | torch.device): Device the model and tensors live on.
        model_kwargs (ConfigDict): Model-specific config passed to the architecture.
        training_kwargs (ConfigDict): Training hyperparameters (epochs, optimizer, scheduler, loss).
        model (torch.nn.Module | transformers.PreTrainedModel): The instantiated model.
        loss (torch.nn.Module | None): Loss function; None until `training_kwargs` provides one.
        optimizer (torch.optim.Optimizer): Optimiser instance.
        scheduler (torch.optim.lr_scheduler.LRScheduler | None): Optional LR scheduler.
        concept_matrix (pd.DataFrame | None): Concept mapping matrix for CBM/concept modes.
        conceptid2labelid (dict[int, int] | None): Maps concept IDs to class label IDs.
    """

    run_name: str
    model_name: str
    num_classes: int
    device: str | torch.device
    model_kwargs: ConfigDict
    training_kwargs: ConfigDict
    model: torch.nn.Module | transformers.PreTrainedModel
    loss: torch.nn.Module | None
    optimizer: torch.optim.Optimizer
    scheduler: torch.optim.lr_scheduler.LRScheduler | None
    concept_matrix: pd.DataFrame | None
    conceptid2labelid: dict[int, int] | None

    def __init__(
        self,
        run_name: str,
        num_classes: int,
        num_concepts: int | None = None,
        model_kwargs: ConfigDict | None = None,
        device: str | torch.device = "cuda",
        model_checkpoint: str | None = None,
        training_mode: str = "standard",  # One of "standard", "concept", "concept-bottleneck"
        training_kwargs: ConfigDict | None = None,
        concept_matrix: pd.DataFrame | None = None,
        conceptid2labelid: dict[int, int] | None = None,
    ):
        self.run_name = run_name
        self.num_classes = num_classes
        self.num_concepts = num_concepts
        self.device = device

        if model_kwargs is None:
            model_kwargs = ConfigDict({})
        if training_kwargs is None:
            training_kwargs = ConfigDict(
                {
                    "epochs": 50,
                    "optimizer": {
                        "type": "AdamW",
                        "lr": 0.001,
                        "weight_decay": 0.01,
                    },
                }
            )

        self.model_name = model_kwargs.pop("name", None)
        self.model_kwargs = model_kwargs
        self.model_checkpoint = model_checkpoint

        self.training_mode = training_mode
        assert self.training_mode in [
            "standard",
            "concept",
            "concept-bottleneck",
        ], f"Invalid training_mode: {self.training_mode}"

        self.training_kwargs = training_kwargs
        self.concept_matrix = concept_matrix
        self.conceptid2labelid = conceptid2labelid
        if self.training_mode == "concept-bottleneck":
            self.model = getattr(mermaidseg.model.models, self.model_name)(
                num_classes=self.num_classes,
                num_concepts=self.num_concepts,
                **model_kwargs,
            )
        elif self.training_mode == "concept":
            self.model = getattr(mermaidseg.model.models, self.model_name)(
                num_classes=self.num_concepts, **model_kwargs
            )
        else:
            self.model = getattr(mermaidseg.model.models, self.model_name)(
                num_classes=self.num_classes, **model_kwargs
            )

        if model_checkpoint:
            checkpoint = torch.load(model_checkpoint)
            if "model_state_dict" in checkpoint:
                self.model.load_state_dict(checkpoint["model_state_dict"])
            else:
                self.model.load_state_dict(checkpoint)

        self.model = self.model.to(device)
        if "loss" in training_kwargs:
            loss = training_kwargs.loss.pop("type", None)
            self.loss = getattr(mermaidseg.model.loss, loss)(**training_kwargs.loss)
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
        target_dim: tuple[int, int] | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        """Perform batch prediction using the model.

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

        segmentation_outputs = self.model(inputs)

        if self.training_mode == "concept-bottleneck":
            concept_outputs = segmentation_outputs.hidden_states
            outputs = segmentation_outputs.logits
            concept_outputs = torch.sigmoid(concept_outputs)
        elif self.training_mode == "concept":
            concept_outputs = segmentation_outputs.logits
            outputs = postprocess_predicted_concepts(
                concept_outputs.detach().cpu().numpy(),
                self.concept_matrix,
                self.conceptid2labelid,
            ).to(self.device)
            concept_outputs = torch.sigmoid(concept_outputs)
        else:
            outputs = segmentation_outputs.logits
            concept_outputs = None

        assert isinstance(outputs, torch.Tensor)
        return outputs, concept_outputs

    def batch_predict_loss(
        self,
        batch: tuple[torch.Tensor, torch.Tensor] | dict[str, torch.Tensor],
        target_dim: tuple[int, int] | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor | None, dict[str, float]]:
        """Perform batch prediction and compute the loss.

        Args:
            batch: A tuple of (inputs, labels) or a dict of named tensors.
            target_dim: Target spatial dimensions for output resizing.
                Defaults to the input spatial dimensions.
        Returns:
            A 4-tuple of (loss, outputs, concept_outputs, loss_components)
            where loss_components is a dict of detached scalar values for
            logging (non-empty only in concept-bottleneck mode).
        """
        loss = None
        outputs = None
        concept_outputs = None
        loss_components: dict[str, float] = {}

        inputs, labels = batch

        if target_dim is None:
            target_dim = (inputs.size(-2), inputs.size(-1))

        inputs = inputs.to(self.device).float()
        labels = labels.long().to(self.device)
        segmentation_outputs = self.model(inputs)

        if self.training_mode == "concept-bottleneck":
            concept_outputs = segmentation_outputs.hidden_states
            outputs = segmentation_outputs.logits
            concept_labels = labels_to_concepts(labels, self.concept_matrix)
            loss, loss_components = self.loss(outputs, labels, concept_outputs, concept_labels)
            concept_outputs = torch.sigmoid(concept_outputs)

        elif self.training_mode == "concept":
            concept_outputs = segmentation_outputs.logits
            concept_labels = labels_to_concepts(labels, self.concept_matrix)
            loss = self.loss(concept_outputs, concept_labels, labels)
            concept_outputs = torch.sigmoid(concept_outputs)
            outputs = postprocess_predicted_concepts(
                concept_outputs.detach().cpu().numpy(),
                self.concept_matrix,
                self.conceptid2labelid,
            ).to(self.device)

        else:
            outputs = segmentation_outputs.logits
            concept_outputs = None
            loss = self.loss(outputs, labels)

        assert loss is not None, "Loss is not computed for the given batch."
        assert isinstance(outputs, torch.Tensor)

        return loss, outputs, concept_outputs, loss_components

    def train_epoch(
        self,
        train_loader: DataLoader[tuple[torch.Tensor, torch.Tensor] | dict[str, torch.Tensor]],
        evaluator: Any
        | None = None,  # TODO: Should be Evaluator - but this leads to circular import, fix
    ) -> tuple[float, dict[str, float | NDArray[np.float64]], dict[str, float | int]]:
        """Trains the model for one epoch using the provided data loader.

        Args:
            train_loader: DataLoader providing batches of training data.
            evaluator: Optional evaluator for computing per-epoch metrics.

        Returns:
            A 3-tuple of (average_loss, metric_results, timing) where timing
            contains ``"data_loading_sec"``, ``"forward_sec"``,
            ``"backward_sec"``, and ``"num_samples"``.
        """
        running_loss = 0.0
        running_loss_components: dict[str, float] = {}
        metric_results: dict[str, float | NDArray[np.float64]] = {}
        use_cuda = self.device != "cpu" and torch.cuda.is_available()

        data_time_total = 0.0
        forward_time_total = 0.0
        backward_time_total = 0.0
        num_samples = 0

        if use_cuda:
            torch.cuda.synchronize()
        batch_end = time.perf_counter()

        for data in tqdm(train_loader):
            if use_cuda:
                torch.cuda.synchronize()
            data_time_total += time.perf_counter() - batch_end

            _, labels = data
            labels = labels.long().to(self.device)

            if use_cuda:
                torch.cuda.synchronize()
            forward_start = time.perf_counter()

            loss, outputs, concept_outputs, loss_components = self.batch_predict_loss(data)

            if use_cuda:
                torch.cuda.synchronize()
            forward_time_total += time.perf_counter() - forward_start

            assert isinstance(loss, torch.Tensor), "Loss must be a torch.Tensor"

            if use_cuda:
                torch.cuda.synchronize()
            backward_start = time.perf_counter()

            loss.backward()
            self.optimizer.step()
            self.optimizer.zero_grad()

            if use_cuda:
                torch.cuda.synchronize()
            backward_time_total += time.perf_counter() - backward_start

            running_loss += loss.item()
            for k, v in loss_components.items():
                running_loss_components[k] = running_loss_components.get(k, 0.0) + v
            num_samples += labels.size(0)

            if evaluator is not None:
                if self.training_mode in ("concept"):
                    concept_labels = labels_to_concepts(labels, self.concept_matrix)
                    print(concept_labels.shape, concept_labels.dtype)
                    concept_labels = postprocess_predicted_concepts(
                        concept_labels,
                        self.concept_matrix,
                        self.conceptid2labelid,
                    )
                    concept_labels = torch.from_numpy(concept_labels).long().to(self.device)
                    for metric in evaluator.metric_dict.values():
                        metric.update(outputs, concept_labels)
                else:
                    if outputs.ndim > 3:
                        outputs = outputs.argmax(dim=1)
                    for metric in evaluator.metric_dict.values():
                        metric.update(outputs, labels)

            if self.training_mode in ("concept-bottleneck", "concept"):
                for metric in evaluator.concept_metric_dict.values():
                    concept_outputs = (concept_outputs > 0.5).float()
                    concept_outputs += 1
                    concept_outputs *= labels.unsqueeze(1) != 0

                    concept_labels = labels_to_concepts(labels, self.concept_matrix)
                    concept_labels += 1
                    concept_labels *= labels.unsqueeze(1) != 0

                    for metric in evaluator.concept_metric_dict.values():
                        metric.update(concept_outputs, concept_labels)

            if use_cuda:
                torch.cuda.synchronize()
            batch_end = time.perf_counter()

        if evaluator is not None:
            for metric_name in evaluator.metric_dict:
                metric_results[metric_name] = (
                    evaluator.metric_dict[metric_name].compute().cpu().numpy()
                )
                if metric_results[metric_name].ndim == 0:
                    metric_results[metric_name] = metric_results[metric_name].item()
                evaluator.metric_dict[metric_name].reset()

            if self.training_mode in ("concept-bottleneck", "concept"):
                for metric_name in evaluator.concept_metric_dict:
                    metric_results[metric_name] = (
                        evaluator.concept_metric_dict[metric_name].compute().cpu().numpy()
                    )
                    if metric_results[metric_name].ndim == 0:
                        metric_results[metric_name] = metric_results[metric_name].item()
                    evaluator.concept_metric_dict[metric_name].reset()

        last_loss = running_loss / len(train_loader)
        avg_loss_components = {k: v / len(train_loader) for k, v in running_loss_components.items()}
        for k, v in avg_loss_components.items():
            metric_results[f"loss/{k}"] = v
        timing = {
            "data_loading_sec": data_time_total,
            "forward_sec": forward_time_total,
            "backward_sec": backward_time_total,
            "num_samples": num_samples,
        }
        return last_loss, metric_results, timing

    @torch.no_grad()
    def validation_epoch(
        self,
        val_loader: DataLoader[tuple[torch.Tensor, torch.Tensor] | dict[str, torch.Tensor]],
        evaluator: Any
        | None = None,  # TODO: Should be Evaluator - but this leads to circular import, fix
    ) -> tuple[float, dict[str, float | NDArray[np.float64]]]:
        """Calculate the validation loss and metrics for one epoch.

        Args:
            val_loader: DataLoader providing batches of validation data.
            evaluator: Optional evaluator for computing per-epoch metrics.
        Returns:
            A 2-tuple of (average_loss, metric_results).
        """
        running_loss = 0.0
        metric_results: dict[str, float | NDArray[np.float64]] = {}

        for data in tqdm(val_loader):
            _, labels = data
            labels = labels.long().to(self.device)

            loss, outputs, concept_outputs, _ = self.batch_predict_loss(data)
            assert isinstance(loss, torch.Tensor), "Loss must be a torch.Tensor"
            running_loss += loss.item()

            if evaluator is not None:
                if self.training_mode in ("concept"):
                    concept_labels = labels_to_concepts(labels, self.concept_matrix)
                    concept_labels = postprocess_predicted_concepts(
                        concept_labels, self.concept_matrix, self.conceptid2labelid
                    )
                    concept_labels = torch.from_numpy(concept_labels).long().to(self.device)
                    for metric in evaluator.metric_dict.values():
                        metric.update(outputs, concept_labels)
                else:
                    if outputs.ndim > 3:
                        outputs = outputs.argmax(dim=1)
                    # Update metrics
                    for metric in evaluator.metric_dict.values():
                        metric.update(outputs, labels)

            if self.training_mode in ("concept-bottleneck", "concept"):
                # Update metrics
                for metric in evaluator.concept_metric_dict.values():
                    concept_outputs = (concept_outputs > 0.5).float()
                    concept_outputs += 1
                    concept_outputs *= labels.unsqueeze(1) != 0

                    concept_labels = labels_to_concepts(labels, self.concept_matrix)
                    concept_labels += 1
                    concept_labels *= labels.unsqueeze(1) != 0

                    for metric in evaluator.concept_metric_dict.values():
                        metric.update(concept_outputs, concept_labels)

        # Compute metrics
        if evaluator is not None:
            for metric_name in evaluator.metric_dict:
                metric_results[metric_name] = (
                    evaluator.metric_dict[metric_name].compute().cpu().numpy()
                )
                if metric_results[metric_name].ndim == 0:
                    metric_results[metric_name] = metric_results[metric_name].item()
                evaluator.metric_dict[metric_name].reset()

        if self.training_mode in ("concept-bottleneck", "concept"):
            for metric_name in evaluator.concept_metric_dict:
                metric_results[metric_name] = (
                    evaluator.concept_metric_dict[metric_name].compute().cpu().numpy()
                )
                if metric_results[metric_name].ndim == 0:
                    metric_results[metric_name] = metric_results[metric_name].item()

                evaluator.concept_metric_dict[metric_name].reset()

        last_loss = running_loss / len(val_loader)
        return last_loss, metric_results

    @torch.no_grad()  # type:ignore
    def predict(
        self,
        image: torch.Tensor | NDArray[Any],
        transform: A.BasicTransform | None = None,
    ) -> NDArray[Any]:
        """Predicts the output for a given input image using the model.

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
        return pred.argmax(dim=1).cpu().numpy()[0]
