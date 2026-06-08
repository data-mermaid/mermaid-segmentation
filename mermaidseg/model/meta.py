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
from mermaidseg.dataset_reconciliation.concepts import (
    postprocess_predicted_concepts,
    source_labels_to_concepts,
)
from mermaidseg.dataset_reconciliation.label_mapping import (
    source_labels_to_target_labels,
)
from mermaidseg.io import ConfigDict


class MetaModel:
    """Wrapper for training and inference of segmentation models.

    Handles model initialisation, optimizer/scheduler/loss wiring, and the
    train/validation loop primitives. Supports three training modes:

    - ``"standard"``: standard cross-entropy segmentation.
    - ``"concept"``: outputs are concept logits mapped back to class predictions.
    - ``"concept-bottleneck"``: joint segmentation + concept supervision.

    Source-label datasets emit ``source_labels`` in the joint global source
    space (see :class:`SourceLabelRegistry`); ``MetaModel`` converts them to
    ``target_labels`` (and optional ``concept_labels``) on-device via long-tensor
    lookups passed via ``source_to_target_lookup`` and
    ``source_to_concepts_lookup``.

    Attributes:
        run_name (str): Identifier for the current run/experiment.
        model_name (str): Name of the model architecture (looked up in `mermaidseg.model.models`).
        num_classes (int): Number of segmentation output (target) classes.
        device (str | torch.device): Device the model and tensors live on.
        model_kwargs (ConfigDict): Model-specific config passed to the architecture.
        training_kwargs (ConfigDict): Training hyperparameters
            (epochs, iterations_per_train_epoch, iterations_per_val_epoch, optimizer, scheduler, loss).
        model (torch.nn.Module | transformers.PreTrainedModel): The instantiated model.
        loss (torch.nn.Module | None): Loss function; None until `training_kwargs` provides one.
        optimizer (torch.optim.Optimizer): Optimiser instance.
        scheduler (torch.optim.lr_scheduler.LRScheduler | None): Optional LR scheduler.
        source_to_target_lookup (torch.Tensor | None): 1-D long tensor of shape ``(N+1,)``
            mapping global source IDs to target label IDs. ``None`` enables identity passthrough
            (synthetic / single-source pipelines whose source space already matches target space).
        source_to_concepts_lookup (torch.Tensor | None): Float tensor of shape ``(N+1, C)`` for
            CBM/concept modes.
        concept_matrix (pd.DataFrame | None): Pandas concept matrix retained for
            :func:`postprocess_predicted_concepts` (uses MultiIndex level metadata).
        conceptid2labelid (dict[int, int] | None): Maps concept IDs to target label IDs.
        concept_value2id (dict[str, dict[str, int]] | None): Maps concept names and values to IDs.
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
    source_to_target_lookup: torch.Tensor | None
    source_to_concepts_lookup: torch.Tensor | None
    concept_matrix: pd.DataFrame | None
    conceptid2labelid: dict[int, int] | None
    concept_value2id: dict[str, dict[str, int]] | None

    def __init__(
        self,
        run_name: str,
        num_classes: int,
        num_concepts: int | None = None,
        model_kwargs: ConfigDict | None = None,
        device: str | torch.device = "cuda",
        model_checkpoint: str | None = None,
        training_kwargs: ConfigDict | None = None,
        source_to_target_lookup: torch.Tensor | None = None,
        source_to_concepts_lookup: torch.Tensor | None = None,
        concept_matrix: pd.DataFrame | None = None,
        conceptid2labelid: dict[int, int] | None = None,
        concept_value2id: dict[str, dict[str, int]] | None = None,
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

        self.training_mode = training_kwargs.pop("training_mode", None)
        assert self.training_mode in [
            "standard",
            "concept",
            "concept-bottleneck",
        ], f"Invalid training_mode: {self.training_mode}"

        freeze_encoder = training_kwargs.pop(
            "freeze_encoder",
            self.training_mode == "concept-bottleneck",
        )

        self.training_kwargs = training_kwargs
        self.iterations_per_train_epoch = training_kwargs.get("iterations_per_train_epoch")
        self._train_loader_iter = None
        self._train_loader = None
        self.iterations_per_val_epoch = training_kwargs.get("iterations_per_val_epoch")
        self._val_loader_iter = None
        self._val_loader = None
        self.source_to_target_lookup = (
            source_to_target_lookup.to(device).long()
            if source_to_target_lookup is not None
            else None
        )
        self.source_to_concepts_lookup = (
            source_to_concepts_lookup.to(device).float()
            if source_to_concepts_lookup is not None
            else None
        )
        self.concept_matrix = concept_matrix
        self.conceptid2labelid = conceptid2labelid
        self.concept_value2id = concept_value2id

        model_cls = getattr(mermaidseg.model.models, self.model_name)
        model_kwargs.setdefault("num_classes", self.num_classes)
        if self.training_mode == "concept-bottleneck":
            model_kwargs.setdefault("num_concepts", self.num_concepts)
            model_kwargs.setdefault("concept_value2id", self.concept_value2id)
        elif self.training_mode == "concept":
            model_kwargs.setdefault(
                "num_classes", self.num_concepts
            )  # Overwrite the number of classes to be the number of concepts, since the model is only predicting concepts which are then mapped to classes via postprocessing
        self.model = model_cls(**model_kwargs)

        if model_checkpoint:
            checkpoint = torch.load(model_checkpoint)
            if "model_state_dict" in checkpoint:
                self.model.load_state_dict(checkpoint["model_state_dict"])
            else:
                self.model.load_state_dict(checkpoint)

        self.model = self.model.to(device)
        self.freeze_encoder = freeze_encoder
        if freeze_encoder and hasattr(self.model, "freeze_encoder"):
            self.model.freeze_encoder()

        if "loss" in training_kwargs:
            loss_cls = getattr(mermaidseg.model.loss, training_kwargs.loss.pop("type", None))
            loss_kwargs = dict(training_kwargs.loss)
            if self.concept_value2id is not None and self.training_mode in (
                "concept",
                "concept-bottleneck",
            ):
                loss_kwargs.setdefault("concept_value2id", self.concept_value2id)
            self.loss = loss_cls(**loss_kwargs)

        optimizer_cls = getattr(torch.optim, training_kwargs.optimizer.pop("type", None))
        trainable_params = [p for p in self.model.parameters() if p.requires_grad]
        self.optimizer = optimizer_cls(params=trainable_params, **training_kwargs.optimizer)

        if "scheduler" in training_kwargs:
            scheduler_cls = getattr(
                torch.optim.lr_scheduler, training_kwargs.scheduler.pop("type", None)
            )
            self.scheduler = scheduler_cls(self.optimizer, **training_kwargs.scheduler)

    def _to_target_labels(self, source_labels: torch.Tensor) -> torch.Tensor:
        """Map source-space labels to target-space labels via the lookup tensor.

        When ``source_to_target_lookup`` is ``None``, source labels are assumed
        to already be in target space (identity passthrough — useful for
        single-source or synthetic pipelines).
        """
        if self.source_to_target_lookup is None:
            return source_labels
        return source_labels_to_target_labels(source_labels, self.source_to_target_lookup)

    def _to_concept_labels(self, source_labels: torch.Tensor) -> torch.Tensor:
        if self.source_to_concepts_lookup is None:
            raise RuntimeError(
                "MetaModel.source_to_concepts_lookup is not set; cannot run concept modes."
            )
        return source_labels_to_concepts(source_labels, self.source_to_concepts_lookup)

    def batch_predict(
        self,
        inputs: torch.Tensor,
        target_dim: tuple[int, int] | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        """Perform batch prediction using the model.

        Args:
            inputs (torch.Tensor): Image tensor batch.
            target_dim (Optional[tuple[int, int]], optional): Spatial size to
                resize logits to. Defaults to the input spatial size.
        Returns:
            tuple[torch.Tensor, torch.Tensor | None]: ``(outputs, concept_outputs)``.
        """
        inputs = inputs.to(self.device).float()
        if target_dim is None:
            target_dim = (inputs.size(-2), inputs.size(-1))

        segmentation_outputs = self.model(inputs)

        if self.training_mode == "concept-bottleneck":
            concept_outputs = segmentation_outputs.hidden_states
            outputs = segmentation_outputs.logits
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
        images: torch.Tensor,
        target_labels: torch.Tensor,
        target_concepts: torch.Tensor | None = None,
        target_dim: tuple[int, int] | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor | None, dict[str, float]]:
        """Perform batch prediction and compute the loss.

        Args:
            images: Image tensor batch (already on-device, float).
            target_labels: Target-space label tensor (already on-device, long).
            target_concepts: Target concept tensor. Required for ``"concept"``
                and ``"concept-bottleneck"`` modes; unused otherwise.
            target_dim: Target spatial dimensions for output resizing.
                Defaults to the input spatial dimensions.
        Returns:
            A 4-tuple of ``(loss, outputs, concept_outputs, loss_components)``.
        """

        loss = None
        outputs = None
        concept_outputs = None
        loss_components: dict[str, float] = {}

        if target_dim is None:
            target_dim = (images.size(-2), images.size(-1))

        segmentation_outputs = self.model(images)

        if self.training_mode == "concept-bottleneck":
            assert target_concepts is not None, (
                "target_concepts must be provided in 'concept-bottleneck' mode"
            )
            concept_outputs = segmentation_outputs.hidden_states
            outputs = segmentation_outputs.logits
            loss, loss_components = self.loss(
                outputs, target_labels, concept_outputs, target_concepts
            )

        elif self.training_mode == "concept":
            assert target_concepts is not None, "target_concepts must be provided in 'concept' mode"
            concept_outputs = segmentation_outputs.logits
            loss, loss_components = self.loss(concept_outputs, target_concepts, target_labels)
            concept_outputs = torch.sigmoid(concept_outputs)
            outputs = postprocess_predicted_concepts(
                concept_outputs.detach().cpu().numpy(),
                self.concept_matrix,
                self.conceptid2labelid,
            ).to(self.device)

        else:
            outputs = segmentation_outputs.logits
            concept_outputs = None
            loss, loss_components = self.loss(outputs, target_labels)

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
            train_loader: DataLoader yielding ``(inputs, source_labels)`` batches.
            evaluator: Optional evaluator for computing per-epoch metrics.

        Returns:
            A 3-tuple of ``(average_loss, metric_results, timing)``.
        """
        if self.freeze_encoder and hasattr(self.model, "freeze_encoder"):
            self.model.freeze_encoder()

        iterations_per_train_epoch = self.iterations_per_train_epoch
        if iterations_per_train_epoch is None:
            iterations_per_train_epoch = len(train_loader)
        if iterations_per_train_epoch <= 0:
            raise ValueError("iterations_per_train_epoch must be > 0.")

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

        if self._train_loader is not train_loader:
            self._train_loader_iter = iter(train_loader)
            self._train_loader = train_loader

        for _ in tqdm(range(iterations_per_train_epoch)):
            assert self._train_loader_iter is not None
            try:
                data = next(self._train_loader_iter)
            except StopIteration:
                self._train_loader_iter = iter(train_loader)
                self._train_loader = train_loader
                data = next(self._train_loader_iter)

            if use_cuda:
                torch.cuda.synchronize()
            data_time_total += time.perf_counter() - batch_end

            images, source_labels = data
            images = images.to(self.device).float()
            source_labels = source_labels.long().to(self.device)
            target_labels = self._to_target_labels(source_labels)
            target_concepts = (
                self._to_concept_labels(source_labels)
                if self.training_mode in ("concept", "concept-bottleneck")
                else None
            )

            if use_cuda:
                torch.cuda.synchronize()
            forward_start = time.perf_counter()

            loss, outputs, concept_outputs, loss_components = self.batch_predict_loss(
                images, target_labels, target_concepts
            )

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
            num_samples += target_labels.size(0)

            if evaluator is not None:
                if self.training_mode == "concept":
                    target_concept_preds = postprocess_predicted_concepts(
                        target_concepts.detach().cpu().numpy(),
                        self.concept_matrix,
                        self.conceptid2labelid,
                    ).to(self.device)
                    for metric in evaluator.metric_dict.values():
                        metric.update(outputs, target_concept_preds)
                else:
                    if outputs.ndim > 3:
                        outputs = outputs.argmax(dim=1)
                    for metric in evaluator.metric_dict.values():
                        metric.update(outputs, target_labels)

            if evaluator is not None and self.training_mode in ("concept-bottleneck", "concept"):
                evaluator.evaluate_concepts(concept_outputs, target_concepts)
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
                metric_results.update(evaluator.compute_concept_metric_results())

        last_loss = running_loss / iterations_per_train_epoch
        avg_loss_components = {
            k: v / iterations_per_train_epoch for k, v in running_loss_components.items()
        }
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
        """Calculate the validation loss and metrics for one epoch."""

        iterations_per_val_epoch = self.iterations_per_val_epoch
        if iterations_per_val_epoch is None:
            iterations_per_val_epoch = len(val_loader)
        if iterations_per_val_epoch <= 0:
            raise ValueError("iterations_per_val_epoch must be > 0.")

        running_loss = 0.0
        running_loss_components: dict[str, float] = {}
        metric_results: dict[str, float | NDArray[np.float64]] = {}

        if self._val_loader is not val_loader:
            self._val_loader_iter = iter(val_loader)
            self._val_loader = val_loader

        for _ in tqdm(range(iterations_per_val_epoch)):
            assert self._val_loader_iter is not None
            try:
                data = next(self._val_loader_iter)
            except StopIteration:
                self._val_loader_iter = iter(val_loader)
                self._val_loader = val_loader
                data = next(self._val_loader_iter)
            images, source_labels = data
            images = images.to(self.device).float()
            source_labels = source_labels.long().to(self.device)
            target_labels = self._to_target_labels(source_labels)
            target_concepts = (
                self._to_concept_labels(source_labels)
                if self.training_mode in ("concept", "concept-bottleneck")
                else None
            )

            loss, outputs, concept_outputs, loss_components = self.batch_predict_loss(
                images, target_labels, target_concepts
            )
            assert isinstance(loss, torch.Tensor), "Loss must be a torch.Tensor"
            running_loss += loss.item()
            for k, v in loss_components.items():
                running_loss_components[k] = running_loss_components.get(k, 0.0) + v

            if evaluator is not None:
                if self.training_mode == "concept":
                    target_concept_preds = postprocess_predicted_concepts(
                        target_concepts.detach().cpu().numpy(),
                        self.concept_matrix,
                        self.conceptid2labelid,
                    ).to(self.device)
                    for metric in evaluator.metric_dict.values():
                        metric.update(outputs, target_concept_preds)
                else:
                    if outputs.ndim > 3:
                        outputs = outputs.argmax(dim=1)
                    for metric in evaluator.metric_dict.values():
                        metric.update(outputs, target_labels)

            if evaluator is not None and self.training_mode in ("concept-bottleneck", "concept"):
                evaluator.evaluate_concepts(concept_outputs, target_concepts)

        if evaluator is not None:
            for metric_name in evaluator.metric_dict:
                metric_results[metric_name] = (
                    evaluator.metric_dict[metric_name].compute().cpu().numpy()
                )
                if metric_results[metric_name].ndim == 0:
                    metric_results[metric_name] = metric_results[metric_name].item()
                evaluator.metric_dict[metric_name].reset()

            if self.training_mode in ("concept-bottleneck", "concept"):
                metric_results.update(evaluator.compute_concept_metric_results())

        last_loss = running_loss / iterations_per_val_epoch
        avg_loss_components = {
            k: v / iterations_per_val_epoch for k, v in running_loss_components.items()
        }
        for k, v in avg_loss_components.items():
            metric_results[f"loss/{k}"] = v
        return last_loss, metric_results

    @torch.no_grad()  # type:ignore
    def predict(
        self,
        image: torch.Tensor | NDArray[Any],
        transform: A.BasicTransform | None = None,
    ) -> NDArray[Any]:
        """Predict the (target-space) segmentation for a single image."""
        if transform:
            image = transform(image=image)["image"]
        inputs = torch.tensor(image).unsqueeze(0)

        pred = self.batch_predict(inputs)
        return pred.argmax(dim=1).cpu().numpy()[0]
