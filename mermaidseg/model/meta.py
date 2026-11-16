import logging
import os
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
from mermaidseg.dataset_reconciliation.morphology import source_labels_to_morphology
from mermaidseg.io import ConfigDict
from mermaidseg.model import checkpoint as checkpoint_io
from mermaidseg.model.loss import DualTaxonomicalLoss
from mermaidseg.model.training_mode import StandardMode, StandardTrainingConfig, TrainingMode

logger = logging.getLogger(__name__)

# tqdm in SageMaker/CloudWatch (non-TTY) writes a line per refresh, flooding the logs with
# thousands of progress lines per epoch. Throttle to one refresh per this many seconds. Tune via
# MERMAID_TQDM_MININTERVAL (no image rebuild needed); set very high to effectively silence
# per-iteration progress — per-epoch loss/metrics still emit via `logging`.
_TQDM_MININTERVAL = float(os.getenv("MERMAID_TQDM_MININTERVAL", "60"))


def _resolve_amp_dtype(dtype_config: Any | None) -> torch.dtype:
    """Map config strings to torch autocast dtypes."""
    if dtype_config is None:
        return torch.float16
    if isinstance(dtype_config, torch.dtype):
        return dtype_config

    dtype_name = str(dtype_config).lower().replace("-", "").replace("_", "")
    if dtype_name in {"bf16", "bfloat16"}:
        return torch.bfloat16
    if dtype_name in {"fp16", "float16", "half"}:
        return torch.float16
    raise ValueError(
        f"Unsupported mixed_precision_dtype={dtype_config!r}; expected 'bfloat16' or 'float16'."
    )


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
        id2label: dict[int, str] | None = None,
        benthic_hierarchy: dict[str, str | None] | None = None,
        source_to_morphology_lookup: torch.Tensor | None = None,
    ):
        self.run_name = run_name
        self.num_classes = num_classes
        self.num_concepts = num_concepts
        self.device = device
        self.id2label = id2label
        self.benthic_hierarchy = benthic_hierarchy
        self.source_to_morphology_lookup = (
            source_to_morphology_lookup.to(device).float()
            if source_to_morphology_lookup is not None
            else None
        )

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
        self._mode: TrainingMode | None = (
            StandardMode() if self.training_mode == "standard" else None
        )
        if self.training_mode == "standard":
            # Fail-fast type/typo check on a snapshot — does not replace the .pop() reads below.
            StandardTrainingConfig.model_validate({**training_kwargs, "training_mode": "standard"})

        freeze_encoder = training_kwargs.pop(
            "freeze_encoder",
            self.training_mode == "concept-bottleneck",
        )
        detach_concepts = training_kwargs.pop(
            "detach_concepts",
            self.training_mode == "concept-bottleneck",
        )

        self.training_kwargs = training_kwargs
        mixed_precision = training_kwargs.pop("mixed_precision", False)
        mixed_precision_dtype = training_kwargs.pop("mixed_precision_dtype", None)
        max_grad_norm = training_kwargs.pop("max_grad_norm", 1.0)
        self.max_grad_norm = float(max_grad_norm) if max_grad_norm is not None else None
        self.amp_dtype = _resolve_amp_dtype(mixed_precision_dtype)
        self.use_amp = bool(mixed_precision) and torch.cuda.is_available()
        if self.use_amp and self.amp_dtype == torch.bfloat16 and not torch.cuda.is_bf16_supported():
            logger.warning("CUDA bf16 autocast is not supported on this GPU; disabling AMP.")
            self.use_amp = False
        self.scaler = torch.amp.GradScaler(enabled=self.use_amp and self.amp_dtype == torch.float16)
        self.iterations_per_train_epoch = training_kwargs.get("iterations_per_train_epoch")
        self.iterations_per_val_epoch = training_kwargs.get("iterations_per_val_epoch")
        self._loader_state = {
            "train": {"loader": None, "iter": None},
            "val": {"loader": None, "iter": None},
        }
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
            model_kwargs.pop("detach_concepts", None)
            model_kwargs["detach_concepts"] = detach_concepts
        elif self.training_mode == "concept":
            model_kwargs.setdefault(
                "num_classes", self.num_concepts
            )  # Overwrite the number of classes to be the number of concepts, since the model is only predicting concepts which are then mapped to classes via postprocessing
        self.model = model_cls(**model_kwargs)

        if model_checkpoint:
            checkpoint = torch.load(model_checkpoint)
            checkpoint_io.load_into(self.model, checkpoint)

        self.model = self.model.to(device)
        self.freeze_encoder = freeze_encoder
        self.detach_concepts = detach_concepts
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
            loss_name = getattr(loss_cls, "__name__", "")
            if loss_name in ("TaxonomicalLoss", "DualTaxonomicalLoss"):
                if self.id2label is None:
                    raise ValueError(
                        f"{loss_name} requires id2label; pass it to MetaModel from the registry"
                    )
                loss_kwargs.setdefault("id2label", self.id2label)
                if self.benthic_hierarchy is not None:
                    loss_kwargs.setdefault("hierarchy", self.benthic_hierarchy)
            self.loss = loss_cls(**loss_kwargs)
            # Move the loss onto the training device so any registered buffers (e.g.
            # TaxonomicalLoss's distance matrix / level remaps) are co-located with the
            # on-device targets they index — otherwise indexing a CPU buffer with a CUDA
            # target raises on the first GPU batch.
            if isinstance(self.loss, torch.nn.Module):
                self.loss = self.loss.to(device)

        optimizer_cls = getattr(torch.optim, training_kwargs.optimizer.pop("type", None))
        optimizer_kwargs = dict(training_kwargs.optimizer)
        lora_lr = optimizer_kwargs.pop("lora_lr", None)
        self._trainable_params = [p for p in self.model.parameters() if p.requires_grad]
        if lora_lr is not None and getattr(self.model, "use_lora", False):
            base_lr = float(optimizer_kwargs.pop("lr"))
            head_params, lora_params = [], []
            for name, p in self.model.named_parameters():
                if p.requires_grad:
                    (lora_params if "lora_" in name else head_params).append(p)

            param_groups = []
            if head_params:
                param_groups.append({"params": head_params, "lr": base_lr})
            if lora_params:
                param_groups.append({"params": lora_params, "lr": float(lora_lr)})
            self.optimizer = optimizer_cls(param_groups, **optimizer_kwargs)
        else:
            self.optimizer = optimizer_cls(params=self._trainable_params, **optimizer_kwargs)

        if "scheduler" in training_kwargs:
            scheduler_cfg = dict(training_kwargs.scheduler)
            warmup_iters = int(scheduler_cfg.pop("warmup_iters", 2000))
            warmup_start_factor = float(scheduler_cfg.pop("warmup_start_factor", 0.01))
            scheduler_cls = getattr(torch.optim.lr_scheduler, scheduler_cfg.pop("type", None))
            self.scheduler = scheduler_cls(self.optimizer, **scheduler_cfg)
            self.warmup_iters = warmup_iters
            self._warmup_iters_completed = 0
            if warmup_iters > 0:
                self.warmup_scheduler = torch.optim.lr_scheduler.LinearLR(
                    self.optimizer,
                    start_factor=warmup_start_factor,
                    total_iters=warmup_iters,
                )
            else:
                self.warmup_scheduler = None
        else:
            self.scheduler = None
            self.warmup_iters = 0
            self.warmup_scheduler = None
            self._warmup_iters_completed = 0

    def _step_warmup_scheduler(self) -> None:
        """Step the linear LR warmup once per training iteration."""
        if self.warmup_scheduler is None or self._warmup_iters_completed >= self.warmup_iters:
            return
        self.warmup_scheduler.step()
        self._warmup_iters_completed += 1

    def _optimizer_step(self, loss: torch.Tensor) -> bool:
        """Run backward + AMP optimizer step with optional gradient clipping.

        Returns True when the optimizer step was applied (not skipped by the scaler).
        """
        if not torch.isfinite(loss):
            logger.warning("train_epoch: skipping non-finite loss (value=%s)", loss.item())
            self.optimizer.zero_grad(set_to_none=True)
            return False

        self.scaler.scale(loss).backward()
        scale_before = self.scaler.get_scale()
        if self.max_grad_norm is not None and self.max_grad_norm > 0:
            self.scaler.unscale_(self.optimizer)
            grad_norm = torch.nn.utils.clip_grad_norm_(self._trainable_params, self.max_grad_norm)
            if not torch.isfinite(grad_norm):
                logger.warning(
                    "train_epoch: skipping optimizer step due to non-finite grad norm (value=%s)",
                    grad_norm.item(),
                )
                if self.scaler.is_enabled():
                    self.scaler.update()
                self.optimizer.zero_grad(set_to_none=True)
                return False
        self.scaler.step(self.optimizer)
        self.scaler.update()
        self.optimizer.zero_grad(set_to_none=True)
        return self.scaler.get_scale() >= scale_before

    def _to_target_labels(self, source_labels: torch.Tensor) -> torch.Tensor:
        """Map source-space labels to target-space labels via the lookup tensor.

        When ``source_to_target_lookup`` is ``None``, source labels are assumed to
        already be in target space (identity passthrough — useful for single-source or
        synthetic pipelines).
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

    def _concepts_to_label_map(self, concept_scores: torch.Tensor) -> torch.Tensor:
        """Reduce per-pixel concept probabilities to a target-label map on-device.

        Wraps :func:`postprocess_predicted_concepts` (hierarchical argmax over the
        concept matrix). Shared by ``batch_predict``, ``batch_predict_loss``, and the
        concept-mode metric accumulation. All callers pass sigmoid probabilities (in
        ``[0, 1]``), which ``postprocess_predicted_concepts`` thresholds at 0.5.
        """
        return postprocess_predicted_concepts(
            concept_scores.detach().cpu().numpy(),
            self.concept_matrix,
            self.conceptid2labelid,
        ).to(self.device)

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

        with torch.autocast(device_type="cuda", dtype=self.amp_dtype, enabled=self.use_amp):
            segmentation_outputs = self.model(inputs)

        if self.training_mode == "concept-bottleneck":
            concept_outputs = segmentation_outputs.concept_outputs
            outputs = segmentation_outputs.logits
        elif self.training_mode == "concept":
            # Cast to fp32 then sigmoid BEFORE postprocess, exactly as batch_predict_loss does.
            # `.float()` matters under AMP: the autocast logits can be fp16/bf16, and (a) a
            # low-precision sigmoid near the boundary can round to exactly 0.5 and flip the
            # strict >0.5 assignment, (b) bf16 cannot pass through postprocess's .cpu().numpy().
            # postprocess_predicted_concepts thresholds at 0.5 as a probability, so raw logits
            # must be squashed first — feeding raw logits here previously made eval/inference
            # disagree with the training loop's concept->label maps.
            concept_outputs = torch.sigmoid(segmentation_outputs.logits.float())
            outputs = self._concepts_to_label_map(concept_outputs)
        else:
            outputs, concept_outputs = self._mode.predict(segmentation_outputs)

        assert isinstance(outputs, torch.Tensor)
        return outputs, concept_outputs

    def batch_predict_loss(
        self,
        images: torch.Tensor,
        target_labels: torch.Tensor,
        target_concepts: torch.Tensor | None = None,
        target_dim: tuple[int, int] | None = None,
        source_labels: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor | None, dict[str, float]]:
        """Perform batch prediction and compute the loss.

        Args:
            images: Image tensor batch (already on-device, float).
            target_labels: Target-space label tensor (already on-device, long).
            target_concepts: Target concept tensor. Required for ``"concept"``
                and ``"concept-bottleneck"`` modes; unused otherwise.
            target_dim: Target spatial dimensions for output resizing.
                Defaults to the input spatial dimensions.
            source_labels: Global source-label map; required for dual-head morphology
                supervision when ``source_to_morphology_lookup`` is set.
        Returns:
            A 4-tuple of ``(loss, outputs, concept_outputs, loss_components)``.
        """
        loss = None
        outputs = None
        concept_outputs = None
        loss_components: dict[str, float] = {}

        if target_dim is None:
            target_dim = (images.size(-2), images.size(-1))

        with torch.autocast(device_type="cuda", dtype=self.amp_dtype, enabled=self.use_amp):
            segmentation_outputs = self.model(images)

        if self.training_mode == "concept-bottleneck":
            assert target_concepts is not None, (
                "target_concepts must be provided in 'concept-bottleneck' mode"
            )
            concept_outputs = segmentation_outputs.concept_outputs.detach()
            concept_logits = segmentation_outputs.concept_logits.float()
            outputs = segmentation_outputs.logits.float()
            loss, loss_components = self.loss(
                outputs, target_labels, concept_logits, target_concepts
            )

        elif self.training_mode == "concept":
            assert target_concepts is not None, "target_concepts must be provided in 'concept' mode"
            concept_outputs = segmentation_outputs.logits.float()
            loss, loss_components = self.loss(concept_outputs, target_concepts, target_labels)
            concept_outputs = torch.sigmoid(concept_outputs)
            outputs = self._concepts_to_label_map(concept_outputs)

        elif (
            isinstance(self.loss, DualTaxonomicalLoss)
            and getattr(segmentation_outputs, "morphology_logits", None) is not None
        ):
            if source_labels is None or self.source_to_morphology_lookup is None:
                raise ValueError(
                    "DualTaxonomicalLoss requires source_labels and source_to_morphology_lookup"
                )
            outputs = segmentation_outputs.logits.float()
            morph_logits = segmentation_outputs.morphology_logits.float()
            morph_targets = source_labels_to_morphology(
                source_labels, self.source_to_morphology_lookup
            )
            loss, loss_components = self.loss(outputs, target_labels, morph_logits, morph_targets)
            concept_outputs = None

        else:
            loss, outputs, concept_outputs, loss_components = self._mode.predict_and_loss(
                segmentation_outputs, self.loss, target_labels, target_concepts
            )

        assert loss is not None, "Loss is not computed for the given batch."
        assert isinstance(outputs, torch.Tensor)

        return loss, outputs, concept_outputs, loss_components

    @property
    def has_concepts(self) -> bool:
        """True for the concept / concept-bottleneck modes (which carry concept
        targets)."""
        return self.training_mode in ("concept", "concept-bottleneck")

    def _next_batch(
        self, loader: DataLoader, role: str
    ) -> tuple[torch.Tensor, torch.Tensor] | dict[str, torch.Tensor]:
        """Pull the next batch from a per-role persistent iterator, re-iterating on
        exhaustion.

        ``role`` is ``"train"`` or ``"val"``; each keeps its own iterator in
        ``_loader_state`` so the train and validation positions never interfere —
        preserving the pre-unification behavior.
        """
        state = self._loader_state[role]
        if state["loader"] is not loader:
            state["loader"] = loader
            state["iter"] = iter(loader)
        try:
            return next(state["iter"])
        except StopIteration:
            state["iter"] = iter(loader)
            return next(state["iter"])

    def _run_epoch(
        self,
        loader: DataLoader[tuple[torch.Tensor, torch.Tensor] | dict[str, torch.Tensor]],
        evaluator: Any | None,
        *,
        role: str,
    ) -> tuple[float, dict[str, float | NDArray[np.float64]], dict[str, float | int]]:
        """Shared train/validation epoch body — the single owner of the per-epoch loop.

        ``role="train"`` runs the optimizer step + warmup, records timing, and counts
        ``num_samples``; ``role="val"`` (invoked under ``@torch.no_grad()`` by
        :meth:`validation_epoch`) does none of those. Everything else — loader iteration,
        empty-batch skip, per-mode prediction/loss, metric accumulation, and loss
        averaging — is identical for both, so train and validation can no longer drift
        apart (the structural cause of the historical ``val = 0.0`` bug).
        """
        train = role == "train"
        if train and self.freeze_encoder and hasattr(self.model, "freeze_encoder"):
            self.model.freeze_encoder()

        iterations = self.iterations_per_train_epoch if train else self.iterations_per_val_epoch
        if iterations is None:
            iterations = len(loader)
        if iterations <= 0:
            raise ValueError(f"iterations_per_{role}_epoch must be > 0.")

        running_loss = 0.0
        running_loss_components: dict[str, float] = {}
        metric_results: dict[str, float | NDArray[np.float64]] = {}
        # Timing (and its cuda syncs) is train-only; validation returns no timing.
        use_cuda = train and self.device != "cpu" and torch.cuda.is_available()
        data_time_total = forward_time_total = backward_time_total = 0.0
        num_samples = 0

        def _now() -> float:
            if use_cuda:
                torch.cuda.synchronize()
            return time.perf_counter()

        batch_end = _now()

        for _ in tqdm(range(iterations), mininterval=_TQDM_MININTERVAL):
            data = self._next_batch(loader, role)
            data_time_total += _now() - batch_end

            images, source_labels = data
            if images.numel() == 0:
                logger.warning(
                    "%s: skipping an empty batch (all items failed to load).",
                    "train_epoch" if train else "validation_epoch",
                )
                continue
            images = images.to(self.device).float()
            source_labels = source_labels.long().to(self.device)
            target_labels = self._to_target_labels(source_labels)
            target_concepts = self._to_concept_labels(source_labels) if self.has_concepts else None

            forward_start = _now()
            loss, outputs, concept_outputs, loss_components = self.batch_predict_loss(
                images,
                target_labels,
                target_concepts,
                source_labels=source_labels,
            )
            forward_time_total += _now() - forward_start

            assert isinstance(loss, torch.Tensor), "Loss must be a torch.Tensor"

            if train:
                backward_start = _now()
                if self._optimizer_step(loss):
                    self._step_warmup_scheduler()
                backward_time_total += _now() - backward_start

                if not torch.isfinite(loss):
                    batch_end = _now()
                    continue

            running_loss += loss.item()
            for k, v in loss_components.items():
                running_loss_components[k] = running_loss_components.get(k, 0.0) + v
            if train:
                num_samples += target_labels.size(0)

            if evaluator is not None:
                if self.training_mode == "concept":
                    target_concept_preds = self._concepts_to_label_map(target_concepts)
                    evaluator.accumulate(outputs, target_concept_preds)
                else:
                    evaluator.accumulate(outputs, target_labels)
                if self.has_concepts:
                    evaluator.evaluate_concepts(concept_outputs.detach(), target_concepts)

            batch_end = _now()

        if evaluator is not None:
            metric_results.update(evaluator.compute_and_reset(include_concepts=self.has_concepts))

        last_loss = running_loss / iterations
        avg_loss_components = {k: v / iterations for k, v in running_loss_components.items()}
        for k, v in avg_loss_components.items():
            metric_results[f"loss/{k}"] = v
        timing = {
            "data_loading_sec": data_time_total,
            "forward_sec": forward_time_total,
            "backward_sec": backward_time_total,
            "num_samples": num_samples,
        }
        return last_loss, metric_results, timing

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
        return self._run_epoch(train_loader, evaluator, role="train")

    @torch.no_grad()
    def validation_epoch(
        self,
        val_loader: DataLoader[tuple[torch.Tensor, torch.Tensor] | dict[str, torch.Tensor]],
        evaluator: Any
        | None = None,  # TODO: Should be Evaluator - but this leads to circular import, fix
    ) -> tuple[float, dict[str, float | NDArray[np.float64]]]:
        """Calculate the validation loss and metrics for one epoch."""
        last_loss, metric_results, _timing = self._run_epoch(val_loader, evaluator, role="val")
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
