import contextlib
import faulthandler
import logging
import os
import threading
import time
from collections.abc import Callable, Iterable, Iterator
from typing import Any

import albumentations as A
import numpy as np
import pandas as pd
import torch
import transformers

# from mermaidseg.model.eval import Evaluator
from numpy.typing import NDArray
from torch.nn.parallel import DistributedDataParallel
from torch.utils.data import DataLoader
from tqdm import tqdm

import mermaidseg.model.loss
import mermaidseg.model.models
from mermaidseg.dataset_reconciliation.concepts import (
    DEFAULT_CLASS_TO_CONCEPTS_CSV,
    postprocess_predicted_concepts,
    source_labels_to_concepts,
)
from mermaidseg.dataset_reconciliation.label_mapping import (
    source_labels_to_target_labels,
)
from mermaidseg.dataset_reconciliation.taxonomy_tree import build_taxonomy_tree
from mermaidseg.io import ConfigDict
from mermaidseg.model.data_parallel import (
    DDPLossStep,
    MetricShard,
    ShardReplicas,
    ShardStep,
    apply_concept_bottleneck_loss,
    build_shard_replicas,
    parallel_forward_loss,
    require_data_parallel_gpus,
)
from mermaidseg.model.distance_loss import TaxonomicDistanceLoss
from mermaidseg.model.distributed import (
    all_ranks_finite,
    current_context,
    is_distributed,
    is_main_process,
    reduce_components,
    require_world_size,
)
from mermaidseg.model.lr_schedule import ConstantThenLinearLR
from mermaidseg.model.taxonomy_loss import TaxonomicLoss

logger = logging.getLogger(__name__)

_BATCH_LOSS_KEY = "__batch_loss__"

STALL_DUMP_ENV = "MERMAIDSEG_STALL_DUMP_SEC"
DEFAULT_STALL_DUMP_SEC = 180.0
SLOW_FETCH_WARN_SEC = 60.0


def stall_dump_seconds_from_env() -> float:
    """Seconds one optimizer update may run before every thread's stack is dumped.

    ``MERMAIDSEG_STALL_DUMP_SEC`` (default 180). 0 or a negative value disables the dump.
    """
    raw = os.environ.get(STALL_DUMP_ENV, "").strip()
    seconds = DEFAULT_STALL_DUMP_SEC if not raw else float(raw)
    return max(0.0, seconds)


def watch_for_stalls(updates: Iterable[int], seconds: float) -> Iterator[int]:
    """Yield ``updates``, re-arming a faulthandler stack dump before each one.

    The timer runs in C, so it still fires when the main thread holds the GIL.
    It is cancelled when iteration ends, breaks, or raises.
    """
    if seconds <= 0:
        yield from updates
        return
    try:
        for update in updates:
            faulthandler.dump_traceback_later(seconds, repeat=False, exit=False)
            yield update
    finally:
        faulthandler.cancel_dump_traceback_later()


def records_train_metrics(microbatch_index: int, every: int) -> bool:
    """True for the training microbatches whose predictions update the metrics.

    ``microbatch_index`` counts microbatches since this ``MetaModel`` was built,
    starting at 0, so the first microbatch is always scored.
    """
    return int(microbatch_index) % max(1, int(every)) == 0


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


def next_epoch_from_checkpoint(checkpoint: dict[str, Any]) -> int:
    """Return the epoch index a resumed run should start at.

    Epoch-boundary checkpoints, including every checkpoint written before step
    checkpoints existed, omit ``epoch_complete`` or set it true. Those resume at
    ``epoch + 1``. A mid-epoch ``model_step*`` checkpoint sets ``epoch_complete``
    false and continues the saved epoch.
    """
    saved = int(checkpoint.get("epoch", -1))
    if bool(checkpoint.get("epoch_complete", True)):
        return saved + 1
    return max(saved, 0)


def restore_warmup_progress(meta_model: Any, checkpoint: dict[str, Any]) -> None:
    """Restore how many warmup steps have already run.

    New checkpoints store ``warmup_iters_completed``. Older ones do not; any of
    those is an epoch boundary, and warmup finishes inside epoch 0, so mark it done.
    """
    if "warmup_iters_completed" in checkpoint:
        meta_model._warmup_iters_completed = int(checkpoint["warmup_iters_completed"])
        return
    if getattr(meta_model, "warmup_iters", 0) > 0:
        meta_model._warmup_iters_completed = meta_model.warmup_iters


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
        self.global_step = 0

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
        # Fail before the encoder loads. A 1-GPU job with the 768px minibatch
        # would run the KL term on the whole batch.
        self.data_parallel = bool(training_kwargs.pop("data_parallel", False))
        self._metric_shards = None
        self._shard_replicas: ShardReplicas | None = None
        # One process per GPU (torchrun). The process group must already exist;
        # each rank's loader yields that rank's shard of the global microbatch.
        self.distributed = bool(training_kwargs.pop("distributed", False))
        self._ddp_step: DistributedDataParallel | None = None
        self.world_size = 1
        if self.data_parallel and self.distributed:
            raise ValueError("data_parallel and distributed are mutually exclusive.")
        if self.data_parallel:
            require_data_parallel_gpus(torch.cuda.device_count())
            logger.info(
                "data_parallel: %s GPUs; each shard computes its own loss.",
                torch.cuda.device_count(),
            )
        if self.distributed:
            if not is_distributed():
                raise RuntimeError(
                    "distributed requires an initialized process group (launch with torchrun)."
                )
            self.world_size = current_context().world_size
            require_world_size(self.world_size)
            logger.info(
                "distributed: %s ranks; each rank computes its shard's loss.", self.world_size
            )

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
        self.grad_accum_steps = max(1, int(training_kwargs.get("grad_accum_steps", 1)))
        # Training metrics only; validation scores every batch.
        self.train_metrics_every = max(1, int(training_kwargs.get("train_metrics_every", 1)))
        self._train_microbatches_seen = 0
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
            model_kwargs.pop("detach_concepts", None)
            model_kwargs["detach_concepts"] = detach_concepts
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
            self.loss = loss_cls(**loss_kwargs)

        # Optional taxonomy-aware terms (partial-label CE, KL coherence) added to the concept loss.
        self.taxonomic_loss = None
        taxonomic_cfg = training_kwargs.get("taxonomic_loss")
        if taxonomic_cfg is not None:
            if self.training_mode != "concept-bottleneck":
                raise ValueError("taxonomic_loss requires training_mode 'concept-bottleneck'.")
            taxonomic_kwargs = dict(taxonomic_cfg)
            # Reduce the partial loss like the concept loss's per-rank cross-entropy.
            for key in ("per_pixel_loss_weight", "per_image_loss_weight"):
                if key in training_kwargs.get("loss", {}):
                    taxonomic_kwargs.setdefault(key, training_kwargs.loss[key])
            # Same tree as the hierarchical metrics: class_to_concepts.csv, permissive by default.
            taxonomy = build_taxonomy_tree(
                training_kwargs.get("concept_mapping_path") or DEFAULT_CLASS_TO_CONCEPTS_CSV,
                ambiguity_policy=taxonomic_kwargs.pop("ambiguity_policy", "permissive"),
            )
            self.taxonomic_loss = TaxonomicLoss(
                taxonomy, self.concept_value2id, **taxonomic_kwargs
            ).to(device)

        # Optional expected tree-distance loss over the rank heads, independent of the above.
        self.distance_loss = None
        distance_cfg = training_kwargs.get("distance_loss")
        if distance_cfg is not None and distance_cfg.get("weight", 0.0) > 0:
            if self.training_mode != "concept-bottleneck":
                raise ValueError("distance_loss requires training_mode 'concept-bottleneck'.")
            distance_kwargs = dict(distance_cfg)
            for key in ("per_pixel_loss_weight", "per_image_loss_weight"):
                if key in training_kwargs.get("loss", {}):
                    distance_kwargs.setdefault(key, training_kwargs.loss[key])
            taxonomy = build_taxonomy_tree(
                training_kwargs.get("concept_mapping_path") or DEFAULT_CLASS_TO_CONCEPTS_CSV,
                ambiguity_policy=distance_kwargs.pop("ambiguity_policy", "permissive"),
            )
            self.distance_loss = TaxonomicDistanceLoss(
                taxonomy, self.concept_value2id, **distance_kwargs
            ).to(device)

        optimizer_cfg = dict(training_kwargs.optimizer)
        optimizer_cls = getattr(torch.optim, optimizer_cfg.pop("type", None))
        # Optional differential learning rates: LoRA adapter params vs. everything else
        # (DPT head + concept/class heads). Split by whether the param name contains
        # "lora_" (matches how ``freeze_encoder`` partitions params in models.py).
        lora_lr = optimizer_cfg.pop("lora_lr", None)
        dpt_lr = optimizer_cfg.pop("dpt_lr", None)
        default_lr = optimizer_cfg.get("lr")

        named_trainable = [(n, p) for n, p in self.model.named_parameters() if p.requires_grad]
        self._trainable_params = [p for _, p in named_trainable]  # kept for grad clipping

        if lora_lr is not None or dpt_lr is not None:
            lora_params = [p for n, p in named_trainable if "lora_" in n]
            other_params = [p for n, p in named_trainable if "lora_" not in n]
            groups = []
            if lora_params:
                groups.append(
                    {"params": lora_params, "lr": lora_lr if lora_lr is not None else default_lr}
                )
            if other_params:
                groups.append(
                    {"params": other_params, "lr": dpt_lr if dpt_lr is not None else default_lr}
                )
            self.optimizer = optimizer_cls(groups, **optimizer_cfg)
        else:
            self.optimizer = optimizer_cls(params=self._trainable_params, **optimizer_cfg)

        self._configure_schedulers()

    def _configure_schedulers(self) -> None:
        """Build the epoch/iteration scheduler and the optional per-iteration warmup.

        ``scheduler.step_every`` is ``epoch`` (default; stepped once in ``train_model``) or
        ``iteration`` (stepped after every optimizer step). Warmup keys are not forwarded to
        the PyTorch scheduler constructor.
        """
        training_kwargs = self.training_kwargs
        if "scheduler" not in training_kwargs:
            self.scheduler = None
            self.scheduler_step_every = "epoch"
            self.warmup_iters = 0
            self.warmup_scheduler = None
            self._warmup_iters_completed = 0
            return

        scheduler_cfg = dict(training_kwargs.scheduler)
        warmup_iters = int(scheduler_cfg.pop("warmup_iters", 2000))
        warmup_start_factor = float(scheduler_cfg.pop("warmup_start_factor", 0.01))
        step_every = str(scheduler_cfg.pop("step_every", "epoch"))
        if step_every not in {"epoch", "iteration"}:
            raise ValueError(
                f"scheduler.step_every must be 'epoch' or 'iteration', got {step_every!r}"
            )
        scheduler_type = scheduler_cfg.pop("type", None)
        if scheduler_type == "ConstantThenLinearLR":
            scheduler_cls = ConstantThenLinearLR
        else:
            scheduler_cls = getattr(torch.optim.lr_scheduler, scheduler_type)
        self.scheduler = scheduler_cls(self.optimizer, **scheduler_cfg)
        self.scheduler_step_every = step_every
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

    def rebuild_scheduler(self) -> None:
        """Drop scheduler state and rebuild it from the current optimizer learning rates.

        Used when a new run is initialized from a checkpoint (cooldown): the loaded optimizer
        already holds the peak learning rates and Adam moments, and the previous schedule must
        not be restored. ``initial_lr`` left by the old scheduler is cleared so the new
        schedule's base rates are the restored group learning rates.
        """
        for group in self.optimizer.param_groups:
            group.pop("initial_lr", None)
        self._configure_schedulers()

    def _step_warmup_scheduler(self) -> None:
        """Step the linear LR warmup once per training iteration."""
        if self.warmup_scheduler is None or self._warmup_iters_completed >= self.warmup_iters:
            return
        self.warmup_scheduler.step()
        self._warmup_iters_completed += 1

    def _step_iteration_scheduler(self) -> None:
        """Step an iteration-level schedule once per optimizer step, after warmup."""
        if self.scheduler is None or self.scheduler_step_every != "iteration":
            return
        if self.warmup_iters > 0 and self._warmup_iters_completed < self.warmup_iters:
            return
        self.scheduler.step()

    def _accumulate_loss(self, loss: torch.Tensor) -> None:
        """Backward one microbatch, averaged in fp32 across the accumulation window.

        ``GradScaler`` is enabled only for fp16. bf16 autocast leaves it disabled, so
        this division is the only scale applied to the loss. Callers must pass a
        finite loss; a non-finite window is discarded before this runs.
        """
        if loss.dtype != torch.float32:
            loss = loss.float()
        self.scaler.scale(loss / self.grad_accum_steps).backward()

    def _abort_accumulation(self, did_backward: bool) -> None:
        """Drop a partial window so it cannot become a short optimizer step.

        When fp16 scaling already ran, ``update()`` resets the scaler so the next
        window may call ``scale().backward()`` again. Disabled scalers (bf16) no-op.
        """
        if did_backward and self.scaler.is_enabled():
            self.scaler.update()
        self.optimizer.zero_grad(set_to_none=True)
        if self._shard_replicas is not None:
            self._shard_replicas.zero_grads()

    def _apply_accumulated_step(self) -> bool:
        """Clip once, then step Adam/SGD on the averaged gradient.

        With data parallel, the replicas' gradients are first summed onto the
        primary parameters, so clipping sees the whole window's gradient.

        Returns True when the optimizer step was applied (not skipped by the scaler
        or a non-finite grad norm).
        """
        if self._shard_replicas is not None:
            self._shard_replicas.reduce_grads()
            self._shard_replicas.invalidate()
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

        When ``source_to_target_lookup`` is ``None``, source labels are assumed to already be in
        target space (identity passthrough — useful for single-source or synthetic pipelines).
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

    def _update_evaluator(
        self,
        evaluator: Any,
        outputs: torch.Tensor,
        target_labels: torch.Tensor,
        concept_outputs: torch.Tensor | None,
        target_concepts: torch.Tensor | None,
    ) -> None:
        """Update ``evaluator`` with one batch already small enough to sit on its device."""
        device = getattr(evaluator, "device", self.device)
        outputs = outputs.detach().to(device)
        target_labels = target_labels.to(device)
        if concept_outputs is not None:
            concept_outputs = concept_outputs.detach().to(device)
        if target_concepts is not None:
            target_concepts = target_concepts.to(device)

        if self.training_mode == "concept":
            assert target_concepts is not None
            target_concept_preds = postprocess_predicted_concepts(
                target_concepts.detach().cpu().numpy(),
                self.concept_matrix,
                self.conceptid2labelid,
            ).to(device)
            for metric in evaluator.metric_dict.values():
                metric.update(outputs, target_concept_preds)
        else:
            preds = outputs.argmax(dim=1) if outputs.ndim > 3 else outputs
            for metric in evaluator.metric_dict.values():
                metric.update(preds, target_labels)
        if self.training_mode in ("concept-bottleneck", "concept"):
            evaluator.evaluate_concepts(concept_outputs, target_concepts)

    def _consume_metric_shards(self, evaluator: Any) -> bool:
        """Update metrics from one GPU shard at a time.

        Returns False when this step was not data-parallel. Shards are dropped
        after they are consumed so a 768px map does not stay on GPU 0 alongside
        the next shard.
        """
        shards = self._metric_shards
        self._metric_shards = None
        if not shards:
            return False
        for shard in shards:
            outputs = shard.outputs
            target_labels = shard.target_labels
            concept_outputs = shard.concept_outputs
            target_concepts = shard.target_concepts
            shard.outputs = None
            shard.concept_outputs = None
            shard.target_labels = None
            shard.target_concepts = None
            assert outputs is not None
            assert target_labels is not None
            self._update_evaluator(
                evaluator, outputs, target_labels, concept_outputs, target_concepts
            )
        return True

    def _record_batch_metrics(
        self,
        evaluator: Any | None,
        outputs: torch.Tensor | None,
        target_labels: torch.Tensor | None,
        concept_outputs: torch.Tensor | None,
        target_concepts: torch.Tensor | None,
    ) -> None:
        """Record train or val metrics for the batch that just finished.

        ``evaluator=None`` records nothing and drops any data-parallel shards.
        """
        if evaluator is None:
            self._metric_shards = None
            return
        if self.distributed and getattr(evaluator, "rank_output_consumers", None):
            raise RuntimeError("rank output consumers are not reduced across ranks.")
        if self._consume_metric_shards(evaluator):
            return
        if self.data_parallel or self.distributed:
            raise RuntimeError("data-parallel forward did not return metric shards.")
        assert outputs is not None
        assert target_labels is not None
        self._update_evaluator(evaluator, outputs, target_labels, concept_outputs, target_concepts)

    def _get_shard_replicas(self) -> ShardReplicas:
        """Per-GPU copies of the model, losses, and lookups, built on first use.

        Built lazily so they copy weights loaded after construction
        (``--init-weights`` / ``--checkpoint``). Rebuilt if ``model`` or a loss
        module has been replaced since.
        """
        replicas = self._shard_replicas
        if replicas is not None:
            primary = replicas.modules[0]
            if (
                primary.model is self.model
                and primary.loss_fn is self.loss
                and primary.taxonomic_loss is self.taxonomic_loss
                and primary.distance_loss is self.distance_loss
            ):
                return replicas
        if self.loss is None:
            raise RuntimeError("data_parallel requires a loss.")
        if self.source_to_concepts_lookup is None:
            raise RuntimeError("data_parallel requires concept labels.")
        self._shard_replicas = build_shard_replicas(
            self.model,
            self.loss,
            self.taxonomic_loss,
            self.distance_loss,
            self.source_to_target_lookup,
            self.source_to_concepts_lookup,
            use_amp=self.use_amp,
            amp_dtype=self.amp_dtype,
        )
        return self._shard_replicas

    def _get_ddp_step(self) -> DistributedDataParallel:
        """This rank's lookup, forward, and loss, wrapped in DDP, built on first use.

        Built lazily, like the replicas, so it wraps weights loaded after
        construction and the ``requires_grad`` set by ``freeze_encoder``.
        Rebuilt if ``model`` or a loss module has been replaced since.
        """
        ddp = self._ddp_step
        if ddp is not None:
            step = ddp.module.step
            if (
                step.model is self.model
                and step.loss_fn is self.loss
                and step.taxonomic_loss is self.taxonomic_loss
                and step.distance_loss is self.distance_loss
            ):
                return ddp
        if self.loss is None:
            raise RuntimeError("distributed requires a loss.")
        if self.source_to_concepts_lookup is None:
            raise RuntimeError("distributed requires concept labels.")
        step = DDPLossStep(
            ShardStep(
                self.model,
                self.loss,
                self.taxonomic_loss,
                self.distance_loss,
                self.source_to_target_lookup,
                self.source_to_concepts_lookup,
                self.use_amp,
                self.amp_dtype,
            )
        )
        device = torch.device(self.device)
        # Buffers (lookups, loss matrices, model buffers) are identical on every
        # rank and never updated, so they are not re-broadcast each forward.
        # A bucket view of the 1x1 DPT/concept conv grads is not stride-compatible
        # (see the reducer's "Grad strides do not match bucket view strides"
        # warning), so leave the grads in their own storage.
        self._ddp_step = DistributedDataParallel(
            step,
            device_ids=[device.index] if device.type == "cuda" else None,
            broadcast_buffers=False,
        )
        return self._ddp_step

    def _grad_sync_context(self, sync: bool) -> contextlib.AbstractContextManager[Any]:
        """``no_sync`` for every microbatch but the last of a distributed window."""
        if not self.distributed or sync:
            return contextlib.nullcontext()
        return self._get_ddp_step().no_sync()

    def _finish_skipped_reduction(self) -> None:
        """Finish the reduction a synced forward prepared when its backward is skipped.

        A synced DDP forward arms the gradient hooks for the next backward. If
        that backward never runs, the next ``no_sync`` backward would all-reduce.
        Back-propagating zero through every trainable parameter completes the
        reduction on all ranks (they skip together); the window's gradients
        are then discarded.
        """
        params = [p for p in self._get_ddp_step().module.parameters() if p.requires_grad]
        if params:
            sum(p.sum() * 0.0 for p in params).backward()

    def batch_predict_loss(
        self,
        images: torch.Tensor,
        target_labels: torch.Tensor | None = None,
        target_concepts: torch.Tensor | None = None,
        target_dim: tuple[int, int] | None = None,
        *,
        source_labels: torch.Tensor | None = None,
        return_maps: bool = True,
    ) -> tuple[torch.Tensor, torch.Tensor | None, torch.Tensor | None, dict[str, float]]:
        """Perform batch prediction and compute the loss.

        Args:
            images: Image tensor batch (already on-device, float).
            target_labels: Target-space label tensor (already on-device, long).
                Unused with ``data_parallel``.
            target_concepts: Target concept tensor. Required for ``"concept"``
                and ``"concept-bottleneck"`` modes; unused otherwise and with
                ``data_parallel``.
            target_dim: Target spatial dimensions for output resizing.
                Defaults to the input spatial dimensions.
            source_labels: Global source-label map (on cuda:0, long). Required
                with ``data_parallel``: each GPU maps its shard to target labels
                and concepts itself.
            return_maps: With ``data_parallel``, keep the per-GPU maps in
                ``_metric_shards``. Pass False when the batch is not scored.
        Returns:
            A 4-tuple of ``(loss, outputs, concept_outputs, loss_components)``.
            With ``data_parallel``, ``outputs`` and ``concept_outputs`` are
            ``None``; the maps stay on each GPU in ``_metric_shards`` so they
            are not concatenated onto GPU 0 before backward.
        """
        self._metric_shards = None
        loss = None
        outputs = None
        concept_outputs = None
        loss_components: dict[str, float] = {}

        if self.data_parallel:
            if self.training_mode != "concept-bottleneck":
                raise RuntimeError("data_parallel is implemented for concept-bottleneck training.")
            if source_labels is None:
                raise RuntimeError("data_parallel requires source labels.")
            parallel = parallel_forward_loss(
                self._get_shard_replicas(),
                images,
                source_labels,
                return_maps=return_maps,
            )
            self._metric_shards = parallel.shards if return_maps else None
            return parallel.loss, None, None, parallel.components

        if self.distributed:
            if self.training_mode != "concept-bottleneck":
                raise RuntimeError("distributed is implemented for concept-bottleneck training.")
            if source_labels is None:
                raise RuntimeError("distributed requires source labels.")
            ddp = self._get_ddp_step()
            # Validation (no grad) bypasses the DDP wrapper; it has nothing to reduce.
            # Training must backward the tensor DDP returned, not a copy from the
            # stashed tuple, so the reducer hooks fire.
            if torch.is_grad_enabled():
                loss = ddp(images, source_labels, return_maps=return_maps)
                output = ddp.module.last
                if output is None:
                    raise RuntimeError("distributed forward did not stash its outputs.")
                _, outputs, concept_outputs, labels, concepts, loss_components = output
            else:
                loss, outputs, concept_outputs, labels, concepts, loss_components = ddp.module.step(
                    images,
                    source_labels,
                    return_maps=return_maps,
                )
            self._metric_shards = (
                [
                    MetricShard(
                        outputs=outputs,
                        concept_outputs=concept_outputs,
                        target_labels=labels,
                        target_concepts=concepts,
                    )
                ]
                if return_maps
                else None
            )
            return loss, None, None, loss_components

        with torch.autocast(device_type="cuda", dtype=self.amp_dtype, enabled=self.use_amp):
            segmentation_outputs = self.model(images)

        if self.training_mode == "concept-bottleneck":
            assert target_concepts is not None, (
                "target_concepts must be provided in 'concept-bottleneck' mode"
            )
            assert self.loss is not None
            loss, outputs, concept_outputs, loss_components = apply_concept_bottleneck_loss(
                segmentation_outputs,
                self.loss,
                self.taxonomic_loss,
                self.distance_loss,
                target_labels,
                target_concepts,
            )

        elif self.training_mode == "concept":
            assert target_concepts is not None, "target_concepts must be provided in 'concept' mode"
            concept_outputs = segmentation_outputs.logits.float()
            loss, loss_components = self.loss(concept_outputs, target_concepts, target_labels)
            concept_outputs = torch.sigmoid(concept_outputs)
            outputs = postprocess_predicted_concepts(
                concept_outputs.detach().cpu().numpy(),
                self.concept_matrix,
                self.conceptid2labelid,
            ).to(self.device)

        else:
            outputs = segmentation_outputs.logits.float()
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
        run_logger: Any | None = None,
        log_interval: int = 1,
        epoch: int | None = None,
        on_optimizer_step: Callable[[Any], None] | None = None,
    ) -> tuple[float, dict[str, float | NDArray[np.float64]], dict[str, float | int]]:
        """Trains the model for one epoch using the provided data loader.

        One iteration is one optimizer update. ``grad_accum_steps`` microbatches are
        forwarded and averaged in fp32 before that update. ``global_step`` advances
        once per successful update, which is once per batch when accumulation is 1.

        A resumed epoch continues at ``global_step - epoch * iterations``. Epoch 0 of
        a fresh run starts at 0. The next epoch in the same process starts at 0 as
        well, because ``global_step`` already counts the updates of earlier epochs.

        Args:
            train_loader: DataLoader yielding ``(inputs, source_labels)`` batches.
                Each batch is one microbatch.
            evaluator: Optional evaluator for computing per-epoch metrics.
            run_logger: Optional experiment logger; when provided, per-update train
                losses are logged against the monotonic ``self.global_step`` axis.
                (Named ``run_logger`` to avoid shadowing the module-level ``logger``.)
            log_interval: Log per-update train losses every ``log_interval`` optimizer
                steps. Defaults to 1 (every update).
            epoch: Current epoch index. Used to place a mid-epoch resume inside this
                epoch's iteration range.
            on_optimizer_step: Called after each successful optimizer update, once
                ``global_step`` and the LR schedule have advanced.

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
        # The phase barriers only make the data/forward/backward totals exact.
        # With data parallel they would also stall the other GPUs, so those
        # totals are host-side times there; perf/step_sec stays wall time.
        sync_timing = use_cuda and not (self.data_parallel or self.distributed)

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

        epoch_index = 0 if epoch is None else int(epoch)
        start_update = int(self.global_step) - epoch_index * int(iterations_per_train_epoch)
        if start_update < 0 or start_update > iterations_per_train_epoch:
            logger.warning(
                "train_epoch: resume offset %s is outside 0..%s; starting this epoch at 0.",
                start_update,
                iterations_per_train_epoch,
            )
            start_update = 0
        n_updates = iterations_per_train_epoch - start_update

        rank = current_context().rank
        stall_dump_sec = stall_dump_seconds_from_env()
        if stall_dump_sec > 0:
            # faulthandler labels each stack by thread id only.
            logger.warning(
                "rank %d: an update running over %.0f s dumps all stacks to stderr; "
                "this rank's main thread is 0x%016x",
                rank,
                stall_dump_sec,
                threading.get_ident(),
            )

        for _update_idx in watch_for_stalls(
            tqdm(range(start_update, iterations_per_train_epoch), disable=not is_main_process()),
            stall_dump_sec,
        ):
            step_start = time.perf_counter()
            step_data_wait = 0.0
            self.optimizer.zero_grad(set_to_none=True)
            micro_losses: list[torch.Tensor] = []
            micro_components: list[dict[str, float]] = []
            window_failed = False
            did_backward = False

            for _micro in range(self.grad_accum_steps):
                assert self._train_loader_iter is not None
                fetch_start = time.perf_counter()
                try:
                    data = next(self._train_loader_iter)
                except StopIteration:
                    self._train_loader_iter = iter(train_loader)
                    self._train_loader = train_loader
                    data = next(self._train_loader_iter)
                fetch_sec = time.perf_counter() - fetch_start
                step_data_wait += fetch_sec
                if fetch_sec > SLOW_FETCH_WARN_SEC:
                    logger.warning(
                        "rank %d: train loader took %.0f s for one microbatch (update %d)",
                        rank,
                        fetch_sec,
                        int(self.global_step) + 1,
                    )

                if sync_timing:
                    torch.cuda.synchronize()
                data_time_total += time.perf_counter() - batch_end

                images, source_labels = data
                if images.numel() == 0:
                    if self.distributed:
                        # Skipping on one rank would leave the others waiting in a collective.
                        raise RuntimeError("train_epoch: empty batch in a distributed run.")
                    logger.warning(
                        "train_epoch: skipping an empty batch (all items failed to load)."
                    )
                    window_failed = True
                    batch_end = time.perf_counter()
                    break
                record_metrics = evaluator is not None and records_train_metrics(
                    self._train_microbatches_seen, self.train_metrics_every
                )
                self._train_microbatches_seen += 1
                images = images.to(self.device, non_blocking=True).float()
                source_labels = source_labels.to(self.device, non_blocking=True).long()
                if self.data_parallel or self.distributed:
                    # Each GPU maps its own shard of source labels.
                    target_labels = None
                    target_concepts = None
                else:
                    target_labels = self._to_target_labels(source_labels)
                    target_concepts = (
                        self._to_concept_labels(source_labels)
                        if self.training_mode in ("concept", "concept-bottleneck")
                        else None
                    )

                # Distributed: gradients are all-reduced once, in the window's last backward.
                sync_grads = _micro == self.grad_accum_steps - 1
                with self._grad_sync_context(sync_grads):
                    if sync_timing:
                        torch.cuda.synchronize()
                    forward_start = time.perf_counter()
                    loss, outputs, concept_outputs, loss_components = self.batch_predict_loss(
                        images,
                        target_labels,
                        target_concepts,
                        source_labels=source_labels,
                        return_maps=record_metrics,
                    )
                    if sync_timing:
                        torch.cuda.synchronize()
                    forward_time_total += time.perf_counter() - forward_start
                    assert isinstance(loss, torch.Tensor), "Loss must be a torch.Tensor"

                    # Distributed: skip the window if any rank's shard loss is non-finite,
                    # as the single-process path does when the mean over shards is.
                    finite = (
                        all_ranks_finite(loss) if self.distributed else bool(torch.isfinite(loss))
                    )
                    if not finite:
                        logger.warning(
                            "train_epoch: skipping non-finite loss (value=%s)", loss.item()
                        )
                        self._metric_shards = None
                        del loss
                        if self.distributed and sync_grads:
                            self._finish_skipped_reduction()
                        window_failed = True
                        batch_end = time.perf_counter()
                        break

                    if sync_timing:
                        torch.cuda.synchronize()
                    backward_start = time.perf_counter()
                    self._accumulate_loss(loss)
                    did_backward = True
                    if sync_timing:
                        torch.cuda.synchronize()
                    backward_time_total += time.perf_counter() - backward_start

                micro_losses.append(loss.detach())
                micro_components.append(loss_components)
                num_samples += int(images.size(0)) * self.world_size
                # Drop the graph before metrics. Data-parallel shards are already
                # detached, so this frees activations before the next shard's maps
                # are copied onto the evaluator device.
                if outputs is not None:
                    outputs = outputs.detach()
                if concept_outputs is not None:
                    concept_outputs = concept_outputs.detach()
                del loss

                self._record_batch_metrics(
                    evaluator if record_metrics else None,
                    outputs,
                    target_labels,
                    concept_outputs,
                    target_concepts,
                )

                if sync_timing:
                    torch.cuda.synchronize()
                batch_end = time.perf_counter()

            if window_failed or not micro_losses:
                self._abort_accumulation(did_backward)
                if sync_timing:
                    torch.cuda.synchronize()
                batch_end = time.perf_counter()
                continue

            n_micro = len(micro_losses)
            batch_loss = sum(torch.stack(micro_losses).double().tolist()) / n_micro
            mean_components: dict[str, float] = {}
            for comps in micro_components:
                for key, value in comps.items():
                    mean_components[key] = mean_components.get(key, 0.0) + value
            mean_components = {key: value / n_micro for key, value in mean_components.items()}
            if self.distributed:
                # Mean over ranks of the per-rank window means: the mean over shards
                # the single-process data-parallel path logs.
                mean_components = reduce_components(
                    {_BATCH_LOSS_KEY: batch_loss, **mean_components}
                )
                batch_loss = mean_components.pop(_BATCH_LOSS_KEY)

            if sync_timing:
                torch.cuda.synchronize()
            backward_start = time.perf_counter()
            # Decay starts on the update after warmup finishes. Stepping both on the
            # warmup boundary would consume a decay step while the LR is still rising.
            warmup_done = (
                self.warmup_scheduler is None or self._warmup_iters_completed >= self.warmup_iters
            )
            step_applied = self._apply_accumulated_step()
            if step_applied:
                self.global_step += 1
                self._step_warmup_scheduler()
                if warmup_done:
                    self._step_iteration_scheduler()
            if sync_timing:
                torch.cuda.synchronize()
            backward_time_total += time.perf_counter() - backward_start

            if not step_applied:
                batch_end = time.perf_counter()
                continue

            running_loss += batch_loss
            for key, value in mean_components.items():
                running_loss_components[key] = running_loss_components.get(key, 0.0) + value

            if run_logger is not None and self.global_step % max(log_interval, 1) == 0:
                run_logger.log(
                    {
                        "train/batch_loss": batch_loss,
                        **{f"train/batch_{key}": value for key, value in mean_components.items()},
                        # Host wall time for this update, and the part of it spent
                        # blocked on the train loader.
                        "perf/step_sec": time.perf_counter() - step_start,
                        "perf/data_wait_sec": step_data_wait,
                    },
                    step=self.global_step,
                )
            if on_optimizer_step is not None:
                on_optimizer_step(self)
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

        denom = n_updates if n_updates > 0 else 1
        last_loss = running_loss / denom
        avg_loss_components = {k: v / denom for k, v in running_loss_components.items()}
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

        for _ in tqdm(range(iterations_per_val_epoch), disable=not is_main_process()):
            assert self._val_loader_iter is not None
            try:
                data = next(self._val_loader_iter)
            except StopIteration:
                self._val_loader_iter = iter(val_loader)
                self._val_loader = val_loader
                data = next(self._val_loader_iter)
            images, source_labels = data
            if images.numel() == 0:
                if self.distributed:
                    raise RuntimeError("validation_epoch: empty batch in a distributed run.")
                logger.warning(
                    "validation_epoch: skipping an empty batch (all items failed to load)."
                )
                continue
            images = images.to(self.device, non_blocking=True).float()
            source_labels = source_labels.to(self.device, non_blocking=True).long()
            if self.data_parallel or self.distributed:
                target_labels = None
                target_concepts = None
            else:
                target_labels = self._to_target_labels(source_labels)
                target_concepts = (
                    self._to_concept_labels(source_labels)
                    if self.training_mode in ("concept", "concept-bottleneck")
                    else None
                )

            loss, outputs, concept_outputs, loss_components = self.batch_predict_loss(
                images,
                target_labels,
                target_concepts,
                source_labels=source_labels,
                return_maps=evaluator is not None,
            )
            assert isinstance(loss, torch.Tensor), "Loss must be a torch.Tensor"
            running_loss += loss.item()
            for k, v in loss_components.items():
                running_loss_components[k] = running_loss_components.get(k, 0.0) + v
            del loss

            self._record_batch_metrics(
                evaluator, outputs, target_labels, concept_outputs, target_concepts
            )

        if self.distributed:
            # Mean over ranks of the per-rank sums: the per-batch mean over shards, summed.
            running_loss_components = reduce_components(
                {_BATCH_LOSS_KEY: running_loss, **running_loss_components}
            )
            running_loss = running_loss_components.pop(_BATCH_LOSS_KEY)

        # Distributed: torchmetrics sums each metric's state over ranks in compute().
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
