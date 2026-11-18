"""MetaModel: model, loss, optimizer and train/validation primitives for concept-bottleneck runs."""

import contextlib
import faulthandler
import logging
import threading
import time
from collections.abc import Callable, Iterable, Iterator, Mapping
from typing import Any, cast

import numpy as np
import torch
import transformers

# from mermaidseg.model.eval import Evaluator
from numpy.typing import NDArray
from torch.nn.parallel import DistributedDataParallel
from torch.utils.data import DataLoader
from torch.utils.data.dataloader import _BaseDataLoaderIter
from tqdm import tqdm

from mermaidseg.config import lookup_class
from mermaidseg.dataset_reconciliation.concept_source_mask import apply_concept_source_mask
from mermaidseg.dataset_reconciliation.concepts import source_labels_to_concepts
from mermaidseg.dataset_reconciliation.label_mapping import (
    source_labels_to_target_labels,
)
from mermaidseg.dataset_reconciliation.taxonomy_tree import build_taxonomy_tree
from mermaidseg.inference.predictor import class_predictions
from mermaidseg.io import ConfigDict
from mermaidseg.model.data_parallel import (
    DDPLossStep,
    MetricShard,
    ShardStep,
    apply_concept_bottleneck_loss,
)
from mermaidseg.model.distributed import (
    all_ranks_finite,
    current_context,
    is_distributed,
    is_main_process,
    reduce_components,
    require_world_size,
)
from mermaidseg.model.loss import LOSS_CLASSES
from mermaidseg.model.models import MODEL_CLASSES, load_pretrained_encoder
from mermaidseg.model.taxonomy_loss import TaxonomicLoss

logger = logging.getLogger(__name__)

_BatchLoader = DataLoader[tuple[torch.Tensor, torch.Tensor] | dict[str, torch.Tensor]]

_BATCH_LOSS_KEY = "__batch_loss__"


def _sum_components(parts: Iterable[Mapping[str, torch.Tensor]]) -> dict[str, torch.Tensor]:
    """Key-wise float64 sum of scalar tensors; a key missing from a part adds nothing."""
    totals: dict[str, torch.Tensor] = {}
    for part in parts:
        for key, value in part.items():
            value = value.detach().double()
            totals[key] = totals[key] + value if key in totals else value
    return totals


SLOW_FETCH_WARN_SEC = 60.0

OPTIMIZER_CLASSES: dict[str, type[torch.optim.Optimizer]] = {"AdamW": torch.optim.AdamW}
SCHEDULER_CLASSES: dict[str, type[torch.optim.lr_scheduler.LRScheduler]] = {
    "LinearLR": torch.optim.lr_scheduler.LinearLR,
}


def stall_dump_seconds(training_kwargs: ConfigDict) -> float:
    """Seconds one optimizer update may run before every thread's stack is dumped.

    Args:
        training_kwargs: Training config holding ``stall_dump_sec``.

    Returns:
        ``training.stall_dump_sec``, or 0 when it is 0 or negative, which disables the dump.
    """
    return max(0.0, float(training_kwargs.stall_dump_sec))


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


def _resolve_amp_dtype(dtype_config: Any) -> torch.dtype:
    """Map a config value to the torch autocast dtype; only bfloat16 is supported."""
    if isinstance(dtype_config, torch.dtype):
        return dtype_config

    dtype_name = str(dtype_config).lower().replace("-", "").replace("_", "")
    if dtype_name in {"bf16", "bfloat16"}:
        return torch.bfloat16
    raise ValueError(f"Unsupported mixed_precision_dtype={dtype_config!r}; expected 'bfloat16'.")


def next_epoch_from_checkpoint(checkpoint: dict[str, Any]) -> int:
    """Return the epoch index a resumed run should start at.

    An epoch-boundary checkpoint (``epoch_complete`` true) resumes at ``epoch + 1``. A
    mid-epoch ``model_step*`` checkpoint (``epoch_complete`` false) continues the saved epoch.
    """
    if checkpoint["epoch_complete"]:
        return int(checkpoint["epoch"]) + 1
    return int(checkpoint["epoch"])


def restore_warmup_progress(meta_model: Any, checkpoint: dict[str, Any]) -> None:
    """Restore how many warmup steps have already run."""
    meta_model._warmup_iters_completed = int(checkpoint["warmup_iters_completed"])
    # LinearLR is chainable: its next factor depends on last_epoch, which the optimizer's
    # restored LR already reflects.
    if meta_model.warmup_scheduler is not None:
        meta_model.warmup_scheduler.last_epoch = meta_model._warmup_iters_completed


class MetaModel:
    """Wrapper for training and inference of segmentation models.

    Handles model initialisation, optimizer/scheduler/loss wiring, and the
    train/validation loop primitives. The only training mode is
    ``"concept-bottleneck"``: joint segmentation + concept supervision.

    Source-label datasets emit ``source_labels`` in the joint global source
    space (see :class:`SourceLabelRegistry`); ``MetaModel`` converts them to
    ``target_labels`` (and optional ``concept_labels``) on-device via long-tensor
    lookups passed via ``source_to_target_lookup`` and
    ``source_to_concepts_lookup``.

    Attributes:
        run_name: Identifier for the current run/experiment.
        model_name: Name of the model architecture (a key of ``MODEL_CLASSES``).
        num_classes: Number of segmentation output (target) classes.
        device: Device the model and tensors live on.
        model_kwargs: Model-specific config passed to the architecture.
        training_kwargs: Training hyperparameters
            (epochs, iterations_per_train_epoch, iterations_per_val_epoch, optimizer, scheduler, loss).
        model: The instantiated model.
        loss: Loss function.
        optimizer: Optimiser instance.
        scheduler: LR scheduler.
        source_to_target_lookup: 1-D long tensor of shape ``(N+1,)``
            mapping global source IDs to target label IDs. ``None`` enables identity passthrough
            (synthetic / single-source pipelines whose source space already matches target space).
        source_to_concepts_lookup: Float tensor of shape ``(N+1, C)``.
        conceptid2labelid: Maps concept IDs to target label IDs.
        concept_value2id: Maps concept names and values to IDs.
        resuming_within_epoch: True after loading a mid-epoch step checkpoint.
    """

    run_name: str
    model_name: str
    num_classes: int
    device: str | torch.device
    model_kwargs: ConfigDict
    training_kwargs: ConfigDict
    model: torch.nn.Module | transformers.PreTrainedModel
    loss: torch.nn.Module
    optimizer: torch.optim.Optimizer
    scheduler: torch.optim.lr_scheduler.LRScheduler
    source_to_target_lookup: torch.Tensor | None
    source_to_concepts_lookup: torch.Tensor | None
    conceptid2labelid: dict[int, int] | None
    concept_value2id: dict[str, dict[str, list[int]]] | None
    resuming_within_epoch: bool

    def __init__(
        self,
        run_name: str,
        num_classes: int,
        num_concepts: int | None = None,
        *,
        model_kwargs: ConfigDict,
        training_kwargs: ConfigDict,
        device: str | torch.device,
        source_to_target_lookup: torch.Tensor | None = None,
        source_to_concepts_lookup: torch.Tensor | None = None,
        conceptid2labelid: dict[int, int] | None = None,
        concept_value2id: dict[str, dict[str, list[int]]] | None = None,
    ):
        """Build the model, loss, optimizer, scheduler, and label lookups.

        Args:
            run_name: Identifier for the current run/experiment.
            num_classes: Number of segmentation output (target) classes.
            num_concepts: Number of concept channels.
            model_kwargs: Model-specific config passed to the architecture.
            training_kwargs: Training hyperparameters.
            device: Device the model and tensors live on.
            source_to_target_lookup: Source-to-target label lookup, or ``None`` for identity.
            source_to_concepts_lookup: Source-to-concept lookup.
            conceptid2labelid: Maps concept IDs to target label IDs.
            concept_value2id: Maps concept names and values to IDs.
        """
        self.run_name = run_name
        self.num_classes = num_classes
        self.num_concepts = num_concepts
        self.device = device
        self.global_step = 0
        self.resuming_within_epoch = False

        self.model_name = model_kwargs.pop("name")
        self.model_kwargs = model_kwargs

        self.training_mode = training_kwargs.pop("training_mode")
        assert self.training_mode == "concept-bottleneck", (
            f"Invalid training_mode: {self.training_mode}"
        )
        self._metric_shards: list[MetricShard] | None = None
        # One process per GPU (torchrun). The process group must already exist;
        # each rank's loader yields that rank's shard of the global microbatch.
        self.distributed = bool(training_kwargs.pop("distributed"))
        self._ddp_step: DistributedDataParallel | None = None
        # Keep table of training.concept_source_mask, set by set_concept_source_mask.
        self.concept_source_keep: torch.Tensor | None = None
        self.world_size = 1
        if self.distributed:
            if not is_distributed():
                raise RuntimeError(
                    "distributed requires an initialized process group (launch with torchrun)."
                )
            self.world_size = current_context().world_size
            require_world_size(self.world_size, int(training_kwargs.world_size))
            logger.info(
                "distributed: %s ranks; each rank computes its shard's loss.", self.world_size
            )

        detach_concepts = training_kwargs.pop("detach_concepts")

        self.training_kwargs = training_kwargs
        mixed_precision = training_kwargs.pop("mixed_precision")
        mixed_precision_dtype = training_kwargs.pop("mixed_precision_dtype")
        self.max_grad_norm = float(training_kwargs.pop("max_grad_norm"))
        self.amp_dtype = _resolve_amp_dtype(mixed_precision_dtype)
        self.use_amp = bool(mixed_precision)
        self.device_type = torch.device(device).type
        if (
            self.use_amp
            and self.device_type == "cuda"
            and self.amp_dtype == torch.bfloat16
            and not torch.cuda.is_bf16_supported()
        ):
            raise RuntimeError(
                "mixed_precision_dtype='bfloat16' is not supported on this GPU; "
                "set training.mixed_precision to false."
            )
        self.grad_accum_steps = max(1, int(training_kwargs.grad_accum_steps))
        # Training metrics only; validation scores every batch.
        self.train_metrics_every = max(1, int(training_kwargs.train_metrics_every))
        self._train_microbatches_seen = 0
        self.iterations_per_train_epoch = int(training_kwargs.iterations_per_train_epoch)
        self._train_loader_iter: _BaseDataLoaderIter | None = None
        self._train_loader: _BatchLoader | None = None
        self.iterations_per_val_epoch = int(training_kwargs.iterations_per_val_epoch)
        self._val_loader_iter: _BaseDataLoaderIter | None = None
        self._val_loader: _BatchLoader | None = None
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
        self.conceptid2labelid = conceptid2labelid
        self.concept_value2id = concept_value2id

        model_cls = lookup_class(MODEL_CLASSES, "model", self.model_name)
        model_kwargs["num_classes"] = self.num_classes
        model_kwargs["num_concepts"] = self.num_concepts
        model_kwargs["concept_value2id"] = self.concept_value2id
        model_kwargs["detach_concepts"] = detach_concepts
        # The encoder is built here; the model class takes it, not its hub name.
        constructor_kwargs = {k: v for k, v in model_kwargs.items() if k != "encoder_name"}
        self.model = model_cls(
            base_encoder=load_pretrained_encoder(model_kwargs["encoder_name"]),
            **constructor_kwargs,
        )

        self.model = self.model.to(device)
        self.detach_concepts = detach_concepts

        loss_cls = lookup_class(LOSS_CLASSES, "loss", training_kwargs.loss.pop("type"))
        loss_kwargs = dict(training_kwargs.loss)
        if self.concept_value2id is not None:
            loss_kwargs.setdefault("concept_value2id", self.concept_value2id)
        self.loss = loss_cls(**loss_kwargs)

        # Taxonomic KL coherence term added to the concept loss. The trainer drops the
        # section when its kl_weight is not positive.
        self.taxonomic_loss = None
        taxonomic_cfg = training_kwargs.get("taxonomic_loss")
        if taxonomic_cfg is not None:
            taxonomic_kwargs = dict(taxonomic_cfg)
            # Same tree as the hierarchical metrics.
            taxonomy = build_taxonomy_tree(
                training_kwargs.concept_mapping_path,
                ambiguity_policy=taxonomic_kwargs.pop("ambiguity_policy"),
            )
            self.taxonomic_loss = TaxonomicLoss(
                taxonomy,
                cast("dict[str, dict[str, list[int]]]", self.concept_value2id),
                **taxonomic_kwargs,
            ).to(device)

        optimizer_cfg = dict(training_kwargs.optimizer)
        optimizer_cls = lookup_class(OPTIMIZER_CLASSES, "optimizer", optimizer_cfg.pop("type"))
        # Differential learning rates: LoRA adapter params vs. everything else
        # (DPT head + concept/class heads). Split by whether the param name contains
        # "lora_" (matches how the model freezes non-LoRA encoder params).
        lora_lr = optimizer_cfg.pop("lora_lr")
        dpt_lr = optimizer_cfg.pop("dpt_lr")

        named_trainable = [(n, p) for n, p in self.model.named_parameters() if p.requires_grad]
        self._trainable_params = [p for _, p in named_trainable]  # kept for grad clipping

        lora_params = [p for n, p in named_trainable if "lora_" in n]
        other_params = [p for n, p in named_trainable if "lora_" not in n]
        groups = []
        if lora_params:
            groups.append({"params": lora_params, "lr": lora_lr})
        if other_params:
            groups.append({"params": other_params, "lr": dpt_lr})
        self.optimizer = optimizer_cls(groups, **optimizer_cfg)

        self._configure_schedulers()

    def _configure_schedulers(self) -> None:
        """Build the per-iteration scheduler and the optional per-iteration warmup.

        ``scheduler.step_every`` must be ``iteration``: the schedule steps after every optimizer
        step. Warmup keys are not forwarded to the PyTorch scheduler constructor.
        """
        scheduler_cfg = dict(self.training_kwargs.scheduler)
        warmup_iters = int(scheduler_cfg.pop("warmup_iters"))
        warmup_start_factor = float(scheduler_cfg.pop("warmup_start_factor"))
        step_every = str(scheduler_cfg.pop("step_every"))
        if step_every != "iteration":
            raise ValueError(f"scheduler.step_every must be 'iteration', got {step_every!r}")
        scheduler_cls = lookup_class(SCHEDULER_CLASSES, "scheduler", scheduler_cfg.pop("type"))
        self.scheduler = scheduler_cls(self.optimizer, **scheduler_cfg)
        self.warmup_iters = warmup_iters
        self._warmup_iters_completed = 0
        self.warmup_scheduler: torch.optim.lr_scheduler.LinearLR | None
        if warmup_iters > 0:
            self.warmup_scheduler = torch.optim.lr_scheduler.LinearLR(
                self.optimizer,
                start_factor=warmup_start_factor,
                total_iters=warmup_iters,
            )
        else:
            self.warmup_scheduler = None

    def _step_warmup_scheduler(self) -> None:
        """Step the linear LR warmup once per training iteration."""
        if self.warmup_scheduler is None or self._warmup_iters_completed >= self.warmup_iters:
            return
        self.warmup_scheduler.step()
        self._warmup_iters_completed += 1

    def _step_iteration_scheduler(self) -> None:
        """Step an iteration-level schedule once per optimizer step, after warmup."""
        if self.warmup_iters > 0 and self._warmup_iters_completed < self.warmup_iters:
            return
        self.scheduler.step()

    def _accumulate_loss(self, loss: torch.Tensor) -> None:
        """Backward one microbatch, averaged in fp32 across the accumulation window.

        Callers must pass a finite loss; a non-finite window is discarded before
        this runs.
        """
        if loss.dtype != torch.float32:
            loss = loss.float()
        (loss / self.grad_accum_steps).backward()

    def _abort_accumulation(self) -> None:
        """Drop a partial window so it cannot become a short optimizer step."""
        self.optimizer.zero_grad(set_to_none=True)

    def _apply_accumulated_step(self) -> bool:
        """Clip once, then step Adam/SGD on the averaged gradient.

        Returns True when the optimizer step was applied (not skipped by a
        non-finite grad norm).
        """
        if self.max_grad_norm > 0:
            grad_norm = torch.nn.utils.clip_grad_norm_(self._trainable_params, self.max_grad_norm)
            if not torch.isfinite(grad_norm):
                logger.warning(
                    "train_epoch: skipping optimizer step due to non-finite grad norm (value=%s)",
                    grad_norm.item(),
                )
                self.optimizer.zero_grad(set_to_none=True)
                return False
        self.optimizer.step()
        self.optimizer.zero_grad(set_to_none=True)
        return True

    def _to_target_labels(self, source_labels: torch.Tensor) -> torch.Tensor:
        """Map source-space labels to target-space labels via the lookup tensor.

        When ``source_to_target_lookup`` is ``None``, source labels are assumed to already be in
        target space (identity passthrough — useful for single-source or synthetic pipelines).
        """
        if self.source_to_target_lookup is None:
            return source_labels
        return source_labels_to_target_labels(source_labels, self.source_to_target_lookup)

    def set_concept_source_mask(self, keep: torch.Tensor | None) -> None:
        """Mask source x concept training targets (``training.concept_source_mask``).

        ``keep`` is the ``(S + 1, C)`` bool table of
        :class:`~mermaidseg.dataset_reconciliation.concept_source_mask.ConceptSourceMask`. Training
        batches must then carry a keep-table row per sample (``attach_source_rows``). Validation is
        not masked.
        """
        if keep is not None:
            if self.num_concepts is not None and keep.shape[1] != self.num_concepts:
                raise ValueError(
                    f"concept source mask has {keep.shape[1]} channels, the model {self.num_concepts}"
                )
            keep = keep.to(device=self.device, dtype=torch.bool)
        self.concept_source_keep = keep

    def _to_concept_labels(self, source_labels: torch.Tensor) -> torch.Tensor:
        if self.source_to_concepts_lookup is None:
            raise RuntimeError(
                "MetaModel.source_to_concepts_lookup is not set; cannot run concept modes."
            )
        return source_labels_to_concepts(source_labels, self.source_to_concepts_lookup)

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

        preds = class_predictions(outputs) if outputs.ndim > 3 else outputs
        for metric in evaluator.metric_dict.values():
            metric.update(preds, target_labels)
        evaluator.evaluate_concepts(concept_outputs, target_concepts)

    def _consume_metric_shards(self, evaluator: Any) -> bool:
        """Update metrics from one GPU shard at a time.

        Returns False when this step was not distributed. Shards are dropped
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

        ``evaluator=None`` records nothing and drops any distributed shards.
        """
        if evaluator is None:
            self._metric_shards = None
            return
        if self._consume_metric_shards(evaluator):
            return
        if self.distributed:
            raise RuntimeError("distributed forward did not return metric shards.")
        assert outputs is not None
        assert target_labels is not None
        self._update_evaluator(evaluator, outputs, target_labels, concept_outputs, target_concepts)

    def _get_ddp_step(self) -> DistributedDataParallel:
        """This rank's lookup, forward, and loss, wrapped in DDP, built on first use.

        Built lazily so it wraps weights loaded after
        construction and the ``requires_grad`` set at model construction.
        Rebuilt if ``model`` or a loss module has been replaced since.
        """
        ddp = self._ddp_step
        if ddp is not None:
            step = ddp.module.step
            if (
                step.model is self.model
                and step.loss_fn is self.loss
                and step.taxonomic_loss is self.taxonomic_loss
                and step.concept_keep is self.concept_source_keep
            ):
                return ddp
        if self.source_to_concepts_lookup is None:
            raise RuntimeError("distributed requires concept labels.")
        step = DDPLossStep(
            ShardStep(
                self.model,
                self.loss,
                self.taxonomic_loss,
                self.source_to_target_lookup,
                self.source_to_concepts_lookup,
                self.use_amp,
                self.amp_dtype,
                self.concept_source_keep,
            )
        )
        device = torch.device(self.device)
        # Buffers (lookups, loss matrices, model buffers) are identical on every
        # rank and never updated, so they are not re-broadcast each forward.
        # A bucket view of the 1x1 DPT/concept conv grads is not stride-compatible
        # (see the reducer's "Grad strides do not match bucket view strides"
        # warning), so leave the grads in their own storage.
        # Built outside inference mode: a first build inside validation_epoch would
        # make the reducer's buckets inference tensors that the next backward
        # cannot update in place.
        with torch.inference_mode(False):
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
        *,
        source_labels: torch.Tensor | None = None,
        return_maps: bool = True,
        source_rows: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor | None, torch.Tensor | None, dict[str, torch.Tensor]]:
        """Perform batch prediction and compute the loss.

        Args:
            images: Image tensor batch (already on-device, float).
            target_labels: Target-space label tensor (already on-device, long).
                Unused with ``distributed``.
            target_concepts: Target concept tensor. Required on a single GPU;
                unused with ``distributed``.
            source_labels: Global source-label map (long). Required with
                ``distributed``: each rank maps its shard to target labels and
                concepts itself.
            return_maps: With ``distributed``, keep this rank's maps in
                ``_metric_shards``. Pass False when the batch is not scored.
            source_rows: Keep-table row per sample, for ``concept_source_keep``
                with ``distributed`` (each rank masks its own shard). On a
                single GPU, ``train_epoch`` masks ``target_concepts`` before
                calling this.

        Returns:
            A 4-tuple of ``(loss, outputs, concept_outputs, loss_components)``.
            With ``distributed``, ``outputs`` and ``concept_outputs`` are
            ``None``; the maps stay in ``_metric_shards``.
        """
        self._metric_shards = None
        loss = None
        outputs = None
        concept_outputs = None
        loss_components: dict[str, torch.Tensor] = {}

        if self.distributed:
            if source_labels is None:
                raise RuntimeError("distributed requires source labels.")
            ddp = self._get_ddp_step()
            # Validation (no grad) bypasses the DDP wrapper; it has nothing to reduce.
            # Training must backward the tensor DDP returned, not a copy from the
            # stashed tuple, so the reducer hooks fire.
            if torch.is_grad_enabled():
                loss = ddp(images, source_labels, return_maps=return_maps, source_rows=source_rows)
                output = ddp.module.last
                if output is None:
                    raise RuntimeError("distributed forward did not stash its outputs.")
                _, outputs, concept_outputs, labels, concepts, loss_components = output
            else:
                loss, outputs, concept_outputs, labels, concepts, loss_components = ddp.module.step(
                    images,
                    source_labels,
                    return_maps=return_maps,
                    source_rows=source_rows,
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

        with torch.autocast(
            device_type=self.device_type, dtype=self.amp_dtype, enabled=self.use_amp
        ):
            segmentation_outputs = self.model(images)

        assert target_concepts is not None, (
            "target_concepts must be provided in 'concept-bottleneck' mode"
        )
        loss, outputs, concept_outputs, loss_components = apply_concept_bottleneck_loss(
            segmentation_outputs,
            self.loss,
            self.taxonomic_loss,
            cast("torch.Tensor", target_labels),
            target_concepts,
        )

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
            A 3-tuple of ``(average_loss, metric_results, timing)``. The data, forward
            and backward totals in ``timing`` are host-side wall times: they do not
            wait for queued GPU work, so asynchronous kernels are charged to whichever
            phase next blocks on the device.
        """
        self.model.train()

        iterations_per_train_epoch = self.iterations_per_train_epoch
        if iterations_per_train_epoch <= 0:
            raise ValueError("iterations_per_train_epoch must be > 0.")

        # Per-update losses and components stay on the device until they are logged
        # or the epoch ends; the batch loss rides along under ``_BATCH_LOSS_KEY``.
        running: dict[str, torch.Tensor] = {}
        metric_results: dict[str, float | NDArray[np.float64]] = {}

        data_time_total = 0.0
        forward_time_total = 0.0
        backward_time_total = 0.0
        num_samples = 0

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
        stall_dump_sec = stall_dump_seconds(self.training_kwargs)
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
            micro_components: list[dict[str, torch.Tensor]] = []
            window_failed = False

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

                data_time_total += time.perf_counter() - batch_end

                images, source_labels, *extra = data
                source_rows = extra[0] if extra else None
                if self.concept_source_keep is not None and source_rows is None:
                    raise RuntimeError(
                        "concept_source_mask is set but the train batch has no source rows; "
                        "wrap the train datasets with attach_source_rows."
                    )
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
                if self.concept_source_keep is None:
                    source_rows = None
                else:
                    source_rows = torch.as_tensor(source_rows).to(self.device).long()
                if self.distributed:
                    # Each GPU maps (and masks) its own shard of source labels.
                    target_labels = None
                    target_concepts = None
                else:
                    target_labels = self._to_target_labels(source_labels)
                    target_concepts = self._to_concept_labels(source_labels)
                    if source_rows is not None:
                        target_concepts = apply_concept_source_mask(
                            target_concepts,
                            cast("torch.Tensor", self.concept_source_keep),
                            source_rows,
                        )

                # Distributed: gradients are all-reduced once, in the window's last backward.
                sync_grads = _micro == self.grad_accum_steps - 1
                with self._grad_sync_context(sync_grads):
                    forward_start = time.perf_counter()
                    loss, outputs, concept_outputs, loss_components = self.batch_predict_loss(
                        images,
                        target_labels,
                        target_concepts,
                        source_labels=source_labels,
                        return_maps=record_metrics,
                        source_rows=source_rows,
                    )
                    forward_time_total += time.perf_counter() - forward_start
                    assert isinstance(loss, torch.Tensor), "Loss must be a torch.Tensor"

                    # Distributed: skip the window if any rank's shard loss is non-finite,
                    # as the single-process path does when the mean over shards is.
                    # The per-microbatch finite check is a host sync that cannot move to a
                    # logging interval: a non-finite window must not reach the optimizer.
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

                    backward_start = time.perf_counter()
                    self._accumulate_loss(loss)
                    backward_time_total += time.perf_counter() - backward_start

                micro_losses.append(loss.detach())
                micro_components.append(loss_components)
                num_samples += int(images.size(0)) * self.world_size
                # Drop the graph before metrics. Distributed shards are already
                # detached, so this frees activations before the shard's maps
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

                batch_end = time.perf_counter()

            if window_failed or not micro_losses:
                self._abort_accumulation()
                batch_end = time.perf_counter()
                continue

            n_micro = len(micro_losses)
            window_sum = _sum_components(
                [{_BATCH_LOSS_KEY: loss_i} for loss_i in micro_losses] + micro_components
            )
            window_mean = {key: value / n_micro for key, value in window_sum.items()}

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
            backward_time_total += time.perf_counter() - backward_start

            if not step_applied:
                batch_end = time.perf_counter()
                continue

            running = _sum_components([running, window_mean])

            # Distributed: every rank joins the reduction, whether or not it has a logger.
            if self.global_step % max(log_interval, 1) == 0 and (
                run_logger is not None or self.distributed
            ):
                logged = reduce_components(window_mean)
                if run_logger is not None:
                    run_logger.log(
                        {
                            "train/batch_loss": logged.pop(_BATCH_LOSS_KEY),
                            **{f"train/batch_{key}": value for key, value in logged.items()},
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
                metric_array = evaluator.metric_dict[metric_name].compute().cpu().numpy()
                metric_results[metric_name] = (
                    metric_array.item() if metric_array.ndim == 0 else metric_array
                )
                evaluator.metric_dict[metric_name].reset()

            metric_results.update(evaluator.compute_concept_metric_results())

        denom = n_updates if n_updates > 0 else 1
        # Summed per-update means are linear in the rank mean, so one reduction here
        # equals the mean over ranks of each update's mean.
        epoch_totals = reduce_components(running)
        last_loss = epoch_totals.pop(_BATCH_LOSS_KEY, 0.0) / denom
        for k, v in epoch_totals.items():
            metric_results[f"loss/{k}"] = v / denom
        timing = {
            "data_loading_sec": data_time_total,
            "forward_sec": forward_time_total,
            "backward_sec": backward_time_total,
            "num_samples": num_samples,
        }
        return last_loss, metric_results, timing

    @torch.inference_mode()
    def validation_epoch(
        self,
        val_loader: DataLoader[tuple[torch.Tensor, torch.Tensor] | dict[str, torch.Tensor]],
        evaluator: Any
        | None = None,  # TODO: Should be Evaluator - but this leads to circular import, fix
    ) -> tuple[float, dict[str, float | NDArray[np.float64]]]:
        """Calculate the validation loss and metrics for one epoch."""
        self.model.eval()
        iterations_per_val_epoch = self.iterations_per_val_epoch
        if iterations_per_val_epoch <= 0:
            raise ValueError("iterations_per_val_epoch must be > 0.")

        running: dict[str, torch.Tensor] = {}
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
            if self.distributed:
                target_labels = None
                target_concepts = None
            else:
                target_labels = self._to_target_labels(source_labels)
                target_concepts = self._to_concept_labels(source_labels)

            loss, outputs, concept_outputs, loss_components = self.batch_predict_loss(
                images,
                target_labels,
                target_concepts,
                source_labels=source_labels,
                return_maps=evaluator is not None,
            )
            assert isinstance(loss, torch.Tensor), "Loss must be a torch.Tensor"
            running = _sum_components([running, {_BATCH_LOSS_KEY: loss}, loss_components])
            del loss

            self._record_batch_metrics(
                evaluator, outputs, target_labels, concept_outputs, target_concepts
            )

        # Distributed: the mean over ranks of the per-rank sums is the per-batch mean over
        # shards, summed.
        epoch_totals = reduce_components(running)

        # Distributed: torchmetrics sums each metric's state over ranks in compute().
        if evaluator is not None:
            for metric_name in evaluator.metric_dict:
                metric_array = evaluator.metric_dict[metric_name].compute().cpu().numpy()
                metric_results[metric_name] = (
                    metric_array.item() if metric_array.ndim == 0 else metric_array
                )
                evaluator.metric_dict[metric_name].reset()

            metric_results.update(evaluator.compute_concept_metric_results())

        last_loss = epoch_totals.pop(_BATCH_LOSS_KEY, 0.0) / iterations_per_val_epoch
        for k, v in epoch_totals.items():
            metric_results[f"loss/{k}"] = v / iterations_per_val_epoch
        return last_loss, metric_results
