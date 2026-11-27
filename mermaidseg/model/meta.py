"""MetaModel: model, loss, optimizer and train/validation primitives for concept-bottleneck runs."""

import contextlib
import faulthandler
import logging
import threading
import time
from collections.abc import Callable, Iterable, Iterator, Mapping
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

import torch
import transformers
from torch.nn.parallel import DistributedDataParallel
from torch.utils.data import DataLoader
from torch.utils.data.dataloader import _BaseDataLoaderIter
from tqdm import tqdm

from mermaidseg.config import lookup_class
from mermaidseg.inference.predictor import class_predictions
from mermaidseg.io import ConfigDict
from mermaidseg.model.data_parallel import DDPLossStep, MetricShard, ShardStep, build_step
from mermaidseg.model.distributed import (
    all_ranks_finite,
    current_context,
    is_distributed,
    is_main_process,
    reduce_components,
    require_world_size,
)
from mermaidseg.model.loss import build_loss
from mermaidseg.model.models import MODEL_CLASSES, load_pretrained_encoder
from mermaidseg.model.taxonomy_loss import build_taxonomic_loss

if TYPE_CHECKING:
    from mermaidseg.model.eval import Evaluator

logger = logging.getLogger(__name__)

_BatchLoader = DataLoader[tuple[torch.Tensor, ...]]

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


def _cuda_bf16_supported(device: torch.device) -> bool:
    # Native bf16 only (pre-Ampere GPUs emulate it slowly), on the model's device, not the current one.
    with torch.cuda.device(device):
        return torch.cuda.is_bf16_supported(including_emulation=False)


def resolve_amp_dtype(dtype_config: Any) -> torch.dtype:
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


@dataclass
class _EpochTally:
    """Host wall times and sample counts summed over one training epoch."""

    data_sec: float = 0.0
    forward_sec: float = 0.0
    backward_sec: float = 0.0
    num_samples: int = 0
    num_loaded_samples: int = 0
    # When the last microbatch or update finished; data time runs from here to the next fetch.
    batch_end: float = field(default_factory=time.perf_counter)

    def timing(self) -> dict[str, float | int]:
        """The ``timing`` dict ``train_epoch`` returns."""
        return {
            "data_loading_sec": self.data_sec,
            "forward_sec": self.forward_sec,
            "backward_sec": self.backward_sec,
            "num_samples": self.num_samples,
            "num_loaded_samples": self.num_loaded_samples,
        }


class MetaModel:
    """Wrapper for training and inference of segmentation models.

    Handles model initialisation, optimizer/scheduler/loss wiring, and the
    train/validation loop primitives: joint segmentation + concept supervision.

    Source-label datasets emit ``source_labels`` in the joint global source
    space (see :class:`SourceLabelRegistry`); ``step`` converts them to
    ``target_labels`` and ``concept_labels`` on-device via the lookups
    passed as ``source_to_target_lookup`` and ``source_to_concepts_lookup``.

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
            mapping global source IDs to target label IDs.
        source_to_concepts_lookup: Float tensor of shape ``(N+1, C)``.
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
    num_concepts: int
    source_to_target_lookup: torch.Tensor
    source_to_concepts_lookup: torch.Tensor
    concept_value2id: dict[str, dict[str, list[int]]]
    resuming_within_epoch: bool

    def __init__(
        self,
        run_name: str,
        num_classes: int,
        num_concepts: int,
        *,
        model_kwargs: ConfigDict,
        training_kwargs: ConfigDict,
        device: str | torch.device,
        source_to_target_lookup: torch.Tensor,
        source_to_concepts_lookup: torch.Tensor,
        concept_value2id: dict[str, dict[str, list[int]]],
        model: torch.nn.Module | None = None,
    ):
        """Build the model, loss, optimizer, scheduler, and label lookups.

        Args:
            run_name: Identifier for the current run/experiment.
            num_classes: Number of segmentation output (target) classes.
            num_concepts: Number of concept channels.
            model_kwargs: Model-specific config passed to the architecture.
            training_kwargs: Training hyperparameters.
            device: Device the model and tensors live on.
            source_to_target_lookup: Source-to-target label lookup.
            source_to_concepts_lookup: Source-to-concept lookup.
            concept_value2id: Maps concept names and values to IDs.
            model: A prebuilt model to use as is; no encoder is loaded and no model is built.
        """
        self.run_name = run_name
        self.num_classes = num_classes
        self.num_concepts = num_concepts
        self.device = device
        self.global_step = 0
        self.resuming_within_epoch = False

        self.model_name = model_kwargs.pop("name")
        self.model_kwargs = model_kwargs

        self._metric_shards: list[MetricShard] | None = None
        # One process per GPU (torchrun). The process group must already exist;
        # each rank's loader yields that rank's shard of the global microbatch.
        self.distributed = bool(training_kwargs.pop("distributed"))
        self._step: ShardStep | None = None
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
        self.amp_dtype = resolve_amp_dtype(mixed_precision_dtype)
        self.use_amp = bool(mixed_precision)
        self.device_type = torch.device(device).type
        if (
            self.use_amp
            and self.device_type == "cuda"
            and self.amp_dtype == torch.bfloat16
            and not _cuda_bf16_supported(torch.device(device))
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
        self.source_to_target_lookup = source_to_target_lookup.to(device).long()
        self.source_to_concepts_lookup = source_to_concepts_lookup.to(device).float()
        self.concept_value2id = concept_value2id

        model_kwargs["num_classes"] = self.num_classes
        model_kwargs["num_concepts"] = self.num_concepts
        model_kwargs["concept_value2id"] = self.concept_value2id
        model_kwargs["detach_concepts"] = detach_concepts
        if model is not None:
            self.model = model
        else:
            model_cls = lookup_class(MODEL_CLASSES, "model", self.model_name)
            # The encoder is built here; the model class takes it, not its hub name.
            constructor_kwargs = {k: v for k, v in model_kwargs.items() if k != "encoder_name"}
            self.model = model_cls(
                base_encoder=load_pretrained_encoder(model_kwargs["encoder_name"]),
                **constructor_kwargs,
            )

        self.model = self.model.to(device)
        self.detach_concepts = detach_concepts

        self.loss = build_loss(training_kwargs.loss, self.concept_value2id)

        # Taxonomic KL coherence term added to the concept loss; None when its kl_weight is
        # not positive.
        taxonomic_loss = build_taxonomic_loss(training_kwargs, self.concept_value2id)
        self.taxonomic_loss = taxonomic_loss.to(device) if taxonomic_loss is not None else None

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

    def set_concept_source_mask(self, keep: torch.Tensor | None) -> None:
        """Mask source x concept training targets (``training.concept_source_mask``).

        ``keep`` is the ``(S + 1, C)`` bool table of
        :class:`~mermaidseg.dataset_reconciliation.concept_source_mask.ConceptSourceMask`. Training
        batches must then carry a keep-table row per sample (``attach_source_rows``). Validation is
        not masked. The step is rebuilt with the new table on its next use.
        """
        if keep is not None:
            if keep.shape[1] != self.num_concepts:
                raise ValueError(
                    f"concept source mask has {keep.shape[1]} channels, the model {self.num_concepts}"
                )
            keep = keep.to(device=self.device, dtype=torch.bool)
        self.concept_source_keep = keep

    def _update_evaluator(
        self,
        evaluator: Any,
        outputs: torch.Tensor,
        target_labels: torch.Tensor,
        concept_outputs: torch.Tensor,
        target_concepts: torch.Tensor,
    ) -> None:
        """Update ``evaluator`` with one batch already small enough to sit on its device."""
        device = getattr(evaluator, "device", self.device)
        outputs = outputs.detach().to(device)
        target_labels = target_labels.to(device)
        concept_outputs = concept_outputs.detach().to(device)
        target_concepts = target_concepts.to(device)

        preds = class_predictions(outputs) if outputs.ndim > 3 else outputs
        for metric in evaluator.metric_dict.values():
            metric.update(preds, target_labels)
        evaluator.evaluate_concepts(concept_outputs, target_concepts)

    def _consume_metric_shards(self, evaluator: Any) -> bool:
        """Update metrics from one GPU shard at a time.

        Returns False when the last forward stored no shards. Shards are dropped
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
            assert concept_outputs is not None
            assert target_concepts is not None
            self._update_evaluator(
                evaluator, outputs, target_labels, concept_outputs, target_concepts
            )
        return True

    def _record_batch_metrics(self, evaluator: Any | None) -> None:
        """Record train or val metrics from the shards the last forward stored.

        ``evaluator=None`` records nothing and drops the shards.
        """
        if evaluator is None:
            self._metric_shards = None
            return
        if not self._consume_metric_shards(evaluator):
            raise RuntimeError("the forward did not store metric shards.")

    @property
    def step(self) -> ShardStep:
        """This process's lookup, forward, and loss over the current model and losses.

        Rebuilt on first use after ``model``, ``loss``, ``taxonomic_loss`` or
        ``concept_source_keep`` has been replaced.
        """
        step = self._step
        if (
            step is None
            or step.model is not self.model
            or step.loss_fn is not self.loss
            or step.taxonomic_loss is not self.taxonomic_loss
            or step.concept_keep is not self.concept_source_keep
        ):
            step = build_step(
                self.model,
                self.loss,
                self.taxonomic_loss,
                self.source_to_target_lookup,
                self.source_to_concepts_lookup,
                torch.device(self.device),
                self.amp_dtype if self.use_amp else None,
                self.concept_source_keep,
            )
            self._step = step
        return step

    def _get_ddp_step(self) -> DistributedDataParallel:
        """``self.step`` wrapped in DDP, built on first use.

        Built lazily so it wraps weights loaded after
        construction and the ``requires_grad`` set at model construction.
        Rebuilt whenever ``self.step`` has been rebuilt.
        """
        step = self.step
        ddp = self._ddp_step
        if ddp is not None and ddp.module.step is step:
            return ddp
        device = torch.device(self.device)
        # Buffers (lookups, loss matrices, model buffers) are identical on every
        # rank and never updated, so they are not re-broadcast each forward.
        # A bucket view of the 1x1 DPT/concept conv grads is not stride-compatible
        # (see the reducer's "Grad strides do not match bucket view strides"
        # warning), so leave the grads in their own storage.
        # Built outside inference mode, so the reducer's buckets are never
        # inference tensors that a backward cannot update in place.
        with torch.inference_mode(False):
            self._ddp_step = DistributedDataParallel(
                DDPLossStep(step),
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
        source_labels: torch.Tensor,
        source_rows: torch.Tensor | None,
        *,
        return_maps: bool = True,
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        """Run ``self.step`` on one batch and keep its maps for the metrics.

        Args:
            images: Image batch (already on-device, float).
            source_labels: Global source-label map (already on-device, long).
            source_rows: Keep-table row per sample for ``concept_source_keep``;
                ``None`` leaves the concept targets unmasked.
            return_maps: Keep the batch's maps in ``_metric_shards``. Pass False
                when the batch is not scored.

        Returns:
            The loss and its components. The maps stay in ``_metric_shards``.
        """
        self._metric_shards = None
        # Training under DDP must backward the tensor DDP returned, not a copy from
        # the stashed tuple, so the reducer hooks fire. Without grad there is nothing
        # to reduce, so the wrapper is bypassed.
        if self.distributed and torch.is_grad_enabled():
            ddp = self._get_ddp_step()
            loss = ddp(images, source_labels, return_maps=return_maps, source_rows=source_rows)
            output = ddp.module.last
            if output is None:
                raise RuntimeError("distributed forward did not stash its outputs.")
            _, outputs, concept_outputs, labels, concepts, loss_components = output
        else:
            loss, outputs, concept_outputs, labels, concepts, loss_components = self.step(
                images, source_labels, return_maps=return_maps, source_rows=source_rows
            )
        if return_maps:
            self._metric_shards = [
                MetricShard(
                    outputs=outputs,
                    concept_outputs=concept_outputs,
                    target_labels=labels,
                    target_concepts=concepts,
                )
            ]
        return loss, loss_components

    def _resume_offset(self, epoch: int | None) -> int:
        """The update this epoch starts at: ``global_step - epoch * iterations``.

        ``global_step`` counts applied updates only, so skipped windows in earlier epochs
        make the offset negative; the epoch then starts at 0.

        Raises:
            ValueError: If the offset is past ``iterations_per_train_epoch``.
        """
        epoch_index = 0 if epoch is None else int(epoch)
        offset = int(self.global_step) - epoch_index * self.iterations_per_train_epoch
        if offset > self.iterations_per_train_epoch:
            raise ValueError(
                f"train_epoch: resume offset {offset} (global_step {self.global_step}, "
                f"epoch {epoch_index}) is past {self.iterations_per_train_epoch}."
            )
        if offset < 0:
            logger.warning(
                "train_epoch: resume offset %s is outside 0..%s; starting this epoch at 0.",
                offset,
                self.iterations_per_train_epoch,
            )
            return 0
        return offset

    def _warn_stall_dump(self, stall_dump_sec: float) -> None:
        """Log the main thread's id, which a faulthandler stack dump labels its stack with."""
        if stall_dump_sec > 0:
            logger.warning(
                "rank %d: an update running over %.0f s dumps all stacks to stderr; "
                "this rank's main thread is 0x%016x",
                current_context().rank,
                stall_dump_sec,
                threading.get_ident(),
            )

    def _next_batch(
        self, train_loader: _BatchLoader, tally: _EpochTally
    ) -> tuple[tuple[torch.Tensor, ...], float]:
        """Fetch the next training microbatch, restarting the loader when it runs out.

        Returns:
            The batch and the seconds spent fetching it.
        """
        assert self._train_loader_iter is not None
        fetch_start = time.perf_counter()
        try:
            data = next(self._train_loader_iter)
        except StopIteration:
            self._train_loader_iter = iter(train_loader)
            self._train_loader = train_loader
            data = next(self._train_loader_iter)
        fetch_sec = time.perf_counter() - fetch_start
        if fetch_sec > SLOW_FETCH_WARN_SEC:
            logger.warning(
                "rank %d: train loader took %.0f s for one microbatch (update %d)",
                current_context().rank,
                fetch_sec,
                int(self.global_step) + 1,
            )
        tally.data_sec += time.perf_counter() - tally.batch_end
        return data, fetch_sec

    def _prepare_targets(
        self, data: tuple[torch.Tensor, ...]
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor | None] | None:
        """Move one training microbatch to the device, or return None for an empty batch.

        Returns:
            The images, source labels, and source rows (None without a concept source mask).

        Raises:
            RuntimeError: If a concept source mask is set and the batch has no source rows,
                or the batch is empty in a distributed run.
        """
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
            logger.warning("train_epoch: skipping an empty batch (all items failed to load).")
            return None
        images = images.to(self.device, non_blocking=True).float()
        source_labels = source_labels.to(self.device, non_blocking=True).long()
        if self.concept_source_keep is None:
            source_rows = None
        else:
            source_rows = torch.as_tensor(source_rows).to(self.device).long()
        return images, source_labels, source_rows

    def _run_window(
        self, train_loader: _BatchLoader, evaluator: "Evaluator | None", tally: _EpochTally
    ) -> tuple[dict[str, torch.Tensor], float] | None:
        """Forward and backward one accumulation window of ``grad_accum_steps`` microbatches.

        Returns:
            The window's mean loss (under ``_BATCH_LOSS_KEY``) and components, and the
            seconds spent waiting on the loader; None when the window is dropped for an
            empty batch or a non-finite loss.
        """
        micro_losses: list[torch.Tensor] = []
        micro_components: list[dict[str, torch.Tensor]] = []
        data_wait = 0.0
        for micro in range(self.grad_accum_steps):
            data, fetch_sec = self._next_batch(train_loader, tally)
            data_wait += fetch_sec
            tally.num_loaded_samples += int(data[0].size(0)) * self.world_size
            batch = self._prepare_targets(data)
            if batch is None:
                tally.batch_end = time.perf_counter()
                return None
            images, source_labels, source_rows = batch
            record_metrics = evaluator is not None and records_train_metrics(
                self._train_microbatches_seen, self.train_metrics_every
            )
            self._train_microbatches_seen += 1

            # Distributed: gradients are all-reduced once, in the window's last backward.
            sync_grads = micro == self.grad_accum_steps - 1
            with self._grad_sync_context(sync_grads):
                forward_start = time.perf_counter()
                loss, loss_components = self.batch_predict_loss(
                    images, source_labels, source_rows, return_maps=record_metrics
                )
                tally.forward_sec += time.perf_counter() - forward_start

                # Skip the window if any rank's shard loss is non-finite. The
                # per-microbatch check is a host sync that cannot move to a logging
                # interval: a non-finite window must not reach the optimizer.
                if not all_ranks_finite(loss):
                    logger.warning("train_epoch: skipping non-finite loss (value=%s)", loss.item())
                    self._metric_shards = None
                    del loss
                    if self.distributed and sync_grads:
                        self._finish_skipped_reduction()
                    tally.batch_end = time.perf_counter()
                    return None

                backward_start = time.perf_counter()
                self._accumulate_loss(loss)
                tally.backward_sec += time.perf_counter() - backward_start

            micro_losses.append(loss.detach())
            micro_components.append(loss_components)
            tally.num_samples += int(images.size(0)) * self.world_size
            # Drop the graph before the maps are copied onto the evaluator device.
            del loss
            self._record_batch_metrics(evaluator if record_metrics else None)
            tally.batch_end = time.perf_counter()

        window_sum = _sum_components(
            [{_BATCH_LOSS_KEY: loss_i} for loss_i in micro_losses] + micro_components
        )
        n_micro = len(micro_losses)
        return {key: value / n_micro for key, value in window_sum.items()}, data_wait

    def _apply_window(self, completed: bool, tally: _EpochTally) -> bool:
        """Step the optimizer on a completed window, or drop a failed one.

        On an applied step, ``global_step`` and the LR schedules advance.

        Returns:
            True when the optimizer step was applied.
        """
        if not completed:
            self._abort_accumulation()
            return False
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
        tally.backward_sec += time.perf_counter() - backward_start
        return step_applied

    def _log_window(
        self,
        window_mean: Mapping[str, torch.Tensor],
        run_logger: Any | None,
        log_interval: int,
        step_sec: float,
        data_wait_sec: float,
    ) -> None:
        """Log one applied update's loss and components at the ``log_interval`` cadence."""
        if self.global_step % max(log_interval, 1) != 0:
            return
        # Distributed: every rank joins the reduction, whether or not it has a logger.
        if run_logger is None and not self.distributed:
            return
        logged = reduce_components(window_mean)
        if run_logger is not None:
            run_logger.log(
                {
                    "train/batch_loss": logged.pop(_BATCH_LOSS_KEY),
                    **{f"train/batch_{key}": value for key, value in logged.items()},
                    # Host wall time for this update, and the part of it spent
                    # blocked on the train loader.
                    "perf/step_sec": step_sec,
                    "perf/data_wait_sec": data_wait_sec,
                },
                step=self.global_step,
            )

    def train_epoch(
        self,
        train_loader: _BatchLoader,
        evaluator: "Evaluator | None" = None,
        run_logger: Any | None = None,
        log_interval: int = 1,
        epoch: int | None = None,
        on_optimizer_step: Callable[[Any], None] | None = None,
    ) -> tuple[float, dict[str, float], dict[str, float | int]]:
        """Trains the model for one epoch using the provided data loader.

        One iteration is one optimizer update. ``grad_accum_steps`` microbatches are
        forwarded and averaged in fp32 before that update. ``global_step`` advances
        once per successful update, which is once per batch when accumulation is 1.

        A resumed epoch continues at ``global_step - epoch * iterations``. Epoch 0 of
        a fresh run starts at 0. The next epoch in the same process starts at 0 as
        well, because ``global_step`` already counts the updates of earlier epochs.

        Args:
            train_loader: DataLoader yielding ``(inputs, source_labels)`` batches, plus
                source rows with a concept source mask. Each batch is one microbatch.
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
            phase next blocks on the device. ``num_samples`` counts the samples of
            microbatches that reached backward; ``num_loaded_samples`` counts every
            sample the loader delivered, including windows later skipped for a
            non-finite loss or an empty batch. Both are summed over ranks.

        Raises:
            ValueError: If ``iterations_per_train_epoch`` is not positive, or the resume
                offset is past ``iterations_per_train_epoch``. A negative offset does not
                raise; it logs a warning and starts the epoch at 0.
        """
        self.model.train()
        if self.iterations_per_train_epoch <= 0:
            raise ValueError("iterations_per_train_epoch must be > 0.")
        start_update = self._resume_offset(epoch)

        # Per-update losses and components stay on the device until they are logged
        # or the epoch ends; the batch loss rides along under ``_BATCH_LOSS_KEY``.
        running: dict[str, torch.Tensor] = {}
        applied_updates = 0
        tally = _EpochTally()

        if self._train_loader is not train_loader:
            self._train_loader_iter = iter(train_loader)
            self._train_loader = train_loader

        stall_dump_sec = stall_dump_seconds(self.training_kwargs)
        self._warn_stall_dump(stall_dump_sec)
        updates = range(start_update, self.iterations_per_train_epoch)
        for _update_idx in watch_for_stalls(
            tqdm(updates, disable=not is_main_process()), stall_dump_sec
        ):
            step_start = time.perf_counter()
            self.optimizer.zero_grad(set_to_none=True)
            window = self._run_window(train_loader, evaluator, tally)
            if not self._apply_window(window is not None, tally):
                tally.batch_end = time.perf_counter()
                continue
            assert window is not None
            window_mean, data_wait = window
            running = _sum_components([running, window_mean])
            applied_updates += 1
            self._log_window(
                window_mean, run_logger, log_interval, time.perf_counter() - step_start, data_wait
            )
            if on_optimizer_step is not None:
                on_optimizer_step(self)
            tally.batch_end = time.perf_counter()

        metric_results = self._drain_metrics(evaluator)
        # Skipped updates add nothing to ``running``, so they stay out of the mean.
        denom = applied_updates if applied_updates > 0 else 1
        # Summed per-update means are linear in the rank mean, so one reduction here
        # equals the mean over ranks of each update's mean.
        epoch_totals = reduce_components(running)
        last_loss = epoch_totals.pop(_BATCH_LOSS_KEY, 0.0) / denom
        for k, v in epoch_totals.items():
            metric_results[f"loss/{k}"] = v / denom
        return last_loss, metric_results, tally.timing()

    @staticmethod
    def _drain_metrics(evaluator: "Evaluator | None") -> dict[str, float]:
        """Compute and reset every metric of ``evaluator``; empty without one.

        Distributed: torchmetrics sums each metric's state over ranks in ``compute()``.
        """
        if evaluator is None:
            return {}
        results: dict[str, float] = {}
        for name, metric in evaluator.metric_dict.items():
            results[name] = metric.compute().item()
            metric.reset()
        results.update(evaluator.compute_concept_metric_results())
        return results

    @torch.inference_mode()
    def validation_epoch(
        self,
        val_loader: _BatchLoader,
        evaluator: "Evaluator | None" = None,
    ) -> tuple[float, dict[str, float]]:
        """Calculate the validation loss and metrics for one epoch, without concept masking."""
        self.model.eval()
        iterations_per_val_epoch = self.iterations_per_val_epoch
        if iterations_per_val_epoch <= 0:
            raise ValueError("iterations_per_val_epoch must be > 0.")

        running: dict[str, torch.Tensor] = {}

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

            loss, loss_components = self.batch_predict_loss(
                images, source_labels, None, return_maps=evaluator is not None
            )
            running = _sum_components([running, {_BATCH_LOSS_KEY: loss}, loss_components])
            del loss
            self._record_batch_metrics(evaluator)

        # Distributed: the mean over ranks of the per-rank sums is the per-batch mean over
        # shards, summed.
        epoch_totals = reduce_components(running)
        metric_results = self._drain_metrics(evaluator)
        last_loss = epoch_totals.pop(_BATCH_LOSS_KEY, 0.0) / iterations_per_val_epoch
        for k, v in epoch_totals.items():
            metric_results[f"loss/{k}"] = v / iterations_per_val_epoch
        return last_loss, metric_results
