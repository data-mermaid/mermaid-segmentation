"""Single-process data parallel for the 768px fine-tune.

Each visible GPU runs the label lookup, the forward, and the
concept/taxonomy loss on its own shard. Only the int source labels and the
images are scattered, and only the scalar losses are averaged onto the primary
GPU, so neither the 768px concept targets nor the predicted maps are gathered
there. ``scatter`` splits a divisible batch into contiguous shards, which is the
same grouping as that many serial microbatches.

The per-GPU copies of the model and losses persist across steps
(:class:`ShardReplicas`). Their gradients are summed onto the primary
parameters once per optimizer step, before clipping.
"""

from __future__ import annotations

import copy
from collections.abc import Sequence
from dataclasses import dataclass

import torch
from torch import nn
from torch.nn.parallel import parallel_apply
from torch.nn.parallel.scatter_gather import scatter

from mermaidseg.dataset_reconciliation.concepts import source_labels_to_concepts
from mermaidseg.dataset_reconciliation.label_mapping import source_labels_to_target_labels

DATA_PARALLEL_GPUS = 4

_MIRRORED_ATTR_TYPES = (bool, int, float, str)


def require_data_parallel_gpus(device_count: int, expected: int = DATA_PARALLEL_GPUS) -> None:
    """Raise unless exactly ``expected`` GPUs are visible.

    A one-GPU launch of the 768px minibatch would run the KL term on every
    image at once. Failing here is the alternative to that OOM.
    """
    if int(device_count) != int(expected):
        raise RuntimeError(f"data_parallel expects {expected} visible GPUs, found {device_count}.")


def require_divisible_batch(batch_size: int, n_gpus: int = DATA_PARALLEL_GPUS) -> int:
    """Return images per GPU, or raise if ``batch_size`` does not split evenly."""
    if int(batch_size) % int(n_gpus) != 0:
        raise RuntimeError(
            f"data_parallel batch size {batch_size} is not divisible by {n_gpus} GPUs."
        )
    return int(batch_size) // int(n_gpus)


def contiguous_shard_bounds(
    batch_size: int, n_gpus: int = DATA_PARALLEL_GPUS
) -> list[tuple[int, int]]:
    """Contiguous ``[start, end)`` ranges ``scatter`` uses for a divisible batch."""
    per_gpu = require_divisible_batch(batch_size, n_gpus)
    return [(start, start + per_gpu) for start in range(0, int(batch_size), per_gpu)]


def mean_shard_losses(losses: list[torch.Tensor] | tuple[torch.Tensor, ...]) -> torch.Tensor:
    """Average scalar shard losses on the first loss's device.

    With equal shard sizes this is the same reduction as averaging that many
    serial microbatch losses. Autograd sends ``1 / n`` of the gradient to each
    shard, which is what the old accumulation window did by dividing each
    microbatch loss before ``backward``.
    """
    if len(losses) == 0:
        raise ValueError("mean_shard_losses requires at least one loss.")
    primary = losses[0].device
    return torch.stack([loss.to(primary) for loss in losses]).mean()


def average_component_dicts(components: list[dict[str, float]]) -> dict[str, float]:
    """Mean of per-shard logging dicts, divided by the number of shards."""
    if not components:
        return {}
    totals: dict[str, float] = {}
    for shard_components in components:
        for key, value in shard_components.items():
            totals[key] = totals.get(key, 0.0) + float(value)
    n_shards = len(components)
    return {key: value / n_shards for key, value in totals.items()}


def apply_concept_bottleneck_loss(
    segmentation_outputs: object,
    loss_fn: nn.Module,
    taxonomic_loss: nn.Module | None,
    distance_loss: nn.Module | None,
    target_labels: torch.Tensor,
    target_concepts: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, dict[str, float]]:
    """Concept, taxonomy, and distance terms for one concept-bottleneck batch.

    This is the loss body of ``MetaModel.batch_predict_loss``. The single-GPU
    path and each data-parallel shard both call it, outside autocast, on
    fp32 logits.
    """
    concept_outputs = segmentation_outputs.hidden_states.detach()
    concept_logits = segmentation_outputs.concept_logits.float()
    outputs = segmentation_outputs.logits.float()
    loss, loss_components = loss_fn(outputs, target_labels, concept_logits, target_concepts)
    if taxonomic_loss is not None:
        taxonomic_value, taxonomic_components = taxonomic_loss(
            concept_logits,
            target_concepts,
            target_labels,
        )
        loss = loss + taxonomic_value
        loss_components.update(taxonomic_components)
    if distance_loss is not None:
        distance_value, distance_components = distance_loss(concept_logits, target_concepts)
        loss = loss + distance_value
        loss_components.update(distance_components)
    return loss, outputs, concept_outputs, loss_components


def shard_targets(
    source_labels: torch.Tensor,
    target_lookup: torch.Tensor | None,
    concept_lookup: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Target labels and concept maps for one shard, built on that shard's device.

    Same lookups as ``MetaModel._to_target_labels`` and ``_to_concept_labels``.
    A ``None`` target lookup is the identity, as in ``MetaModel``.
    """
    if target_lookup is None:
        target_labels = source_labels
    else:
        target_labels = source_labels_to_target_labels(source_labels, target_lookup)
    target_concepts = source_labels_to_concepts(source_labels, concept_lookup)
    return target_labels, target_concepts


@dataclass
class MetricShard:
    """Detached predictions for one GPU. Maps stay on that GPU until metrics."""

    outputs: torch.Tensor | None
    concept_outputs: torch.Tensor | None
    target_labels: torch.Tensor | None
    target_concepts: torch.Tensor | None


@dataclass
class ParallelLoss:
    """Mean scalar loss, averaged components, and per-GPU metric shards.

    ``shards`` is empty when the forward ran with ``return_maps=False``.
    """

    loss: torch.Tensor
    components: dict[str, float]
    shards: list[MetricShard]


class ShardStep(nn.Module):
    """Label lookup, forward, and concept-bottleneck loss for one GPU's shard.

    The primary step's children are ``MetaModel``'s own model and losses, and
    its lookup buffers share storage with ``MetaModel``'s lookups. Registering
    them here does not re-parent ``MetaModel.model``. The other GPUs' steps are
    deep copies (:class:`ShardReplicas`).
    """

    def __init__(
        self,
        model: nn.Module,
        loss_fn: nn.Module,
        taxonomic_loss: nn.Module | None,
        distance_loss: nn.Module | None,
        target_lookup: torch.Tensor | None,
        concept_lookup: torch.Tensor,
        use_amp: bool,
        amp_dtype: torch.dtype,
    ) -> None:
        super().__init__()
        self.model = model
        self.loss_fn = loss_fn
        self.taxonomic_loss = taxonomic_loss
        self.distance_loss = distance_loss
        self.register_buffer("target_lookup", target_lookup, persistent=False)
        self.register_buffer("concept_lookup", concept_lookup, persistent=False)
        self.use_amp = use_amp
        self.amp_dtype = amp_dtype

    def forward(
        self,
        images: torch.Tensor,
        source_labels: torch.Tensor,
        return_maps: bool = True,
    ) -> tuple[
        torch.Tensor,
        torch.Tensor | None,
        torch.Tensor | None,
        torch.Tensor | None,
        torch.Tensor | None,
        dict[str, float],
    ]:
        target_labels, target_concepts = shard_targets(
            source_labels, self.target_lookup, self.concept_lookup
        )
        with torch.autocast(device_type="cuda", dtype=self.amp_dtype, enabled=self.use_amp):
            segmentation_outputs = self.model(images)
        loss, outputs, concept_outputs, components = apply_concept_bottleneck_loss(
            segmentation_outputs,
            self.loss_fn,
            self.taxonomic_loss,
            self.distance_loss,
            target_labels,
            target_concepts,
        )
        if not return_maps:
            return loss, None, None, None, None, components
        # Detach before returning so metric maps do not keep the autograd graph
        # after the scalar loss is dropped. Storage is shared with the graph
        # until that loss is freed.
        return (
            loss,
            outputs.detach(),
            concept_outputs.detach(),
            target_labels,
            target_concepts,
            components,
        )


class DDPLossStep(nn.Module):
    """DDP wrapper that returns only the scalar loss.

    ``ShardStep.forward`` also returns detached maps, label tensors, and a
    logging dict. DDP treats every tensor in that tuple as a graph output, so
    the unused-parameter reducer never finishes and the next iteration raises
    ``Expected to have finished reduction in the prior iteration``.
    """

    def __init__(self, step: ShardStep) -> None:
        super().__init__()
        self.step = step
        self.last: (
            tuple[
                torch.Tensor,
                torch.Tensor | None,
                torch.Tensor | None,
                torch.Tensor | None,
                torch.Tensor | None,
                dict[str, float],
            ]
            | None
        ) = None

    def forward(
        self,
        images: torch.Tensor,
        source_labels: torch.Tensor,
        return_maps: bool = True,
    ) -> torch.Tensor:
        self.last = self.step(images, source_labels, return_maps=bool(return_maps))
        return self.last[0]


class ShardReplicas:
    """One persistent copy of a module per device; ``devices[0]`` holds the original.

    ``sync`` copies a primary parameter or buffer onto the other devices when
    it may have changed: trainable parameters after ``invalidate``, and any
    tensor whose version counter or storage pointer differs from the last
    copy. In-place updates (optimizer steps, ``load_state_dict``) bump the
    version; assigning ``.data`` changes the storage. A write that does
    neither is not seen. Primitive module attributes (``training`` and flags
    such as ``_encoder_frozen``) and ``requires_grad`` are mirrored on every
    ``sync``.

    The copies' gradients accumulate across a window. ``reduce_grads`` adds
    them into the primary gradients and clears them.
    """

    def __init__(self, module: nn.Module, devices: Sequence[torch.device | str]) -> None:
        if len(devices) == 0:
            raise ValueError("ShardReplicas requires at least one device.")
        self.devices = [torch.device(device) for device in devices]
        self.modules: list[nn.Module] = [module]
        for device in self.devices[1:]:
            replica = copy.deepcopy(module).to(device)
            for param in replica.parameters():
                param.grad = None
            self.modules.append(replica)

        named_params = [list(m.named_parameters()) for m in self.modules]
        named_buffers = [list(m.named_buffers()) for m in self.modules]
        self._params: list[tuple[torch.Tensor, list[torch.Tensor]]] = []
        for index, (name, param) in enumerate(named_params[0]):
            copies = [named[index] for named in named_params[1:]]
            if any(copy_name != name for copy_name, _ in copies):
                raise RuntimeError(f"replica parameter order differs at {name!r}.")
            self._params.append((param, [tensor for _, tensor in copies]))
        self._buffers: list[tuple[torch.Tensor, list[torch.Tensor]]] = []
        for index, (name, buffer) in enumerate(named_buffers[0]):
            copies = [named[index] for named in named_buffers[1:]]
            if any(copy_name != name for copy_name, _ in copies):
                raise RuntimeError(f"replica buffer order differs at {name!r}.")
            self._buffers.append((buffer, [tensor for _, tensor in copies]))

        self._module_pairs: list[tuple[nn.Module, list[nn.Module], list[str]]] = []
        for primary, *copies in zip(*(m.modules() for m in self.modules), strict=True):
            keys = [
                key
                for key, value in vars(primary).items()
                if isinstance(value, _MIRRORED_ATTR_TYPES)
            ]
            self._module_pairs.append((primary, list(copies), keys))

        self._param_seen = [self._stamp(param) for param, _ in self._params]
        self._buffer_seen = [self._stamp(buffer) for buffer, _ in self._buffers]

    @staticmethod
    def _stamp(tensor: torch.Tensor) -> tuple[int, int]:
        return (int(tensor._version), int(tensor.data_ptr()))

    def invalidate(self) -> None:
        """Mark every trainable primary parameter as changed (call after an optimizer step)."""
        for index, (param, _) in enumerate(self._params):
            if param.requires_grad:
                self._param_seen[index] = (-1, -1)

    @torch.no_grad()
    def sync(self) -> None:
        """Bring the copies up to date with the primary module."""
        for index, (param, copies) in enumerate(self._params):
            for replica in copies:
                if replica.requires_grad != param.requires_grad:
                    replica.requires_grad_(param.requires_grad)
            stamp = self._stamp(param)
            if stamp != self._param_seen[index]:
                for replica in copies:
                    replica.copy_(param)
                self._param_seen[index] = stamp
        for index, (buffer, copies) in enumerate(self._buffers):
            stamp = self._stamp(buffer)
            if stamp != self._buffer_seen[index]:
                for replica in copies:
                    replica.copy_(buffer)
                self._buffer_seen[index] = stamp
        for primary, copies, keys in self._module_pairs:
            state = vars(primary)
            for replica in copies:
                replica_state = vars(replica)
                for key in keys:
                    replica_state[key] = state[key]

    @torch.no_grad()
    def reduce_grads(self) -> None:
        """Add the copies' gradients into the primary gradients, then clear the copies'."""
        for param, copies in self._params:
            for replica in copies:
                if replica.grad is None:
                    continue
                grad = replica.grad.to(param.device)
                if param.grad is None:
                    param.grad = grad
                else:
                    param.grad.add_(grad)
                replica.grad = None

    def zero_grads(self) -> None:
        """Drop the copies' gradients without reducing them."""
        for _, copies in self._params:
            for replica in copies:
                replica.grad = None


def _require_primary_cuda(module: nn.Module, name: str) -> None:
    """Require every parameter and buffer on ``cuda:0``.

    A module with neither is left alone. The replicas are copied from these
    tensors and the scalar losses are averaged on ``cuda:0``.
    """
    for tensor in list(module.parameters()) + list(module.buffers()):
        device = tensor.device
        if device.type != "cuda" or device.index not in (None, 0):
            raise RuntimeError(f"data_parallel requires {name} on cuda:0, found {device}.")


def build_shard_replicas(
    model: nn.Module,
    loss_fn: nn.Module,
    taxonomic_loss: nn.Module | None,
    distance_loss: nn.Module | None,
    target_lookup: torch.Tensor | None,
    concept_lookup: torch.Tensor,
    *,
    use_amp: bool,
    amp_dtype: torch.dtype,
) -> ShardReplicas:
    """Copy the model, losses, and label lookups onto each of the 4 GPUs."""
    require_data_parallel_gpus(torch.cuda.device_count())
    _require_primary_cuda(model, "the model")
    _require_primary_cuda(loss_fn, "the loss")
    if taxonomic_loss is not None:
        _require_primary_cuda(taxonomic_loss, "the taxonomic loss")
    if distance_loss is not None:
        _require_primary_cuda(distance_loss, "the distance loss")
    for lookup, name in (
        (target_lookup, "the target lookup"),
        (concept_lookup, "the concept lookup"),
    ):
        if lookup is not None and (
            lookup.device.type != "cuda" or lookup.device.index not in (None, 0)
        ):
            raise RuntimeError(f"data_parallel requires {name} on cuda:0, found {lookup.device}.")
    step = ShardStep(
        model,
        loss_fn,
        taxonomic_loss,
        distance_loss,
        target_lookup,
        concept_lookup,
        use_amp,
        amp_dtype,
    )
    return ShardReplicas(step, [torch.device("cuda", index) for index in range(DATA_PARALLEL_GPUS)])


def parallel_forward_loss(
    replicas: ShardReplicas,
    images: torch.Tensor,
    source_labels: torch.Tensor,
    *,
    return_maps: bool = True,
) -> ParallelLoss:
    """Run lookup, forward, and loss on 4 GPUs and average the scalar losses.

    With ``return_maps``, full-resolution maps and targets are returned
    detached on the GPU that computed them. Callers should drop the scalar
    loss (freeing the graph) before copying those maps one shard at a time
    for metrics.
    """
    require_data_parallel_gpus(torch.cuda.device_count())
    batch_size = int(images.shape[0])
    require_divisible_batch(batch_size)
    if source_labels.shape[0] != batch_size:
        raise RuntimeError(
            "data_parallel images and source labels must share a batch dimension, "
            f"got {batch_size} and {source_labels.shape[0]}."
        )
    if images.device.type != "cuda" or images.device.index not in (None, 0):
        raise RuntimeError(f"data_parallel expects the batch on cuda:0, found {images.device}.")

    replicas.sync()
    device_ids = [device.index for device in replicas.devices]
    scattered = scatter((images, source_labels), device_ids, dim=0)
    kwargs_tup = tuple({"return_maps": bool(return_maps)} for _ in device_ids)
    shard_outputs = parallel_apply(
        replicas.modules, scattered, kwargs_tup=kwargs_tup, devices=device_ids
    )

    losses: list[torch.Tensor] = []
    components: list[dict[str, float]] = []
    shards: list[MetricShard] = []
    for shard in shard_outputs:
        loss, outputs, concept_outputs, labels, concepts, shard_components = shard
        losses.append(loss)
        components.append(shard_components)
        if return_maps:
            shards.append(
                MetricShard(
                    outputs=outputs,
                    concept_outputs=concept_outputs,
                    target_labels=labels,
                    target_concepts=concepts,
                )
            )
    return ParallelLoss(
        loss=mean_shard_losses(losses),
        components=average_component_dicts(components),
        shards=shards,
    )
