"""Checks for torchrun DDP: sampler shards, collectives, and gradient parity.

The 768px job splits each 12-image microbatch into four contiguous shards of 3
and averages the shard losses. These tests lock that split, the loss-component
reduce, the non-finite abort, and a CPU (gloo) window that must match a serial
single-process accumulation. The last test runs only with 4 visible GPUs.
"""

from __future__ import annotations

import copy
from datetime import timedelta
from types import SimpleNamespace
from typing import Any, cast
from unittest.mock import patch

import pytest
import torch
import torch.distributed as dist
from torch import nn
from torch.distributed.algorithms.ddp_comm_hooks import default_hooks
from torch.utils.data import (
    BatchSampler,
    DataLoader,
    RandomSampler,
    TensorDataset,
)

import mermaidseg.model.meta as meta_module
import mermaidseg.model.models as models_module
from mermaidseg.model.data_parallel import ShardStep
from mermaidseg.model.distributed import (
    DistContext,
    ShardedBatchSampler,
    all_ranks_finite,
    broadcast_generator_from_rank0,
    init_distributed_from_env,
    on_main_process,
    reduce_components,
    require_world_size,
)
from tests.conftest import (
    CBM_CONCEPT_VALUE2ID,
    CBM_NUM_CONCEPTS,
    IMAGE_SIZE,
    NUM_CLASSES,
    cbm_concept_lookup,
    cbm_test_config,
)
from tests.mermaidseg.conftest import init_gloo, spawn_gloo

_NUM_SOURCE = 9
_NUM_CONCEPTS = 5
_NUM_CLASSES = 4
_WORLD = 4
_GLOBAL_BATCH = 12
_PER_RANK = _GLOBAL_BATCH // _WORLD
_ACCUM = 3


class _IndexDataset:
    def __init__(self, length: int) -> None:
        self.length = length

    def __len__(self) -> int:
        return self.length

    def __getitem__(self, index: int) -> int:
        return int(index)


class _TinyCBM(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.stem = nn.Conv2d(3, 6, 1)
        self.concept_head = nn.Conv2d(6, _NUM_CONCEPTS, 1)
        self.classifier = nn.Conv2d(_NUM_CONCEPTS, _NUM_CLASSES, 1)
        for param in self.stem.parameters():
            param.requires_grad = False

    def forward(self, x: torch.Tensor) -> models_module.ConceptBottleneckOutput:
        concept_logits = self.concept_head(torch.relu(self.stem(x)))
        concept_outputs = torch.sigmoid(concept_logits)
        return models_module.ConceptBottleneckOutput(
            logits=self.classifier(concept_outputs),
            concept_logits=concept_logits,
            concept_probs=concept_outputs,
        )


class _TinyLoss(nn.Module):
    def forward(self, outputs, target_labels, concept_logits, target_concepts):
        class_loss = nn.functional.cross_entropy(outputs, target_labels)
        concept_loss = nn.functional.binary_cross_entropy_with_logits(
            concept_logits, target_concepts
        )
        loss = class_loss + concept_loss
        return loss, {"concepts": concept_loss.detach(), "classification": class_loss.detach()}


def _lookups(seed: int = 0) -> tuple[torch.Tensor, torch.Tensor]:
    generator = torch.Generator().manual_seed(seed)
    target = torch.randint(0, _NUM_CLASSES, (_NUM_SOURCE + 1,), generator=generator)
    target[0] = 0
    concepts = torch.rand(_NUM_SOURCE + 1, _NUM_CONCEPTS, generator=generator)
    concepts[0] = 0
    return target, concepts


def _batch(batch_size: int = 36, seed: int = 1) -> tuple[torch.Tensor, torch.Tensor]:
    generator = torch.Generator().manual_seed(seed)
    images = torch.randn(batch_size, 3, 8, 8, generator=generator)
    source_labels = torch.randint(0, _NUM_SOURCE + 1, (batch_size, 8, 8), generator=generator)
    return images, source_labels


def test_require_world_size_matches_expected():
    require_world_size(4, 4)
    with pytest.raises(RuntimeError, match="ranks"):
        require_world_size(2, 4)


def test_init_distributed_single_process_when_one_rank_expected(monkeypatch):
    monkeypatch.delenv("WORLD_SIZE", raising=False)
    ctx = init_distributed_from_env(expected_world_size=1, nccl_timeout_min=30)
    assert ctx == DistContext()
    assert not ctx.distributed


def test_init_distributed_rejects_mismatched_world_size_before_any_group_starts(monkeypatch):
    monkeypatch.setenv("WORLD_SIZE", "2")
    started: list[object] = []
    monkeypatch.setattr(dist, "init_process_group", lambda *a, **k: started.append(k))
    with pytest.raises(RuntimeError, match="4 ranks"):
        init_distributed_from_env(expected_world_size=4, nccl_timeout_min=30)
    assert started == []


def test_init_distributed_requires_cuda_for_a_multi_rank_job(monkeypatch):
    monkeypatch.setenv("WORLD_SIZE", "2")
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    started: list[object] = []
    monkeypatch.setattr(dist, "init_process_group", lambda *a, **k: started.append(k))

    with pytest.raises(RuntimeError, match="requires CUDA"):
        init_distributed_from_env(expected_world_size=2, nccl_timeout_min=30)

    assert started == []


@pytest.mark.parametrize(("minutes", "expected"), [(45.0, timedelta(minutes=45))])
def test_init_distributed_passes_nccl_timeout_minutes(monkeypatch, minutes, expected):
    monkeypatch.setenv("WORLD_SIZE", "2")
    monkeypatch.setenv("RANK", "1")
    monkeypatch.setenv("LOCAL_RANK", "1")
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch.cuda, "set_device", lambda index: None)
    monkeypatch.setattr(dist, "is_initialized", lambda: False)
    seen: dict[str, object] = {}
    monkeypatch.setattr(dist, "init_process_group", lambda **kwargs: seen.update(kwargs))
    ctx = init_distributed_from_env(expected_world_size=2, nccl_timeout_min=minutes)
    assert seen == {"backend": "nccl", "timeout": expected}
    assert (ctx.rank, ctx.world_size, ctx.local_rank) == (1, 2, 1)


def test_distributed_init_requires_a_process_group(minimal_config, make_meta_model):
    cfg = copy.deepcopy(minimal_config)
    cfg.training.distributed = True
    with pytest.raises(RuntimeError, match="process group"):
        make_meta_model(cfg)


def test_sharded_sampler_matches_single_process_shuffle():
    dataset = _IndexDataset(96)
    expected = [
        list(batch)
        for batch in DataLoader(
            dataset,
            shuffle=True,
            batch_size=12,
            drop_last=True,
            generator=torch.Generator().manual_seed(7),
        )
    ]

    rank_batches: list[list[list[int]]] = []
    for rank in range(_WORLD):
        generator = torch.Generator().manual_seed(7)
        sampler = ShardedBatchSampler(
            dataset, 12, rank, _WORLD, generator=generator, broadcast_state=lambda _: None
        )
        loader = DataLoader(dataset, batch_sampler=sampler, generator=generator)
        rank_batches.append([list(batch) for batch in loader])
    reconstructed = [sum(slices, []) for slices in zip(*rank_batches, strict=True)]
    assert reconstructed == expected


def test_sharded_sampler_shuffles_from_the_broadcast_state():
    dataset = _IndexDataset(60)
    rank0_state = torch.Generator().manual_seed(12345).get_state()
    expected = list(
        BatchSampler(
            RandomSampler(dataset, generator=torch.Generator().manual_seed(12345)),
            12,
            drop_last=True,
        )
    )
    iters = [
        iter(
            ShardedBatchSampler(
                dataset,
                12,
                rank,
                _WORLD,
                generator=torch.Generator().manual_seed(rank),
                broadcast_state=lambda generator: generator.set_state(rank0_state),
            )
        )
        for rank in range(_WORLD)
    ]
    reconstructed = [sum((next(it) for it in iters), []) for _ in expected]
    assert reconstructed == expected


def test_sharded_sampler_length_is_the_number_of_whole_global_batches():
    sampler = ShardedBatchSampler(
        _IndexDataset(100), 12, 1, _WORLD, generator=torch.Generator().manual_seed(0)
    )

    # 100 images make 8 whole batches of 12; the 4 left over are dropped.
    assert len(sampler) == 8
    assert [len(batch) for batch in sampler] == [_PER_RANK] * 8


def _broadcast_generator_worker(rank: int, world_size: int, init_method: str, queue) -> None:
    init_gloo(rank, world_size, init_method)
    try:
        generator = torch.Generator().manual_seed(100 + rank)
        broadcast_generator_from_rank0(generator)
        queue.put((rank, torch.rand(3, generator=generator).tolist()))
    finally:
        dist.destroy_process_group()


def test_broadcast_generator_gives_every_rank_rank0_state():
    expected = torch.rand(3, generator=torch.Generator().manual_seed(100)).tolist()
    assert spawn_gloo(_broadcast_generator_worker, 2) == [expected, expected]


def _reduce_components_worker(rank: int, world_size: int, init_method: str, queue) -> None:
    init_gloo(rank, world_size, init_method)
    try:
        values = {f"k{i}": float(rank + 1 + i) for i in range(3)}
        values["shared"] = float(rank)
        reduced = reduce_components(values)
        queue.put((rank, reduced))
    finally:
        dist.destroy_process_group()


def _on_main_process_worker(rank: int, world_size: int, init_method: str, queue) -> None:
    init_gloo(rank, world_size, init_method)
    try:
        ran = 0

        def body() -> None:
            nonlocal ran
            ran += 1

        on_main_process(body)
        queue.put((rank, ran))
    finally:
        dist.destroy_process_group()


def test_on_main_process_runs_on_one_rank():
    counts = spawn_gloo(_on_main_process_worker, _WORLD)
    assert counts == [1, 0, 0, 0]


def _mismatched_keys_worker(rank: int, world_size: int, init_method: str, queue) -> None:
    init_gloo(rank, world_size, init_method)
    try:
        values = {"classification": 1.0, "concepts": 2.0 if rank == 0 else 3.0}
        if rank == 1:
            values["extra"] = 4.0
        try:
            reduce_components(values)
        except RuntimeError as error:
            queue.put((rank, str(error)))
        else:
            queue.put((rank, "reduced"))
    finally:
        dist.destroy_process_group()


def test_reduce_components_raises_on_every_rank_when_the_keys_differ():
    messages = spawn_gloo(_mismatched_keys_worker, 2)

    assert messages == [
        "ranks logged different loss components; this rank has ['classification', 'concepts'].",
        "ranks logged different loss components; "
        "this rank has ['classification', 'concepts', 'extra'].",
    ]


def test_reduce_components_is_the_mean_over_ranks():
    per_rank = [
        {f"k{i}": float(rank + 1 + i) for i in range(3)} | {"shared": float(rank)}
        for rank in range(_WORLD)
    ]
    expected = {key: sum(d[key] for d in per_rank) / _WORLD for key in per_rank[0]}
    reduced = spawn_gloo(_reduce_components_worker, _WORLD)
    for payload in reduced:
        assert payload == pytest.approx(expected)


def _all_ranks_finite_worker(rank: int, world_size: int, init_method: str, queue) -> None:
    init_gloo(rank, world_size, init_method)
    try:
        finite_everywhere = torch.tensor([1.0, 2.0])
        bad_on_rank_one = torch.tensor([1.0, float("inf") if rank == 1 else 2.0])
        queue.put(
            (
                rank,
                {
                    "clean": all_ranks_finite(finite_everywhere),
                    "one_rank_bad": all_ranks_finite(bad_on_rank_one),
                },
            )
        )
    finally:
        dist.destroy_process_group()


def test_a_non_finite_loss_on_one_rank_reads_as_non_finite_on_every_rank():
    results = spawn_gloo(_all_ranks_finite_worker, _WORLD)

    assert results == [{"clean": True, "one_rank_bad": False}] * _WORLD


def _tiny_meta(state: dict[str, torch.Tensor], grad_accum_steps: int, updates: int):
    """A distributed MetaModel over ``_TinyCBM`` (weights ``state``) and ``_TinyLoss``.

    Clip 0.5 and AdamW at lr 1e-3 with no weight decay; the schedule holds the lr constant.
    """
    cfg = cbm_test_config()
    cfg.training.distributed = True
    cfg.training.world_size = _WORLD
    cfg.training.grad_accum_steps = grad_accum_steps
    cfg.training.iterations_per_train_epoch = updates
    cfg.training.max_grad_norm = 0.5
    cfg.training.optimizer.dpt_lr = 1e-3
    target, concepts = _lookups()
    model = _TinyCBM()
    model.load_state_dict(state)
    meta = meta_module.MetaModel(
        run_name="ddp-window",
        num_classes=_NUM_CLASSES,
        num_concepts=_NUM_CONCEPTS,
        model_kwargs=cfg.model,
        training_kwargs=cfg.training,
        device="cpu",
        source_to_target_lookup=target,
        source_to_concepts_lookup=concepts,
        concept_value2id=cast("dict[str, dict[str, list[int]]]", CBM_CONCEPT_VALUE2ID),
        model=model,
    )
    meta.loss = _TinyLoss()
    return meta


def _rank_microbatches(
    rank: int, images: torch.Tensor, source_labels: torch.Tensor, micros: range
) -> list[tuple[torch.Tensor, torch.Tensor]]:
    """This rank's contiguous shard of each global microbatch in ``micros``."""
    shards = []
    for micro in micros:
        start = micro * _GLOBAL_BATCH + rank * _PER_RANK
        shards.append((images[start : start + _PER_RANK], source_labels[start : start + _PER_RANK]))
    return shards


def _record_step_gradients(meta) -> dict[str, Any]:
    """Capture the averaged gradients and their norm at the moment ``meta`` applies its step."""
    captured: dict[str, Any] = {}
    real = meta._apply_accumulated_step

    def _capture() -> bool:
        trainable = {n: p for n, p in meta.model.named_parameters() if p.grad is not None}
        captured["grads"] = {n: p.grad.detach().clone().numpy() for n, p in trainable.items()}
        norms = torch.stack([p.grad.detach().norm() for p in trainable.values()])
        captured["norm"] = float(norms.norm())
        return real()

    meta._apply_accumulated_step = _capture
    return captured


def _record_forwards_and_reductions(meta) -> list[str]:
    """Event log of ``"forward"`` per microbatch and ``"reduce"`` per DDP gradient bucket reduced."""
    events: list[str] = []
    ddp = meta._get_ddp_step()

    def _hook(_state, bucket):
        events.append("reduce")
        return default_hooks.allreduce_hook(None, bucket)

    ddp.register_comm_hook(None, _hook)
    real = meta.batch_predict_loss

    def _forward(*args, **kwargs):
        events.append("forward")
        return real(*args, **kwargs)

    meta.batch_predict_loss = _forward
    return events


def _trainable_weights(meta) -> dict[str, Any]:
    # numpy, not tensors: queued tensors go through shared memory, which can be gone by the
    # time the parent reads them once this worker exits (EOFError in queue.get).
    return {
        name: param.detach().clone().numpy()
        for name, param in meta.model.named_parameters()
        if param.requires_grad
    }


def _ddp_window_worker(
    rank: int,
    world_size: int,
    init_method: str,
    queue,
    state: dict[str, torch.Tensor],
    images: torch.Tensor,
    source_labels: torch.Tensor,
) -> None:
    init_gloo(rank, world_size, init_method)
    try:
        meta = _tiny_meta(state, grad_accum_steps=_ACCUM, updates=1)
        captured = _record_step_gradients(meta)
        events = _record_forwards_and_reductions(meta)
        batches = _rank_microbatches(rank, images, source_labels, range(_ACCUM))
        meta.train_epoch(batches, epoch=0)
        queue.put(
            (
                rank,
                {
                    "events": events,
                    "grads": captured["grads"],
                    "clip": captured["norm"],
                    "weights": _trainable_weights(meta),
                },
            )
        )
    finally:
        dist.destroy_process_group()


def _serial_window(
    state: dict[str, torch.Tensor],
    images: torch.Tensor,
    source_labels: torch.Tensor,
    micros: range,
):
    """One optimizer update on a single process: the mean over ranks' shard losses per micro."""
    target, concepts = _lookups()
    model = _TinyCBM()
    model.load_state_dict(state)
    step = ShardStep(model, _TinyLoss(), None, target, concepts, False, torch.bfloat16)
    for micro in micros:
        shard_losses = []
        for rank in range(_WORLD):
            start = micro * _GLOBAL_BATCH + rank * _PER_RANK
            loss = step(
                images[start : start + _PER_RANK],
                source_labels[start : start + _PER_RANK],
                return_maps=False,
            )[0]
            shard_losses.append(loss)
        (torch.stack(shard_losses).mean() / len(micros)).backward()
    trainable = [param for param in step.parameters() if param.requires_grad]
    grads = {
        name: cast(torch.Tensor, param.grad).detach().clone()
        for name, param in step.named_parameters()
        if param.requires_grad
    }
    clip = torch.nn.utils.clip_grad_norm_(trainable, 0.5)
    torch.optim.AdamW(trainable, lr=1e-3, weight_decay=0.0).step()
    weights = {
        name: param.detach().clone()
        for name, param in step.named_parameters()
        if param.requires_grad
    }
    return grads, weights, float(clip)


def _tiny_state() -> dict[str, torch.Tensor]:
    torch.manual_seed(0)
    return {name: t.detach().clone() for name, t in _TinyCBM().state_dict().items()}


def _assert_matches_serial(payload, serial_grads, serial_weights, serial_clip) -> None:
    assert payload["clip"] == pytest.approx(serial_clip, rel=1e-5, abs=1e-6)
    for name, grad in serial_grads.items():
        # The serial reference names parameters through ShardStep ("model." prefix).
        torch.testing.assert_close(
            torch.from_numpy(payload["grads"][name.removeprefix("model.")]),
            grad,
            rtol=1e-5,
            atol=1e-6,
        )
    for name, weight in serial_weights.items():
        torch.testing.assert_close(
            torch.from_numpy(payload["weights"][name.removeprefix("model.")]),
            weight,
            rtol=1e-5,
            atol=1e-6,
        )


def test_train_epoch_window_matches_serial_microbatches():
    """Three microbatches per update, synced only on the last, equal the serial mean-shard step."""
    state = _tiny_state()
    images, source_labels = _batch()
    serial = _serial_window(state, images, source_labels, range(_ACCUM))

    payloads = spawn_gloo(_ddp_window_worker, _WORLD, extra=(state, images, source_labels))

    for payload in payloads:
        _assert_matches_serial(payload, *serial)
        # Gradients are all-reduced once, in the last microbatch's backward.
        assert payload["events"] == ["forward"] * _ACCUM + ["reduce"]


def _skipped_window_worker(
    rank: int,
    world_size: int,
    init_method: str,
    queue,
    state: dict[str, torch.Tensor],
    images: torch.Tensor,
    source_labels: torch.Tensor,
) -> None:
    init_gloo(rank, world_size, init_method)
    try:
        meta = _tiny_meta(state, grad_accum_steps=2, updates=2)
        captured = _record_step_gradients(meta)
        events = _record_forwards_and_reductions(meta)
        micro0, micro1, micro2, micro3 = _rank_microbatches(rank, images, source_labels, range(4))
        if rank == 1:
            micro1 = (torch.full_like(micro1[0], float("inf")), micro1[1])
        try:
            meta.train_epoch([micro0, micro1, micro2, micro3], epoch=0)
        except RuntimeError as error:
            queue.put((rank, {"error": str(error)}))
            return
        queue.put(
            (
                rank,
                {
                    "events": events,
                    "global_step": meta.global_step,
                    "grads": captured["grads"],
                    "clip": captured["norm"],
                    "weights": _trainable_weights(meta),
                },
            )
        )
    finally:
        dist.destroy_process_group()


def test_non_finite_shard_skips_the_window_on_every_rank_and_the_next_window_still_syncs():
    """Rank 1's second shard of window one is non-finite: every rank drops that window.

    The next window trains on microbatches 2 and 3. The dropped window still completes its
    reduction (``_finish_skipped_reduction``), so the next window reduces only on its last backward.
    """
    state = _tiny_state()
    images, source_labels = _batch(batch_size=4 * _GLOBAL_BATCH)
    serial = _serial_window(state, images, source_labels, range(2, 4))

    payloads = spawn_gloo(_skipped_window_worker, _WORLD, extra=(state, images, source_labels))

    assert [payload.get("error") for payload in payloads] == [None] * _WORLD
    assert [payload["global_step"] for payload in payloads] == [1] * _WORLD
    for payload in payloads:
        _assert_matches_serial(payload, *serial)
        assert payload["events"] == ["forward", "forward", "reduce", "forward", "forward", "reduce"]


def test_resume_at_an_epoch_boundary_then_validate_then_train(
    minimal_config, make_meta_model, tmp_path
):
    """A resume at an epoch's last update builds DDP inside validation, then trains."""
    cfg = copy.deepcopy(minimal_config)
    cfg.training.distributed = True
    cfg.training.iterations_per_train_epoch = 1
    cfg.training.iterations_per_val_epoch = 1
    init_gloo(0, 1, f"file://{tmp_path}/rendezvous")
    try:
        meta = make_meta_model(cfg, run_name="resume-boundary")
        meta.global_step = 1
        images = torch.randn(2, 3, *IMAGE_SIZE)
        labels = torch.ones(2, *images.shape[-2:], dtype=torch.long)
        loader = DataLoader(TensorDataset(images, labels), batch_size=2)
        meta.train_epoch(loader, evaluator=None, epoch=0)
        meta.validation_epoch(loader)
        loss, _, _ = meta.train_epoch(loader, evaluator=None, epoch=1)
        assert torch.isfinite(torch.tensor(loss))
    finally:
        dist.destroy_process_group()


class _MockEncoder(nn.Module):
    """Tiny DINOv3 stand-in with a ``q_proj`` LoRA target, as ``mock_dinov3_encoder`` builds."""

    def __init__(self) -> None:
        super().__init__()
        self.config = SimpleNamespace(
            hidden_size=16, patch_size=14, num_hidden_layers=4, num_register_tokens=4
        )
        self.embed = nn.Linear(3, 16)
        self.q_proj = nn.Linear(16, 16)

    def forward(self, pixel_values: torch.Tensor, **_: object) -> SimpleNamespace:
        patches = nn.functional.avg_pool2d(pixel_values, 14).flatten(2).transpose(1, 2)
        tokens = self.embed(patches)
        prefix = tokens.new_zeros(tokens.shape[0], 5, tokens.shape[2])
        states = self.q_proj(torch.cat([prefix, tokens], dim=1))
        return SimpleNamespace(hidden_states=(states,) * 5)


class _StepLogger:
    def __init__(self) -> None:
        self.steps: list[int] = []

    def log(self, metrics: dict[str, object], step: int) -> None:
        self.steps.append(step)


def _distributed_meta_model(world_size: int) -> meta_module.MetaModel:
    cfg = cbm_test_config()
    cfg.training.distributed = True
    cfg.training.world_size = world_size
    cfg.training.grad_accum_steps = 2
    cfg.training.iterations_per_train_epoch = 4
    cfg.training.iterations_per_val_epoch = 2
    torch.manual_seed(0)
    auto_model = SimpleNamespace(from_pretrained=lambda *_args, **_kwargs: _MockEncoder())
    with patch.object(models_module, "AutoModel", auto_model):
        return meta_module.MetaModel(
            run_name=f"ddp-train-validate-{world_size}",
            num_classes=NUM_CLASSES,
            model_kwargs=cfg.model,
            training_kwargs=cfg.training,
            device="cpu",
            num_concepts=CBM_NUM_CONCEPTS,
            concept_value2id=cast("dict[str, dict[str, list[int]]]", CBM_CONCEPT_VALUE2ID),
            source_to_target_lookup=torch.arange(NUM_CLASSES),
            source_to_concepts_lookup=cbm_concept_lookup(),
        )


def _train_then_validate_worker(rank: int, world_size: int, init_method: str, queue) -> None:
    init_gloo(rank, world_size, init_method)
    try:
        meta = _distributed_meta_model(world_size)
        run_logger = _StepLogger() if rank == 0 else None
        generator = torch.Generator().manual_seed(rank)
        images = torch.randn(16, 3, *IMAGE_SIZE, generator=generator)
        labels = torch.randint(1, NUM_CLASSES, (16, *IMAGE_SIZE), generator=generator)
        loader: DataLoader[Any] = DataLoader(TensorDataset(images, labels), batch_size=2)
        meta.train_epoch(loader, run_logger=run_logger, log_interval=2, epoch=0)
        meta.validation_epoch(loader)
        logged = None if run_logger is None else run_logger.steps
        queue.put((rank, {"global_step": meta.global_step, "logged": logged}))
    finally:
        dist.destroy_process_group()


def test_logger_only_on_rank_zero_trains_and_validates_without_desync():
    """Every rank joins the log-interval reduction, or rank 0 blocks in it alone."""
    results = spawn_gloo(_train_then_validate_worker, 2)
    assert [payload["global_step"] for payload in results] == [4, 4]
    assert [payload["logged"] for payload in results] == [[2, 4], None]
