"""Run settings of the trainer: dataset repetition, and a one-step CPU run."""

from __future__ import annotations

import argparse
import functools
import json
import multiprocessing
import os
import signal
import subprocess
import sys
import time
import types
from pathlib import Path
from types import SimpleNamespace
from typing import Any
from unittest.mock import MagicMock

import pytest
import torch
import torch.distributed as dist
import yaml
from torch.utils.data import ConcatDataset, DataLoader, Dataset, TensorDataset

import mermaidseg.model.models
from mermaidseg import training
from mermaidseg.datasets.factory import build_datasets
from mermaidseg.datasets.local_cache import create_cache_stats
from mermaidseg.inference.preprocess import IMAGENET_MEAN, IMAGENET_STD
from mermaidseg.io import ConfigDict, load_config, resolve_repo_path
from mermaidseg.model.distributed import DistContext
from mermaidseg.run_setup import RESUME_CHECKPOINT_KEYS, read_checkpoint
from mermaidseg.training import (
    VAL_GENERATOR_SEED_OFFSET,
    _assemble_train_datasets,
    _encoder_config,
    _lineage_params,
    _sigterm_raises_system_exit,
    load_training_checkpoint,
    make_shuffled_loader,
    seed_run,
)
from scripts.train import build_parser
from tests.conftest import cbm_test_config
from tests.mermaidseg.conftest import init_gloo, spawn_gloo


def _datasets() -> dict[tuple[str, str], str]:
    return {
        ("a", "train"): "a",
        ("coralscapes_v2", "train"): "cs",
        ("b", "train"): "b",
        ("b", "val"): "bv",
    }


def test_train_datasets_repeat_the_configured_dataset_after_all_trains():
    data_cfg = ConfigDict(
        {"a": {"train": {}}, "coralscapes_v2": {"train": {"repeat": 6}}, "b": {"train": {}}}
    )
    assert _assemble_train_datasets(_datasets(), data_cfg) == [
        "a",
        "cs",
        "b",
        "cs",
        "cs",
        "cs",
        "cs",
        "cs",
    ]


@pytest.mark.parametrize("repeat", [0, 2.5, True])
def test_train_datasets_reject_an_invalid_repeat_naming_the_dataset(repeat):
    data_cfg = ConfigDict(
        {"a": {"train": {}}, "coralscapes_v2": {"train": {"repeat": repeat}}, "b": {"train": {}}}
    )
    with pytest.raises(ValueError, match="coralscapes_v2"):
        _assemble_train_datasets(_datasets(), data_cfg)


# ---------- shuffle order ----------


class _Indices(Dataset):
    def __len__(self) -> int:
        return 16

    def __getitem__(self, index: int) -> int:
        return index


def _order(generator: torch.Generator) -> list[int]:
    loader = make_shuffled_loader(
        DistContext(), ConcatDataset([_Indices()]), batch_size=4, generator=generator
    )
    return [int(i) for batch in loader for i in batch]


def test_seed_run_seeds_val_generator_at_the_offset():
    train_generator, val_generator = torch.Generator(), torch.Generator()
    seed_run(7, train_generator, val_generator)
    assert train_generator.initial_seed() == 7
    assert val_generator.initial_seed() == 7 + VAL_GENERATOR_SEED_OFFSET
    assert VAL_GENERATOR_SEED_OFFSET != 0


def test_seed_run_reseeds_a_built_loader_to_the_resume_order():
    train_generator, val_generator = torch.Generator(), torch.Generator()
    seed_run(7, train_generator, val_generator)
    loader = make_shuffled_loader(
        DistContext(), ConcatDataset([_Indices()]), batch_size=4, generator=train_generator
    )
    seed_run(9, train_generator, val_generator)
    resumed = [int(i) for batch in loader for i in batch]
    assert resumed == _order(torch.Generator().manual_seed(9))
    assert resumed != _order(torch.Generator().manual_seed(7))


# ---------- one-step smoke run ----------

SMOKE_RESOLUTION = 56
SMOKE_RUN = "smoke"


class _SmokeCoralscapesV2(Dataset):
    """Two fully labelled images per split, one class each, in coralscapes_v2's label space."""

    SOURCE_NAME = "coralscapes_v2"

    def __init__(self, **_split_kwargs: Any) -> None:
        names = ["pocillopora alive", "sand"]
        self.source_id2name = dict(enumerate(names, start=1))
        self.source_name2id = {name: i for i, name in self.source_id2name.items()}
        self.num_source_classes = len(names) + 1
        self._global_offset = 0

    def set_global_offset(self, offset: int) -> None:
        self._global_offset = offset

    def __len__(self) -> int:
        return 2

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        size = (SMOKE_RESOLUTION, SMOKE_RESOLUTION)
        label = self._global_offset + 1 + idx
        return torch.zeros(3, *size), torch.full(size, label, dtype=torch.long)


class _FakeWandbRun:
    id = "smoke-id"
    url = "https://wandb.invalid/smoke"
    tags: list[str] = []

    def __init__(self) -> None:
        self.params: dict[str, Any] = {}
        self.config = SimpleNamespace(update=lambda d, **_: self.params.update(d))
        self.exit_code: int | None = None

    def log(self, log_dict: dict, step: int) -> None:
        pass

    def log_artifact(self, artifact: Any) -> None:
        pass

    def finish(self, exit_code: int | None = None) -> None:
        self.exit_code = exit_code


def _fake_wandb(run: _FakeWandbRun) -> types.ModuleType:
    fake = types.ModuleType("wandb")
    fake.api = SimpleNamespace(api_key="smoke-key")  # type: ignore[attr-defined]
    fake.init = lambda **_: run  # type: ignore[attr-defined]
    fake.Artifact = MagicMock()  # type: ignore[attr-defined]
    return fake


def _plain(cfg: ConfigDict) -> dict[str, Any]:
    return json.loads(json.dumps(cfg))


def _write_section(path: Path, section: str, values: dict[str, Any]) -> str:
    path.write_text(yaml.safe_dump({section: values}))
    return str(path)


def _smoke_configs(tmp_path: Path, epochs: int = 1) -> list[str]:
    """Tiny training, model, data and logger YAMLs derived from the final run's configs."""
    training = load_config(resolve_repo_path("configs/training_config_cbm_final.yaml")).training
    training.update(
        epochs=epochs,
        iterations_per_train_epoch=1,
        iterations_per_val_epoch=1,
        batch_size=2,
        checkpoint_every_steps=1,
        world_size=1,
        seed=7,
    )
    training.scheduler.update(warmup_iters=0, total_iters=epochs)
    model = load_config(resolve_repo_path("configs/model_config_cbm_dpt_lora_vitl.yaml")).model
    model.update(
        concept_feature_dim=8,
        neck_hidden_sizes=[8, 8, 8, 8],
        fusion_hidden_size=8,
        lora_r=2,
        lora_alpha=4,
        lora_target_modules=["q_proj"],
    )
    logger = load_config(resolve_repo_path("configs/logger_config.yaml")).logger
    data = {
        "local_cache_write_through": True,
        "coralscapes_v2": {"train": {"split": "train"}, "val": {"split": "val"}},
    }
    return [
        "--training-config",
        _write_section(tmp_path / "training.yaml", "training", _plain(training)),
        "--model-config",
        _write_section(tmp_path / "model.yaml", "model", _plain(model)),
        "--data-config",
        _write_section(tmp_path / "data.yaml", "data", data),
        "--logger-config",
        _write_section(tmp_path / "logger.yaml", "logger", _plain(logger)),
    ]


def test_run_rejects_a_world_size_that_the_launcher_did_not_provide(monkeypatch, tmp_path):
    monkeypatch.chdir(tmp_path)
    monkeypatch.delenv("WORLD_SIZE", raising=False)
    monkeypatch.delenv("WANDB_RUN_ID", raising=False)
    fake_wandb = _fake_wandb(_FakeWandbRun())
    monkeypatch.setitem(sys.modules, "wandb", fake_wandb)

    def _unreachable(*args: Any, **kwargs: Any) -> Any:
        raise AssertionError("run() built datasets before checking the world size")

    monkeypatch.setattr(training, "build_datasets", _unreachable)
    configs = _smoke_configs(tmp_path)
    training_path = Path(configs[configs.index("--training-config") + 1])
    section = yaml.safe_load(training_path.read_text())
    section["training"]["world_size"] = 2
    training_path.write_text(yaml.safe_dump(section))
    args = build_parser().parse_args(
        [
            "--run-name",
            SMOKE_RUN,
            "--num-workers",
            "1",
            "--output-dir",
            str(tmp_path / "out"),
            *configs,
        ]
    )
    with pytest.raises(RuntimeError, match="2 ranks"):
        training.run(args)


def _install_smoke_fakes(
    monkeypatch, tmp_path: Path, mock_dinov3_encoder
) -> tuple[Path, _FakeWandbRun, list[Any]]:
    """Patch wandb, the encoder, the cache and the datasets; return cwd, wandb run, cache roots."""
    workdir = tmp_path / "cwd"
    workdir.mkdir()
    monkeypatch.chdir(workdir)
    monkeypatch.delenv("WORLD_SIZE", raising=False)
    monkeypatch.delenv("WANDB_RUN_ID", raising=False)

    mock_dinov3_encoder(mermaidseg.model.models)
    wandb_run = _FakeWandbRun()
    fake_wandb = _fake_wandb(wandb_run)
    monkeypatch.setitem(sys.modules, "wandb", fake_wandb)
    cache_roots: list[Any] = []

    def _record_cache(data_cfg: Any) -> Any:
        cache_roots.append(data_cfg.local_cache_dir)
        return create_cache_stats()

    monkeypatch.setattr(training, "setup_local_cache", _record_cache)
    monkeypatch.setattr(
        training,
        "build_datasets",
        functools.partial(
            build_datasets,
            dataset_classes={"coralscapes_v2": _SmokeCoralscapesV2},
            verbose=False,
        ),
    )
    return workdir, wandb_run, cache_roots


def _smoke_args(
    tmp_path: Path,
    output_dir: Path,
    cache_dir: Path,
    *,
    epochs: int = 1,
    checkpoint: Path | None = None,
) -> argparse.Namespace:
    cli = ["--checkpoint", str(checkpoint)] if checkpoint else []
    return build_parser().parse_args(
        [
            "--run-name",
            SMOKE_RUN,
            "--num-workers",
            "1",
            "--resolution",
            str(SMOKE_RESOLUTION),
            "--output-dir",
            str(output_dir),
            "--cache-dir",
            str(cache_dir),
            *cli,
            *_smoke_configs(tmp_path, epochs),
        ]
    )


_FORK_ONLY = pytest.mark.skipif(
    multiprocessing.get_start_method() != "fork",
    reason=(
        "spawn-started DataLoader workers hang pytest at interpreter exit: torch_shm_manager "
        "inherits the multiprocessing resource-tracker pipe (tracked follow-up)"
    ),
)


@_FORK_ONLY
def test_one_step_run_writes_its_checkpoint_under_the_output_dir(
    monkeypatch, tmp_path, mock_dinov3_encoder
):
    workdir, wandb_run, cache_roots = _install_smoke_fakes(
        monkeypatch, tmp_path, mock_dinov3_encoder
    )
    output_dir = tmp_path / "out"
    cache_dir = tmp_path / "cache"
    sigterm_before = signal.getsignal(signal.SIGTERM)
    training.run(_smoke_args(tmp_path, output_dir, cache_dir))
    assert signal.getsignal(signal.SIGTERM) is sigterm_before

    run_dir = output_dir / "model_checkpoints" / SMOKE_RUN
    assert (run_dir / "model_step1").is_file()
    assert not (run_dir / "id2label.json").exists()
    assert not (run_dir / "concept_id2name.json").exists()
    sidecar = json.loads((run_dir / "model.json").read_text())
    assert sidecar["id2label"]["0"] == "ignore"
    assert "sand" in sidecar["id2label"].values()
    assert sidecar["concept_id2name"]
    assert sidecar["lineage"]["seed"] == 7
    assert sidecar["architecture"]["encoder_config"]["hidden_size"] == 16
    assert sidecar["preprocessing"]["mean"] == list(IMAGENET_MEAN)
    assert sidecar["preprocessing"]["std"] == list(IMAGENET_STD)
    resolved = yaml.safe_load((run_dir / "config.yaml").read_text())
    assert resolved["run_name"] == SMOKE_RUN
    assert resolved["model"]["input_size"] == [SMOKE_RESOLUTION, SMOKE_RESOLUTION]
    assert sidecar["preprocessing"]["size_hw"] == resolved["model"]["input_size"]
    assert resolved["training"]["concept_mapping_path"] == "configs/class_to_concepts.csv"
    checkpoint = torch.load(run_dir / "model_step1", weights_only=True)
    assert checkpoint["config"] == resolved
    assert checkpoint["lineage"]["seed"] == 7
    assert not (workdir / "model_checkpoints").exists()
    assert cache_roots == [str(cache_dir)]
    assert wandb_run.params["cli/num_workers"] == 1
    assert wandb_run.params["cli/output_dir"] == str(output_dir)
    assert wandb_run.params["data/seed"] == 7
    assert {k for k in wandb_run.params if k.startswith("lineage/")} >= {
        "lineage/git_sha",
        "lineage/git_dirty",
        "lineage/uv_lock_sha256",
        "lineage/data_config_sha1",
    }
    assert wandb_run.params["training_concept_mapping_path"] == "configs/class_to_concepts.csv"
    assert wandb_run.exit_code == 0


def _raise_boom(**_: Any) -> None:
    raise RuntimeError("boom")


def _send_sigterm(**_: Any) -> None:
    os.kill(os.getpid(), signal.SIGTERM)


def _guard_sigterm(signum: int, frame: Any) -> None:
    raise AssertionError("SIGTERM reached the guard handler")


@_FORK_ONLY
@pytest.mark.parametrize(
    ("train_step", "expected_exit"),
    [
        pytest.param(_raise_boom, RuntimeError, id="exception"),
        pytest.param(_send_sigterm, SystemExit, id="sigterm"),
    ],
)
def test_run_ends_the_tracker_run_as_failed_when_training_dies_mid_run(
    monkeypatch, tmp_path, mock_dinov3_encoder, train_step, expected_exit
):
    _, wandb_run, _ = _install_smoke_fakes(monkeypatch, tmp_path, mock_dinov3_encoder)
    monkeypatch.setattr(training, "train_model", train_step)
    # Without run()'s handler, a self-sent SIGTERM would otherwise kill pytest.
    original_sigterm = signal.signal(signal.SIGTERM, _guard_sigterm)
    original_sigchld = signal.getsignal(signal.SIGCHLD)
    try:
        with pytest.raises(expected_exit) as raised:
            training.run(_smoke_args(tmp_path, tmp_path / "out", tmp_path / "cache"))
        assert signal.getsignal(signal.SIGTERM) is _guard_sigterm
    finally:
        signal.signal(signal.SIGTERM, original_sigterm)
        signal.signal(signal.SIGCHLD, original_sigchld)

    if expected_exit is SystemExit:
        assert raised.value.code == 143
    assert wandb_run.exit_code == 1


class _EndlessSlowDataset(Dataset):
    def __len__(self) -> int:
        return 10**6

    def __getitem__(self, index: int) -> torch.Tensor:
        time.sleep(0.01)
        return torch.zeros(4)


class _RecordingExit:
    """Stand-in for a tracker's end_run: Python work in ``__exit__`` that reports how it ended."""

    def __init__(self, conn: Any) -> None:
        self.conn = conn

    def __enter__(self) -> _RecordingExit:
        return self

    def __exit__(self, *exc_info: Any) -> None:
        try:
            deadline = time.monotonic() + 1.0
            while time.monotonic() < deadline:
                sum(range(1000))
        except Exception as error:
            self.conn.send(f"{type(error).__name__}: {error}")
        else:
            self.conn.send("completed")


def _iterate_until_group_sigterm(conn: Any) -> None:
    # Own process group, so the group-wide SIGTERM cannot reach pytest.
    os.setpgid(0, 0)
    loader = DataLoader(
        _EndlessSlowDataset(),
        batch_size=2,
        num_workers=2,
        persistent_workers=True,
        multiprocessing_context="fork",
    )
    with _sigterm_raises_system_exit(True), _RecordingExit(conn):
        for index, _ in enumerate(loader):
            if index == 5:
                # Sent by another process, as Slurm and torchrun do, so the workers die too.
                subprocess.Popen(["/bin/kill", "-TERM", "--", f"-{os.getpgrp()}"])


@_FORK_ONLY
def test_group_sigterm_lets_the_enclosing_exit_finish_despite_dying_dataloader_workers():
    context = multiprocessing.get_context("fork")
    receiver, sender = context.Pipe(duplex=False)
    scenario = context.Process(target=_iterate_until_group_sigterm, args=(sender,))
    scenario.start()
    sender.close()
    try:
        assert receiver.poll(60), "the scenario never reached its __exit__"
        outcome = receiver.recv()
        scenario.join(60)
    finally:
        if scenario.is_alive():
            os.killpg(scenario.pid, signal.SIGKILL)
            scenario.join()

    assert outcome == "completed"
    assert scenario.exitcode == 143


def _resume_checkpoint(meta) -> dict[str, Any]:
    """A checkpoint with the keys ``Logger.save_model_checkpoint`` writes."""
    return {
        "config": {},
        "model_state_dict": meta.model.state_dict(),
        "optimizer_state_dict": meta.optimizer.state_dict(),
        "epoch": 2,
        "epoch_complete": False,
        "global_step": 7,
        "warmup_iters_completed": 5,
        "timestamp": "20260101000000",
        "metrics": {},
        "scheduler_state_dict": meta.scheduler.state_dict(),
    }


def test_load_training_checkpoint_restores_the_resume_state(
    minimal_config, make_meta_model, tmp_path
):
    meta = make_meta_model(minimal_config, run_name="resume-ok")
    path = tmp_path / "ckpt.pt"
    torch.save(_resume_checkpoint(meta), path)
    start_epoch = load_training_checkpoint(
        meta, read_checkpoint(path, RESUME_CHECKPOINT_KEYS), DistContext()
    )
    assert start_epoch == 2
    assert meta.global_step == 7
    assert meta._warmup_iters_completed == 5
    assert meta.resuming_within_epoch is True


def _warmup_config() -> ConfigDict:
    minimal_config = cbm_test_config()
    minimal_config.training.scheduler.update(
        type="LinearLR",
        start_factor=1.0,
        end_factor=0.1,
        total_iters=20,
        warmup_iters=10,
        warmup_start_factor=0.01,
    )
    return minimal_config


def _lr_after_steps(meta, steps: int) -> list[float]:
    """Run ``steps`` optimizer steps with the trainer's scheduler order; return each LR."""
    lrs = []
    for _ in range(steps):
        meta.optimizer.step()
        meta._step_warmup_scheduler()
        meta._step_iteration_scheduler()
        lrs.append(meta.optimizer.param_groups[0]["lr"])
    return lrs


def test_resumed_mid_warmup_lr_matches_the_uninterrupted_run(make_meta_model, tmp_path):
    continuous = _lr_after_steps(make_meta_model(_warmup_config(), run_name="continuous"), 16)

    first = make_meta_model(_warmup_config(), run_name="first-half")
    _lr_after_steps(first, 5)
    checkpoint = _resume_checkpoint(first)
    checkpoint["warmup_iters_completed"] = first._warmup_iters_completed
    path = tmp_path / "ckpt.pt"
    torch.save(checkpoint, path)

    resumed = make_meta_model(_warmup_config(), run_name="second-half")
    load_training_checkpoint(resumed, read_checkpoint(path, RESUME_CHECKPOINT_KEYS), DistContext())
    assert _lr_after_steps(resumed, 11) == pytest.approx(continuous[5:])


def test_run_rejects_a_bad_checkpoint_before_building_datasets(monkeypatch, tmp_path):
    def _unreachable(*args: Any, **kwargs: Any) -> Any:
        raise AssertionError("run() reached dataset setup with an unloadable checkpoint")

    monkeypatch.chdir(tmp_path)
    monkeypatch.delenv("WORLD_SIZE", raising=False)
    monkeypatch.delenv("WANDB_RUN_ID", raising=False)
    fake_wandb = _fake_wandb(_FakeWandbRun())
    monkeypatch.setitem(sys.modules, "wandb", fake_wandb)
    monkeypatch.setattr(training, "setup_local_cache", _unreachable)
    monkeypatch.setattr(training, "build_datasets", _unreachable)
    bad = tmp_path / "bad.pt"
    bad.write_bytes(b"not a checkpoint")
    args = _smoke_args(tmp_path, tmp_path / "out", tmp_path / "cache", checkpoint=bad)
    with pytest.raises(RuntimeError, match="bad.pt"):
        training.run(args)


@_FORK_ONLY
def test_resumed_run_continues_the_global_step_from_its_checkpoint(
    monkeypatch, tmp_path, mock_dinov3_encoder
):
    _install_smoke_fakes(monkeypatch, tmp_path, mock_dinov3_encoder)
    output_dir = tmp_path / "out"
    cache_dir = tmp_path / "cache"
    run_dir = output_dir / "model_checkpoints" / SMOKE_RUN

    training.run(_smoke_args(tmp_path, output_dir, cache_dir))
    first_checkpoint = run_dir / "model_step1"
    assert first_checkpoint.is_file()
    assert not (run_dir / "model_step2").exists()
    first_written = first_checkpoint.stat().st_mtime_ns

    # Run 1 consumed epoch 0 of 1; a second epoch lets the resumed run take update 2.
    training.run(
        _smoke_args(tmp_path, output_dir, cache_dir, epochs=2, checkpoint=first_checkpoint)
    )
    # The checkpoint is this project's own, written by training.run() earlier in this test.
    resumed = torch.load(run_dir / "model_step2", map_location="cpu", weights_only=False)
    assert resumed["global_step"] == 2
    # A run that restarted from step 0 would rewrite model_step1.
    assert first_checkpoint.stat().st_mtime_ns == first_written


def test_lineage_params_flatten_the_lineage_dict():
    lineage = {
        "git_sha": "abc123",
        "git_dirty": False,
        "uv_lock_sha256": "deadbeef",
        "data_version": {
            "data_config_sha1": "c0ffee",
            "manifests": {"coralnet.train": "m/train.parquet", "coralnet.val": "m/val.parquet"},
        },
    }
    assert _lineage_params(lineage) == {
        "lineage/git_sha": "abc123",
        "lineage/git_dirty": False,
        "lineage/uv_lock_sha256": "deadbeef",
        "lineage/data_config_sha1": "c0ffee",
        "lineage/manifests/coralnet.train": "m/train.parquet",
        "lineage/manifests/coralnet.val": "m/val.parquet",
    }


def _model_with_encoder_config(config: dict):
    base = SimpleNamespace(config=SimpleNamespace(to_dict=lambda: dict(config)))
    return SimpleNamespace(encoder=SimpleNamespace(get_base_model=lambda: base))


def test_stored_encoder_config_drops_environment_keys_so_a_resume_passes():
    arch = {"hidden_size": 16, "num_hidden_layers": 4}
    old = _model_with_encoder_config(
        {**arch, "transformers_version": "4.50.0", "_name_or_path": "/cache/a"}
    )
    assert _encoder_config(old) == arch


_LOADER_SEED = 21
_LOADER_ITEMS = 24
_LOADER_BATCH = 4


def _index_dataset() -> ConcatDataset:
    return ConcatDataset([TensorDataset(torch.arange(_LOADER_ITEMS))])


def _shuffled_loader_worker(rank: int, world_size: int, init_method: str, queue) -> None:
    init_gloo(rank, world_size, init_method)
    try:
        # Rank 1 starts from a different seed; the broadcast must align it with rank 0.
        generator = torch.Generator().manual_seed(_LOADER_SEED + rank)
        loader = make_shuffled_loader(
            DistContext(rank=rank, world_size=world_size, local_rank=rank),
            _index_dataset(),
            generator,
            batch_size=_LOADER_BATCH,
            drop_last=True,
            num_workers=0,
        )
        passes = [[batch[0].tolist() for batch in loader] for _ in range(2)]
        queue.put((rank, passes))
    finally:
        dist.destroy_process_group()


def test_distributed_shuffled_loader_matches_single_process_over_two_passes():
    reference_loader = DataLoader(
        _index_dataset(),
        shuffle=True,
        batch_size=_LOADER_BATCH,
        drop_last=True,
        generator=torch.Generator().manual_seed(_LOADER_SEED),
    )
    expected = [[batch[0].tolist() for batch in reference_loader] for _ in range(2)]
    rank0, rank1 = spawn_gloo(_shuffled_loader_worker, 2)
    for pass_index in range(2):
        interleaved = [
            shard0 + shard1
            for shard0, shard1 in zip(rank0[pass_index], rank1[pass_index], strict=True)
        ]
        assert interleaved == expected[pass_index]
