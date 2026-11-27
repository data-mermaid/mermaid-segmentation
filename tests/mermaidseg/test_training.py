"""Run settings of the trainer: dataset repetition, and a one-step CPU run."""

from __future__ import annotations

import argparse
import dataclasses
import datetime
import hashlib
import json
import multiprocessing
import os
import signal
import subprocess
import sys
import time
import types
import unittest.mock
from collections.abc import Sequence
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Self
from unittest.mock import MagicMock

import numpy as np
import pyarrow as pa
import pytest
import torch
import yaml
from torch.utils.data import ConcatDataset, DataLoader, Dataset, TensorDataset

import mermaidseg.model.models
from mermaidseg import training
from mermaidseg.data.config import DataSection
from mermaidseg.data.factory import open_dataset_version
from mermaidseg.dataset_reconciliation.concept_schema import ConceptSchema
from mermaidseg.inference.preprocess import IMAGENET_MEAN, IMAGENET_STD
from mermaidseg.io import ConfigDict, build_train_parser, load_section, resolve_repo_path
from mermaidseg.lineage import data_version
from mermaidseg.model.distributed import DistContext
from mermaidseg.run_setup import (
    RESUME_CHECKPOINT_KEYS,
    concept_mapping_path,
    concept_removal_list,
    read_checkpoint,
)
from mermaidseg.training import (
    _assemble_train_datasets,
    _copy_label_space,
    _encoder_config,
    _lineage_params,
    _sigterm_raises_system_exit,
    load_training_checkpoint,
    make_shuffled_loader,
    seed_run,
)
from segdata_spec import stage as segdata_stage
from tests.conftest import cbm_test_config, dinov3_encoder_mocker
from tests.mermaidseg.conftest import write_test_label_space
from tests.mermaidseg.data.conftest import ImageSpec, write_taxonomy, write_version


def _datasets() -> dict[tuple[str, str], str]:
    return {
        ("a", "train"): "a",
        ("coralscapes_v2", "train"): "cs",
        ("b", "train"): "b",
        ("b", "val"): "bv",
    }


def _train_section(**train_blocks: dict[str, Any]) -> DataSection:
    pin = {"version": "v", "manifest_sha256": "s"}
    return DataSection.model_validate(
        {"datasets": {name: {**pin, "train": block} for name, block in train_blocks.items()}}
    )


def test_train_datasets_repeat_the_configured_dataset_after_all_trains():
    section = _train_section(
        a={"groups": "all"},
        coralscapes_v2={"groups": "all", "repeat": 6},
        b={"groups": "all"},
    )
    assert _assemble_train_datasets(_datasets(), section) == [
        "a",
        "cs",
        "b",
        "cs",
        "cs",
        "cs",
        "cs",
        "cs",
    ]


# ---------- shuffle order ----------


class _Indices(Dataset):
    def __len__(self) -> int:
        return 16

    def __getitem__(self, index: int) -> int:
        return index


def _source_files(tmp_path: Path) -> list[Path]:
    source = tmp_path / "source"
    source.mkdir()
    paths = [source / "classes.csv", source / "meta.yaml"]
    for path in paths:
        path.write_text(f"{path.name} A")
    return paths


def _sha256(text: str) -> str:
    return hashlib.sha256(text.encode()).hexdigest()


def test_copy_label_space_returns_the_hashes_of_the_copy(tmp_path):
    files = _source_files(tmp_path)
    target = tmp_path / "run" / "label_space"
    target.parent.mkdir()

    hashes = _copy_label_space(files, target)

    assert hashes == {"classes.csv": _sha256("classes.csv A"), "meta.yaml": _sha256("meta.yaml A")}
    assert (target / "meta.yaml").read_text() == "meta.yaml A"


def test_copy_label_space_replaces_a_leftover_partial_copy(tmp_path):
    files = _source_files(tmp_path)
    target = tmp_path / "label_space"
    partial = tmp_path / ".label_space.partial"
    partial.mkdir()
    (partial / "stale.csv").write_text("stale")

    _copy_label_space(files, target)

    assert sorted(p.name for p in target.iterdir()) == ["classes.csv", "meta.yaml"]
    assert not partial.exists()


def test_copy_label_space_keeps_an_identical_existing_copy(tmp_path):
    files = _source_files(tmp_path)
    target = tmp_path / "label_space"
    _copy_label_space(files, target)
    marker = target / "classes.csv"
    mtime = marker.stat().st_mtime_ns

    hashes = _copy_label_space(files, target)

    assert hashes["classes.csv"] == _sha256("classes.csv A")
    assert marker.stat().st_mtime_ns == mtime


def test_copy_label_space_refuses_an_existing_copy_that_differs_naming_the_file(tmp_path):
    files = _source_files(tmp_path)
    target = tmp_path / "label_space"
    _copy_label_space(files, target)
    files[1].write_text("meta.yaml A2")

    with pytest.raises(ValueError, match=r"\['meta\.yaml'\]"):
        _copy_label_space(files, target)


def _order(generator: torch.Generator) -> list[int]:
    loader = make_shuffled_loader(
        DistContext(), ConcatDataset([_Indices()]), batch_size=4, generator=generator
    )
    return [int(i) for batch in loader for i in batch]


def test_seed_run_seeds_val_generator_at_the_offset():
    seeds = []
    for seed in (7, 9):
        train_generator, val_generator = torch.Generator(), torch.Generator()
        seed_run(seed, train_generator, val_generator)
        assert train_generator.initial_seed() == seed
        assert val_generator.initial_seed() != seed
        seeds.append(val_generator.initial_seed())
    assert seeds[1] - seeds[0] == 2


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
_SMOKE_TAXONOMY = pa.table(
    {"id": ["1", "2"], "name": ["Pocillopora", "Sand"], "parent_id": [None, None]}
)


class _FrozenDatetime(datetime.datetime):
    """``datetime`` whose ``now`` is fixed, so staged manifests (and their pins) repeat exactly."""

    @classmethod
    def now(cls, tz: datetime.tzinfo | None = None) -> Self:
        return cls(2026, 1, 1, tzinfo=tz)


def _write_smoke_versions(tmp_path: Path) -> dict[str, Any]:
    """Write a tiny coralscapes_v2 version and the taxonomy into ``<tmp_path>/cache``.

    Each split holds two fully labelled images, one class each. Returns the data section; the
    pins are the same on every call, as a resume requires.
    """
    with unittest.mock.patch.object(segdata_stage, "datetime", _FrozenDatetime):
        return _stage_smoke_versions(tmp_path)


def _stage_smoke_versions(tmp_path: Path) -> dict[str, Any]:
    size = (SMOKE_RESOLUTION, SMOKE_RESOLUTION)
    images = [
        ImageSpec(f"{split}{k}", group=split, mask=np.full(size, k + 1, np.uint8))
        for split in ("train", "val")
        for k in range(2)
    ]
    labels = [
        {"label_id": 1, "name": "pocillopora alive", "mermaid_benthic_attribute": "Pocillopora"},
        {"label_id": 2, "name": "sand", "mermaid_benthic_attribute": "Sand"},
    ]
    cache_dir, _, pin = write_version(
        tmp_path, kind="dense", images=images, labels=labels, name="coralscapes_v2"
    )
    taxonomy_pin = write_taxonomy(cache_dir, tmp_path, _SMOKE_TAXONOMY)
    return {
        "taxonomy": {
            "version": taxonomy_pin.version,
            "manifest_sha256": taxonomy_pin.manifest_sha256,
        },
        "datasets": {
            "coralscapes_v2": {
                "version": pin.version,
                "manifest_sha256": pin.manifest_sha256,
                "train": {"groups": ["train"]},
                "val": {"groups": ["val"]},
            }
        },
    }


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


def _write_smoke_label_space(tmp_path: Path, training: ConfigDict, data: dict[str, Any]) -> Path:
    """Write the label space of the smoke data into ``<tmp_path>/label_space``; return its path.

    Its meta matches ``training`` (with the run's concept removal and source mask files).
    """
    section = DataSection.model_validate(data)
    mapping = concept_mapping_path(training)
    schema_sources = set(
        load_section(resolve_repo_path("configs/data_config.yaml"), "data")["datasets"]
    )
    schema = ConceptSchema.from_csv(
        mapping, sources=schema_sources, removed_concepts=concept_removal_list(training, mapping)
    )
    directory = tmp_path / "label_space"
    write_test_label_space(
        directory,
        {"coralscapes_v2": open_dataset_version(section, "coralscapes_v2", tmp_path / "cache")},
        training,
        schema,
        section,
        taxonomy=_SMOKE_TAXONOMY,
    )
    return directory


def _smoke_configs(
    tmp_path: Path, epochs: int = 1, training_overrides: dict[str, Any] | None = None
) -> list[str]:
    """Tiny training, model, data and logger YAMLs derived from the final run's configs.

    The data config pins tiny versions written into ``<tmp_path>/cache``.
    """
    training = ConfigDict(
        load_section(resolve_repo_path("configs/training_config_cbm_final.yaml"), "training")
    )
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
    training.update(training_overrides or {})
    model = ConfigDict(
        load_section(resolve_repo_path("configs/model_config_cbm_dpt_lora_vitl.yaml"), "model")
    )
    model.update(
        concept_feature_dim=8,
        neck_hidden_sizes=[8, 8, 8, 8],
        fusion_hidden_size=8,
        lora_r=2,
        lora_alpha=4,
        lora_target_modules=["q_proj"],
    )
    logger = ConfigDict(load_section(resolve_repo_path("configs/logger_config.yaml"), "logger"))
    data = _write_smoke_versions(tmp_path)
    training.label_space = str(_write_smoke_label_space(tmp_path, training, data))
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
    args = build_train_parser().parse_args(
        [
            "--run-name",
            SMOKE_RUN,
            "--num-workers",
            "1",
            "--output-dir",
            str(tmp_path / "out"),
            "--cache-dir",
            str(tmp_path / "cache"),
            *configs,
        ]
    )
    with pytest.raises(RuntimeError, match="2 ranks"):
        training.run(args)


def _install_smoke_fakes(
    monkeypatch, tmp_path: Path, mock_dinov3_encoder
) -> tuple[Path, _FakeWandbRun]:
    """Patch wandb and the encoder; return the cwd and the wandb run."""
    workdir = tmp_path / "cwd"
    workdir.mkdir()
    monkeypatch.chdir(workdir)
    monkeypatch.delenv("WORLD_SIZE", raising=False)
    monkeypatch.delenv("WANDB_RUN_ID", raising=False)

    mock_dinov3_encoder(mermaidseg.model.models)
    wandb_run = _FakeWandbRun()
    fake_wandb = _fake_wandb(wandb_run)
    monkeypatch.setitem(sys.modules, "wandb", fake_wandb)
    return workdir, wandb_run


def _smoke_args(
    tmp_path: Path,
    output_dir: Path,
    cache_dir: Path,
    *,
    epochs: int = 1,
    checkpoint: Path | None = None,
    cli: Sequence[str] = (),
    training_overrides: dict[str, Any] | None = None,
) -> argparse.Namespace:
    checkpoint_cli = ["--checkpoint", str(checkpoint)] if checkpoint else []
    return build_train_parser().parse_args(
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
            *checkpoint_cli,
            *cli,
            *_smoke_configs(tmp_path, epochs, training_overrides),
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
    workdir, wandb_run = _install_smoke_fakes(monkeypatch, tmp_path, mock_dinov3_encoder)
    output_dir = tmp_path / "out"
    cache_dir = tmp_path / "cache"
    sigterm_before = signal.getsignal(signal.SIGTERM)
    args = _smoke_args(tmp_path, output_dir, cache_dir)
    configured = tmp_path / "label_space"
    (configured / "acknowledged.csv").write_text("dataset,native_name,kind,reason\n")
    (configured / ".DS_Store").write_text("leftover")
    training.run(args)
    assert signal.getsignal(signal.SIGTERM) is sigterm_before

    run_dir = output_dir / "model_checkpoints" / SMOKE_RUN
    copied = {p.name: p.read_bytes() for p in (run_dir / "label_space").iterdir()}
    assert sorted(copied) == [
        "acknowledged.csv",
        "classes.csv",
        "concepts.csv",
        "label_space.meta.yaml",
        "source_labels.csv",
    ]
    assert all(data == (configured / name).read_bytes() for name, data in copied.items())
    assert (run_dir / "model_step1").is_file()
    assert not (run_dir / "id2label.json").exists()
    assert not (run_dir / "concept_id2name.json").exists()
    sidecar = json.loads((run_dir / "model.json").read_text())
    assert sidecar["id2label"]["0"] == "ignore"
    assert "sand" in sidecar["id2label"].values()
    assert sidecar["concept_id2name"]
    assert sidecar["lineage"]["seed"] == 7
    assert "parent" not in sidecar["lineage"]
    assert sidecar["architecture"]["encoder_config"]["hidden_size"] == 16
    assert sidecar["preprocessing"]["mean"] == list(IMAGENET_MEAN)
    assert sidecar["preprocessing"]["std"] == list(IMAGENET_STD)
    assert sidecar["label_space"] == {
        name: hashlib.sha256(data).hexdigest() for name, data in copied.items()
    }
    resolved = yaml.safe_load((run_dir / "config.yaml").read_text())
    assert resolved["run_name"] == SMOKE_RUN
    assert resolved["model"]["input_size"] == [SMOKE_RESOLUTION, SMOKE_RESOLUTION]
    assert sidecar["preprocessing"]["size_hw"] == resolved["model"]["input_size"]
    assert resolved["training"]["concept_mapping_path"] == "configs/class_to_concepts.csv"
    checkpoint = torch.load(run_dir / "model_step1", weights_only=True)
    assert checkpoint["config"] == resolved
    assert checkpoint["lineage"]["seed"] == 7
    assert checkpoint["config"] == checkpoint["lineage"]["resolved_config"]
    assert not (workdir / "model_checkpoints").exists()
    assert wandb_run.params["cli/num_workers"] == 1
    assert wandb_run.params["cli/output_dir"] == str(output_dir)
    assert wandb_run.params["data/seed"] == 7
    assert wandb_run.exit_code == 0


def _raise_boom(*_: Any, **__: Any) -> None:
    raise RuntimeError("boom")


def _send_sigterm(*_: Any, **__: Any) -> None:
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
    _, wandb_run = _install_smoke_fakes(monkeypatch, tmp_path, mock_dinov3_encoder)
    monkeypatch.setattr(training, "run_epoch", train_step)
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


def _checkpoint_lineage(**overrides: Any) -> dict[str, Any]:
    return {
        "git_sha": "abc123",
        "git_dirty": False,
        "seed": 1,
        "data_version": {"datasets": {"ds": {"version": "v1"}}, "taxonomy": None},
        **overrides,
    }


def _resume_checkpoint(meta) -> dict[str, Any]:
    """A checkpoint with the keys ``write_checkpoint`` writes."""
    return {
        "lineage": _checkpoint_lineage(),
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


def _patch_git_dirty(monkeypatch: pytest.MonkeyPatch, git_dirty: bool | None) -> None:
    """Make ``collect_lineage`` report ``git_dirty`` whatever the checkout looks like."""
    real_collect = training.collect_lineage

    def _collect(**kwargs: Any) -> dict[str, Any]:
        return {**real_collect(**kwargs), "git_dirty": git_dirty}

    monkeypatch.setattr(training, "collect_lineage", _collect)


@_FORK_ONLY
def test_resumed_runs_propagate_a_dirty_ancestor_through_a_shallow_parent(
    monkeypatch, tmp_path, mock_dinov3_encoder
):
    _install_smoke_fakes(monkeypatch, tmp_path, mock_dinov3_encoder)
    output_dir = tmp_path / "out"
    cache_dir = tmp_path / "cache"
    run_dir = output_dir / "model_checkpoints" / SMOKE_RUN

    _patch_git_dirty(monkeypatch, True)
    training.run(_smoke_args(tmp_path, output_dir, cache_dir))
    first_checkpoint = run_dir / "model_step1"
    first_lineage = torch.load(first_checkpoint, map_location="cpu", weights_only=True)["lineage"]

    _patch_git_dirty(monkeypatch, False)
    training.run(
        _smoke_args(tmp_path, output_dir, cache_dir, epochs=2, checkpoint=first_checkpoint)
    )

    resumed = torch.load(run_dir / "model_step2", map_location="cpu", weights_only=True)["lineage"]
    sidecar = json.loads((run_dir / "model.json").read_text())["lineage"]
    for lineage in (resumed, sidecar):
        assert lineage["git_dirty"] is True
        assert lineage["parent"] == {
            "git_sha": first_lineage["git_sha"],
            "git_dirty": first_lineage["git_dirty"],
            "data_version": first_lineage["data_version"],
        }

    # A clean third run still inherits the dirty verdict from its parent's own inherited flag.
    training.run(
        _smoke_args(tmp_path, output_dir, cache_dir, epochs=3, checkpoint=run_dir / "model_step2")
    )
    third = torch.load(run_dir / "model_step3", map_location="cpu", weights_only=True)["lineage"]
    assert third["git_dirty"] is True
    assert third["parent"] == {
        "git_sha": resumed["git_sha"],
        "git_dirty": True,
        "data_version": resumed["data_version"],
    }


@pytest.mark.parametrize("flag", ["--checkpoint", "--init-weights"])
def test_a_checkpoint_without_lineage_is_refused_before_training(
    flag, run_scenario, marked_checkpoint
):
    saved = torch.load(marked_checkpoint.path, weights_only=True)
    del saved["lineage"]
    source_path = run_scenario.scratch_dir() / "no_lineage.pt"
    torch.save(saved, source_path)

    with pytest.raises(RuntimeError, match="lineage"):
        run_scenario(cli=[flag, str(source_path)])


def test_init_weights_run_from_a_clean_checkpoint_records_a_shallow_parent(
    run_scenario, marked_checkpoint
):
    saved = torch.load(marked_checkpoint.path, weights_only=True)
    saved["lineage"] = _checkpoint_lineage(git_sha="abc123")
    source_path = run_scenario.scratch_dir() / "with_lineage.pt"
    torch.save(saved, source_path)
    with pytest.MonkeyPatch.context() as monkeypatch:
        _patch_git_dirty(monkeypatch, False)
        result = run_scenario(cli=["--init-weights", str(source_path)])

    lineage = json.loads((result.run_dir / "model.json").read_text())["lineage"]
    assert lineage["git_dirty"] is False
    assert lineage["parent"] == {
        "git_sha": "abc123",
        "git_dirty": False,
        "data_version": saved["lineage"]["data_version"],
    }


def _checkpoint_with_changed_data(first_checkpoint: Path, path: Path) -> dict[str, Any]:
    """Save a copy of ``first_checkpoint`` that pins another version of its dataset."""
    saved = torch.load(first_checkpoint, map_location="cpu", weights_only=True)
    saved["lineage"]["data_version"]["datasets"]["coralscapes_v2"]["version"] = "other-version"
    torch.save(saved, path)
    return saved


@_FORK_ONLY
def test_resume_with_changed_data_pins_is_refused_before_building_datasets(
    monkeypatch, tmp_path, mock_dinov3_encoder
):
    _install_smoke_fakes(monkeypatch, tmp_path, mock_dinov3_encoder)
    output_dir, cache_dir = tmp_path / "out", tmp_path / "cache"
    training.run(_smoke_args(tmp_path, output_dir, cache_dir))
    changed = tmp_path / "changed.pt"
    _checkpoint_with_changed_data(
        output_dir / "model_checkpoints" / SMOKE_RUN / "model_step1", changed
    )

    def _unreachable(*args: Any, **kwargs: Any) -> Any:
        raise AssertionError("run() built datasets before checking the data pins")

    monkeypatch.setattr(training, "build_datasets", _unreachable)
    with pytest.raises(ValueError, match="coralscapes_v2"):
        training.run(_smoke_args(tmp_path, output_dir, cache_dir, epochs=2, checkpoint=changed))


@_FORK_ONLY
def test_init_weights_from_a_checkpoint_with_other_data_trains_and_records_its_pins(
    monkeypatch, tmp_path, mock_dinov3_encoder
):
    _install_smoke_fakes(monkeypatch, tmp_path, mock_dinov3_encoder)
    output_dir, cache_dir = tmp_path / "out", tmp_path / "cache"
    training.run(_smoke_args(tmp_path, output_dir, cache_dir))
    changed = tmp_path / "changed.pt"
    saved = _checkpoint_with_changed_data(
        output_dir / "model_checkpoints" / SMOKE_RUN / "model_step1", changed
    )

    training.run(
        _smoke_args(tmp_path, tmp_path / "out2", cache_dir, cli=["--init-weights", str(changed)])
    )

    sidecar = json.loads(
        (tmp_path / "out2" / "model_checkpoints" / SMOKE_RUN / "model.json").read_text()
    )
    assert sidecar["lineage"]["parent"]["data_version"] == saved["lineage"]["data_version"]


def _lineage(taxonomy: dict[str, str] | None) -> dict[str, Any]:
    pin = {"version": "v1", "manifest_sha256": "aa11", "resize": None}
    return {
        "git_sha": "abc123",
        "git_dirty": False,
        "uv_lock_sha256": "deadbeef",
        "data_version": {"datasets": {"coralnet": pin}, "taxonomy": taxonomy},
    }


@pytest.mark.parametrize(
    ("taxonomy", "taxonomy_params"),
    [
        (
            {"version": "t1", "manifest_sha256": "tt"},
            {"lineage/taxonomy/version": "t1", "lineage/taxonomy/manifest_sha256": "tt"},
        ),
        (None, {}),
    ],
    ids=["pinned-taxonomy", "absent-taxonomy"],
)
def test_lineage_params_flatten_the_dataset_and_taxonomy_pins(taxonomy, taxonomy_params):
    assert _lineage_params(_lineage(taxonomy)) == {
        "lineage/git_sha": "abc123",
        "lineage/git_dirty": False,
        "lineage/uv_lock_sha256": "deadbeef",
        "lineage/datasets/coralnet/version": "v1",
        "lineage/datasets/coralnet/manifest_sha256": "aa11",
        **taxonomy_params,
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


def test_distributed_shuffled_loader_matches_single_process_over_two_passes(lockstep_ranks):
    reference_loader = DataLoader(
        _index_dataset(),
        shuffle=True,
        batch_size=_LOADER_BATCH,
        drop_last=True,
        generator=torch.Generator().manual_seed(_LOADER_SEED),
    )
    expected = [[batch[0].tolist() for batch in reference_loader] for _ in range(2)]

    def two_passes(rank: int) -> list[list[list[int]]]:
        # Rank 1 starts from a different seed; the broadcast must align it with rank 0.
        generator = torch.Generator().manual_seed(_LOADER_SEED + rank)
        loader = make_shuffled_loader(
            DistContext(rank=rank, world_size=2, local_rank=rank),
            _index_dataset(),
            generator,
            batch_size=_LOADER_BATCH,
            drop_last=True,
            num_workers=0,
        )
        return [[batch[0].tolist() for batch in loader] for _ in range(2)]

    rank0, rank1 = lockstep_ranks(2).run(two_passes)
    for pass_index in range(2):
        interleaved = [
            shard0 + shard1
            for shard0, shard1 in zip(rank0[pass_index], rank1[pass_index], strict=True)
        ]
        assert interleaved == expected[pass_index]


# ---------- run() with a recording stand-in for run_epoch ----------
# The loaders are built but never iterated, so none of these tests starts a DataLoader worker.


class _RecordingTrainModel:
    """Stands in for ``run_epoch``: records each call and fires ``on_optimizer_step`` at ``steps``."""

    def __init__(self, steps: Sequence[int] = (), validation_metrics: dict | None = None) -> None:
        self.steps = steps
        self.validation_metrics = validation_metrics or {}
        self.calls: list[dict[str, Any]] = []

    def __call__(
        self,
        meta_model: Any,
        evaluator: Any,
        train_loader: Any,
        val_loader: Any,
        logger: Any,
        *,
        epoch: int,
        train_log_interval: int,
        on_optimizer_step: Any,
    ) -> dict[str, Any]:
        self.calls.append({"meta_model": meta_model, "epoch": epoch})
        for step in self.steps:
            meta_model.global_step = step
            on_optimizer_step(meta_model)
        return {"train_metrics": {}, "validation_metrics": self.validation_metrics}


@dataclasses.dataclass
class _RunResult:
    """What one ``run()`` left behind: the recorder, seeds, tracker params, files and exit."""

    recorder: _RecordingTrainModel
    seeds: list[int]
    wandb_run: _FakeWandbRun
    run_dir: Path
    system_exit: SystemExit | None = None
    save_atomic: dict[str, bool] = dataclasses.field(default_factory=dict)

    def resolved_config(self) -> dict[str, Any]:
        return yaml.safe_load((self.run_dir / "config.yaml").read_text())


class _ScenarioRunner:
    """Runs ``run()`` once per call in its own directory, with the smoke fakes scoped to the call."""

    def __init__(self, tmp_path_factory: pytest.TempPathFactory) -> None:
        self._tmp_path_factory = tmp_path_factory

    def scratch_dir(self) -> Path:
        return self._tmp_path_factory.mktemp("scratch")

    def __call__(
        self,
        recorder: _RecordingTrainModel | None = None,
        *,
        catch_exit: bool = False,
        spy_saves: bool = False,
        **args_kwargs: Any,
    ) -> _RunResult:
        tmp_path = self._tmp_path_factory.mktemp("run")
        output_dir = tmp_path / "out"
        result = _RunResult(
            recorder=recorder or _RecordingTrainModel(),
            seeds=[],
            wandb_run=_FakeWandbRun(),
            run_dir=output_dir / "model_checkpoints" / SMOKE_RUN,
        )
        with pytest.MonkeyPatch.context() as monkeypatch:
            _, result.wandb_run = _install_smoke_fakes(
                monkeypatch, tmp_path, dinov3_encoder_mocker(monkeypatch)
            )
            self._record_seeds(monkeypatch, result.seeds)
            if spy_saves:
                self._spy_on_saves(monkeypatch, result.save_atomic)
            monkeypatch.setattr(training, "run_epoch", result.recorder)
            args = _smoke_args(tmp_path, output_dir, tmp_path / "cache", **args_kwargs)
            try:
                training.run(args)
            except SystemExit as exit_:
                if not catch_exit:
                    raise
                result.system_exit = exit_
        return result

    @staticmethod
    def _record_seeds(monkeypatch: pytest.MonkeyPatch, seeds: list[int]) -> None:
        real_seed_run = training.seed_run

        def _record_seed(seed: int, train_generator: Any, val_generator: Any) -> None:
            seeds.append(seed)
            real_seed_run(seed, train_generator, val_generator)

        monkeypatch.setattr(training, "seed_run", _record_seed)

    @staticmethod
    def _spy_on_saves(monkeypatch: pytest.MonkeyPatch, atomic_by_name: dict[str, bool]) -> None:
        real_write = training.write_checkpoint

        def _record(path, meta_model, **kwargs):
            atomic_by_name[Path(path).name] = kwargs.get("atomic", False)
            real_write(path, meta_model, **kwargs)

        monkeypatch.setattr(training, "write_checkpoint", _record)


@pytest.fixture(scope="module")
def run_scenario(tmp_path_factory) -> _ScenarioRunner:
    return _ScenarioRunner(tmp_path_factory)


@pytest.fixture(scope="module")
def fresh_run(run_scenario) -> _RunResult:
    """The default one-epoch run, shared by the tests that need a built model or its settings."""
    return run_scenario()


def _save_resume_checkpoint(meta, path: Path, **overrides: Any) -> Path:
    """A resumable checkpoint whose lineage pins the data the smoke runs use."""
    smoke_data = data_version(DataSection.model_validate(_write_smoke_versions(path.parent)))
    lineage = _checkpoint_lineage(data_version=smoke_data)
    torch.save({**_resume_checkpoint(meta), "lineage": lineage, **overrides}, path)
    return path


# ---------- --init-weights and --checkpoint ----------


def _weights_only_checkpoint(path: Path) -> Path:
    torch.save(
        {"model_state_dict": torch.nn.Linear(2, 2).state_dict(), "lineage": _checkpoint_lineage()},
        path,
    )
    return path


def test_init_weights_reads_a_file_that_holds_only_model_weights(tmp_path):
    path = _weights_only_checkpoint(tmp_path / "weights.pt")
    args = SimpleNamespace(init_weights=str(path), checkpoint=None)
    assert set(training._read_run_checkpoint(args)) == {"model_state_dict", "lineage"}  # type: ignore[arg-type]


def test_resume_checkpoint_must_hold_the_optimizer_and_progress_keys(tmp_path):
    path = _weights_only_checkpoint(tmp_path / "weights.pt")
    args = SimpleNamespace(init_weights=None, checkpoint=str(path))
    with pytest.raises(RuntimeError, match=r"weights\.pt.*optimizer_state_dict"):
        training._read_run_checkpoint(args)  # type: ignore[arg-type]


@dataclasses.dataclass
class _MarkedCheckpoint:
    path: Path
    tensor_name: str
    marked_tensor: torch.Tensor


@pytest.fixture(scope="module")
def marked_checkpoint(fresh_run, run_scenario) -> _MarkedCheckpoint:
    """The fresh run's weights, +1 on one tensor, saved with a foreign optimizer and progress."""
    meta = fresh_run.recorder.calls[0]["meta_model"]
    checkpoint = _resume_checkpoint(meta)
    checkpoint["model_state_dict"] = {
        key: value.detach().clone() for key, value in meta.model.state_dict().items()
    }
    name = next(k for k, v in checkpoint["model_state_dict"].items() if v.is_floating_point())
    checkpoint["model_state_dict"][name] += 1.0
    checkpoint["optimizer_state_dict"]["param_groups"][0]["lr"] = 123.0
    checkpoint["global_step"] = 99
    path = run_scenario.scratch_dir() / "init.pt"
    torch.save(checkpoint, path)
    return _MarkedCheckpoint(path, name, checkpoint["model_state_dict"][name])


@pytest.fixture(scope="module")
def init_weights_run(run_scenario, marked_checkpoint) -> _RunResult:
    return run_scenario(cli=["--init-weights", str(marked_checkpoint.path)])


def test_init_weights_run_loads_the_weights_into_a_fresh_optimizer_at_epoch_zero(
    fresh_run, init_weights_run, marked_checkpoint
):
    configured_lrs = [
        g["lr"] for g in fresh_run.recorder.calls[0]["meta_model"].optimizer.param_groups
    ]

    call = init_weights_run.recorder.calls[0]
    meta = call["meta_model"]
    assert call["epoch"] == 0
    assert (meta.global_step, meta.resuming_within_epoch) == (0, False)
    assert [g["lr"] for g in meta.optimizer.param_groups] == configured_lrs
    torch.testing.assert_close(
        meta.model.state_dict()[marked_checkpoint.tensor_name], marked_checkpoint.marked_tensor
    )


def test_init_weights_run_reseeds_to_the_run_seed_not_the_checkpoint_progress(init_weights_run):
    # Checkpoint global_step is 99 and epoch 2; the new run starts from the plain run seed.
    assert init_weights_run.seeds == [7, 7]


@pytest.mark.parametrize(
    ("epoch_complete", "global_step", "epochs", "expected_resume_seed"),
    [
        pytest.param(False, 5, 2, 7 + 5, id="mid-epoch-adds-the-global-step"),
        pytest.param(True, 5, 3, 7 + 1, id="epoch-boundary-adds-the-start-epoch"),
    ],
)
def test_resumed_run_reseeds_with_the_run_seed_plus_its_progress(
    fresh_run, run_scenario, epoch_complete, global_step, epochs, expected_resume_seed
):
    checkpoint = _save_resume_checkpoint(
        fresh_run.recorder.calls[0]["meta_model"],
        run_scenario.scratch_dir() / "resume.pt",
        epoch=0,
        epoch_complete=epoch_complete,
        global_step=global_step,
    )
    resumed = run_scenario(epochs=epochs, checkpoint=checkpoint)
    assert resumed.seeds == [7, expected_resume_seed]
    assert resumed.recorder.calls[0]["epoch"] == (1 if epoch_complete else 0)


def test_resuming_a_checkpoint_that_covers_every_epoch_exits_with_status_zero(
    fresh_run, run_scenario
):
    checkpoint = _save_resume_checkpoint(
        fresh_run.recorder.calls[0]["meta_model"],
        run_scenario.scratch_dir() / "done.pt",
        epoch=0,
        epoch_complete=True,
    )
    resumed = run_scenario(epochs=1, checkpoint=checkpoint, catch_exit=True)
    assert resumed.system_exit is not None
    assert resumed.system_exit.code == 0
    assert resumed.recorder.calls == []


def test_resuming_the_last_unfinished_epoch_still_trains_it(fresh_run, run_scenario):
    checkpoint = _save_resume_checkpoint(
        fresh_run.recorder.calls[0]["meta_model"],
        run_scenario.scratch_dir() / "mid.pt",
        epoch=0,
        epoch_complete=False,
    )
    resumed = run_scenario(epochs=1, checkpoint=checkpoint)
    assert [c["epoch"] for c in resumed.recorder.calls] == [0]


# ---------- config overrides ----------


def test_a_zero_kl_weight_keeps_the_taxonomic_section_and_logs_the_zero(run_scenario):
    result = run_scenario(training_overrides={"taxonomic_loss": {"kl_weight": 0}})
    assert result.resolved_config()["training"]["taxonomic_loss"]["kl_weight"] == 0
    assert result.wandb_run.params["loss/kl_weight"] == 0


def test_a_positive_kl_weight_is_kept_and_logged(run_scenario):
    result = run_scenario(training_overrides={"taxonomic_loss": {"kl_weight": 0.25}})
    assert result.resolved_config()["training"]["taxonomic_loss"]["kl_weight"] == 0.25
    assert result.wandb_run.params["loss/kl_weight"] == 0.25


def test_the_grad_accum_flag_overrides_the_config_value(run_scenario):
    result = run_scenario(
        cli=["--grad-accum-steps", "4"], training_overrides={"grad_accum_steps": 1}
    )
    assert result.resolved_config()["training"]["grad_accum_steps"] == 4
    assert result.wandb_run.params["train/grad_accum_steps"] == 4


# ---------- checkpoints written by run() ----------


def _checkpoint_flags(path: Path) -> tuple[int, bool]:
    saved = torch.load(path, weights_only=True)
    return saved["global_step"], saved["epoch_complete"]


def test_a_step_checkpoint_marks_its_epoch_complete_only_at_the_last_update(run_scenario):
    result = run_scenario(
        _RecordingTrainModel(steps=[3, 4]),
        training_overrides={"checkpoint_every_steps": 1, "iterations_per_train_epoch": 4},
    )
    assert sorted(p.name for p in result.run_dir.glob("model_step*")) == [
        "model_step3",
        "model_step4",
    ]
    assert _checkpoint_flags(result.run_dir / "model_step3") == (3, False)
    assert _checkpoint_flags(result.run_dir / "model_step4") == (4, True)


@pytest.fixture(scope="module")
def off_cadence_run(run_scenario) -> _RunResult:
    """Three updates with a checkpoint cadence of four: no interval checkpoint is due."""
    return run_scenario(
        _RecordingTrainModel(steps=[1, 2, 3]),
        training_overrides={"checkpoint_every_steps": 4, "iterations_per_train_epoch": 3},
    )


def test_a_run_always_ends_with_a_final_checkpoint_of_its_last_step(off_cadence_run):
    assert list(off_cadence_run.run_dir.glob("model_step*")) == []
    assert _checkpoint_flags(off_cadence_run.run_dir / "model_final") == (3, True)


def test_resuming_the_final_checkpoint_exits_with_status_zero_without_training(
    run_scenario, off_cadence_run
):
    resumed = run_scenario(checkpoint=off_cadence_run.run_dir / "model_final", catch_exit=True)
    assert resumed.system_exit is not None
    assert resumed.system_exit.code == 0
    assert resumed.recorder.calls == []


def test_steps_off_the_checkpoint_cadence_write_nothing(run_scenario):
    result = run_scenario(
        _RecordingTrainModel(steps=[1, 2, 3]),
        training_overrides={"checkpoint_every_steps": 4, "iterations_per_train_epoch": 100},
        cli=["--preempt-checkpoint-every-steps", "5"],
    )
    assert list(result.run_dir.glob("model_step*")) == []
    assert list(result.run_dir.glob("model_preempt*")) == []


@pytest.fixture(scope="module")
def step_and_preempt_run(run_scenario) -> _RunResult:
    """Step checkpoints every 2 and preempt checkpoints every 3, across updates 2, 3 and 6."""
    return run_scenario(
        _RecordingTrainModel(steps=[2, 3, 6]),
        training_overrides={"checkpoint_every_steps": 2, "iterations_per_train_epoch": 100},
        cli=["--preempt-checkpoint-every-steps", "3"],
        spy_saves=True,
    )


def test_only_preempt_and_final_checkpoints_are_written_atomically(step_and_preempt_run):
    assert step_and_preempt_run.save_atomic == {
        "model_step2": False,
        "model_preempt3": True,
        "model_step6": False,
        "model_preempt6": True,
        "model_final": True,
    }


def test_preempt_checkpoints_roll_and_leave_step_checkpoints_alone(step_and_preempt_run):
    run_dir = step_and_preempt_run.run_dir
    assert sorted(p.name for p in run_dir.glob("model_step*")) == [
        "model_step2",
        "model_step6",
    ]
    # model_preempt3 was pruned once model_preempt6 landed; no .partial is left behind.
    assert sorted(p.name for p in run_dir.glob("model_preempt*")) == ["model_preempt6"]
    assert _checkpoint_flags(run_dir / "model_preempt6") == (6, False)


def test_preempt_checkpoints_are_written_without_interval_checkpoints(run_scenario):
    result = run_scenario(
        _RecordingTrainModel(steps=[3]),
        training_overrides={"checkpoint_every_steps": 1000, "iterations_per_train_epoch": 100},
        cli=["--preempt-checkpoint-every-steps", "3"],
    )
    assert [p.name for p in result.run_dir.glob("model_preempt*")] == ["model_preempt3"]
    assert list(result.run_dir.glob("model_step*")) == []


def test_the_preempt_checkpoint_of_the_last_update_marks_the_epoch_complete(run_scenario):
    result = run_scenario(
        _RecordingTrainModel(steps=[4]),
        training_overrides={"checkpoint_every_steps": 1000, "iterations_per_train_epoch": 4},
        cli=["--preempt-checkpoint-every-steps", "2"],
    )
    assert _checkpoint_flags(result.run_dir / "model_preempt4") == (4, True)
