"""Run settings of the trainer: dataset repetition, and a one-step CPU run."""

from __future__ import annotations

import argparse
import functools
import json
import multiprocessing
import sys
import types
from pathlib import Path
from types import SimpleNamespace
from typing import Any
from unittest.mock import MagicMock

import pytest
import torch
import yaml
from torch.utils.data import Dataset

import mermaidseg.logger
import mermaidseg.model.models
from mermaidseg import training
from mermaidseg.datasets.factory import DATASET_CLASSES, build_datasets
from mermaidseg.datasets.local_cache import create_cache_stats
from mermaidseg.io import ConfigDict, load_config
from mermaidseg.run_setup import resolve_repo_path
from mermaidseg.training import _assemble_train_datasets
from scripts.train import build_parser


def _datasets() -> dict[tuple[str, str], str]:
    return {
        ("a", "train"): "a",
        ("coralscapes_v2", "train"): "cs",
        ("b", "train"): "b",
        ("b", "val"): "bv",
    }


class _Stub:
    def __init__(self, **kwargs):
        pass

    def __len__(self) -> int:
        return 1


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


def test_train_datasets_without_repeat_are_not_repeated():
    data_cfg = ConfigDict(
        {"a": {"train": {}}, "coralscapes_v2": {"train": {"split": "train"}}, "b": {"train": {}}}
    )
    assert _assemble_train_datasets(_datasets(), data_cfg) == ["a", "cs", "b"]


@pytest.mark.parametrize("repeat", [0, -1, 2.5, True])
def test_train_datasets_reject_an_invalid_repeat_naming_the_dataset(repeat):
    data_cfg = ConfigDict(
        {"a": {"train": {}}, "coralscapes_v2": {"train": {"repeat": repeat}}, "b": {"train": {}}}
    )
    with pytest.raises(ValueError, match="coralscapes_v2"):
        _assemble_train_datasets(_datasets(), data_cfg)


def test_trainval_config_trains_on_six_coralscapes_v2_copies():
    data_cfg = load_config(resolve_repo_path("configs/data_config_trainval.yaml"))["data"]
    classes = dict.fromkeys(DATASET_CLASSES, _Stub)
    built = build_datasets(data_cfg, dataset_classes=classes, splits=["train"], verbose=False)
    train = _assemble_train_datasets(built, data_cfg)
    assert sum(ds is built[("coralscapes_v2", "train")] for ds in train) == 6


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
        self.finished = False

    def log(self, log_dict: dict, step: int) -> None:
        pass

    def log_artifact(self, artifact: Any) -> None:
        pass

    def finish(self) -> None:
        self.finished = True


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
    data = {"coralscapes_v2": {"train": {"split": "train"}, "val": {"split": "val"}}}
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
    monkeypatch.setattr(mermaidseg.logger, "wandb", fake_wandb)

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
    monkeypatch.setattr(mermaidseg.logger, "wandb", fake_wandb)
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
    training.run(_smoke_args(tmp_path, output_dir, cache_dir))

    run_dir = output_dir / "model_checkpoints" / SMOKE_RUN
    assert (run_dir / "model_step1").is_file()
    assert (run_dir / "id2label.json").is_file()
    assert (run_dir / "concept_id2name.json").is_file()
    assert not (workdir / "model_checkpoints").exists()
    assert cache_roots == [str(cache_dir)]
    assert wandb_run.params["cli/num_workers"] == 1
    assert wandb_run.params["cli/output_dir"] == str(output_dir)
    assert wandb_run.params["data/seed"] == 7
    mapping_path = Path(wandb_run.params["training_concept_mapping_path"])
    assert mapping_path.is_absolute() and mapping_path.is_file()
    assert wandb_run.finished


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
