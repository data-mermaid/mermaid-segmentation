"""Tests for mermaidseg.checkpoint."""

from __future__ import annotations

import os
from typing import Any

import numpy as np
import pytest
import torch

from mermaidseg.checkpoint import checkpoint_dir, prune_rolling_checkpoints, write_checkpoint
from tests.conftest import FakeMetaModel

LINEAGE = {"git_sha": "abc", "git_dirty": False}


class _MoveRecordingLinear(torch.nn.Linear):
    """A ``Linear`` that records every ``.to()``, ``.cpu()`` or ``.cuda()`` applied to it."""

    moves: list[str]

    def _apply(self, fn, recurse=True):  # type: ignore[no-untyped-def]
        self.moves.append("moved")
        return super()._apply(fn, recurse)


def _meta_with_move_recording_model(run_name: str) -> FakeMetaModel:
    meta = FakeMetaModel(run_name=run_name)
    model = _MoveRecordingLinear(4, meta.num_classes)
    model.moves = []
    meta.model = model
    meta.optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
    meta.scheduler = torch.optim.lr_scheduler.StepLR(meta.optimizer, step_size=1)
    return meta


def _spy_torch_save(monkeypatch: pytest.MonkeyPatch) -> list[tuple[dict, str]]:
    """Record the (object, target path) of every ``torch.save`` while still writing the file."""
    calls: list[tuple[dict, str]] = []
    real_save = torch.save

    def _save(obj: dict, f: Any, *args: Any, **kwargs: Any) -> None:
        calls.append((obj, str(f)))
        real_save(obj, f, *args, **kwargs)

    monkeypatch.setattr(torch, "save", _save)
    return calls


def _spy_os_replace(monkeypatch: pytest.MonkeyPatch) -> list[tuple[str, str]]:
    """Record every ``os.replace`` (source, destination) while still renaming."""
    calls: list[tuple[str, str]] = []
    real_replace = os.replace

    def _replace(src: Any, dst: Any) -> None:
        calls.append((str(src), str(dst)))
        real_replace(src, dst)

    monkeypatch.setattr(os, "replace", _replace)
    return calls


def _write(path, meta, config, **overrides: Any) -> None:
    kwargs: dict[str, Any] = {
        "config": config,
        "lineage": LINEAGE,
        "epoch": 1,
        "epoch_complete": True,
        "metrics": {"loss": 0.1},
    }
    kwargs.update(overrides)
    write_checkpoint(path, meta, **kwargs)


def test_checkpoint_dir_is_the_run_folder_under_model_checkpoints(tmp_path):
    assert checkpoint_dir(tmp_path, "run-a") == tmp_path / "model_checkpoints" / "run-a"


class TestWriteCheckpoint:
    def test_step_checkpoint_carries_resume_state(self, tmp_path, make_config):
        meta = FakeMetaModel(run_name="step-ckpt")
        meta.global_step = 25000
        meta._warmup_iters_completed = 4000
        path = tmp_path / "step-ckpt" / "model_step25000"
        _write(path, meta, make_config(), epoch=0, epoch_complete=False)
        assert path.is_file()
        ckpt = torch.load(path, weights_only=True)
        assert ckpt["global_step"] == 25000
        assert ckpt["warmup_iters_completed"] == 4000
        assert ckpt["epoch_complete"] is False
        assert ckpt["epoch"] == 0

    def test_checkpoint_always_carries_the_lineage(self, tmp_path, make_config):
        meta = FakeMetaModel()
        path = tmp_path / "model_step1"
        _write(path, meta, make_config())
        assert torch.load(path, weights_only=True)["lineage"] == LINEAGE

    def test_checkpoint_metrics_and_config_load_as_plain_python(self, tmp_path, make_config):
        path = tmp_path / "model_step1"
        metrics = {
            "iou": np.array([0.5, 0.25]),
            "acc": np.float32(0.75),
            "loss": torch.tensor(0.125),
            "step": 3,
        }
        _write(path, FakeMetaModel(), make_config(), metrics=metrics)
        ckpt = torch.load(path, weights_only=True)
        assert ckpt["metrics"] == {"iou": [0.5, 0.25], "acc": 0.75, "loss": 0.125, "step": 3}
        assert type(ckpt["config"]) is dict
        assert type(ckpt["config"]["training"]) is dict
        assert ckpt["config"]["training"]["optimizer"] == {"type": "SGD", "lr": 0.01}

    def test_save_leaves_parameters_in_place(self, tmp_path, make_config, monkeypatch):
        meta = _meta_with_move_recording_model("inplace-ckpt")
        weight = meta.model.weight
        ptr = weight.data_ptr()
        values = weight.detach().clone()
        meta.model.moves.clear()
        saved = _spy_torch_save(monkeypatch)
        path = tmp_path / "model_step1"
        _write(path, meta, make_config())

        assert meta.model.moves == []
        assert meta.model.weight is weight
        assert meta.model.weight.data_ptr() == ptr
        checkpoint = saved[0][0]
        for key, live in meta.model.state_dict().items():
            assert checkpoint["model_state_dict"][key].data_ptr() != live.data_ptr()
        saved_weight = torch.load(path, weights_only=True)["model_state_dict"]["weight"]
        assert saved_weight.device.type == "cpu"
        torch.testing.assert_close(saved_weight, values)

    def test_interval_checkpoint_is_written_directly_to_its_final_path(
        self, tmp_path, make_config, monkeypatch
    ):
        saved = _spy_torch_save(monkeypatch)
        replaced = _spy_os_replace(monkeypatch)
        final = tmp_path / "direct-ckpt" / "model_step5"
        _write(final, FakeMetaModel(), make_config(), epoch=0, metrics={})
        assert [target for _, target in saved] == [str(final)]
        assert replaced == []

    def test_atomic_checkpoint_renames_a_partial_into_place(
        self, tmp_path, make_config, monkeypatch
    ):
        meta = FakeMetaModel(run_name="atomic-ckpt")
        meta.global_step = 1000
        saved = _spy_torch_save(monkeypatch)
        replaced = _spy_os_replace(monkeypatch)
        directory = tmp_path / "atomic-ckpt"
        final = directory / "model_preempt1000"
        partial = directory / "model_preempt1000.partial"
        _write(final, meta, make_config(), epoch=0, epoch_complete=False, metrics={}, atomic=True)

        assert [target for _, target in saved] == [str(partial)]
        assert replaced == [(str(partial), str(final))]
        assert final.is_file()
        assert not partial.exists()
        ckpt = torch.load(final, weights_only=True)
        assert ckpt["global_step"] == 1000
        assert ckpt["epoch_complete"] is False
        assert ckpt["optimizer_state_dict"]
        assert ckpt["scheduler_state_dict"]
        assert ckpt["model_state_dict"]

    def test_an_existing_checkpoint_is_overwritten_with_a_warning(
        self, tmp_path, make_config, caplog
    ):
        path = tmp_path / "model_step1"
        _write(path, FakeMetaModel(), make_config(), epoch=1)
        with caplog.at_level("WARNING"):
            _write(path, FakeMetaModel(), make_config(), epoch=2)
        assert torch.load(path, weights_only=True)["epoch"] == 2
        assert "already exists" in caplog.text


class TestPruneRollingCheckpoints:
    def test_only_other_preempt_files_are_removed(self, tmp_path):
        keep = tmp_path / "model_preempt1000"
        keep.write_bytes(b"keep")
        (tmp_path / "model_preempt500").write_bytes(b"stale")
        (tmp_path / "model_preempt500.partial").write_bytes(b"partial")
        (tmp_path / "model_step25000").write_bytes(b"keep")
        (tmp_path / "model_epoch0").write_bytes(b"keep")

        prune_rolling_checkpoints(tmp_path, keep_name="model_preempt1000")

        assert sorted(p.name for p in tmp_path.iterdir()) == [
            "model_epoch0",
            "model_preempt1000",
            "model_step25000",
        ]

    def test_a_missing_directory_is_ignored(self, tmp_path):
        prune_rolling_checkpoints(tmp_path / "absent", keep_name="model_preempt1")
