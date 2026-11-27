"""Tests for mermaidseg.logger."""

from __future__ import annotations

import json
import logging
import os
import subprocess
import sys
import types
from pathlib import Path
from types import SimpleNamespace
from typing import Any
from unittest.mock import MagicMock

import numpy as np
import pytest
import torch

import mermaidseg.logger
from mermaidseg.io import ConfigDict
from mermaidseg.logger import (
    Logger,
    MlflowTracker,
    WandbTracker,
    _mlflow_key,
    _mlflow_run_url,
    _safe_wandb_artifact_name,
    make_tracker,
    prune_rolling_checkpoints,
)
from tests.conftest import FakeMetaModel

REPO_ROOT = Path(__file__).resolve().parents[2]


class FakeTracker:
    """Records every ``Tracker`` call; built by ``make_tracker`` through ``TRACKER_CLASSES``."""

    run_id = "fake-id"
    run_url = "https://tracker.invalid/fake"

    def __init__(self, **kwargs: Any) -> None:
        self.kwargs = kwargs
        self.params: list[dict[str, Any]] = []
        self.metrics: list[tuple[dict[str, float], int]] = []
        self.dicts: list[tuple[str, dict[str, Any]]] = []
        self.ended: list[bool] = []

    def log_params(self, params: dict[str, Any]) -> None:
        self.params.append(params)

    def log_metrics(self, metrics: dict[str, float], step: int) -> None:
        self.metrics.append((metrics, step))

    def log_dict(self, data: dict[str, Any], artifact_path: str) -> None:
        self.dicts.append((artifact_path, data))

    def end_run(self, failed: bool) -> None:
        self.ended.append(failed)


@pytest.fixture()
def fake_tracker(monkeypatch: pytest.MonkeyPatch) -> list[FakeTracker]:
    """Route ``make_tracker``'s ``wandb`` entry to ``FakeTracker``; return the built trackers."""
    built: list[FakeTracker] = []

    def _build(logger_cfg: Any, run_name: str, config: Any) -> FakeTracker:
        tracker = FakeTracker(
            project=logger_cfg.wandb_project,
            entity=logger_cfg.wandb_entity,
            run_name=run_name,
            config=config,
        )
        built.append(tracker)
        return tracker

    monkeypatch.setitem(mermaidseg.logger.TRACKER_CLASSES, "wandb", _build)
    return built


def _logger(make_config, meta=None, **kwargs):
    kwargs.setdefault("id2concept", {})
    lgr = Logger(config=make_config(), meta_model=meta or FakeMetaModel(), **kwargs)
    tracker = lgr.tracker
    assert isinstance(tracker, FakeTracker)
    return lgr, tracker


# ===================================================================
# WandbTracker (stub ``wandb`` module)
# ===================================================================
class _StubArtifact:
    def __init__(self, name: str, type: str) -> None:
        self.name = name
        self.type = type
        self.files: dict[str, Any] = {}

    def add_file(self, local_path: str, name: str) -> None:
        self.files[name] = json.loads(Path(local_path).read_text())


class _StubRun:
    id = "stub-id"
    url = "https://wandb.invalid/stub"

    def __init__(self) -> None:
        self.config_updates: list[tuple[dict[str, Any], bool]] = []
        self.config = SimpleNamespace(
            update=lambda d, allow_val_change=False: self.config_updates.append(
                (d, allow_val_change)
            )
        )
        self.logged: list[tuple[dict[str, float], int]] = []
        self.artifacts: list[_StubArtifact] = []
        self.exit_code: int | None = None

    def log(self, data: dict[str, float], step: int) -> None:
        self.logged.append((data, step))

    def log_artifact(self, artifact: _StubArtifact) -> None:
        self.artifacts.append(artifact)

    def finish(self, exit_code: int | None = None) -> None:
        self.exit_code = exit_code


@pytest.fixture()
def stub_wandb(monkeypatch: pytest.MonkeyPatch) -> types.ModuleType:
    """A ``wandb`` module whose ``init`` records its kwargs and returns a ``_StubRun``."""
    stub = types.ModuleType("wandb")
    stub.run = _StubRun()  # type: ignore[attr-defined]
    stub.init_kwargs = {}  # type: ignore[attr-defined]

    def _init(**kwargs: Any) -> _StubRun:
        stub.init_kwargs = kwargs  # type: ignore[attr-defined]
        return stub.run  # type: ignore[attr-defined]

    stub.init = _init  # type: ignore[attr-defined]
    stub.api = SimpleNamespace(api_key="stub-key")  # type: ignore[attr-defined]
    stub.Artifact = _StubArtifact  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "wandb", stub)
    monkeypatch.delenv("WANDB_RUN_ID", raising=False)
    return stub


def _wandb_tracker() -> WandbTracker:
    return WandbTracker(project="mermaidseg", entity="team", run_name="run-a", config={"k": 1})


class TestWandbTracker:
    def test_calls_reach_the_wandb_run(self, stub_wandb):
        tracker = _wandb_tracker()
        run = stub_wandb.run

        tracker.log_metrics({"val/iou": 0.5}, step=5)
        tracker.log_params({"lr": 0.1})
        tracker.log_dict({"a": 1}, "metadata/model.json")
        tracker.end_run(failed=False)

        assert stub_wandb.init_kwargs == {
            "project": "mermaidseg",
            "entity": "team",
            "name": "run-a",
            "config": {"k": 1},
            "id": None,
            "resume": None,
        }
        assert run.logged == [({"val/iou": 0.5}, 5)]
        assert run.config_updates == [({"lr": 0.1}, True)]
        assert [(a.name, a.files) for a in run.artifacts] == [("model", {"model.json": {"a": 1}})]
        assert run.exit_code == 0
        assert (tracker.run_id, tracker.run_url) == ("stub-id", "https://wandb.invalid/stub")

    def test_failed_end_run_finishes_with_exit_code_1(self, stub_wandb):
        tracker = _wandb_tracker()
        tracker.end_run(failed=True)
        assert stub_wandb.run.exit_code == 1

    @pytest.mark.parametrize(
        ("value", "run_id"),
        [("r1-cbm_lossw_pix0.5_img0.5", "r1-cbm_lossw_pix0.5_img0.5"), ("  run-1 \n", "run-1")],
        ids=["plain", "stripped"],
    )
    def test_wandb_run_id_env_resumes(self, stub_wandb, monkeypatch, value, run_id):
        monkeypatch.setenv("WANDB_RUN_ID", value)
        _wandb_tracker()
        assert stub_wandb.init_kwargs["id"] == run_id
        assert stub_wandb.init_kwargs["resume"] == "allow"

    @pytest.mark.parametrize("value", ["", "   "])
    def test_a_blank_wandb_run_id_env_starts_a_new_run(self, stub_wandb, monkeypatch, value):
        monkeypatch.setenv("WANDB_RUN_ID", value)
        _wandb_tracker()
        assert stub_wandb.init_kwargs["id"] is None
        assert stub_wandb.init_kwargs["resume"] is None

    @pytest.mark.parametrize(
        ("method", "args", "broken", "warning"),
        [
            ("log_params", ({"lr": 0.1},), ("config", "update"), "Failed to log params to wandb"),
            (
                "log_dict",
                ({"a": 1}, "metadata/model.json"),
                (None, "log_artifact"),
                "Failed to log model to wandb",
            ),
            ("end_run", (False,), (None, "finish"), "Failed to end wandb run"),
        ],
    )
    def test_a_failing_backend_call_is_warned_not_raised(
        self, stub_wandb, monkeypatch, caplog, method, args, broken, warning
    ):
        caplog.set_level(logging.WARNING)
        tracker = _wandb_tracker()
        owner_name, attribute = broken
        owner = getattr(stub_wandb.run, owner_name) if owner_name else stub_wandb.run
        monkeypatch.setattr(owner, attribute, _raise_connection_error)
        getattr(tracker, method)(*args)
        warnings = [r.getMessage() for r in caplog.records if r.levelno == logging.WARNING]
        assert len(warnings) == 1
        assert warning in warnings[0]
        assert "tracker down" in warnings[0]

    def test_missing_api_key_raises_before_starting_a_run(self, stub_wandb):
        stub_wandb.api = SimpleNamespace(api_key=None)
        with pytest.raises(RuntimeError, match="not authenticated"):
            _wandb_tracker()
        assert stub_wandb.init_kwargs == {}

    def test_missing_wandb_names_the_install_command(self, monkeypatch):
        monkeypatch.setitem(sys.modules, "wandb", None)
        with pytest.raises(ImportError, match=r"uv sync --extra wandb"):
            _wandb_tracker()


# ===================================================================
# MlflowTracker (real mlflow on a local sqlite store)
# ===================================================================
@pytest.fixture()
def mlflow_store(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    """Point ``MLFLOW_TRACKING_URI`` at a sqlite store under ``tmp_path``; yield a client."""
    mlflow = pytest.importorskip("mlflow")
    from mlflow.tracking import MlflowClient

    uri = f"sqlite:///{tmp_path / 'mlflow.db'}"
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("MLFLOW_TRACKING_URI", uri)
    yield MlflowClient(tracking_uri=uri)
    mlflow.end_run()


class TestMlflowTracker:
    def test_params_metrics_and_dict_land_in_the_named_run(self, mlflow_store):
        mlflow = pytest.importorskip("mlflow")
        tracker = MlflowTracker(experiment_name="seg-exp", run_name="run-b")
        tracker.log_params({"lr": 0.1, "cli/classes": ["a", "b"], "bad(key)": 1})
        tracker.log_metrics({"val/iou/acropora (branching)": 0.25}, step=3)
        tracker.log_dict({"a": 1}, "metadata/model.json")
        tracker.end_run(failed=False)

        run = mlflow_store.get_run(tracker.run_id)
        assert mlflow_store.get_experiment(run.info.experiment_id).name == "seg-exp"
        assert run.info.run_name == "run-b"
        assert run.info.status == "FINISHED"
        assert run.data.params == {"lr": "0.1", "cli/classes": "['a', 'b']", "bad_key_": "1"}
        history = mlflow_store.get_metric_history(tracker.run_id, "val/iou/acropora _branching_")
        assert [(m.step, m.value) for m in history] == [(3, 0.25)]
        loaded = mlflow.artifacts.load_dict(f"runs:/{tracker.run_id}/metadata/model.json")
        assert loaded == {"a": 1}
        assert tracker.run_url is None

    def test_params_and_metrics_are_logged_asynchronously_and_flushed_by_end_run(
        self, mlflow_store, monkeypatch
    ):
        mlflow = pytest.importorskip("mlflow")
        synchronous_args: dict[str, Any] = {}

        def _spy(name: str):
            real = getattr(mlflow, name)

            def _call(*args: Any, **kwargs: Any) -> Any:
                synchronous_args[name] = kwargs.get("synchronous", True)
                return real(*args, **kwargs)

            return _call

        monkeypatch.setattr(mlflow, "log_params", _spy("log_params"))
        monkeypatch.setattr(mlflow, "log_metrics", _spy("log_metrics"))
        tracker = MlflowTracker(experiment_name="seg-exp", run_name="run-b")
        tracker.log_params({"lr": 0.1})
        tracker.log_metrics({"loss": 0.5}, step=1)
        tracker.end_run(failed=False)

        assert synchronous_args == {"log_params": False, "log_metrics": False}
        run = mlflow_store.get_run(tracker.run_id)
        assert run.data.params == {"lr": "0.1"}
        assert [
            (m.step, m.value) for m in mlflow_store.get_metric_history(tracker.run_id, "loss")
        ] == [(1, 0.5)]

    def test_unset_tracking_uri_raises(self, monkeypatch):
        monkeypatch.delenv("MLFLOW_TRACKING_URI", raising=False)
        with pytest.raises(RuntimeError, match="MLFLOW_TRACKING_URI"):
            MlflowTracker(experiment_name="seg-exp", run_name="run-b")

    def test_arn_uri_without_sagemaker_mlflow_names_the_install_command(self, monkeypatch):
        monkeypatch.setenv(
            "MLFLOW_TRACKING_URI", "arn:aws:sagemaker:us-east-1:000000000000:mlflow-app/app"
        )
        monkeypatch.setitem(sys.modules, "sagemaker_mlflow", None)
        with pytest.raises(ImportError, match=r"uv sync --extra mlflow"):
            MlflowTracker(experiment_name="seg-exp", run_name="run-b")


def _raise_connection_error(*args: Any, **kwargs: Any) -> None:
    raise ConnectionError("tracker down")


@pytest.fixture(params=["wandb", "mlflow"])
def failing_metrics_tracker(request: pytest.FixtureRequest, monkeypatch: pytest.MonkeyPatch):
    """A started tracker of each backend whose ``log_metrics`` backend call raises."""
    if request.param == "wandb":
        stub_wandb = request.getfixturevalue("stub_wandb")
        monkeypatch.setattr(stub_wandb.run, "log", _raise_connection_error)
        return _wandb_tracker()
    request.getfixturevalue("mlflow_store")
    import mlflow

    tracker = MlflowTracker(experiment_name="seg-exp", run_name="run-b")
    monkeypatch.setattr(mlflow, "log_metrics", _raise_connection_error)
    return tracker


def _metric_failure_warnings(caplog: pytest.LogCaptureFixture) -> list[str]:
    return [
        r.getMessage()
        for r in caplog.records
        if r.levelno == logging.WARNING and "Failed to log metrics" in r.getMessage()
    ]


class TestMetricFailureRateLimit:
    def test_warnings_carry_the_running_count_and_end_run_the_total(
        self, failing_metrics_tracker, caplog
    ):
        caplog.set_level(logging.WARNING)
        for step in range(201):
            failing_metrics_tracker.log_metrics({"loss": 0.5}, step=step)
        for message, count in zip(_metric_failure_warnings(caplog), (1, 100, 200), strict=True):
            assert f"(failure {count})" in message

        caplog.clear()
        failing_metrics_tracker.end_run(failed=False)
        totals = [r.getMessage() for r in caplog.records if r.levelno == logging.WARNING]
        assert len(totals) == 1
        assert "201" in totals[0]

    def test_end_run_without_metric_failures_logs_no_warning(self, stub_wandb, caplog):
        caplog.set_level(logging.WARNING)
        tracker = _wandb_tracker()
        tracker.log_metrics({"loss": 0.5}, step=1)
        tracker.end_run(failed=False)
        assert [r for r in caplog.records if r.levelno >= logging.WARNING] == []


class TestMlflowFailureModes:
    def test_a_failing_connectivity_probe_stops_before_an_experiment_or_run_is_created(
        self, mlflow_store, monkeypatch
    ):
        mlflow = pytest.importorskip("mlflow")
        created: list[str] = []
        monkeypatch.setattr(mlflow, "search_experiments", _raise_connection_error)
        monkeypatch.setattr(mlflow, "set_experiment", lambda *a, **k: created.append("experiment"))
        monkeypatch.setattr(mlflow, "start_run", lambda *a, **k: created.append("run"))
        with pytest.raises(ConnectionError, match="tracker down"):
            MlflowTracker(experiment_name="exp", run_name="run")
        assert created == []

    @pytest.mark.parametrize(
        ("method", "args", "mlflow_function"),
        [
            ("log_params", ({"lr": 0.1},), "log_params"),
            ("log_dict", ({"a": 1}, "metadata/a.json"), "log_dict"),
            ("end_run", (False,), "end_run"),
        ],
    )
    def test_a_failing_backend_call_is_warned_not_raised(
        self, mlflow_store, monkeypatch, caplog, method, args, mlflow_function
    ):
        mlflow = pytest.importorskip("mlflow")
        caplog.set_level(logging.WARNING)
        tracker = MlflowTracker(experiment_name="seg-exp", run_name="run-b")
        with monkeypatch.context() as patch:
            patch.setattr(mlflow, mlflow_function, _raise_connection_error)
            getattr(tracker, method)(*args)
        assert len([r for r in caplog.records if r.levelno == logging.WARNING]) == 1
        mlflow.end_run()


@pytest.mark.parametrize(
    ("key", "expected"),
    [
        ("val/iou/acropora (branching) & sp+", "val/iou/acropora _branching_ _ sp_"),
        ("lineage/manifests/a-b.c: d_e", "lineage/manifests/a-b.c: d_e"),
    ],
)
def test_mlflow_key_replaces_disallowed_characters(key: str, expected: str):
    assert _mlflow_key(key) == expected


@pytest.mark.parametrize(
    ("uri", "expected"),
    [
        ("http://localhost:5000", "http://localhost:5000/#/experiments/7/runs/abc"),
        ("https://mlflow.invalid/", "https://mlflow.invalid/#/experiments/7/runs/abc"),
        ("sqlite:///mlflow.db", None),
    ],
)
def test_mlflow_run_url_only_for_http_servers(uri: str, expected: str | None):
    assert _mlflow_run_url(uri, "7", "abc") == expected


@pytest.mark.parametrize(
    ("name", "expected"),
    [
        ("v1.2-final_x", "v1.2-final_x"),
        ("a b(c)/d:e", "a-b-c--d-e"),
    ],
)
def test_safe_wandb_artifact_name_keeps_only_allowed_characters(name: str, expected: str):
    assert _safe_wandb_artifact_name(name) == expected


# ===================================================================
# make_tracker
# ===================================================================
def test_make_tracker_passes_the_wandb_config_keys(stub_wandb):
    cfg = ConfigDict(
        {
            "tracker": "wandb",
            "wandb_project": "proj",
            "wandb_entity": "ent",
            "train_log_interval": 1,
        }
    )
    tracker = make_tracker(cfg, run_name="run-c", config={"k": 2})
    assert isinstance(tracker, WandbTracker)
    assert stub_wandb.init_kwargs == {
        "project": "proj",
        "entity": "ent",
        "name": "run-c",
        "config": {"k": 2},
        "id": None,
        "resume": None,
    }


def test_make_tracker_passes_the_mlflow_experiment_name(mlflow_store):
    cfg = ConfigDict({"tracker": "mlflow", "experiment_name": "exp-c", "train_log_interval": 1})
    tracker = make_tracker(cfg, run_name="run-c", config={"k": 2})
    assert isinstance(tracker, MlflowTracker)
    run = mlflow_store.get_run(tracker.run_id)
    assert mlflow_store.get_experiment(run.info.experiment_id).name == "exp-c"
    assert run.info.run_name == "run-c"


# ===================================================================
# Logger -> tracker dispatch
# ===================================================================
class TestLoggerInit:
    def test_run_metadata_reaches_the_tracker(self, make_config, fake_tracker):
        meta = FakeMetaModel(run_name="meta-run", num_classes=3, num_concepts=5)
        config = make_config()
        _, tracker = _logger(make_config, meta, id2label={0: "ignore", 1: "coral"})

        assert tracker.kwargs == {
            "project": "mermaidseg",
            "entity": None,
            "run_name": "meta-run",
            "config": config,
        }
        assert tracker.dicts == [
            ("config/config.json", {**config, "num_classes": 3}),
            ("metadata/id2label.json", {0: "ignore", 1: "coral"}),
        ]

    @pytest.mark.parametrize("id2label", [None])
    def test_an_empty_id2label_is_not_logged(self, make_config, fake_tracker, id2label):
        _, tracker = _logger(make_config, id2label=id2label)
        assert [path for path, _ in tracker.dicts] == ["config/config.json"]


class TestLoggerLog:
    """Metric logging: scalars and ndarrays with id maps."""

    def test_scalar_metrics_reach_the_tracker_with_the_step(self, make_config, fake_tracker):
        lgr, tracker = _logger(make_config)
        lgr.log({"loss": 0.3}, step=5)
        assert tracker.metrics == [({"loss": 0.3}, 5)]

    def test_ndarray_class_metrics_unpacked(self, make_config, fake_tracker):
        lgr, tracker = _logger(make_config, id2label={0: "coral", 1: "sand", 2: "rock"})
        lgr.log({"iou": np.array([0.75, 0.5, 0.25])}, step=1)
        assert tracker.metrics == [({"iou/coral": 0.75, "iou/sand": 0.5, "iou/rock": 0.25}, 1)]

    def test_ndarray_without_id_map_uses_index(self, make_config, fake_tracker):
        lgr, tracker = _logger(make_config)
        lgr.log({"dice": np.array([0.5, 0.25])}, step=1)
        assert tracker.metrics == [({"dice/class_0": 0.5, "dice/class_1": 0.25}, 1)]

    def test_ndarray_concept_metrics_name_unmapped_indices_by_index(
        self, make_config, fake_tracker
    ):
        lgr, tracker = _logger(make_config, id2concept={0: "hard"})
        lgr.log({"concept_accuracy": np.array([0.75, 0.5])}, step=1)
        assert tracker.metrics == [
            ({"concept_accuracy/hard": 0.75, "concept_accuracy/concept_1": 0.5}, 1)
        ]

    def test_ndarray_concept_metrics_use_id2concept(self, make_config, fake_tracker):
        lgr, tracker = _logger(
            make_config, id2label={0: "coral", 1: "sand"}, id2concept={0: "hard", 1: "soft"}
        )
        lgr.log({"concept_accuracy": np.array([0.75, 0.5])}, step=1)
        assert tracker.metrics == [
            ({"concept_accuracy/hard": 0.75, "concept_accuracy/soft": 0.5}, 1)
        ]


class TestLoggerDispatch:
    """log_params/log_dict/log_dataloader_params/log_reconciliation reach the tracker."""

    def test_log_reconciliation_logs_each_map_and_the_class_counts(self, make_config, fake_tracker):
        lgr, tracker = _logger(make_config)
        tracker.params.clear()
        tracker.dicts.clear()
        registry = MagicMock()
        registry.global_id2source = {0: ("mermaid", "coral")}
        registry.target_id2label = {1: "coral"}
        registry.dataset_offsets = {"mermaid": 0}
        registry.source_to_target = torch.tensor([0, 1], dtype=torch.long)
        registry.concept_id2name = {0: "hard"}
        registry.num_target_classes = 2
        registry.num_global_source_classes = 3

        lgr.log_reconciliation(registry)

        assert tracker.dicts == [
            ("metadata/global_id2source.json", {"0": ["mermaid", "coral"]}),
            ("metadata/target_id2label.json", {"1": "coral"}),
            ("metadata/dataset_offsets.json", {"mermaid": 0}),
            ("metadata/source_to_target.json", {"0": 0, "1": 1}),
            ("metadata/concept_id2name.json", {"0": "hard"}),
        ]
        assert tracker.params == [{"num_target_classes": 2, "num_global_source_classes": 3}]


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


class TestSaveModelCheckpoint:
    """Checkpoint saving to local files."""

    def test_local_checkpoint_skipped_when_disabled(self, tmp_path, make_config, fake_tracker):
        meta = FakeMetaModel(run_name="ckpt-skip")
        lgr = Logger(
            id2concept={},
            config=make_config(),
            meta_model=meta,
            checkpoint_dir=str(tmp_path),
            log_checkpoint=50,
            save_local_checkpoints=False,
        )
        lgr.save_model_checkpoint(meta, epoch=50, metrics_dict={"loss": 0.2})
        assert not (tmp_path / "model_checkpoints" / "ckpt-skip").exists()

    def test_step_checkpoint_name_and_resume_flags(self, tmp_path, make_config, fake_tracker):
        meta = FakeMetaModel(run_name="step-ckpt")
        meta.global_step = 25000
        meta._warmup_iters_completed = 4000
        config = make_config()
        lgr = Logger(
            id2concept={},
            config=config,
            meta_model=meta,
            checkpoint_dir=str(tmp_path),
            log_checkpoint=1,
        )
        lgr.save_model_checkpoint(
            meta,
            epoch=0,
            metrics_dict={"loss": 0.1},
            checkpoint_name="model_step25000",
            epoch_complete=False,
        )
        path = tmp_path / "model_checkpoints" / "step-ckpt" / "model_step25000"
        assert path.is_file()
        ckpt = torch.load(path, weights_only=True)
        assert ckpt["global_step"] == 25000
        assert ckpt["warmup_iters_completed"] == 4000
        assert ckpt["epoch_complete"] is False
        assert ckpt["epoch"] == 0

    def test_checkpoint_carries_the_loggers_lineage_only_when_set(
        self, tmp_path, make_config, fake_tracker
    ):
        lineage = {"git_sha": "abc", "git_dirty": False}
        for name, kwargs in (("with-lineage", {"lineage": lineage}), ("no-lineage", {})):
            meta = FakeMetaModel(run_name=name)
            lgr = Logger(
                id2concept={},
                config=make_config(),
                meta_model=meta,
                checkpoint_dir=str(tmp_path),
                log_checkpoint=1,
                **kwargs,
            )
            lgr.save_model_checkpoint(meta, epoch=1, metrics_dict={"loss": 0.1})
        saved = {
            name: torch.load(
                tmp_path / "model_checkpoints" / name / "model_epoch1", weights_only=True
            )
            for name in ("with-lineage", "no-lineage")
        }
        assert saved["with-lineage"]["lineage"] == lineage
        assert "lineage" not in saved["no-lineage"]

    def test_checkpoint_metrics_and_config_load_as_plain_python(
        self, tmp_path, make_config, fake_tracker
    ):
        meta = FakeMetaModel(run_name="plain-ckpt")
        lgr = Logger(
            id2concept={},
            config=make_config(),
            meta_model=meta,
            checkpoint_dir=str(tmp_path),
            log_checkpoint=1,
        )
        metrics = {
            "iou": np.array([0.5, 0.25]),
            "acc": np.float32(0.75),
            "loss": torch.tensor(0.125),
            "step": 3,
        }
        lgr.save_model_checkpoint(meta, epoch=1, metrics_dict=metrics)
        path = tmp_path / "model_checkpoints" / "plain-ckpt" / "model_epoch1"
        ckpt = torch.load(path, weights_only=True)
        assert ckpt["metrics"] == {"iou": [0.5, 0.25], "acc": 0.75, "loss": 0.125, "step": 3}
        assert type(ckpt["config"]) is dict
        assert type(ckpt["config"]["training"]) is dict
        assert ckpt["config"]["training"]["optimizer"] == {"type": "SGD", "lr": 0.01}

    def test_save_leaves_parameters_in_place(
        self, tmp_path, make_config, fake_tracker, monkeypatch
    ):
        meta = _meta_with_move_recording_model("inplace-ckpt")
        weight = meta.model.weight
        ptr = weight.data_ptr()
        values = weight.detach().clone()
        lgr = Logger(
            id2concept={},
            config=make_config(),
            meta_model=meta,
            checkpoint_dir=str(tmp_path),
            log_checkpoint=1,
        )
        meta.model.moves.clear()
        saved = _spy_torch_save(monkeypatch)
        lgr.save_model_checkpoint(meta, epoch=1, metrics_dict={"loss": 0.1})

        assert meta.model.moves == []
        assert meta.model.weight is weight
        assert meta.model.weight.data_ptr() == ptr
        checkpoint = saved[0][0]
        for key, live in meta.model.state_dict().items():
            assert checkpoint["model_state_dict"][key].data_ptr() != live.data_ptr()
        path = tmp_path / "model_checkpoints" / "inplace-ckpt" / "model_epoch1"
        ckpt = torch.load(path, weights_only=True)
        saved_weight = ckpt["model_state_dict"]["weight"]
        assert saved_weight.device.type == "cpu"
        torch.testing.assert_close(saved_weight, values)

    def test_atomic_checkpoint_leaves_no_partial(
        self, tmp_path, make_config, fake_tracker, monkeypatch
    ):
        meta = FakeMetaModel(run_name="atomic-ckpt")
        meta.global_step = 1000
        lgr = Logger(
            id2concept={}, config=make_config(), meta_model=meta, checkpoint_dir=str(tmp_path)
        )
        saved = _spy_torch_save(monkeypatch)
        replaced = _spy_os_replace(monkeypatch)
        lgr.save_model_checkpoint(
            meta,
            epoch=0,
            metrics_dict={},
            checkpoint_name="model_preempt1000",
            epoch_complete=False,
            atomic=True,
        )
        directory = tmp_path / "model_checkpoints" / "atomic-ckpt"
        final = directory / "model_preempt1000"
        partial = directory / "model_preempt1000.partial"
        assert [target for _, target in saved] == [str(partial)]
        assert replaced == [(str(partial), str(final))]
        assert final.is_file()
        assert not (directory / "model_preempt1000.partial").exists()
        ckpt = torch.load(final, weights_only=True)
        assert ckpt["global_step"] == 1000
        assert ckpt["epoch_complete"] is False
        assert ckpt["optimizer_state_dict"]
        assert ckpt["scheduler_state_dict"]
        assert ckpt["model_state_dict"]

        (directory / "model_preempt500").write_bytes(b"stale")
        (directory / "model_preempt500.partial").write_bytes(b"partial")
        (directory / "model_step25000").write_bytes(b"keep")
        (directory / "model_epoch0").write_bytes(b"keep")
        prune_rolling_checkpoints(str(directory), keep_name="model_preempt1000")
        assert final.is_file()
        assert (directory / "model_step25000").is_file()
        assert (directory / "model_epoch0").is_file()
        assert not (directory / "model_preempt500").exists()
        assert not (directory / "model_preempt500.partial").exists()


# ===================================================================
# Logger.end_run & context manager
# ===================================================================
class TestLoggerLifecycle:
    """Logger cleanup: end_run and context manager protocol."""

    def test_context_manager_ends_the_run_as_failed_on_exception(self, make_config, fake_tracker):
        lgr, tracker = _logger(make_config)
        with pytest.raises(ValueError, match="boom"), lgr:
            raise ValueError("boom")
        assert tracker.ended == [True]

    def test_context_manager_ends_the_run_as_not_failed_on_clean_exit(
        self, make_config, fake_tracker
    ):
        lgr, tracker = _logger(make_config)
        with lgr:
            pass
        assert tracker.ended == [False]

    def test_context_manager_sets_the_mlflow_run_status_failed(self, make_config, mlflow_store):
        config = make_config(logger={"tracker": "mlflow", "experiment_name": "exp-s"})
        lgr = Logger(id2concept={}, config=config, meta_model=FakeMetaModel())
        run_id = lgr.run_id
        assert run_id is not None
        with pytest.raises(ValueError, match="boom"), lgr:
            raise ValueError("boom")
        assert mlflow_store.get_run(run_id).info.status == "FAILED"


# ===================================================================
# Imports without the tracker extras
# ===================================================================
IMPORT_SCRIPT = """
import sys
for name in ("wandb", "mlflow", "sagemaker_mlflow"):
    sys.modules[name] = None
import mermaidseg
import mermaidseg.logger
import mermaidseg.model.train
import mermaidseg.training
"""


def test_package_imports_without_the_tracker_dependencies():
    result = subprocess.run(
        [sys.executable, "-c", IMPORT_SCRIPT],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0, result.stderr
