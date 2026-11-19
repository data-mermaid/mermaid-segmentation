"""MetaModel: run settings from the training config, optimizer wiring, gradient accumulation."""

from __future__ import annotations

import contextlib
import copy
import json
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
import pytest
import torch
from torch.utils.data import DataLoader, TensorDataset

import mermaidseg.model.meta as meta_module
import mermaidseg.model.models as models_module
from mermaidseg.export import write_sidecar
from mermaidseg.inference.bundle import MODEL_JSON
from mermaidseg.io import REPO_ROOT, ConfigDict
from mermaidseg.model.eval import Evaluator
from mermaidseg.model.meta import next_epoch_from_checkpoint, records_train_metrics
from mermaidseg.model.models import MODEL_CLASSES
from tests.conftest import IMAGE_SIZE, NUM_CLASSES
from tests.mermaidseg.conftest import cbm_meta, concept_keep


@pytest.mark.parametrize(("raw", "expected"), [(180, 180.0), (0, 0.0), (-1, 0.0)])
def test_stall_dump_seconds_reads_training_config(raw, expected):
    assert meta_module.stall_dump_seconds(ConfigDict({"stall_dump_sec": raw})) == expected


def test_records_train_metrics_scores_every_nth_microbatch():
    assert [i for i in range(10) if records_train_metrics(i, 1)] == list(range(10))
    assert [i for i in range(10) if records_train_metrics(i, 3)] == [0, 3, 6, 9]
    assert records_train_metrics(5, 0)


class _FakeFaulthandler:
    def __init__(self) -> None:
        self.armed: list[float] = []
        self.cancelled = 0

    def dump_traceback_later(self, timeout, repeat=False, exit=False):
        assert repeat is False and exit is False
        self.armed.append(timeout)

    def cancel_dump_traceback_later(self):
        self.cancelled += 1


def test_watch_for_stalls_rearms_each_update_and_cancels(monkeypatch):
    fake = _FakeFaulthandler()
    monkeypatch.setattr(meta_module, "faulthandler", fake)
    assert list(meta_module.watch_for_stalls(range(3), 5.0)) == [0, 1, 2]
    assert fake.armed == [5.0, 5.0, 5.0]
    assert fake.cancelled == 1


def test_watch_for_stalls_cancels_on_break_and_error(monkeypatch):
    fake = _FakeFaulthandler()
    monkeypatch.setattr(meta_module, "faulthandler", fake)
    for update in meta_module.watch_for_stalls(range(5), 5.0):
        if update == 1:
            break
    assert fake.cancelled == 1

    def run() -> None:
        for update in meta_module.watch_for_stalls(range(5), 5.0):
            if update == 2:
                raise RuntimeError("stalled")

    with pytest.raises(RuntimeError, match="stalled"):
        run()
    assert fake.cancelled == 2


def test_watch_for_stalls_disabled_never_arms(monkeypatch):
    fake = _FakeFaulthandler()
    monkeypatch.setattr(meta_module, "faulthandler", fake)
    assert list(meta_module.watch_for_stalls(range(2), 0.0)) == [0, 1]
    assert fake.armed == [] and fake.cancelled == 0


def test_optimizer_gives_lora_params_lora_lr_and_the_rest_dpt_lr(minimal_config, make_meta_model):
    minimal_config.training.optimizer.lora_lr = 1e-5
    minimal_config.training.optimizer.dpt_lr = 5e-5
    meta = make_meta_model(minimal_config)
    initial_lr = {
        id(param): group["initial_lr"]
        for group in meta.optimizer.param_groups
        for param in group["params"]
    }
    trainable = [(n, p) for n, p in meta.model.named_parameters() if p.requires_grad]
    lora = {n for n, _ in trainable if "lora_" in n}
    assert lora and len(lora) < len(trainable)
    for name, param in trainable:
        assert initial_lr[id(param)] == (1e-5 if name in lora else 5e-5), name


def _standard_cfg(minimal_config, **training):
    cfg = copy.deepcopy(minimal_config)
    for key, value in training.items():
        cfg.training[key] = value
    return cfg


def _images_labels(n: int) -> tuple[torch.Tensor, torch.Tensor]:
    generator = torch.Generator().manual_seed(0)
    images = torch.randn(n, 3, *IMAGE_SIZE, generator=generator)
    labels = torch.ones(n, *IMAGE_SIZE, dtype=torch.long)
    return images, labels


@pytest.mark.parametrize(
    ("epoch_complete", "expected"),
    [(True, 4), (False, 3)],
    ids=["epoch-boundary-starts-next-epoch", "mid-epoch-continues-same-epoch"],
)
def test_next_epoch_from_checkpoint(epoch_complete, expected):
    assert next_epoch_from_checkpoint({"epoch": 3, "epoch_complete": epoch_complete}) == expected


@pytest.mark.parametrize("name", [None, "float16", "fp16", "half"])
def test_resolve_amp_dtype_rejects_missing_and_fp16_aliases(name):
    with pytest.raises(ValueError, match="mixed_precision_dtype"):
        meta_module.resolve_amp_dtype(name)


def test_resolve_amp_dtype_accepts_bfloat16():
    assert meta_module.resolve_amp_dtype("bfloat16") is torch.bfloat16


def _record_encoder_autocast(meta) -> dict[str, object]:
    """Record the CPU autocast state seen inside the mocked encoder's forward."""
    seen: dict[str, object] = {}

    def _record(_module, _args):
        seen["enabled"] = torch.is_autocast_enabled("cpu")
        seen["dtype"] = torch.get_autocast_dtype("cpu")

    encoder = next(m for m in meta.model.modules() if hasattr(m, "embed"))
    encoder.register_forward_pre_hook(_record)
    return seen


def _train_one_step(meta) -> None:
    images, labels = _images_labels(2)
    loader = DataLoader(TensorDataset(images, labels), batch_size=2)
    meta.train_epoch(loader, evaluator=None, epoch=0)


def test_cpu_train_step_runs_encoder_under_cpu_bf16_autocast(minimal_config, make_meta_model):
    cfg = _standard_cfg(
        minimal_config,
        mixed_precision=True,
        mixed_precision_dtype="bfloat16",
        iterations_per_train_epoch=1,
    )
    meta = make_meta_model(cfg, run_name="cpu-amp")
    seen = _record_encoder_autocast(meta)
    _train_one_step(meta)
    assert seen == {"enabled": True, "dtype": torch.bfloat16}


def test_cpu_train_step_without_mixed_precision_does_not_autocast(minimal_config, make_meta_model):
    cfg = _standard_cfg(minimal_config, mixed_precision=False, iterations_per_train_epoch=1)
    meta = make_meta_model(cfg, run_name="cpu-no-amp")
    seen = _record_encoder_autocast(meta)
    _train_one_step(meta)
    assert seen["enabled"] is False


def test_cuda_bf16_check_targets_the_model_device_without_emulation(minimal_config, monkeypatch):
    support_kwargs: list[dict] = []
    entered_devices: list[torch.device] = []

    def record_support(*args, **kwargs):
        support_kwargs.append(kwargs)
        return False

    @contextlib.contextmanager
    def record_device(device):
        entered_devices.append(device)
        yield

    monkeypatch.setattr(torch.cuda, "is_bf16_supported", record_support)
    monkeypatch.setattr(torch.cuda, "device", record_device)
    cfg = _standard_cfg(minimal_config, mixed_precision=True, mixed_precision_dtype="bfloat16")
    with pytest.raises(RuntimeError, match="mixed_precision_dtype"):
        meta_module.MetaModel(
            run_name="no-bf16-device",
            num_classes=4,
            model_kwargs=cfg.model,
            training_kwargs=cfg.training,
            device=torch.device("cuda:1"),
        )
    assert support_kwargs == [{"including_emulation": False}]
    assert entered_devices == [torch.device("cuda:1")]


def test_accumulate_loss_averages_in_fp32(minimal_config, make_meta_model):
    meta = make_meta_model(_standard_cfg(minimal_config), run_name="scale")
    meta.grad_accum_steps = 2
    param = torch.nn.Parameter(torch.tensor(1.0))
    loss = param * 4
    meta._accumulate_loss(loss)
    assert param.grad is not None
    assert param.grad.dtype == torch.float32
    assert param.grad.item() == pytest.approx(2.0)


def test_two_microbatches_are_one_optimizer_step(minimal_config, make_meta_model):
    cfg = _standard_cfg(minimal_config, iterations_per_train_epoch=1, grad_accum_steps=2)
    meta = make_meta_model(cfg, run_name="accum-count")
    images, labels = _images_labels(4)
    loader = DataLoader(TensorDataset(images, labels), batch_size=2)
    calls = {"n": 0}
    real = meta.batch_predict_loss

    def _count(*args, **kwargs):
        calls["n"] += 1
        return real(*args, **kwargs)

    meta.batch_predict_loss = _count
    with patch("torch.nn.utils.clip_grad_norm_", wraps=torch.nn.utils.clip_grad_norm_) as clip:
        meta.train_epoch(loader, evaluator=None, epoch=0)
    assert calls["n"] == 2
    assert meta.global_step == 1
    assert clip.call_count == 1


def _capture_step_grads(meta) -> dict[str, torch.Tensor]:
    """Gradients each parameter holds when ``meta`` applies its optimizer step."""
    captured: dict[str, torch.Tensor] = {}
    real = meta._apply_accumulated_step

    def _capture() -> bool:
        for name, param in meta.model.named_parameters():
            if param.grad is not None:
                captured[name] = param.grad.detach().clone()
        return real()

    meta._apply_accumulated_step = _capture
    return captured


def test_microbatches_match_a_single_batch(minimal_config, make_meta_model):
    """Per-image mean of two equal microbatches matches one combined batch."""
    images, labels = _images_labels(4)
    full_loader = DataLoader(TensorDataset(images, labels), batch_size=4)
    micro_loader = DataLoader(TensorDataset(images, labels), batch_size=2)

    full = make_meta_model(
        _standard_cfg(minimal_config, iterations_per_train_epoch=1, grad_accum_steps=1),
        run_name="full-batch",
    )
    micro = make_meta_model(
        _standard_cfg(minimal_config, iterations_per_train_epoch=1, grad_accum_steps=2),
        run_name="micro-batch",
    )
    micro.model.load_state_dict(full.model.state_dict())
    # train_epoch enters train mode, so switch dropout off to give both runs the same forward.
    for module in (*full.model.modules(), *micro.model.modules()):
        if isinstance(module, torch.nn.Dropout):
            module.p = 0.0
    before = {name: param.detach().clone() for name, param in full.model.named_parameters()}
    full_grads = _capture_step_grads(full)
    micro_grads = _capture_step_grads(micro)

    full.train_epoch(full_loader, evaluator=None, epoch=0)
    micro.train_epoch(micro_loader, evaluator=None, epoch=0)

    assert full_grads.keys() == micro_grads.keys()
    for name, grad in full_grads.items():
        assert torch.allclose(grad, micro_grads[name], atol=1e-5, rtol=1e-5), name
    assert any(grad.abs().sum() > 0 for grad in full_grads.values())
    assert any(not torch.equal(before[n], p) for n, p in full.model.named_parameters())


def test_nonfinite_microbatch_drops_the_window(minimal_config, make_meta_model):
    cfg = _standard_cfg(minimal_config, iterations_per_train_epoch=1, grad_accum_steps=2)
    meta = make_meta_model(cfg, run_name="nonfinite")
    before = {name: tensor.detach().clone() for name, tensor in meta.model.state_dict().items()}
    calls = {"n": 0}
    real = meta.batch_predict_loss

    def _poison(*args, **kwargs):
        calls["n"] += 1
        loss, outputs, concepts, components = real(*args, **kwargs)
        if calls["n"] == 2:
            loss = torch.tensor(float("nan"))
        return loss, outputs, concepts, components

    meta.batch_predict_loss = _poison
    images, labels = _images_labels(4)
    loader = DataLoader(TensorDataset(images, labels), batch_size=2)
    meta.train_epoch(loader, evaluator=None, epoch=0)

    assert meta.global_step == 0
    after = meta.model.state_dict()
    for name, tensor in before.items():
        assert torch.equal(after[name], tensor)


def test_nonfinite_window_counts_loaded_samples_but_not_trained_samples(
    minimal_config, make_meta_model
):
    cfg = _standard_cfg(minimal_config, iterations_per_train_epoch=2, grad_accum_steps=1)
    meta = make_meta_model(cfg, run_name="loaded-samples")
    calls = {"n": 0}
    real = meta.batch_predict_loss

    def _poison(*args, **kwargs):
        calls["n"] += 1
        loss, outputs, concepts, components = real(*args, **kwargs)
        if calls["n"] == 2:
            loss = torch.tensor(float("nan"))
        return loss, outputs, concepts, components

    meta.batch_predict_loss = _poison
    images, labels = _images_labels(4)
    loader = DataLoader(TensorDataset(images, labels), batch_size=2)
    _, _, timing = meta.train_epoch(loader, evaluator=None, epoch=0)

    # Two updates of batch 2: the second window is skipped, so only the first reaches backward.
    assert timing["num_loaded_samples"] == 4
    assert timing["num_samples"] == 2


@pytest.mark.parametrize(
    ("epoch", "global_step"),
    [(0, 2), (1, 6)],
    ids=["first-epoch", "later-epoch-offsets-by-the-epoch"],
)
def test_resume_continues_the_same_epoch(minimal_config, make_meta_model, epoch, global_step):
    cfg = _standard_cfg(minimal_config, iterations_per_train_epoch=4, grad_accum_steps=1)
    meta = make_meta_model(cfg, run_name="resume-epoch")
    meta.global_step = global_step
    images, labels = _images_labels(8)
    loader = DataLoader(TensorDataset(images, labels), batch_size=2)
    calls = {"n": 0}
    real = meta.batch_predict_loss

    def _count(*args, **kwargs):
        calls["n"] += 1
        return real(*args, **kwargs)

    meta.batch_predict_loss = _count
    meta.train_epoch(loader, evaluator=None, epoch=epoch)
    assert calls["n"] == 2
    assert meta.global_step == global_step + 2


class _TreeBuildStop(Exception):
    """Ends MetaModel construction once the taxonomy path has been captured."""


def test_taxonomic_loss_reads_the_mapping_from_the_repo_not_the_current_directory(
    minimal_config, make_meta_model, monkeypatch, tmp_path
):
    cfg = _standard_cfg(
        minimal_config,
        taxonomic_loss={"ambiguity_policy": "permissive", "kl_weight": 1.0},
    )
    cfg.training.concept_mapping_path = "configs/class_to_concepts.csv"
    seen: list[Path] = []

    def _record(path, **_kwargs):
        seen.append(path)
        raise _TreeBuildStop

    monkeypatch.setattr(meta_module, "build_taxonomy_tree", _record)
    monkeypatch.chdir(tmp_path)
    with pytest.raises(_TreeBuildStop):
        make_meta_model(cfg)
    assert seen == [REPO_ROOT / "configs" / "class_to_concepts.csv"]


def _record_encoder_modes(meta) -> list[dict[str, bool]]:
    """Record ``training`` and inference-mode state at each mocked-encoder forward."""
    seen: list[dict[str, bool]] = []

    def _record(module, _args):
        seen.append({"training": module.training, "inference": torch.is_inference_mode_enabled()})

    encoder = next(m for m in meta.model.modules() if hasattr(m, "embed"))
    encoder.register_forward_pre_hook(_record)
    return seen


def _loader(n: int = 2) -> DataLoader:
    images, labels = _images_labels(n)
    return DataLoader(TensorDataset(images, labels), batch_size=2)


def _loop_cfg(minimal_config):
    return _standard_cfg(minimal_config, iterations_per_train_epoch=1, iterations_per_val_epoch=1)


def test_validation_epoch_runs_forward_in_eval_under_inference_mode(
    minimal_config, make_meta_model
):
    meta = make_meta_model(_loop_cfg(minimal_config), run_name="val-eval")
    meta.model.train()
    seen = _record_encoder_modes(meta)
    meta.validation_epoch(_loader())
    assert seen == [{"training": False, "inference": True}]


def test_train_epoch_after_validation_returns_the_model_to_train_mode(
    minimal_config, make_meta_model
):
    meta = make_meta_model(_loop_cfg(minimal_config), run_name="val-then-train")
    meta.validation_epoch(_loader())
    meta.model.eval()  # whatever left the model in eval, train_epoch must undo it
    seen = _record_encoder_modes(meta)
    meta.train_epoch(_loader(), evaluator=None, epoch=0)
    assert seen == [{"training": True, "inference": False}]


class _RecordingLogger:
    def __init__(self) -> None:
        self.calls: list[tuple[dict[str, object], int]] = []

    def log(self, metrics: dict[str, object], step: int) -> None:
        self.calls.append((metrics, step))


def _record_micro_batches(meta) -> list[tuple[float, dict[str, float]]]:
    """Record each microbatch's loss and components as floats, in call order."""
    records: list[tuple[float, dict[str, float]]] = []
    real = meta.batch_predict_loss

    def _recording(*args, **kwargs):
        loss, outputs, concepts, components = real(*args, **kwargs)
        records.append((float(loss.detach()), {k: float(v) for k, v in components.items()}))
        return loss, outputs, concepts, components

    meta.batch_predict_loss = _recording
    return records


def _window_means(records, accum: int, update: int) -> tuple[float, dict[str, float]]:
    window = records[update * accum : (update + 1) * accum]
    keys = window[0][1].keys()
    means = {k: sum(c[k] for _, c in window) / accum for k in keys}
    return sum(loss for loss, _ in window) / accum, means


def test_train_epoch_logs_float_window_means_only_at_the_log_interval(
    minimal_config, make_meta_model
):
    cfg = _standard_cfg(minimal_config, iterations_per_train_epoch=4, grad_accum_steps=2)
    meta = make_meta_model(cfg, run_name="log-interval")
    records = _record_micro_batches(meta)
    run_logger = _RecordingLogger()
    images, labels = _images_labels(16)
    loader = DataLoader(TensorDataset(images, labels), batch_size=2)

    meta.train_epoch(loader, run_logger=run_logger, log_interval=2, epoch=0)

    assert [step for _, step in run_logger.calls] == [2, 4]
    for metrics, step in run_logger.calls:
        assert all(isinstance(v, float) for k, v in metrics.items() if k.startswith("train/"))
        loss, means = _window_means(records, 2, step - 1)
        assert metrics["train/batch_loss"] == pytest.approx(loss, rel=1e-6)
        for key in ("classification", "concepts"):
            assert metrics[f"train/batch_{key}"] == pytest.approx(means[key], rel=1e-6)


def test_train_epoch_epoch_components_average_the_update_means(minimal_config, make_meta_model):
    cfg = _standard_cfg(minimal_config, iterations_per_train_epoch=4, grad_accum_steps=2)
    meta = make_meta_model(cfg, run_name="epoch-components")
    records = _record_micro_batches(meta)
    images, labels = _images_labels(16)
    loader = DataLoader(TensorDataset(images, labels), batch_size=2)

    last_loss, metrics, _ = meta.train_epoch(loader, epoch=0)

    per_update = [_window_means(records, 2, u) for u in range(4)]
    assert last_loss == pytest.approx(sum(loss for loss, _ in per_update) / 4, rel=1e-6)
    for key in ("classification", "concepts"):
        expected = sum(means[key] for _, means in per_update) / 4
        assert metrics[f"loss/{key}"] == pytest.approx(expected, rel=1e-6, abs=1e-9)


def test_update_evaluator_feeds_metrics_the_argmax_of_class_logits(minimal_config, make_meta_model):
    class _Recorder:
        def __init__(self) -> None:
            self.preds: list[torch.Tensor] = []

        def update(self, preds: torch.Tensor, _target: torch.Tensor) -> None:
            self.preds.append(preds)

    recorder = _Recorder()
    evaluator = SimpleNamespace(
        device="cpu", metric_dict={"m": recorder}, evaluate_concepts=lambda *_a: None
    )
    meta = make_meta_model(minimal_config)
    logits = torch.tensor([[[[1.0, 5.0]], [[3.0, 2.0]], [[0.0, 9.0]]]])  # (1, 3, 1, 2)

    meta._update_evaluator(evaluator, logits, torch.zeros(1, 1, 2, dtype=torch.long), None, None)

    assert [p.tolist() for p in recorder.preds] == [[[[1, 2]]]]


def test_sidecar_architecture_kwargs_rebuild_the_model_from_json(
    minimal_config, make_meta_model, mock_dinov3_encoder, tmp_path
):
    meta = make_meta_model(minimal_config)
    write_sidecar(
        tmp_path,
        run_name="run",
        model_class=meta.model_name,
        model_kwargs=meta.model_kwargs,
        encoder_config={},
        id2label={0: "ignore"},
        concept_id2name={0: "c"},
        input_size=IMAGE_SIZE,
        mean=(0.5, 0.5, 0.5),
        std=(0.25, 0.25, 0.25),
        lineage={},
    )
    architecture = json.loads((tmp_path / MODEL_JSON).read_text())["architecture"]
    kwargs = dict(architecture["kwargs"])

    model = MODEL_CLASSES[architecture["class"]](
        base_encoder=mock_dinov3_encoder(models_module), **kwargs
    )

    out = model(torch.randn(1, 3, *IMAGE_SIZE))
    assert out.logits.shape[1] == NUM_CLASSES


def test_meta_model_loads_the_encoder_named_in_the_model_config(
    minimal_config, make_meta_model, monkeypatch
):
    real = meta_module.load_pretrained_encoder
    calls: list[str] = []

    def _record(name):
        calls.append(name)
        return real(name)

    monkeypatch.setattr(meta_module, "load_pretrained_encoder", _record)
    make_meta_model(minimal_config)
    assert calls == [minimal_config.model["encoder_name"]]


def test_meta_model_uses_a_prebuilt_model_without_loading_an_encoder(
    minimal_config, make_meta_model, monkeypatch
):
    prebuilt = make_meta_model(copy.deepcopy(minimal_config), run_name="donor").model

    def _forbidden(_name):
        raise AssertionError("load_pretrained_encoder must not run for a prebuilt model")

    monkeypatch.setattr(meta_module, "load_pretrained_encoder", _forbidden)
    meta = make_meta_model(minimal_config, run_name="prebuilt", model=prebuilt)
    assert meta.model is prebuilt


# Lightweight integration tests for pipeline wiring: single steps, offline, no network or real S3.
# Run with -m integration, deselect with -m "not integration".


@pytest.mark.integration
def test_metamodel_validation_epoch_returns_loss_and_metrics(
    minimal_config, tiny_loader, make_meta_model
):
    """MetaModel.validation_epoch returns loss and metric dict in eval mode."""
    meta = make_meta_model(minimal_config, run_name="test-val-epoch")
    evaluator = Evaluator(num_classes=NUM_CLASSES, device="cpu")

    val_loss, val_metrics = meta.validation_epoch(tiny_loader, evaluator)

    assert isinstance(val_loss, float)
    assert isinstance(val_metrics, dict)


@pytest.mark.integration
@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
def test_metamodel_runs_on_cuda_when_available(minimal_config, make_meta_model):
    """MetaModel can be instantiated on CUDA device."""
    meta = make_meta_model(minimal_config, run_name="test-cuda", device="cuda")
    assert meta.device == "cuda"
    assert next(meta.model.parameters()).device.type == "cuda"


def test_train_epoch_masks_concept_targets(make_meta_model, monkeypatch, cbm_schema):
    meta = cbm_meta(make_meta_model, cbm_schema)
    meta.set_concept_source_mask(concept_keep(cbm_schema))
    bleached = cbm_schema.channel_names.index("bleached")
    seen: list[torch.Tensor] = []
    original = meta.batch_predict_loss

    def spy(images, target_labels=None, target_concepts=None, *args, **kwargs):
        seen.append(target_concepts.detach().clone())
        return original(images, target_labels, target_concepts, *args, **kwargs)

    monkeypatch.setattr(meta, "batch_predict_loss", spy)
    images = torch.randn(2, 3, *IMAGE_SIZE)
    labels = torch.ones(
        2, *IMAGE_SIZE, dtype=torch.long
    )  # acropora: bleached False, branching True
    rows = torch.tensor([1, 0])
    loss, _, _ = meta.train_epoch(
        DataLoader(TensorDataset(images, labels, rows), batch_size=2), epoch=0
    )
    assert np.isfinite(loss)
    (targets,) = seen
    assert (targets[0, bleached] == 0).all()  # sample 0's source is masked for bleached
    assert (targets[1, bleached] == 1).all()  # sample 1 (row 0) keeps its False target
    unmasked = [i for i in range(cbm_schema.num_channels) if i != bleached]
    assert torch.equal(targets[0, unmasked], targets[1, unmasked])


def test_train_epoch_requires_source_rows_when_masking(make_meta_model, cbm_schema):
    meta = cbm_meta(make_meta_model, cbm_schema)
    meta.set_concept_source_mask(concept_keep(cbm_schema))
    loader = DataLoader(
        TensorDataset(torch.randn(2, 3, *IMAGE_SIZE), torch.ones(2, *IMAGE_SIZE, dtype=torch.long)),
        batch_size=2,
    )
    with pytest.raises(RuntimeError, match="no source rows"):
        meta.train_epoch(loader, epoch=0)
