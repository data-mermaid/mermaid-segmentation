"""MetaModel: run settings from the training config, optimizer wiring, gradient accumulation."""

from __future__ import annotations

import contextlib
import copy
import json
import math
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
import pytest
import torch
from torch.utils.data import DataLoader, TensorDataset

import mermaidseg.model.meta as meta_module
import mermaidseg.model.models as models_module
import mermaidseg.model.taxonomy_loss as taxonomy_loss_module
from mermaidseg.dataset_reconciliation.concept_ranks import TAXONOMIC_CONCEPTS
from mermaidseg.export import write_sidecar
from mermaidseg.inference.bundle import MODEL_JSON
from mermaidseg.io import REPO_ROOT, ConfigDict
from mermaidseg.model.eval import Evaluator
from mermaidseg.model.meta import next_epoch_from_checkpoint, records_train_metrics
from mermaidseg.model.models import MODEL_CLASSES
from tests.conftest import (
    CBM_CONCEPT_VALUE2ID,
    CBM_NUM_CONCEPTS,
    IMAGE_SIZE,
    NUM_CLASSES,
    cbm_concept_lookup,
)
from tests.mermaidseg.conftest import cbm_meta, concept_keep


@pytest.mark.parametrize(("raw", "expected"), [(180, 180.0), (0, 0.0)])
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


@pytest.mark.parametrize("name", [None, "float16"])
def test_resolve_amp_dtype_rejects_missing_and_fp16_aliases(name):
    with pytest.raises(ValueError, match="mixed_precision_dtype"):
        meta_module.resolve_amp_dtype(name)


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
            num_classes=NUM_CLASSES,
            num_concepts=CBM_NUM_CONCEPTS,
            model_kwargs=cfg.model,
            training_kwargs=cfg.training,
            device=torch.device("cuda:1"),
            source_to_target_lookup=torch.arange(NUM_CLASSES),
            source_to_concepts_lookup=cbm_concept_lookup(),
            concept_value2id=CBM_CONCEPT_VALUE2ID,
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
        loss, components = real(*args, **kwargs)
        if calls["n"] == 2:
            loss = torch.tensor(float("nan"))
        return loss, components

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
        loss, components = real(*args, **kwargs)
        if calls["n"] == 2:
            loss = torch.tensor(float("nan"))
        return loss, components

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

    monkeypatch.setattr(taxonomy_loss_module, "build_taxonomy_tree", _record)
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
        loss, components = real(*args, **kwargs)
        records.append((float(loss.detach()), {k: float(v) for k, v in components.items()}))
        return loss, components

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
        label_space={},
        lineage={},
    )
    architecture = json.loads((tmp_path / MODEL_JSON).read_text())["architecture"]
    kwargs = dict(architecture["kwargs"])

    model = MODEL_CLASSES[architecture["class"]](
        base_encoder=mock_dinov3_encoder(models_module), **kwargs
    )

    out = model(torch.randn(1, 3, *IMAGE_SIZE))
    assert out.logits.shape[1] == NUM_CLASSES


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


def _label_batches(*labels: int) -> list[tuple[torch.Tensor, torch.Tensor]]:
    """One single-image batch per label, each a 1 x 2 map of that source label."""
    return [(torch.zeros(1, 3, 1, 2), torch.full((1, 1, 2), label)) for label in labels]


def test_validation_epoch_averages_the_batch_losses_and_their_components(fixed_meta_model):
    """Two batches over fixed class logits (0, ln 3, 0, 0), every concept logit 0.

    Batch 1 is labelled class 1, scoring ln 2 (probability 3/6); batch 2 is class 2, scoring
    ln 6 (1/6) and predicted wrong. Each of the six taxonomic ranks scores ln 2 and enters
    divided by 6; the binary channel scores ln 2, so the concept loss is 2 ln 2 on both batches.
    """
    ln2, ln6 = math.log(2.0), math.log(6.0)
    meta = fixed_meta_model(
        class_logits=[0.0, math.log(3.0), 0.0, 0.0], training={"iterations_per_val_epoch": 2}
    )
    evaluator = Evaluator(
        num_classes=NUM_CLASSES, device="cpu", concept_value2id=CBM_CONCEPT_VALUE2ID
    )

    val_loss, metrics = meta.validation_epoch(_label_batches(1, 2), evaluator)

    assert val_loss == pytest.approx(((ln2 + 2 * ln2) + (ln6 + 2 * ln2)) / 2, rel=1e-5)
    assert metrics == {
        "accuracy": pytest.approx(0.5),
        # Uniform concept scores predict the first value of every rank and the unset binary
        # state, which matches the label-1 batch and misses the label-2 batch.
        **dict.fromkeys((f"accuracy/{rank}" for rank in TAXONOMIC_CONCEPTS), pytest.approx(0.5)),
        "accuracy/multi_hot": pytest.approx(0.5),
        "loss/classification": pytest.approx((ln2 + ln6) / 2, rel=1e-5),
        **dict.fromkeys(
            (f"loss/{rank}" for rank in TAXONOMIC_CONCEPTS), pytest.approx(ln2 / 6, rel=1e-5)
        ),
        "loss/multi_hot": pytest.approx(ln2, rel=1e-5),
        "loss/concepts": pytest.approx(2 * ln2, rel=1e-5),
    }


def test_train_epoch_masks_concept_targets(make_meta_model, monkeypatch, cbm_schema):
    meta = cbm_meta(make_meta_model, cbm_schema)
    meta.set_concept_source_mask(concept_keep(cbm_schema))
    bleached = cbm_schema.channel_names.index("bleached")
    seen: list[torch.Tensor] = []
    original = meta.batch_predict_loss

    def spy(images, source_labels, source_rows, *, return_maps):
        # Keep the maps so the step's concept targets can be read back.
        loss, components = original(images, source_labels, source_rows, return_maps=True)
        seen.append(meta._metric_shards[0].target_concepts.clone())
        return loss, components

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


def _meta_stepping(meta, grad: list[float], max_grad_norm: float) -> torch.nn.Parameter:
    """Make ``meta`` step one parameter [1, 1] holding ``grad`` with plain SGD at lr 1."""
    param = torch.nn.Parameter(torch.ones(2))
    param.grad = torch.tensor(grad)
    meta._trainable_params = [param]
    meta.optimizer = torch.optim.SGD([param], lr=1.0)
    meta.max_grad_norm = max_grad_norm
    return param


@pytest.mark.parametrize(
    ("max_grad_norm", "expected"),
    [
        pytest.param(1.0, [0.4, 0.2], id="norm_5_clipped_to_1"),
        pytest.param(10.0, [-2.0, -3.0], id="norm_below_the_limit_untouched"),
        pytest.param(0.0, [-2.0, -3.0], id="zero_disables_clipping"),
    ],
)
def test_apply_accumulated_step_clips_the_gradient_norm_then_steps(
    fixed_meta_model, max_grad_norm, expected
):
    meta = fixed_meta_model()
    param = _meta_stepping(meta, [3.0, 4.0], max_grad_norm)

    assert meta._apply_accumulated_step() is True

    # SGD from [1, 1]: the (3, 4) gradient has norm 5, so a limit of 1 scales it to (0.6, 0.8).
    assert param.tolist() == pytest.approx(expected)
    assert param.grad is None


@pytest.mark.parametrize("bad", [float("inf")])
def test_apply_accumulated_step_skips_a_non_finite_gradient_norm(fixed_meta_model, bad):
    meta = fixed_meta_model()
    param = _meta_stepping(meta, [bad, 1.0], max_grad_norm=1.0)

    assert meta._apply_accumulated_step() is False

    assert param.tolist() == [1.0, 1.0]
    assert param.grad is None


def test_resume_offset_past_the_epoch_end_raises(fixed_meta_model):
    meta = fixed_meta_model(training={"iterations_per_train_epoch": 4})
    meta.global_step = 5

    with pytest.raises(ValueError, match="resume offset"):
        meta.train_epoch(_label_batches(1), epoch=0)

    assert meta.global_step == 5


def test_resume_offset_before_the_epoch_start_warns_and_runs_the_whole_epoch(
    fixed_meta_model, caplog
):
    """Skipped windows leave ``global_step`` behind ``epoch * iterations``: offset -1 here."""
    meta = fixed_meta_model(training={"iterations_per_train_epoch": 4})
    meta.global_step = 3
    updates: list[int] = []

    with caplog.at_level("WARNING", logger=meta_module.logger.name):
        meta.train_epoch(
            _label_batches(1), epoch=1, on_optimizer_step=lambda m: updates.append(m.global_step)
        )

    assert updates == [4, 5, 6, 7]
    assert any("resume offset -1" in record.getMessage() for record in caplog.records)


@pytest.mark.parametrize(
    ("global_step", "expected_updates"),
    [
        pytest.param(0, 4, id="at_the_epoch_start_runs_the_whole_epoch"),
        pytest.param(4, 0, id="at_the_epoch_end_is_a_finished_epoch"),
    ],
)
def test_resume_offset_at_either_epoch_bound_runs_the_rest(
    fixed_meta_model, global_step, expected_updates
):
    meta = fixed_meta_model(training={"iterations_per_train_epoch": 4})
    meta.global_step = global_step
    updates: list[int] = []

    meta.train_epoch(
        _label_batches(1), epoch=0, on_optimizer_step=lambda m: updates.append(m.global_step)
    )

    assert updates == list(range(global_step + 1, global_step + 1 + expected_updates))


def test_warmup_raises_the_lr_before_the_decay_schedule_starts(fixed_meta_model):
    """Warmup 0.5 -> 1.0 over 2 updates, then decay 1.0 -> 0.5 over 2 updates, base lr 1.

    The first two updates only warm up (0.75, 1.0); decay starts on the update after, so
    the boundary update does not consume a decay step.
    """
    meta = fixed_meta_model(
        training={
            "iterations_per_train_epoch": 4,
            "optimizer": {"dpt_lr": 1.0},
            "scheduler": {
                "start_factor": 1.0,
                "end_factor": 0.5,
                "total_iters": 2,
                "warmup_iters": 2,
                "warmup_start_factor": 0.5,
            },
        }
    )
    lrs: list[float] = []

    meta.train_epoch(
        _label_batches(1),
        epoch=0,
        on_optimizer_step=lambda m: lrs.append(m.optimizer.param_groups[0]["lr"]),
    )

    assert lrs == pytest.approx([0.75, 1.0, 0.75, 0.5])


def test_epoch_means_average_the_applied_updates_only(fixed_meta_model):
    """Three updates, the middle one an empty batch: the means cover the two applied updates.

    Class logits (0, ln 3, 0, 0): a class-1 batch scores 3 ln 2 in total (classification ln 2),
    a class-2 batch ln 6 + 2 ln 2 (classification ln 6); every concept term is the same on both.
    """
    ln2, ln6 = math.log(2.0), math.log(6.0)
    meta = fixed_meta_model(
        class_logits=[0.0, math.log(3.0), 0.0, 0.0], training={"iterations_per_train_epoch": 3}
    )
    empty = (torch.zeros(0, 3, 1, 2), torch.zeros(0, 1, 2, dtype=torch.long))
    first, second = _label_batches(1, 2)

    last_loss, metrics, _ = meta.train_epoch([first, empty, second], epoch=0)

    # The first update moves the offset, which shifts the sigmoid concept channel slightly.
    assert last_loss == pytest.approx(((3 * ln2) + (ln6 + 2 * ln2)) / 2, rel=1e-4)
    assert metrics["loss/classification"] == pytest.approx((ln2 + ln6) / 2, rel=1e-4)
    assert metrics["loss/concepts"] == pytest.approx(2 * ln2, rel=1e-4)
    assert metrics["loss/multi_hot"] == pytest.approx(ln2, rel=1e-4)


def test_on_optimizer_step_fires_after_each_applied_update_and_not_for_a_skipped_window(
    fixed_meta_model,
):
    meta = fixed_meta_model(training={"iterations_per_train_epoch": 3})
    real = meta.batch_predict_loss
    calls = {"n": 0}

    def _poison_second(*args, **kwargs):
        calls["n"] += 1
        loss, components = real(*args, **kwargs)
        if calls["n"] == 2:
            loss = torch.tensor(float("nan"))
        return loss, components

    meta.batch_predict_loss = _poison_second
    seen: list[tuple[bool, int]] = []

    meta.train_epoch(
        _label_batches(1),
        epoch=0,
        on_optimizer_step=lambda m: seen.append((m is meta, m.global_step)),
    )

    # Updates 1 and 3 are applied; update 2 is dropped as non-finite.
    assert seen == [(True, 1), (True, 2)]


def test_a_non_finite_grad_norm_skips_the_update_and_its_callback(fixed_meta_model):
    meta = fixed_meta_model(training={"iterations_per_train_epoch": 2})
    bomb = torch.nn.Parameter(torch.ones(()))
    bomb.register_hook(lambda grad: grad * float("inf"))  # 0 * inf: a NaN gradient, finite loss
    meta._trainable_params.append(bomb)
    real = meta.batch_predict_loss
    calls = {"n": 0}

    def _first_window_gets_a_nan_gradient(*args, **kwargs):
        calls["n"] += 1
        loss, components = real(*args, **kwargs)
        if calls["n"] == 1:
            loss = loss + bomb * 0.0
        else:
            bomb.grad = None  # not an optimizer parameter, so nothing else clears it
        return loss, components

    meta.batch_predict_loss = _first_window_gets_a_nan_gradient
    seen: list[int] = []

    last_loss, _, _ = meta.train_epoch(
        _label_batches(1), epoch=0, on_optimizer_step=lambda m: seen.append(m.global_step)
    )

    assert seen == [1]
    assert meta.global_step == 1
    # The epoch mean covers the one applied window. Zero class logits give ln 4 = 2 ln 2 for
    # classification, plus 2 ln 2 for the concept terms.
    assert last_loss == pytest.approx(4 * math.log(2.0), rel=1e-4)


# MetaModel attribute -> the ShardStep attribute that holds it.
_STEP_PART_OF = {
    "model": "model",
    "loss": "loss_fn",
    "taxonomic_loss": "taxonomic_loss",
    "concept_source_keep": "concept_keep",
}


@pytest.mark.parametrize(
    "swapped",
    [
        pytest.param(lambda meta: ("model", copy.deepcopy(meta.model)), id="model"),
        pytest.param(lambda meta: ("loss", copy.deepcopy(meta.loss)), id="loss"),
        pytest.param(lambda meta: ("taxonomic_loss", torch.nn.Identity()), id="taxonomic_loss"),
        pytest.param(
            lambda meta: ("concept_source_keep", torch.ones(4, 13, dtype=torch.bool)),
            id="concept_source_keep",
        ),
    ],
)
def test_step_is_built_once_and_rebuilt_when_a_part_is_replaced(fixed_meta_model, swapped):
    meta = fixed_meta_model()
    first = meta.step
    assert meta.step is first

    name, replacement = swapped(meta)
    setattr(meta, name, replacement)
    rebuilt = meta.step

    assert rebuilt is not first
    assert getattr(rebuilt, _STEP_PART_OF[name]) is replacement
    assert meta.step is rebuilt


@pytest.mark.parametrize(
    "swapped",
    [
        pytest.param(lambda meta: ("model", copy.deepcopy(meta.model)), id="model"),
        pytest.param(lambda meta: ("loss", copy.deepcopy(meta.loss)), id="loss"),
        pytest.param(lambda meta: ("taxonomic_loss", torch.nn.Identity()), id="taxonomic_loss"),
        pytest.param(
            lambda meta: ("concept_source_keep", torch.ones(4, 13, dtype=torch.bool)),
            id="concept_source_keep",
        ),
    ],
)
def test_ddp_step_is_built_once_and_rebuilt_when_a_part_is_replaced(
    fixed_meta_model, monkeypatch, swapped
):
    # No process group: a stand-in DDP that only keeps the wrapped module is enough here.
    monkeypatch.setattr(meta_module, "is_distributed", lambda: True)
    monkeypatch.setattr(
        meta_module, "DistributedDataParallel", lambda module, **_: SimpleNamespace(module=module)
    )
    meta = fixed_meta_model(training={"distributed": True, "world_size": 1})
    first = meta._get_ddp_step()
    assert meta._get_ddp_step() is first

    name, replacement = swapped(meta)
    setattr(meta, name, replacement)
    rebuilt = meta._get_ddp_step()

    assert rebuilt is not first
    assert meta._get_ddp_step() is rebuilt


class _RecordingDDP:
    """Stand-in for ``DistributedDataParallel``: runs the wrapped step and records each forward
    as synced or inside ``no_sync``. Gradients are not reduced; ``LockstepRanks`` fakes the
    collectives that ``train_epoch`` makes itself."""

    def __init__(self, module: torch.nn.Module, **_: object) -> None:
        self.module = module
        self.forwards: list[str] = []
        self._syncing = True

    def __call__(self, *args: object, **kwargs: object) -> torch.Tensor:
        self.forwards.append("synced" if self._syncing else "no_sync")
        return self.module(*args, **kwargs)

    @contextlib.contextmanager
    def no_sync(self):
        self._syncing = False
        try:
            yield
        finally:
            self._syncing = True


def _two_rank_meta(fixed_meta_model, **training: object):
    return fixed_meta_model(training={"distributed": True, "world_size": 2, **training})


def test_a_non_finite_synced_shard_drops_the_window_on_every_rank(
    fixed_meta_model, lockstep_ranks, monkeypatch
):
    """Rank 1's loss is non-finite on window one's synced microbatch.

    Both ranks drop that window and finish its armed reduction; window two then trains with
    the same ``no_sync``-then-synced pattern, so only one update is applied.
    """
    monkeypatch.setattr(meta_module, "DistributedDataParallel", _RecordingDDP)

    def body(rank: int) -> dict[str, object]:
        meta = _two_rank_meta(fixed_meta_model, grad_accum_steps=2, iterations_per_train_epoch=2)
        finished: list[int] = []
        real_finish = meta._finish_skipped_reduction

        def finish() -> None:
            finished.append(meta.global_step)
            real_finish()

        meta._finish_skipped_reduction = finish
        real_forward = meta.batch_predict_loss
        calls = 0

        def forward(*args, **kwargs):
            nonlocal calls
            calls += 1
            loss, *rest = real_forward(*args, **kwargs)
            if rank == 1 and calls == 2:
                loss = loss * float("inf")
            return (loss, *rest)

        meta.batch_predict_loss = forward
        meta.train_epoch(_label_batches(1, 1, 1, 1), epoch=0)
        return {
            "global_step": meta.global_step,
            "finished": finished,
            "forwards": meta._ddp_step.forwards,
        }

    expected = {"global_step": 1, "finished": [0], "forwards": ["no_sync", "synced"] * 2}
    assert lockstep_ranks(2).run(body) == [expected, expected]


def test_every_rank_joins_the_log_reduction_when_only_rank_zero_has_a_logger(
    fixed_meta_model, lockstep_ranks, monkeypatch
):
    """Rank 1 has no logger but must still join ``reduce_components`` at each log step.

    If it did not, rank 0 would wait in the reduction alone; ``LockstepRanks`` raises instead.
    """
    monkeypatch.setattr(meta_module, "DistributedDataParallel", _RecordingDDP)

    def body(rank: int) -> list[int] | None:
        meta = _two_rank_meta(fixed_meta_model, grad_accum_steps=1, iterations_per_train_epoch=4)
        run_logger = _RecordingLogger() if rank == 0 else None
        meta.train_epoch(_label_batches(1, 1, 1, 1), run_logger=run_logger, log_interval=2, epoch=0)
        return None if run_logger is None else [step for _, step in run_logger.calls]

    assert lockstep_ranks(2).run(body) == [[2, 4], None]


class _RecordingMetric:
    def __init__(self) -> None:
        self.preds: list[list[int]] = []

    def update(self, preds: torch.Tensor, _target: torch.Tensor) -> None:
        self.preds.append(preds.flatten().tolist())


def test_consume_metric_shards_scores_each_shard_in_order_and_releases_its_maps(fixed_meta_model):
    metric = _RecordingMetric()
    evaluator = SimpleNamespace(
        device="cpu", metric_dict={"m": metric}, evaluate_concepts=lambda *_a: None
    )
    meta = fixed_meta_model()
    concepts = torch.zeros(1, CBM_NUM_CONCEPTS, 1, 2)
    # Class scores per pixel: shard one predicts [1, 2], shard two [0, 0].
    shard_one = meta_module.MetricShard(
        outputs=torch.tensor([[[[0.0, 0.0]], [[1.0, 0.0]], [[0.0, 1.0]]]]),
        concept_outputs=concepts,
        target_labels=torch.ones(1, 1, 2, dtype=torch.long),
        target_concepts=concepts,
    )
    shard_two = meta_module.MetricShard(
        outputs=torch.tensor([[[[1.0, 1.0]], [[0.0, 0.0]], [[0.0, 0.0]]]]),
        concept_outputs=concepts,
        target_labels=torch.ones(1, 1, 2, dtype=torch.long),
        target_concepts=concepts,
    )
    meta._metric_shards = [shard_one, shard_two]

    assert meta._consume_metric_shards(evaluator) is True

    assert metric.preds == [[1, 2], [0, 0]]
    assert meta._metric_shards is None
    for shard in (shard_one, shard_two):
        assert shard.outputs is None
        assert shard.target_labels is None


@pytest.mark.parametrize("shards", [None], ids=["none"])
def test_consume_metric_shards_reports_a_non_distributed_step(fixed_meta_model, shards):
    meta = fixed_meta_model()
    meta._metric_shards = shards

    assert meta._consume_metric_shards(SimpleNamespace()) is False
    assert meta._metric_shards is None


@pytest.mark.parametrize("name", ["BF-16"])
def test_resolve_amp_dtype_normalises_case_and_separators(name):
    assert meta_module.resolve_amp_dtype(name) is torch.bfloat16


@pytest.mark.parametrize("dtype", [torch.bfloat16])
def test_resolve_amp_dtype_passes_a_torch_dtype_through(dtype):
    assert meta_module.resolve_amp_dtype(dtype) is dtype
