"""Exporting a trained checkpoint into a self-describing bundle and loading it offline."""

from __future__ import annotations

import json
import os
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pytest
import torch
from safetensors.torch import load_file, save_file

from mermaidseg.inference.bundle import (
    MODEL_JSON,
    WEIGHTS,
    _untied_cpu_state,
    export_bundle,
    load_bundle,
)
from mermaidseg.inference.predictor import CBMPredictor
from tests.mermaidseg.inference import _tiny


@pytest.fixture(scope="module")
def run(tmp_path_factory: pytest.TempPathFactory) -> tuple[Path, torch.nn.Module]:
    return _tiny.write_run(tmp_path_factory.mktemp("run"))


@pytest.fixture(scope="module")
def bundle_dir(run: tuple[Path, torch.nn.Module], tmp_path_factory: pytest.TempPathFactory) -> Path:
    return export_bundle(run[0], tmp_path_factory.mktemp("out") / "bundle")


def test_round_trip_reproduces_the_trained_model_outputs(run, bundle_dir):
    trained = run[1]
    x = torch.randn(1, 3, 64, 64, generator=torch.Generator().manual_seed(1))
    with torch.no_grad():
        expected = trained(x)
        actual = load_bundle(bundle_dir, "cpu").model(x)
    torch.testing.assert_close(actual.logits, expected.logits, atol=1e-5, rtol=0)
    torch.testing.assert_close(actual.hidden_states, expected.hidden_states, atol=1e-5, rtol=0)


def test_bundle_directory_holds_only_the_sidecar_and_weights(bundle_dir):
    assert sorted(p.name for p in bundle_dir.iterdir()) == [MODEL_JSON, WEIGHTS]
    assert [p.name for p in bundle_dir.parent.iterdir()] == [bundle_dir.name]


def test_exported_metadata_extends_the_sidecar_with_weights_and_checkpoint(run, bundle_dir):
    sidecar = json.loads((run[0].parent / MODEL_JSON).read_text())
    exported = json.loads((bundle_dir / MODEL_JSON).read_text())
    assert exported == {
        **sidecar,
        "weights": "weights.safetensors",
        "checkpoint": {"epoch": 3, "global_step": 42},
    }


def test_loaded_bundle_exposes_the_sidecar_contract(bundle_dir):
    bundle = load_bundle(bundle_dir, "cpu")
    assert bundle.path == bundle_dir
    assert bundle.id2label == {0: "ignore", 1: "acropora", 2: "porites", 3: "sand", 4: "algae"}
    assert bundle.concept_id2name == _tiny.concept_id2name()
    assert bundle.input_size == (64, 64)
    assert bundle.mean == (0.485, 0.456, 0.406)
    assert bundle.std == (0.229, 0.224, 0.225)
    assert bundle.config == _tiny.resolved_config()


def test_load_never_calls_the_hub(bundle_dir):
    with patch("transformers.AutoModel.from_pretrained", side_effect=AssertionError("hub")):
        load_bundle(bundle_dir, "cpu")


def test_load_refuses_weights_missing_a_key(bundle_dir, tmp_path):
    broken = tmp_path / "broken"
    broken.mkdir()
    (broken / MODEL_JSON).write_text((bundle_dir / MODEL_JSON).read_text())
    weights = load_file(bundle_dir / WEIGHTS)
    del weights["concept_proj.bias"]
    save_file(weights, broken / WEIGHTS)
    with pytest.raises(RuntimeError, match="concept_proj.bias"):
        load_bundle(broken, "cpu")


@pytest.mark.parametrize(("key", "value"), [("schema_version", 99), ("task", "other")])
def test_load_refuses_a_foreign_schema_version_or_task(bundle_dir, tmp_path, key, value):
    foreign = tmp_path / "foreign"
    foreign.mkdir()
    metadata = json.loads((bundle_dir / MODEL_JSON).read_text())
    (foreign / MODEL_JSON).write_text(json.dumps({**metadata, key: value}))
    with pytest.raises(ValueError, match=key):
        load_bundle(foreign, "cpu")


@pytest.mark.parametrize("git_dirty", [True, None])
def test_export_refuses_a_run_from_a_dirty_or_unknown_tree(tmp_path, git_dirty):
    checkpoint, _ = _tiny.write_run(tmp_path / "run", git_dirty=git_dirty)
    with pytest.raises(ValueError, match="clean tree"):
        export_bundle(checkpoint, tmp_path / "out")
    assert not (tmp_path / "out").exists()


def test_export_refuses_a_non_empty_output_directory(run, tmp_path):
    out = tmp_path / "out"
    out.mkdir()
    (out / "keep.txt").write_text("x")
    with pytest.raises(ValueError, match="not empty"):
        export_bundle(run[0], out)
    assert [p.name for p in out.iterdir()] == ["keep.txt"]


def test_export_accepts_an_existing_empty_output_directory(run, tmp_path):
    out = tmp_path / "out"
    out.mkdir()
    export_bundle(run[0], out)
    assert sorted(p.name for p in out.iterdir()) == sorted([MODEL_JSON, WEIGHTS])


def test_export_refuses_a_checkpoint_whose_config_differs_naming_the_key(tmp_path):
    other = {**_tiny.resolved_config(), "seed": 1}
    checkpoint, _ = _tiny.write_run(tmp_path / "run", config=other)
    with pytest.raises(ValueError, match="seed"):
        export_bundle(checkpoint, tmp_path / "out")
    assert not (tmp_path / "out").exists()


def test_export_without_a_sidecar_raises_naming_its_path(run, tmp_path):
    lone = tmp_path / "lone"
    lone.mkdir()
    checkpoint = lone / "model_step1"
    checkpoint.write_bytes(run[0].read_bytes())
    with pytest.raises(FileNotFoundError, match=str(lone / MODEL_JSON)):
        export_bundle(checkpoint, tmp_path / "out")


def test_a_failed_write_leaves_neither_output_nor_partial_directory(run, tmp_path):
    out = tmp_path / "out"
    with (
        patch("mermaidseg.inference.bundle.save_file", side_effect=OSError("disk full")),
        pytest.raises(OSError, match="disk full"),
    ):
        export_bundle(run[0], out)
    assert not out.exists()
    assert not list(tmp_path.glob("*.partial"))


def _precious_partial(out: Path) -> Path:
    foreign = out.with_name(out.name + ".partial")
    foreign.mkdir()
    (foreign / "precious").write_text("keep")
    return foreign


def test_a_preexisting_partial_sibling_survives_a_failed_export(run, tmp_path):
    out = tmp_path / "out"
    foreign = _precious_partial(out)
    with (
        patch("mermaidseg.inference.bundle.save_file", side_effect=OSError("disk full")),
        pytest.raises(OSError, match="disk full"),
    ):
        export_bundle(run[0], out)
    assert [p.name for p in foreign.iterdir()] == ["precious"]
    assert (foreign / "precious").read_text() == "keep"
    assert [p for p in tmp_path.glob("*.partial") if p != foreign] == []
    assert not out.exists()


def test_export_requires_the_output_parent_to_exist_and_creates_nothing(run, tmp_path):
    with pytest.raises(FileNotFoundError, match="missing"):
        export_bundle(run[0], tmp_path / "missing" / "out")
    assert list(tmp_path.iterdir()) == []


def test_untied_state_shares_no_storage_and_preserves_values():
    shared = torch.arange(6, dtype=torch.float32)
    state = {"a": shared, "b": shared[:3]}
    untied = _untied_cpu_state(state)
    assert untied["a"].untyped_storage().data_ptr() != untied["b"].untyped_storage().data_ptr()
    torch.testing.assert_close(untied["a"], shared)
    torch.testing.assert_close(untied["b"], shared[:3])


def test_untied_state_round_trips_through_save_file(tmp_path):
    shared = torch.arange(6, dtype=torch.float32)
    untied = _untied_cpu_state({"a": shared, "b": shared[:3]})
    save_file(untied, tmp_path / "w.safetensors")
    loaded = load_file(tmp_path / "w.safetensors")
    torch.testing.assert_close(loaded["a"], shared)
    torch.testing.assert_close(loaded["b"], shared[:3])


@pytest.mark.parametrize(("umask", "mode"), [(0o022, 0o755), (0o002, 0o775)])
def test_exported_bundle_directory_has_the_mode_a_plain_mkdir_gives(run, tmp_path, umask, mode):
    previous = os.umask(umask)
    try:
        out = export_bundle(run[0], tmp_path / "out")
    finally:
        os.umask(previous)
    assert out.stat().st_mode & 0o777 == mode


def test_predictor_from_bundle_returns_per_point_class_and_concept_probabilities(bundle_dir):
    predictor = CBMPredictor.from_bundle(load_bundle(bundle_dir, "cpu"), amp_dtype=None)
    image = np.random.default_rng(0).integers(0, 255, (40, 50, 3), dtype=np.uint8)
    class_probs, concept_probs = predictor.predict_points(image, [[5, 5], [30, 40], [10, 20]])
    assert class_probs.shape == (3, len(_tiny.CLASS_NAMES))
    assert concept_probs.shape == (3, len(_tiny.CONCEPT_NAMES))
    torch.testing.assert_close(class_probs.sum(dim=1), torch.ones(3), atol=1e-5, rtol=0)
