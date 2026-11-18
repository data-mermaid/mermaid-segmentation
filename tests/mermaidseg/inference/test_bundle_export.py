"""Exporting a trained checkpoint into a self-describing bundle and loading it offline."""

from __future__ import annotations

import json
import os
import shutil
from pathlib import Path
from unittest.mock import patch

import cv2
import numpy as np
import pytest
import torch
from safetensors.torch import load_file, save_file

from mermaidseg.export import _untied_cpu_state, export_bundle
from mermaidseg.inference.bundle import MODEL_JSON, WEIGHTS, load_bundle
from mermaidseg.inference.predictor import CBMPredictor
from tests.mermaidseg.inference import _tiny


def test_round_trip_reproduces_the_trained_model_outputs(run, bundle_dir):
    trained = run[1]
    x = torch.randn(1, 3, 64, 64, generator=torch.Generator().manual_seed(1))
    with torch.no_grad():
        expected = trained(x)
        actual = load_bundle(bundle_dir, "cpu").model(x)
    torch.testing.assert_close(actual.logits, expected.logits, atol=1e-5, rtol=0)
    torch.testing.assert_close(actual.hidden_states, expected.hidden_states, atol=1e-5, rtol=0)


def test_loaded_bundle_reads_the_binary_concepts_bool_keys_as_strings(bundle_dir):
    loaded = load_bundle(bundle_dir, "cpu").model
    assert loaded.concept_value2id["bleached"] == {"not_given": 0, "False": 1, "True": 2}


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


def test_exported_lineage_is_the_checkpoints_not_the_sidecars(tmp_path):
    own = {**_tiny.resolved_config()}
    lineage = {"git_sha": "from-checkpoint", "git_dirty": False, "resolved_config": own}
    checkpoint, _ = _tiny.write_run(tmp_path / "run", checkpoint_lineage=lineage)
    export_bundle(checkpoint, tmp_path / "out")
    exported = json.loads((tmp_path / "out" / MODEL_JSON).read_text())
    sidecar = json.loads((checkpoint.parent / MODEL_JSON).read_text())
    assert sidecar["lineage"]["git_sha"] == "abc123"
    assert exported["lineage"] == lineage


def test_export_refuses_a_dirty_checkpoint_after_the_sidecar_was_refreshed_clean(tmp_path):
    checkpoint, _ = _tiny.write_run(tmp_path / "run", git_dirty=True)
    sidecar_path = checkpoint.parent / MODEL_JSON
    sidecar = json.loads(sidecar_path.read_text())
    sidecar["lineage"]["git_dirty"] = False
    sidecar_path.write_text(json.dumps(sidecar))
    with pytest.raises(ValueError, match="clean tree"):
        export_bundle(checkpoint, tmp_path / "out")
    assert not (tmp_path / "out").exists()


def test_export_of_a_checkpoint_without_lineage_raises_naming_its_path(tmp_path):
    checkpoint, _ = _tiny.write_run(tmp_path / "run")
    contents = torch.load(checkpoint, map_location="cpu", weights_only=True)
    del contents["lineage"]
    torch.save(contents, checkpoint)
    with pytest.raises(RuntimeError, match="lineage") as excinfo:
        export_bundle(checkpoint, tmp_path / "out")
    assert str(checkpoint) in str(excinfo.value)


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


_MISSING = object()


@pytest.mark.parametrize(
    "weights", ["../weights.safetensors", "/etc/weights.safetensors", _MISSING]
)
def test_load_refuses_a_weights_path_other_than_the_bundle_weights(bundle_dir, tmp_path, weights):
    foreign = tmp_path / "foreign"
    foreign.mkdir()
    metadata = json.loads((bundle_dir / MODEL_JSON).read_text())
    if weights is _MISSING:
        del metadata["weights"]
    else:
        metadata["weights"] = weights
    (foreign / MODEL_JSON).write_text(json.dumps(metadata))
    with pytest.raises(ValueError, match="weights"):
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


def test_export_refuses_a_checkpoint_whose_config_differs_from_its_lineage_naming_the_key(
    tmp_path,
):
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


def test_export_of_a_checkpoint_missing_the_weights_raises_naming_its_path(tmp_path):
    checkpoint, _ = _tiny.write_run(tmp_path / "run")
    contents = torch.load(checkpoint, map_location="cpu", weights_only=True)
    del contents["model_state_dict"]
    torch.save(contents, checkpoint)
    with pytest.raises(RuntimeError, match="model_state_dict") as excinfo:
        export_bundle(checkpoint, tmp_path / "out")
    assert str(checkpoint) in str(excinfo.value)
    assert not (tmp_path / "out").exists()


def test_a_failed_write_leaves_neither_output_nor_partial_directory(run, tmp_path):
    out = tmp_path / "out"
    with (
        patch("mermaidseg.export.save_file", side_effect=OSError("disk full")),
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
        patch("mermaidseg.export.save_file", side_effect=OSError("disk full")),
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


def test_predictor_preprocesses_with_the_bundles_own_statistics(tmp_path):
    checkpoint, _ = _tiny.write_run(tmp_path / "run", mean=[0.5] * 3, std=[0.25] * 3)
    bundle = load_bundle(export_bundle(checkpoint, tmp_path / "bundle"), "cpu")
    predictor = CBMPredictor.from_bundle(bundle, amp_dtype=None)
    image = np.random.default_rng(0).integers(0, 256, (40, 50, 3), dtype=np.uint8)
    resized = cv2.resize(image, (64, 64), interpolation=cv2.INTER_LINEAR)
    expected = (resized.astype(np.float32) / 255 - 0.5) / 0.25
    np.testing.assert_allclose(
        predictor.preprocess(image).numpy(), expected.transpose(2, 0, 1), atol=1e-5
    )


@pytest.mark.parametrize(
    ("key", "foreign"), [("channel_order", "BGR"), ("interpolation", "nearest")]
)
def test_load_rejects_a_foreign_preprocessing_key(bundle_dir, tmp_path, key, foreign):
    copied = tmp_path / "copy"
    shutil.copytree(bundle_dir, copied)
    metadata = json.loads((copied / MODEL_JSON).read_text())
    metadata["preprocessing"][key] = foreign
    (copied / MODEL_JSON).write_text(json.dumps(metadata))
    with pytest.raises(ValueError, match=key):
        load_bundle(copied, "cpu")
