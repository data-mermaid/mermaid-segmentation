"""The sidecar ``model.json`` a run writes at start, and the bundle exported from a run."""

from __future__ import annotations

import copy
import hashlib
import json
import os
from pathlib import Path
from unittest.mock import patch

import pytest
import torch
from safetensors.torch import load_file, save_file

from mermaidseg import export
from mermaidseg.export import _untied_cpu_state, export_bundle, write_sidecar
from mermaidseg.inference.bundle import MODEL_JSON, WEIGHTS
from tests.mermaidseg.inference import _tiny


def _sidecar_kwargs(**overrides):
    kwargs = {
        "run_name": "run-a",
        "model_class": "ConceptBottleneckDPTLoRADINOv3",
        "model_kwargs": {
            "encoder_name": "enc",
            "input_size": (56, 56),
            "num_classes": 3,
            "concept_value2id": {"order": {"not_given": [0, 0]}, "bleached": {False: 1}},
        },
        "encoder_config": {"hidden_size": 16},
        "id2label": {0: "ignore", 1: "coral"},
        "concept_id2name": {0: "a", 1: "b"},
        "input_size": (56, 56),
        "mean": (0.5, 0.5, 0.5),
        "std": (0.25, 0.25, 0.25),
        "lineage": {"seed": 7},
    }
    return {**kwargs, **overrides}


def _read(path: Path) -> dict:
    return json.loads(path.read_text())


def test_sidecar_records_encoder_name_outside_kwargs_without_mutating_the_caller(tmp_path: Path):
    kwargs = _sidecar_kwargs()
    before = copy.deepcopy(kwargs["model_kwargs"])

    architecture = _read(write_sidecar(tmp_path, **kwargs))["architecture"]

    assert architecture["encoder_name"] == "enc"
    assert "encoder_name" not in architecture["kwargs"]
    assert kwargs["model_kwargs"] == before


def test_sidecar_layout(tmp_path: Path):
    path = write_sidecar(tmp_path, **_sidecar_kwargs())
    assert path == tmp_path / MODEL_JSON
    assert _read(path) == {
        "schema_version": 1,
        "task": "segmentation_cbm",
        "run_name": "run-a",
        "architecture": {
            "class": "ConceptBottleneckDPTLoRADINOv3",
            "encoder_name": "enc",
            "kwargs": {
                "input_size": [56, 56],
                "num_classes": 3,
                "concept_value2id": {"order": {"not_given": [0, 0]}, "bleached": {"False": 1}},
            },
            "encoder_config": {"hidden_size": 16},
        },
        "id2label": {"0": "ignore", "1": "coral"},
        "concept_id2name": {"0": "a", "1": "b"},
        "preprocessing": {
            "size_hw": [56, 56],
            "mean": [0.5, 0.5, 0.5],
            "std": [0.25, 0.25, 0.25],
            "channel_order": "RGB",
            "interpolation": "bilinear",
        },
        "lineage": {"seed": 7},
    }


def test_resume_with_the_same_contract_refreshes_the_lineage(tmp_path: Path):
    write_sidecar(tmp_path, **_sidecar_kwargs())
    first = _read(tmp_path / MODEL_JSON)
    write_sidecar(tmp_path, **_sidecar_kwargs(lineage={"seed": 7, "git_sha": "new"}))
    second = _read(tmp_path / MODEL_JSON)
    assert second["lineage"] == {"seed": 7, "git_sha": "new"}
    assert {k: v for k, v in second.items() if k != "lineage"} == {
        k: v for k, v in first.items() if k != "lineage"
    }


@pytest.mark.parametrize(
    ("override", "key"),
    [
        ({"id2label": {0: "ignore", 1: "sand"}}, "id2label"),
        ({"concept_id2name": {0: "z"}}, "concept_id2name"),
        ({"model_class": "Other"}, "architecture"),
        ({"mean": (0.1, 0.1, 0.1)}, "preprocessing"),
    ],
)
def test_resume_with_a_different_contract_names_the_differing_key(tmp_path, override, key):
    write_sidecar(tmp_path, **_sidecar_kwargs())
    before = (tmp_path / MODEL_JSON).read_text()
    with pytest.raises(ValueError, match=key):
        write_sidecar(tmp_path, **_sidecar_kwargs(**override))
    assert (tmp_path / MODEL_JSON).read_text() == before


def test_a_failed_replace_leaves_the_old_file_intact(monkeypatch, tmp_path: Path):
    write_sidecar(tmp_path, **_sidecar_kwargs())
    before = (tmp_path / MODEL_JSON).read_text()

    def broken_replace(*_args):
        raise OSError("disk gone")

    monkeypatch.setattr(export.os, "replace", broken_replace)
    with pytest.raises(OSError, match="disk gone"):
        write_sidecar(tmp_path, **_sidecar_kwargs(lineage={"seed": 99}))
    assert (tmp_path / MODEL_JSON).read_text() == before
    assert sorted(p.name for p in tmp_path.iterdir()) == [MODEL_JSON]


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
        "weights_sha256": hashlib.sha256((bundle_dir / WEIGHTS).read_bytes()).hexdigest(),
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


def test_export_refuses_a_run_from_an_unknown_tree(tmp_path):
    checkpoint, _ = _tiny.write_run(tmp_path / "run", git_dirty=None)
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


@pytest.mark.parametrize(
    ("config", "checkpoint_lineage"),
    [
        pytest.param(["not", "a", "mapping"], None, id="checkpoint-config-is-not-a-mapping"),
        pytest.param(None, {"git_dirty": False}, id="lineage-has-no-resolved-config"),
        pytest.param(
            None,
            {"git_dirty": False, "resolved_config": "not a mapping"},
            id="lineage-resolved-config-is-not-a-mapping",
        ),
    ],
)
def test_export_refuses_configs_that_are_not_both_mappings_without_naming_a_key(
    tmp_path, config, checkpoint_lineage
):
    checkpoint, _ = _tiny.write_run(
        tmp_path / "run", config=config, checkpoint_lineage=checkpoint_lineage
    )
    with pytest.raises(ValueError, match="does not match its lineage's resolved_config") as raised:
        export_bundle(checkpoint, tmp_path / "out")
    assert "at key" not in str(raised.value)
    assert not (tmp_path / "out").exists()


@pytest.mark.parametrize("key", ["config", "epoch", "global_step"])
def test_export_of_a_checkpoint_missing_a_required_key_names_the_key_and_its_path(tmp_path, key):
    checkpoint, _ = _tiny.write_run(tmp_path / "run")
    contents = torch.load(checkpoint, map_location="cpu", weights_only=True)
    del contents[key]
    torch.save(contents, checkpoint)
    with pytest.raises(RuntimeError, match=key) as raised:
        export_bundle(checkpoint, tmp_path / "out")
    assert str(checkpoint) in str(raised.value)
    assert not (tmp_path / "out").exists()


def test_export_refuses_an_output_path_that_is_an_existing_file(run, tmp_path):
    out = tmp_path / "out"
    out.write_text("not a directory")
    with pytest.raises(ValueError, match="not empty"):
        export_bundle(run[0], out)
    assert out.read_text() == "not a directory"


def test_export_without_a_sidecar_raises_naming_its_path(run, tmp_path):
    lone = tmp_path / "lone"
    lone.mkdir()
    checkpoint = lone / "model_step1"
    checkpoint.write_bytes(run[0].read_bytes())
    with pytest.raises(FileNotFoundError, match=str(lone / MODEL_JSON)):
        export_bundle(checkpoint, tmp_path / "out")


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
