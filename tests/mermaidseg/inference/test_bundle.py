"""The sidecar ``model.json`` a run writes at start."""

from __future__ import annotations

import copy
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from mermaidseg import export
from mermaidseg.export import write_sidecar
from mermaidseg.inference.bundle import MODEL_JSON
from mermaidseg.training import _encoder_config


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
        ({"encoder_config": {"hidden_size": 32}}, "architecture"),
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


def _model_with_encoder_config(config: dict):
    base = SimpleNamespace(config=SimpleNamespace(to_dict=lambda: dict(config)))
    return SimpleNamespace(encoder=SimpleNamespace(get_base_model=lambda: base))


def test_stored_encoder_config_drops_environment_keys_so_a_resume_passes(tmp_path: Path):
    arch = {"hidden_size": 16, "num_hidden_layers": 4}
    old = _model_with_encoder_config(
        {**arch, "transformers_version": "4.50.0", "_name_or_path": "/cache/a"}
    )
    new = _model_with_encoder_config(
        {**arch, "transformers_version": "4.57.0", "_name_or_path": "/cache/b"}
    )
    assert _encoder_config(old) == arch

    write_sidecar(tmp_path, **_sidecar_kwargs(encoder_config=_encoder_config(old)))
    write_sidecar(tmp_path, **_sidecar_kwargs(encoder_config=_encoder_config(new)))
    assert _read(tmp_path / MODEL_JSON)["architecture"]["encoder_config"] == arch
