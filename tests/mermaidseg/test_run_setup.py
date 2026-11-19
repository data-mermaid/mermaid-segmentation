"""Run-setup helpers: checkpoint reads, weight loading and the resolved config."""

from __future__ import annotations

import pickle
from pathlib import Path
from types import SimpleNamespace

import albumentations as A
import pytest
import torch
import yaml

from mermaidseg.io import ConfigDict
from mermaidseg.run_setup import (
    MODEL_CHECKPOINT_KEYS,
    build_registry,
    concept_removal_list,
    load_model_weights,
    load_run_concept_source_mask,
    read_checkpoint,
    target_labels,
    write_resolved_config,
)


def test_resolved_config_yaml_holds_the_overridden_values(tmp_path):
    cfg = ConfigDict(
        {
            "run_name": "r1",
            "training": {"grad_accum_steps": 4, "concept_mapping_path": "/abs/map.csv"},
            "model": {"input_size": (56, 56)},
        }
    )
    write_resolved_config(cfg, tmp_path)
    assert yaml.safe_load((tmp_path / "config.yaml").read_text()) == {
        "run_name": "r1",
        "training": {"grad_accum_steps": 4, "concept_mapping_path": "/abs/map.csv"},
        "model": {"input_size": [56, 56]},
    }


def _save_checkpoint(path: Path, state_dict: dict, config: dict | None = None) -> Path:
    torch.save({"config": config or {}, "model_state_dict": state_dict}, path)
    return path


def test_load_model_weights_copies_the_saved_weights(tmp_path):
    source = torch.nn.Linear(2, 2)
    path = _save_checkpoint(tmp_path / "ckpt.pt", source.state_dict(), {"seed": 3})
    target = torch.nn.Linear(2, 2)
    load_model_weights(
        SimpleNamespace(model=target),  # type: ignore[arg-type]
        read_checkpoint(path, MODEL_CHECKPOINT_KEYS),
    )
    torch.testing.assert_close(target.weight, source.weight)


def test_load_model_weights_rejects_a_state_dict_missing_a_model_key(tmp_path):
    state = torch.nn.Linear(2, 2).state_dict()
    del state["bias"]
    checkpoint = read_checkpoint(
        _save_checkpoint(tmp_path / "ckpt.pt", state), ("model_state_dict",)
    )
    with pytest.raises(RuntimeError, match="Missing key"):
        load_model_weights(SimpleNamespace(model=torch.nn.Linear(2, 2)), checkpoint)  # type: ignore[arg-type]


def test_read_checkpoint_refuses_a_checkpoint_that_pickles_a_compose_naming_the_file(tmp_path):
    path = _save_checkpoint(
        tmp_path / "ckpt.pt",
        torch.nn.Linear(2, 2).state_dict(),
        {"transform": A.Compose([A.HorizontalFlip()])},
    )
    with pytest.raises(RuntimeError, match="ckpt.pt") as excinfo:
        read_checkpoint(path, MODEL_CHECKPOINT_KEYS)
    assert isinstance(excinfo.value.__cause__, pickle.UnpicklingError)


def test_read_checkpoint_names_the_file_and_the_missing_key(tmp_path):
    path = tmp_path / "ckpt.pt"
    torch.save({"config": {}}, path)
    with pytest.raises(RuntimeError, match="ckpt.pt.*model_state_dict"):
        read_checkpoint(path, MODEL_CHECKPOINT_KEYS)


@pytest.mark.parametrize("path", [None, ""])
def test_concept_removal_list_is_empty_without_a_removal_file(path):
    assert concept_removal_list({"concept_removal": path}, "unused.csv") == []


@pytest.mark.parametrize("path", [None, ""])
def test_no_concept_source_mask_path_loads_no_mask(path):
    training_cfg = ConfigDict({"concept_source_mask": path})
    assert load_run_concept_source_mask(training_cfg, schema=None) is None  # type: ignore[arg-type]


def test_target_labels_are_lower_cased():
    assert target_labels(ConfigDict({"class_subset": ["Sand", "ACROPORA"]})) == [
        "acropora",
        "sand",
    ]


def test_target_labels_drop_duplicates_that_differ_only_in_case():
    assert target_labels(ConfigDict({"class_subset": ["Sand", "sand", "SAND"]})) == ["sand"]


def test_target_labels_are_sorted_whatever_the_config_order():
    assert target_labels(ConfigDict({"class_subset": ["sand", "porites", "acropora"]})) == [
        "acropora",
        "porites",
        "sand",
    ]


# ---------- build_registry ----------


class _SourceDataset:
    """A dataset with a labelled source vocabulary, as ``build_registry`` wires it."""

    def __init__(self, source_name: str, names: list[str]) -> None:
        self.SOURCE_NAME = source_name
        self.source_id2name = dict(enumerate(names, start=1))
        self.source_name2id = {name: i for i, name in self.source_id2name.items()}
        self.num_source_classes = len(names) + 1
        self.global_offset: int | None = None

    def set_global_offset(self, offset: int) -> None:
        self.global_offset = offset


def _schema_config(tmp_path: Path, *sources: str) -> Path:
    path = tmp_path / "schema.yaml"
    path.write_text(
        yaml.safe_dump(
            {
                "data": {
                    "local_cache_write_through": True,
                    **{name: {"train": {"split": "train"}} for name in sources},
                }
            }
        )
    )
    return path


def _registry_training_cfg(**overrides) -> ConfigDict:
    return ConfigDict(
        {
            "concept_mapping_path": "configs/class_to_concepts.csv",
            "concept_removal": None,
            "class_subset": ["Pocillopora", "Sand"],
            "training_mode": "concept-bottleneck",
            "label_roll_up": False,
            **overrides,
        }
    )


def _coralscapes_dataset_dict() -> dict[tuple[str, str], _SourceDataset]:
    names = ["pocillopora alive", "sand"]
    return {("coralscapes_v2", "train"): _SourceDataset("coralscapes_v2", names)}


def test_build_registry_freezes_concepts_to_the_schema_and_wires_the_datasets(
    tmp_path, monkeypatch
):
    monkeypatch.chdir(tmp_path)
    dataset_dict = _coralscapes_dataset_dict()
    registry, schema = build_registry(
        dataset_dict,
        _registry_training_cfg(),
        schema_config=_schema_config(tmp_path, "coralscapes_v2"),
    )
    assert registry.target_id2label == {1: "pocillopora", 2: "sand"}
    assert schema.num_channels > 0
    assert registry.num_concepts == schema.num_channels
    assert dataset_dict[("coralscapes_v2", "train")].global_offset == 0


def test_build_registry_exits_naming_a_dataset_missing_from_the_concept_schema(tmp_path):
    dataset_dict = {("coralnet", "train"): _SourceDataset("coralnet", ["sand"])}
    with pytest.raises(SystemExit, match=r"\['coralnet'\].*missing from the frozen concept schema"):
        build_registry(
            dataset_dict,
            _registry_training_cfg(),
            schema_config=_schema_config(tmp_path, "coralscapes_v2"),
        )
    assert dataset_dict[("coralnet", "train")].global_offset is None


def test_build_registry_raises_when_the_registry_has_fewer_concepts_than_the_schema(
    tmp_path, monkeypatch
):
    monkeypatch.chdir(tmp_path)
    # Standard mode builds no concept lookup, so the registry has 0 channels against the schema's.
    with pytest.raises(RuntimeError, match=r"registry has 0 concept channels, schema \d+"):
        build_registry(
            _coralscapes_dataset_dict(),
            _registry_training_cfg(training_mode="standard"),
            schema_config=_schema_config(tmp_path, "coralscapes_v2"),
        )
