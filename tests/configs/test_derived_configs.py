"""The derived configs a committed training config names were built from the files and pins it reads."""

from __future__ import annotations

import hashlib
from pathlib import Path

import pytest
import yaml

from mermaidseg.dataset_reconciliation.artifact_meta import read_sidecar
from mermaidseg.dataset_reconciliation.label_space import label_keys_sha1, read_label_space_meta
from mermaidseg.io import ConfigDict

_REPO = Path(__file__).resolve().parents[2]
_DERIVED_KEYS = ("concept_removal", "concept_source_mask")
_DATA_CONFIG = _REPO / "configs" / "data_config.yaml"
_TRAINING_CONFIGS = sorted((_REPO / "configs").glob("training_config*.yaml"))


def _training_section(config: Path) -> dict:
    return yaml.safe_load(config.read_text())["training"]


_LABEL_SPACE_CONFIGS = [c for c in _TRAINING_CONFIGS if _training_section(c).get("label_space")]
# training key -> label space meta key of the sha1 of the file it names.
_LABEL_SPACE_FILES = {
    "concept_mapping_path": "concepts_csv_sha1",
    "concept_removal": "concept_removal_sha1",
    "concept_source_mask": "concept_source_mask_sha1",
}


def _derived_csvs(config: Path) -> list[Path]:
    section = _training_section(config)
    return [_REPO / section[key] for key in _DERIVED_KEYS if section.get(key)]


def test_a_committed_training_config_names_a_derived_csv():
    assert any(_derived_csvs(config) for config in _TRAINING_CONFIGS)


@pytest.mark.parametrize("config", _TRAINING_CONFIGS, ids=lambda path: path.name)
def test_the_derived_csvs_were_built_against_the_configs_concept_table(config):
    table_sha1 = hashlib.sha1(
        (_REPO / _training_section(config)["concept_mapping_path"]).read_bytes()
    ).hexdigest()

    for csv in _derived_csvs(config):
        assert read_sidecar(csv, artifact="derived config")["concepts_csv_sha1"] == table_sha1, (
            csv.name
        )


def _file_sha1_or_none(value: str) -> str | None:
    return hashlib.sha1((_REPO / value).read_bytes()).hexdigest() if value else None


def _label_space_meta(config: Path) -> dict:
    return read_label_space_meta(_REPO / _training_section(config)["label_space"])


def _pins(data_section: dict) -> dict:
    """``{dataset: (version, manifest_sha256), "taxonomy": (...)}`` of a data section."""
    entries = {**data_section["datasets"], "taxonomy": data_section["taxonomy"]}
    return {name: (e["version"], e["manifest_sha256"]) for name, e in entries.items()}


def test_a_committed_training_config_names_a_label_space():
    assert _LABEL_SPACE_CONFIGS


@pytest.mark.parametrize("config", _LABEL_SPACE_CONFIGS, ids=lambda path: path.name)
def test_the_label_space_was_built_from_the_configs_label_keys(config):
    training = ConfigDict(_training_section(config))

    assert _label_space_meta(config)["label_keys_sha1"] == label_keys_sha1(training)


@pytest.mark.parametrize("config", _LABEL_SPACE_CONFIGS, ids=lambda path: path.name)
def test_the_label_space_was_built_from_the_configs_concept_files(config):
    section = _training_section(config)
    meta = _label_space_meta(config)

    for key, meta_key in _LABEL_SPACE_FILES.items():
        assert meta[meta_key] == _file_sha1_or_none(section[key]), key


@pytest.mark.parametrize("config", _LABEL_SPACE_CONFIGS, ids=lambda path: path.name)
def test_the_label_space_was_built_from_the_data_configs_pins(config):
    data_version = _label_space_meta(config)["data_version"]
    data_section = yaml.safe_load(_DATA_CONFIG.read_text())["data"]

    assert _pins(data_version) == _pins(data_section)
