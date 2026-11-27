"""Each derived concept config a committed training config names was built against its concept table."""

from __future__ import annotations

import hashlib
from pathlib import Path

import pytest
import yaml

from mermaidseg.dataset_reconciliation.artifact_meta import read_sidecar

_REPO = Path(__file__).resolve().parents[2]
_DERIVED_KEYS = ("concept_removal", "concept_source_mask")
_TRAINING_CONFIGS = sorted((_REPO / "configs").glob("training_config*.yaml"))


def _training_section(config: Path) -> dict:
    return yaml.safe_load(config.read_text())["training"]


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
