"""Evaluation entrypoint: config files are read through the config schema."""

from __future__ import annotations

from pathlib import Path

import pytest
import yaml

import scripts.evaluate as evaluate_script

_CONFIGS = Path(__file__).resolve().parents[2] / "configs"


def test_training_config_with_an_unknown_key_raises_naming_the_file(tmp_path: Path):
    raw = yaml.safe_load((_CONFIGS / "training_config_cbm_final.yaml").read_text())
    raw["training"]["distance_loss"] = {"weight": 0.1}
    path = tmp_path / "training_unknown_key.yaml"
    path.write_text(yaml.safe_dump(raw))
    with pytest.raises(ValueError, match="training_unknown_key.yaml"):
        evaluate_script.run_removed_concepts(path, _CONFIGS / "class_to_concepts.csv")
