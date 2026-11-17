"""Config loading: section wrappers, plain-data output, and the augmentation allowlist."""

from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path
from typing import Any

import pytest
import yaml

from mermaidseg.io import compile_transform, setup_config

_CONFIGS = Path(__file__).resolve().parents[2] / "configs"


def test_file_without_its_section_wrapper_raises_naming_the_file(tmp_path: Path):
    raw = yaml.safe_load((_CONFIGS / "logger_config.yaml").read_text())
    path = tmp_path / "unwrapped_logger.yaml"
    path.write_text(yaml.safe_dump(raw["logger"]))
    with pytest.raises(ValueError, match="unwrapped_logger.yaml"):
        setup_config({"logger": str(path)})


def _plain(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {key: _plain(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_plain(item) for item in value]
    return value


def test_setup_config_output_is_plain_yaml_data():
    cfg = setup_config({"data": str(_CONFIGS / "data_config.yaml")})
    dumped = yaml.safe_dump(_plain(cfg))
    assert yaml.safe_load(dumped)["data"]["coralnet"]["val"]["transform"] == {
        "Resize": {"width": 256, "height": 256},
        "Normalize": {"mean": [0.485, 0.456, 0.406], "std": [0.229, 0.224, 0.225]},
    }


def test_unknown_augmentation_raises_listing_the_allowed_names():
    with pytest.raises(ValueError, match="'Blur'.*ColorJitter.*VerticalFlip"):
        compile_transform({"Blur": {"blur_limit": 3}})
