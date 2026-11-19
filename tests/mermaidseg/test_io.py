"""Config loading: section wrappers, plain-data output, and the augmentation allowlist."""

from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path
from typing import Any

import albumentations as A
import pytest
import yaml

from mermaidseg.inference.preprocess import IMAGENET_MEAN, IMAGENET_STD
from mermaidseg.io import (
    REPO_ROOT,
    apply_resolution,
    build_eval_transform,
    compile_transform,
    resolve_repo_path,
    setup_config,
)

_CONFIGS = Path(__file__).resolve().parents[2] / "configs"


def _plain(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {key: _plain(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_plain(item) for item in value]
    return value


def test_setup_config_output_is_plain_yaml_data():
    cfg = setup_config({"data": str(_CONFIGS / "data_config.yaml")})
    dumped = yaml.safe_dump(_plain(cfg))
    assert yaml.safe_load(dumped) == _plain(cfg)
    train = _plain(cfg)["data"]["coralnet"]["train"]["transform"]
    assert list(train) == [
        "SmallestMaxSize",
        "RandomResizedCrop",
        "HorizontalFlip",
        "VerticalFlip",
        "ColorJitter",
    ]
    assert train["RandomResizedCrop"]["size"] == [256, 256]


def test_train_transform_ends_in_imagenet_normalize():
    transform = compile_transform({"HorizontalFlip": {"p": 0.5}, "ColorJitter": {"hue": 0.03}})
    assert [type(t) for t in transform.transforms] == [A.HorizontalFlip, A.ColorJitter, A.Normalize]
    normalize = transform.transforms[-1]
    assert tuple(normalize.mean) == pytest.approx(IMAGENET_MEAN)
    assert tuple(normalize.std) == pytest.approx(IMAGENET_STD)


def test_yaml_normalize_is_rejected_by_the_allowlist():
    with pytest.raises(ValueError, match="'Normalize'"):
        compile_transform({"Normalize": {"mean": [0, 0, 0], "std": [1, 1, 1]}})


def test_apply_resolution_sets_only_the_random_resized_crop_size():
    spec = {
        "data": {
            "default": {
                "train": {
                    "transform": {
                        "SmallestMaxSize": {"max_size": 1024},
                        "RandomResizedCrop": {"size": [256, 256], "scale": [0.2, 1]},
                        "Resize": {"width": 10, "height": 12},
                    }
                }
            }
        }
    }
    transform = apply_resolution(spec, 512)["data"]["default"]["train"]["transform"]
    assert transform == {
        "SmallestMaxSize": {"max_size": 1024},
        "RandomResizedCrop": {"size": [512, 512], "scale": [0.2, 1]},
        "Resize": {"width": 10, "height": 12},
    }


def test_eval_transform_resizes_to_the_input_size_then_normalizes():
    resize, normalize = build_eval_transform((32, 48)).transforms
    assert (type(resize), resize.height, resize.width) == (A.Resize, 32, 48)
    assert isinstance(normalize, A.Normalize)


def test_relative_path_resolves_to_the_repo_not_the_current_directory(monkeypatch, tmp_path: Path):
    (tmp_path / "configs").mkdir()
    (tmp_path / "configs" / "class_to_concepts.csv").write_text("decoy\n")
    monkeypatch.chdir(tmp_path)
    assert resolve_repo_path("configs/class_to_concepts.csv") == (
        REPO_ROOT / "configs" / "class_to_concepts.csv"
    )
