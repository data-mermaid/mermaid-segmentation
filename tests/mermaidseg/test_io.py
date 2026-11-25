"""Config loading: section wrappers, plain-data output, and the augmentation allowlist."""

from __future__ import annotations

import datetime
from pathlib import Path
from typing import Any

import albumentations as A
import pytest

from mermaidseg.inference.preprocess import IMAGENET_MEAN, IMAGENET_STD
from mermaidseg.io import (
    REPO_ROOT,
    ConfigDict,
    apply_resolution,
    build_eval_transform,
    compile_transform,
    get_parser,
    load_config,
    preprocess_data_config,
    resolve_repo_path,
    setup_config,
    update_config_with_args,
)

_CONFIGS = Path(__file__).resolve().parents[2] / "configs"


_SCALARS = (str, int, float, bool, type(None), datetime.date)


def _non_plain_nodes(value: Any, path: str = "") -> list[str]:
    """Paths of nodes that are neither a dict (``ConfigDict`` included), a list nor a YAML scalar."""
    if isinstance(value, dict):
        return [bad for k, v in value.items() for bad in _non_plain_nodes(v, f"{path}/{k}")]
    if isinstance(value, list):
        return [bad for i, v in enumerate(value) for bad in _non_plain_nodes(v, f"{path}[{i}]")]
    return [] if isinstance(value, _SCALARS) else [f"{path}: {type(value).__name__}"]


def test_setup_config_output_is_plain_yaml_data():
    paths = {
        "data": str(_CONFIGS / "data_config.yaml"),
        "model": str(_CONFIGS / "model_config_cbm_dpt_lora_vitl.yaml"),
        "training": str(_CONFIGS / "training_config_cbm_final.yaml"),
        "logger": str(_CONFIGS / "logger_config.yaml"),
    }
    cfg = setup_config(paths, resolution=64)
    assert _non_plain_nodes(cfg) == []
    train = cfg["data"]["datasets"]["coralnet"]["train"]["transform"]
    assert list(train) == [
        "SmallestMaxSize",
        "RandomResizedCrop",
        "HorizontalFlip",
        "VerticalFlip",
        "ColorJitter",
    ]
    assert train["RandomResizedCrop"]["size"] == [64, 64]
    assert cfg["model"]["input_size"] == [64, 64]


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


# ---------- preprocess_data_config ----------


def _expand(data: dict[str, Any]) -> ConfigDict:
    return preprocess_data_config(ConfigDict({"data": data})).data


def test_a_dataset_inherits_the_default_splits_and_overrides_one_key():
    expanded = _expand(
        {
            "default": {"train": {"split": "train", "repeat": 1}, "val": {"split": "val"}},
            "ds": {"train": {"repeat": 3}},
        }
    )
    assert expanded == {"ds": {"train": {"split": "train", "repeat": 3}, "val": {"split": "val"}}}


def test_a_null_split_override_disables_that_split():
    expanded = _expand(
        {"default": {"train": {"split": "train"}, "val": {"split": "val"}}, "ds": {"val": None}}
    )
    assert expanded["ds"] == {"train": {"split": "train"}, "val": None}


def test_a_dataset_listed_as_null_gets_the_default_splits():
    expanded = _expand({"default": {"train": {"split": "train"}}, "ds": None})
    assert expanded == {"ds": {"train": {"split": "train"}}}


def test_reserved_data_keys_are_not_expanded_into_datasets():
    expanded = _expand(
        {
            "default": {"train": {"split": "train"}},
            "local_cache_dir": "/cache",
            "local_cache_write_through": True,
            "ds": {},
        }
    )
    assert expanded == {
        "local_cache_dir": "/cache",
        "local_cache_write_through": True,
        "ds": {"train": {"split": "train"}},
    }


def test_datasets_do_not_share_their_expanded_default():
    expanded = _expand({"default": {"train": {"split": "train"}}, "a": {}, "b": {}})
    expanded["a"]["train"]["split"] = "changed"
    assert expanded["b"]["train"]["split"] == "train"


def test_a_config_without_default_is_returned_as_a_copy():
    original = ConfigDict({"data": {"ds": {"train": {"split": "train"}}}})
    expanded = preprocess_data_config(original)
    assert expanded == original
    assert expanded is not original


# ---------- ConfigDict ----------


def test_update_merges_nested_mappings_instead_of_replacing_them():
    cfg = ConfigDict({"a": {"x": 1, "y": 2}, "b": 1})
    cfg.update({"a": {"y": 3, "z": 4}})
    assert cfg == {"a": {"x": 1, "y": 3, "z": 4}, "b": 1}
    assert isinstance(cfg["a"], ConfigDict)


def test_update_merges_nested_mappings_given_as_keywords():
    cfg = ConfigDict({"a": {"x": 1, "y": 2}})
    cfg.update(a={"y": 3, "z": 4})
    assert cfg == {"a": {"x": 1, "y": 3, "z": 4}}


def test_update_replaces_a_mapping_by_a_scalar_and_a_scalar_by_a_mapping():
    cfg = ConfigDict({"a": {"x": 1}, "b": 1})
    cfg.update({"a": None, "b": {"y": 2}})
    assert cfg == {"a": None, "b": {"y": 2}}
    assert isinstance(cfg["b"], ConfigDict)


def test_copy_is_deep():
    cfg = ConfigDict({"a": {"b": [1]}})
    clone = cfg.copy()
    clone.a.b.append(2)
    clone.a.c = 3
    assert cfg == {"a": {"b": [1]}}
    assert isinstance(clone, ConfigDict)
    assert isinstance(clone.a, ConfigDict)


def test_a_missing_attribute_raises_attribute_error_not_key_error():
    cfg = ConfigDict({"present": 1})
    with pytest.raises(AttributeError, match="absent"):
        cfg.absent  # noqa: B018
    assert not hasattr(cfg, "absent")
    assert getattr(cfg, "absent", "fallback") == "fallback"


# ---------- load_config ----------


@pytest.mark.parametrize("content", ["- a\n- b\n", "just a string\n"])
def test_load_config_rejects_a_file_that_is_not_a_mapping_naming_the_file(tmp_path, content):
    path = tmp_path / "cfg.yaml"
    path.write_text(content)
    with pytest.raises(ValueError, match="cfg.yaml.*mapping"):
        load_config(path)


def test_load_config_wraps_a_mapping_file(tmp_path):
    path = tmp_path / "cfg.yaml"
    path.write_text("a:\n  b: 1\n")
    cfg = load_config(path)
    assert cfg.a.b == 1


# ---------- update_config_with_args ----------


def test_batch_size_flag_overrides_the_training_batch_size_and_keeps_its_siblings():
    cfg = ConfigDict({"run_name": "keep", "training": {"batch_size": 2, "epochs": 5}})
    args = get_parser().parse_args(["--batch-size", "8"])
    update_config_with_args(cfg, args)
    assert cfg == {"run_name": "keep", "training": {"batch_size": 8, "epochs": 5}}


def test_run_name_flag_overrides_the_run_name():
    cfg = ConfigDict({"run_name": "old"})
    update_config_with_args(cfg, get_parser().parse_args(["--run-name", "new"]))
    assert cfg.run_name == "new"


def test_flags_that_are_not_given_leave_the_config_untouched():
    cfg = ConfigDict({"run_name": "keep", "training": {"batch_size": 2}})
    update_config_with_args(cfg, get_parser().parse_args([]))
    assert cfg == {"run_name": "keep", "training": {"batch_size": 2}}


def test_batch_size_flag_creates_a_missing_training_section():
    cfg = ConfigDict()
    update_config_with_args(cfg, get_parser().parse_args(["--batch-size", "8"]))
    assert cfg == {"training": {"batch_size": 8}}
