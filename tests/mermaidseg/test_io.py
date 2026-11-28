"""Config loading: section wrappers, plain-data output, and the augmentation allowlist."""

from __future__ import annotations

import argparse
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
    build_train_parser,
    check_train_args,
    compile_transform,
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


def test_relative_path_resolves_to_the_repo_not_the_current_directory(monkeypatch, tmp_path: Path):
    (tmp_path / "configs").mkdir()
    (tmp_path / "configs" / "class_to_concepts.csv").write_text("decoy\n")
    monkeypatch.chdir(tmp_path)
    assert resolve_repo_path("configs/class_to_concepts.csv") == (
        REPO_ROOT / "configs" / "class_to_concepts.csv"
    )


# ---------- ConfigDict ----------


@pytest.mark.parametrize(
    "update",
    [
        lambda cfg: cfg.update({"a": {"y": 3, "z": 4}}),
        lambda cfg: cfg.update(a={"y": 3, "z": 4}),
    ],
    ids=["mapping", "keywords"],
)
def test_update_merges_nested_mappings_instead_of_replacing_them(update):
    cfg = ConfigDict({"a": {"x": 1, "y": 2}, "b": 1})
    update(cfg)
    assert cfg == {"a": {"x": 1, "y": 3, "z": 4}, "b": 1}
    assert isinstance(cfg["a"], ConfigDict)


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


# ---------- build_train_parser / check_train_args ----------

REQUIRED = ["--num-workers", "8", "--cache-dir", "cache"]

CONFIG_FLAGS = (
    "training_config",
    "model_config",
    "data_config",
    "logger_config",
)


def _parse(*argv: str):
    return build_train_parser().parse_args([*REQUIRED, *argv])


def _parse_and_check(argv: list[str]):
    parser = build_train_parser()
    args = parser.parse_args(argv)
    check_train_args(parser, args)
    return args


def test_config_defaults_point_at_existing_repo_files():
    args = _parse()
    for flag in CONFIG_FLAGS:
        assert resolve_repo_path(getattr(args, flag)).is_file(), flag


@pytest.mark.parametrize(
    "argv",
    [
        pytest.param(["--cache-dir", "cache", "--num-workers", "0"], id="zero-num-workers"),
        pytest.param([*REQUIRED, "--nccl-timeout-min", "0"], id="zero-nccl-timeout"),
        pytest.param(
            [*REQUIRED, "--preempt-checkpoint-every-steps", "-1"], id="negative-preempt-every"
        ),
        pytest.param(
            [*REQUIRED, "--checkpoint", "x", "--init-weights", "y"], id="checkpoint-and-init"
        ),
    ],
)
def test_check_train_args_rejects_invalid_arguments_with_exit_2(argv):
    with pytest.raises(SystemExit) as exit_info:
        _parse_and_check(argv)
    assert exit_info.value.code == 2


@pytest.mark.parametrize(
    "argv",
    [
        pytest.param([*REQUIRED, "--nccl-timeout-min", "0.01"], id="small-positive-nccl-timeout"),
        pytest.param([*REQUIRED, "--preempt-checkpoint-every-steps", "0"], id="preempt-disabled"),
        pytest.param(["--cache-dir", "cache", "--num-workers", "1"], id="single-worker"),
    ],
)
def test_check_train_args_accepts_the_edge_of_each_valid_range(argv):
    assert _parse_and_check(argv).cache_dir == "cache"


# ---------- update_config_with_args ----------


def _overrides(**given: Any) -> argparse.Namespace:
    """Parsed flags with the override flags not in ``given`` left unset."""
    return argparse.Namespace(**{"run_name": None, "batch_size": None, **given})


def test_batch_size_flag_overrides_the_training_batch_size_and_keeps_its_siblings():
    cfg = ConfigDict({"run_name": "keep", "training": {"batch_size": 2, "epochs": 5}})
    args = _overrides(batch_size=8)
    update_config_with_args(cfg, args)
    assert cfg == {"run_name": "keep", "training": {"batch_size": 8, "epochs": 5}}


def test_flags_that_are_not_given_leave_the_config_untouched():
    cfg = ConfigDict({"run_name": "keep", "training": {"batch_size": 2}})
    update_config_with_args(cfg, _overrides())
    assert cfg == {"run_name": "keep", "training": {"batch_size": 2}}
