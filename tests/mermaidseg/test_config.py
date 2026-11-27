"""Config schema: committed configs validate; unknown keys, missing keys and unknown names raise."""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path
from typing import Any

import pytest
import torch
import yaml

from mermaidseg.config import lookup_class, to_plain_dict, validate_section
from mermaidseg.io import setup_config

_REPO = Path(__file__).resolve().parents[2]
_CONFIGS = _REPO / "configs"

_FILES = {
    "data": "data_config.yaml",
    "training": "training_config_cbm_final.yaml",
    "model": "model_config_cbm_dpt_lora_vitl.yaml",
    "logger": "logger_config.yaml",
}


def _mutated(
    tmp_path: Path, section: str, mutate: Callable[[dict[str, Any]], None]
) -> dict[str, str | None]:
    """Write the committed ``section`` file after ``mutate`` and return a setup_config input."""
    raw = yaml.safe_load((_CONFIGS / _FILES[section]).read_text())
    mutate(raw[section])
    path = tmp_path / f"{section}.yaml"
    path.write_text(yaml.safe_dump(raw))
    return {section: str(path)}


def _set(*keys: str, value: Any) -> Callable[[dict[str, Any]], None]:
    def mutate(section: dict[str, Any]) -> None:
        for key in keys[:-1]:
            section = section[key]
        section[keys[-1]] = value

    return mutate


@pytest.mark.parametrize(
    ("section", "keys"),
    [
        ("training", ("distance_loss",)),
        ("data", ("datasets", "coralnet", "vall")),
    ],
)
def test_unknown_key_raises_naming_the_file(tmp_path: Path, section: str, keys: tuple[str, ...]):
    paths = _mutated(tmp_path, section, _set(*keys, value={}))
    with pytest.raises(ValueError, match=f"{section}.yaml"):
        setup_config(paths)


@pytest.mark.parametrize(
    ("section", "keys"),
    [
        ("model", ("name",)),
    ],
)
def test_unknown_name_raises(tmp_path: Path, section: str, keys: tuple[str, ...]):
    paths = _mutated(tmp_path, section, _set(*keys, value="Unknown"))
    with pytest.raises(ValueError, match="Unknown"):
        setup_config(paths)


@pytest.mark.parametrize("steps", [0, -1])
def test_a_non_positive_checkpoint_interval_raises_naming_the_file(tmp_path: Path, steps: int):
    paths = _mutated(tmp_path, "training", _set("checkpoint_every_steps", value=steps))
    with pytest.raises(
        ValueError, match=r"(?s)training\.yaml.*checkpoint_every_steps.*greater than 0"
    ):
        setup_config(paths)


def test_a_checkpoint_interval_of_one_validates(tmp_path: Path):
    paths = _mutated(tmp_path, "training", _set("checkpoint_every_steps", value=1))
    assert setup_config(paths).training.checkpoint_every_steps == 1


def test_committed_sagemaker_logger_config_validates():
    cfg = setup_config({"logger": str(_CONFIGS / "logger_config_sagemaker.yaml")})
    assert cfg.logger.tracker == "mlflow"


def test_wandb_key_under_the_mlflow_tracker_raises_naming_the_file():
    raw = {
        "logger": {
            "tracker": "mlflow",
            "experiment_name": "e",
            "train_log_interval": 1,
            "wandb_project": "p",
        }
    }
    with pytest.raises(ValueError, match=r"(?s)^cfg\.yaml: .*wandb_project"):
        validate_section(raw, "logger", "cfg.yaml")


def test_string_none_split_raises(tmp_path: Path):
    paths = _mutated(tmp_path, "data", _set("datasets", "mermaid", "val", value="None"))
    with pytest.raises(ValueError, match="mermaid.val"):
        setup_config(paths)


class _First:
    pass


class _Second:
    pass


def test_lookup_class_rejects_an_unknown_name_listing_the_allowed_names_sorted():
    with pytest.raises(ValueError, match=r"'SGD'.*\['First', 'Second'\]"):
        lookup_class({"Second": _Second, "First": _First}, "optimizer", "SGD")


@pytest.mark.parametrize(
    "training_file", ["training_config_cbm_final.yaml", "training_config_cbm_final_hires.yaml"]
)
def test_resolved_committed_config_loads_under_weights_only_with_paths_as_text(
    tmp_path: Path, training_file: str
):
    cfg = setup_config(
        {
            "data": str(_CONFIGS / _FILES["data"]),
            "training": str(_CONFIGS / training_file),
            "model": str(_CONFIGS / _FILES["model"]),
            "logger": str(_CONFIGS / _FILES["logger"]),
        }
    )
    path = tmp_path / "ckpt.pt"
    torch.save({"config": to_plain_dict(cfg)}, path)
    loaded = torch.load(path, weights_only=True)["config"]
    assert loaded["data"]["datasets"]["coralnet"]["split_file"] == "configs/splits/coralnet.csv"


_DATA_SECTION = {
    "datasets": {
        "mermaid": {
            "version": "v",
            "manifest_sha256": "s",
            "split_file": "splits.csv",
            "train": {"use": ["train"]},
        }
    }
}


def test_validate_section_returns_plain_dict_without_unset_keys():
    result = validate_section({"data": _DATA_SECTION}, "data", "cfg.yaml")
    assert result == _DATA_SECTION
    assert type(result) is dict


@pytest.mark.parametrize(
    "raw",
    [{"model": {}, "extra": 1}],
)
def test_validate_section_rejects_any_top_level_keys_but_the_section(raw: dict[str, Any]):
    with pytest.raises(ValueError, match=r"^cfg\.yaml: expected the single top-level key 'model'"):
        validate_section(raw, "model", "cfg.yaml")
