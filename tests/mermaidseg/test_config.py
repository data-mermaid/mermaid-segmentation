"""Config schema: committed configs validate; unknown keys, missing keys and unknown names raise."""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path
from typing import Any

import pytest
import yaml

from mermaidseg.io import setup_config

_REPO = Path(__file__).resolve().parents[2]
_CONFIGS = _REPO / "configs"

_FILES = {
    "data": "data_config.yaml",
    "training": "training_config_cbm_final.yaml",
    "model": "model_config_cbm_dpt_lora_vitl.yaml",
    "logger": "logger_config.yaml",
}


@pytest.mark.parametrize(
    ("data_file", "training_file"),
    [
        ("data_config_trainval.yaml", "training_config_cbm_final.yaml"),
        ("data_config_trainval.yaml", "training_config_cbm_final_hires.yaml"),
        ("data_config.yaml", "training_config_cbm_final.yaml"),
    ],
)
def test_committed_config_sets_validate(data_file: str, training_file: str):
    cfg = setup_config(
        {
            "data": str(_CONFIGS / data_file),
            "training": str(_CONFIGS / training_file),
            "model": str(_CONFIGS / _FILES["model"]),
            "logger": str(_CONFIGS / _FILES["logger"]),
        }
    )
    assert set(cfg) == {"data", "training", "model", "logger"}


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


def _delete(*keys: str) -> Callable[[dict[str, Any]], None]:
    def mutate(section: dict[str, Any]) -> None:
        for key in keys[:-1]:
            section = section[key]
        del section[keys[-1]]

    return mutate


@pytest.mark.parametrize(
    ("section", "keys"),
    [
        ("training", ("distance_loss",)),
        ("training", ("loss", "unknown")),
        ("training", ("optimizer", "unknown")),
        ("training", ("scheduler", "unknown")),
        ("training", ("taxonomic_loss", "unknown")),
        ("training", ("loader", "unknown")),
        ("model", ("unknown",)),
        ("logger", ("unknown",)),
        ("data", ("coralscapes",)),
        ("data", ("coralnet", "vall")),
    ],
)
def test_unknown_key_raises_naming_the_file(tmp_path: Path, section: str, keys: tuple[str, ...]):
    paths = _mutated(tmp_path, section, _set(*keys, value={}))
    with pytest.raises(ValueError, match=f"{section}.yaml"):
        setup_config(paths)


@pytest.mark.parametrize(
    ("section", "keys"),
    [
        ("training", ("max_grad_norm",)),
        ("training", ("train_metrics_every",)),
        ("training", ("scheduler", "warmup_iters")),
        ("model", ("lora_r",)),
        ("logger", ("enable_wandb",)),
    ],
)
def test_missing_key_raises(tmp_path: Path, section: str, keys: tuple[str, ...]):
    paths = _mutated(tmp_path, section, _delete(*keys))
    with pytest.raises(ValueError, match=keys[-1]):
        setup_config(paths)


@pytest.mark.parametrize(
    ("section", "keys"),
    [
        ("model", ("name",)),
        ("training", ("loss", "type")),
        ("training", ("optimizer", "type")),
        ("training", ("scheduler", "type")),
        ("training", ("training_mode",)),
    ],
)
def test_unknown_name_raises(tmp_path: Path, section: str, keys: tuple[str, ...]):
    paths = _mutated(tmp_path, section, _set(*keys, value="Unknown"))
    with pytest.raises(ValueError, match="Unknown"):
        setup_config(paths)


def test_string_none_split_raises(tmp_path: Path):
    paths = _mutated(tmp_path, "data", _set("mermaid", "val", value="None"))
    with pytest.raises(ValueError, match="mermaid.val"):
        setup_config(paths)
