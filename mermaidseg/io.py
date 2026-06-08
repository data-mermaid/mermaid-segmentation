import argparse
from collections.abc import Mapping
from copy import deepcopy
from pathlib import Path
from typing import Any

import albumentations as A
import pandas as pd
import yaml


class ConfigDict(dict):
    """Dictionary subclass with attribute-style access."""

    def __init__(self, dictionary: Mapping[str, Any] | None = None, **kwargs: Any):
        super().__init__()
        if dictionary is not None:
            self.update(dictionary)
        if kwargs:
            self.update(kwargs)

    @staticmethod
    def _wrap(value: Any) -> Any:
        if isinstance(value, ConfigDict):
            return value
        if isinstance(value, Mapping):
            return ConfigDict(value)
        if isinstance(value, list):
            return [ConfigDict._wrap(item) for item in value]
        if isinstance(value, tuple):
            return tuple(ConfigDict._wrap(item) for item in value)
        return value

    def __setitem__(self, key: str, value: Any) -> None:
        super().__setitem__(key, self._wrap(value))

    def __getattr__(self, attr: str) -> Any:
        try:
            return self[attr]
        except KeyError as exc:
            raise AttributeError(attr) from exc

    def __setattr__(self, key: str, value: Any) -> None:
        self[key] = value

    def update(self, *args: Any, **kwargs: Any) -> None:
        for mapping in args:
            for key, value in dict(mapping).items():
                current = self.get(key)
                if isinstance(current, Mapping) and isinstance(value, Mapping):
                    nested = ConfigDict(current)
                    nested.update(value)
                    self[key] = nested
                else:
                    self[key] = value
        for key, value in kwargs.items():
            current = self.get(key)
            if isinstance(current, Mapping) and isinstance(value, Mapping):
                nested = ConfigDict(current)
                nested.update(value)
                self[key] = nested
            else:
                self[key] = value

    def copy(self) -> "ConfigDict":
        return ConfigDict(deepcopy(dict(self)))


def _ensure_section(config: ConfigDict, *path: str) -> ConfigDict:
    current = config
    for key in path:
        section = current.get(key)
        if not isinstance(section, Mapping):
            section = ConfigDict()
            current[key] = section
        elif not isinstance(section, ConfigDict):
            section = ConfigDict(section)
            current[key] = section
        current = section
    return current


def _set_config_value(config: ConfigDict, path: tuple[str, ...], value: Any) -> None:
    if value is None:
        return
    section = _ensure_section(config, *path[:-1]) if len(path) > 1 else config
    section[path[-1]] = value


def _load_csv_value(value: Any, header: int | None = 0) -> Any:
    """Load CSV paths recursively while keeping non-path values unchanged."""
    if isinstance(value, Mapping):
        return ConfigDict(
            {key: _load_csv_value(subvalue, header=header) for key, subvalue in value.items()}
        )
    if isinstance(value, (str, Path)):
        path = Path(value).expanduser()
        if path.is_file() and path.suffix == ".csv":
            return pd.read_csv(path, header=header).to_numpy().flatten().tolist()
    return value


def load_config(config_path: str | Path) -> ConfigDict:
    """Load configuration from a YAML file."""

    with Path(config_path).open(encoding="utf-8") as file:
        loaded = yaml.safe_load(file) or {}
    if not isinstance(loaded, Mapping):
        raise ValueError(
            f"Configuration file {config_path} must contain a mapping at the top level."
        )
    return ConfigDict(loaded)


def get_parser() -> argparse.ArgumentParser:
    """Create and configure an argument parser for semantic segmentation training."""
    parser = argparse.ArgumentParser(description="Semantic Segmentation Run")

    # model and dataset
    parser.add_argument("--run-name", type=str)
    parser.add_argument("--model", type=str, help="model name")

    # training hyper params
    parser.add_argument("--batch-size", type=int, help="batch size")
    parser.add_argument("--epochs", type=int, help="number of epochs to train")
    parser.add_argument("--lr", type=float, help="learning rate")
    parser.add_argument(
        "--iterations-per-train-epoch", type=int, help="iterations per training epoch"
    )
    parser.add_argument(
        "--iterations-per-val-epoch", type=int, help="iterations per validation epoch"
    )

    # checkpoint and log
    parser.add_argument("--model-checkpoint", type=str, help="path to model checkpoint")

    parser.add_argument("--log-epochs", type=int, help="log every log-epochs")

    parser.add_argument("--config", type=str, help="path to config file")
    return parser


def update_config_with_args(config: ConfigDict, args: argparse.Namespace) -> ConfigDict:
    """Update configuration with command-line overrides."""

    _set_config_value(config, ("run_name",), getattr(args, "run_name", None))
    _set_config_value(config, ("model", "name"), getattr(args, "model", None))
    _set_config_value(config, ("model", "checkpoint"), getattr(args, "model_checkpoint", None))
    _set_config_value(config, ("training", "epochs"), getattr(args, "epochs", None))
    _set_config_value(config, ("training", "optimizer", "lr"), getattr(args, "lr", None))
    _set_config_value(config, ("training", "batch_size"), getattr(args, "batch_size", None))
    _set_config_value(
        config,
        ("training", "iterations_per_train_epoch"),
        getattr(args, "iterations_per_train_epoch", None),
    )
    _set_config_value(
        config,
        ("training", "iterations_per_val_epoch"),
        getattr(args, "iterations_per_val_epoch", None),
    )
    _set_config_value(config, ("logger", "log_epochs"), getattr(args, "log_epochs", None))
    return config


def preprocess_data_config(data_cfg_orig):
    data_cfg = data_cfg_orig.copy()
    if "default" in data_cfg.data:
        default = data_cfg.data.pop("default", None)
        for dataset_name in list(data_cfg.data.keys()):
            dataset_tmp = data_cfg.data.get(dataset_name, None)
            data_cfg.data[dataset_name] = default.copy() if default is not None else {}
            if dataset_tmp is not None:
                data_cfg.data[dataset_name].update(dataset_tmp)

    for dataset_name in data_cfg.data:
        for split_name in data_cfg.data[dataset_name]:
            split_cfg = data_cfg.data[dataset_name][split_name]
            # Use whichever key is present (only one should appear): 'augmentation' or 'transform'
            key = next((k for k in ("augmentation", "transform") if k in split_cfg), None)
            if key:
                transform_spec = split_cfg.pop(key)
                split_cfg.transform = A.Compose(
                    [getattr(A, name)(**params) for name, params in transform_spec.items()]
                )

    return data_cfg


def setup_config(config_path_dict: dict[str, str | None]) -> ConfigDict:
    """Load the base config, optionally merge overrides, and normalize data inputs."""

    cfg = ConfigDict()
    if "data" in config_path_dict:
        data_cfg = load_config(config_path_dict["data"])
        data_cfg = preprocess_data_config(data_cfg)
        if "data" in data_cfg:
            cfg.update(data_cfg)
        else:
            cfg.data = data_cfg
    if "model" in config_path_dict:
        model_cfg = load_config(config_path_dict["model"])
        if "model" in model_cfg:
            cfg.update(model_cfg)
        else:
            cfg.model = model_cfg
    if "training" in config_path_dict:
        training_cfg = load_config(config_path_dict["training"])
        if "training" in training_cfg:
            cfg.update(training_cfg)
        else:
            cfg.training = training_cfg
    if "logger" in config_path_dict:
        logger_cfg = load_config(config_path_dict["logger"])
        if "logger" in logger_cfg:
            cfg.update(logger_cfg)
        else:
            cfg.logger = logger_cfg

    return cfg
