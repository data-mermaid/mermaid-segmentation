"""Configuration loading, merging, and command-line overrides for training runs."""

import argparse
from collections.abc import Mapping, MutableMapping, Sequence
from copy import deepcopy
from pathlib import Path
from typing import Any, cast

import albumentations as A
import yaml

from mermaidseg.config import RESERVED_DATA_KEYS, lookup_class, validate_section
from mermaidseg.inference.preprocess import CV2_INTERPOLATION, IMAGENET_MEAN, IMAGENET_STD

REPO_ROOT = Path(__file__).resolve().parents[1]


def resolve_repo_path(path: str | Path) -> Path:
    """An absolute path as given; a relative path resolved against the repository root.

    The current directory never matters, so entrypoints run from anywhere.
    """
    path = Path(path)
    if path.is_absolute():
        return path
    return REPO_ROOT / path


class ConfigDict(dict):
    """Dictionary subclass with attribute-style access."""

    def __init__(self, dictionary: Mapping[str, Any] | None = None, **kwargs: Any) -> None:
        """Initialize from an optional mapping and keywords, wrapping nested dicts."""
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
        """Set an item, wrapping mappings as ``ConfigDict``."""
        super().__setitem__(key, self._wrap(value))

    def __getattr__(self, attr: str) -> Any:
        """Return the item named ``attr``.

        Raises:
            AttributeError: If the key is missing.
        """
        try:
            return self[attr]
        except KeyError as exc:
            raise AttributeError(attr) from exc

    def __setattr__(self, key: str, value: Any) -> None:
        """Set the item named ``key``."""
        self[key] = value

    def update(self, *args: Any, **kwargs: Any) -> None:
        """Merge mappings and keyword entries, recursively merging nested mappings."""
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
        """Return a deep copy."""
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


def _read_mapping(config_path: str | Path) -> dict[str, Any]:
    with Path(config_path).open(encoding="utf-8") as file:
        loaded = yaml.safe_load(file) or {}
    if not isinstance(loaded, Mapping):
        raise ValueError(
            f"Configuration file {config_path} must contain a mapping at the top level."
        )
    return dict(loaded)


def load_config(config_path: str | Path) -> ConfigDict:
    """Load configuration from a YAML file."""
    return ConfigDict(_read_mapping(config_path))


def load_section(config_path: str | Path, section: str) -> dict[str, Any]:
    """Load one config section from a YAML file wrapped in its section key, and validate it.

    Returns the validated section as plain data. Keys a file leaves out of an optional block
    (a dataset, a split) stay absent.

    Raises:
        ValueError: If the file's only top-level key is not ``section``, or the section does not
            match its schema in ``mermaidseg.config``. The message names the file.
    """
    return validate_section(_read_mapping(config_path), section, config_path)


def get_parser() -> argparse.ArgumentParser:
    """Create and configure an argument parser for semantic segmentation training."""
    parser = argparse.ArgumentParser(description="Semantic Segmentation Run", allow_abbrev=False)

    parser.add_argument("--run-name", type=str)
    parser.add_argument("--batch-size", type=int, help="batch size")
    return parser


def update_config_with_args(config: ConfigDict, args: argparse.Namespace) -> ConfigDict:
    """Update configuration with command-line overrides."""
    _set_config_value(config, ("run_name",), args.run_name)
    _set_config_value(config, ("training", "batch_size"), args.batch_size)
    return config


AUGMENTATION_CLASSES: dict[str, type[A.BasicTransform]] = {
    "SmallestMaxSize": A.SmallestMaxSize,
    "RandomResizedCrop": A.RandomResizedCrop,
    "HorizontalFlip": A.HorizontalFlip,
    "VerticalFlip": A.VerticalFlip,
    "ColorJitter": A.ColorJitter,
}


def _normalize() -> A.Normalize:
    return A.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD)


def compile_transform(spec: Mapping[str, Mapping[str, Any]]) -> A.Compose:
    """Compile a train split's ``transform`` spec (augmentation name -> kwargs) into a ``Compose``.

    ``Normalize`` with the ImageNet statistics is always appended last; a spec cannot set it.

    Raises:
        ValueError: If a name is not in :data:`AUGMENTATION_CLASSES`.
    """
    transforms = [
        lookup_class(AUGMENTATION_CLASSES, "augmentation", name)(**params)
        for name, params in spec.items()
    ]
    return A.Compose([*transforms, _normalize()])


def build_eval_transform(input_size: Sequence[int]) -> A.Compose:
    """Return the val/test transform: ``Resize`` to ``input_size`` (height, width), then Normalize."""
    height, width = int(input_size[0]), int(input_size[1])
    return A.Compose(
        [A.Resize(height=height, width=width, interpolation=CV2_INTERPOLATION, p=1), _normalize()]
    )


def preprocess_data_config(data_cfg_orig: ConfigDict) -> ConfigDict:
    """Expand the per-dataset ``default`` block of the data config (configs/data_config.yaml).

    Every dataset gets the ``default`` splits, overridden by its own. Reserved keys (e.g.
    ``local_cache_dir``) under ``data`` are left untouched. Transform specs stay plain data;
    :func:`compile_transform` compiles them where datasets are built.
    """
    data_cfg = data_cfg_orig.copy()
    if "default" in data_cfg.data:
        default = data_cfg.data.pop("default", None)
        for dataset_name in list(data_cfg.data.keys()):
            if dataset_name in RESERVED_DATA_KEYS:
                continue
            dataset_tmp = data_cfg.data.get(dataset_name, None)
            data_cfg.data[dataset_name] = default.copy() if default is not None else {}
            if dataset_tmp is not None:
                data_cfg.data[dataset_name].update(dataset_tmp)

    return data_cfg


def apply_resolution(spec: Any, resolution: int) -> Any:
    """Set every ``RandomResizedCrop`` size in a raw spec.

    Walks the YAML mapping in place. Every other transform is left unchanged; val/test sizes
    come from ``model.input_size``.
    """
    if isinstance(spec, Mapping):
        for key, value in spec.items():
            if key == "RandomResizedCrop" and isinstance(value, Mapping):
                cast("MutableMapping[str, Any]", value)["size"] = [resolution, resolution]
            else:
                apply_resolution(value, resolution)
    elif isinstance(spec, list):
        for item in spec:
            apply_resolution(item, resolution)
    return spec


def setup_config(
    config_path_dict: dict[str, str | None], resolution: int | None = None
) -> ConfigDict:
    """Load and merge the split config files into a single ConfigDict.

    Takes a mapping of section name -> YAML path, e.g.::

        {"data": "configs/data_config.yaml",
         "model": "configs/model_config_cbm_dpt_lora_vitl.yaml",
         "training": "configs/training_config_cbm_final.yaml",
         "logger": "configs/logger_config.yaml"}

    Each file wraps its contents in its section key and is validated against that section's
    schema (:func:`load_section`). The data config is additionally normalized via
    ``preprocess_data_config`` (per-dataset default expansion). The result is plain data.

    ``resolution`` sets the square train crop (:func:`apply_resolution`) and ``model.input_size``,
    which sizes the val/test resize (:func:`build_eval_transform`).

    Raises:
        ValueError: If a file is unwrapped or does not match its schema.
    """
    cfg = ConfigDict()
    if "data" in config_path_dict:
        data_section = load_section(cast("str | Path", config_path_dict["data"]), "data")
        if resolution is not None:
            apply_resolution(data_section, int(resolution))
        cfg.update(preprocess_data_config(ConfigDict({"data": data_section})))
    if "model" in config_path_dict:
        cfg.model = load_section(cast("str | Path", config_path_dict["model"]), "model")
        if resolution is not None:
            cfg.model.input_size = [int(resolution), int(resolution)]
    if "training" in config_path_dict:
        cfg.training = load_section(cast("str | Path", config_path_dict["training"]), "training")
    if "logger" in config_path_dict:
        cfg.logger = load_section(cast("str | Path", config_path_dict["logger"]), "logger")

    return cfg
