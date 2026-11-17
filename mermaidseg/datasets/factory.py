"""Shared dataset construction used by training and prefetch scripts.

Keeping the ``(name, split) -> Dataset`` build loop in one place guarantees the
training job and the cache-prefetch job agree on exactly which source objects a
run needs, so an offline (AWS-free) training run never hits an unwarmed key.
"""

from __future__ import annotations

import logging
from collections.abc import Collection, Sequence
from typing import Any

from mermaidseg.config import RESERVED_DATA_KEYS
from mermaidseg.datasets.benthos_yuval import BenthosYuvalCoralsDataset
from mermaidseg.datasets.catlin_seaview import CatlinSeaviewDataset
from mermaidseg.datasets.coralnet import CoralNetDataset
from mermaidseg.datasets.coralscapes_v2 import CoralscapesV2Dataset
from mermaidseg.datasets.mermaid import MermaidDataset
from mermaidseg.datasets.moorea_labeled_corals import MooreaLabeledCoralsDataset
from mermaidseg.datasets.pacific_labeled_corals import PacificLabeledCoralsDataset
from mermaidseg.datasets.ucsd_mosaics import UCSDMosaicsDataset
from mermaidseg.io import build_eval_transform, compile_transform

logger = logging.getLogger(__name__)

# Canonical dataset registry (name -> class). Order is deterministic so runs and
# prefetch produce identical iteration order.
DATASET_CLASSES: dict[str, type[Any]] = {
    "pacific_labeled_corals": PacificLabeledCoralsDataset,
    "moorea_labeled_corals": MooreaLabeledCoralsDataset,
    "catlin_seaview": CatlinSeaviewDataset,
    "mermaid": MermaidDataset,
    "coralnet": CoralNetDataset,
    "coralscapes_v2": CoralscapesV2Dataset,
    "benthos_yuval": BenthosYuvalCoralsDataset,
    "ucsd_mosaics": UCSDMosaicsDataset,
}

# These datasets do not accept a ``padding`` argument (dense masks, not points).
NO_PADDING_DATASETS: frozenset[str] = frozenset({"coralscapes_v2", "ucsd_mosaics"})


def _is_skip(split_cfg: Any) -> bool:
    """True when a split config is YAML ``null``, which means "skip this split"."""
    return split_cfg is None


def _check_data_keys(data_cfg: Any, allowed: Collection[str]) -> None:
    """Raise on a ``data`` key that is neither a dataset nor in ``allowed``."""
    unknown = sorted(key for key in data_cfg if key not in allowed)
    if unknown:
        raise ValueError(f"unknown data keys {unknown}; expected one of {sorted(allowed)}")


_CONCEPT_SPLITS = ("train", "val", "test")

# Trainer-only split key: how many times the train split is repeated per epoch.
_REPEAT_KEY = "repeat"


def configured_source_names(
    data_cfg: Any,
    *,
    dataset_classes: dict[str, type[Any]] | None = None,
) -> set[str]:
    """``SOURCE_NAME``s with a non-skipped train, val, or test block.

    Does not instantiate datasets. ``data_cfg`` is the ``data`` section, either
    raw YAML or the preprocessed mapping passed to :func:`build_datasets`.
    A profile that sets a train split to ``None`` drops that dataset only when
    this function is called on the profile. Call it on the unmodified config to
    freeze the concept channels.

    Raises:
        ValueError: If a key is neither a dataset, ``default`` nor a reserved key.
    """
    dataset_classes = dataset_classes or DATASET_CLASSES
    _check_data_keys(data_cfg, {*dataset_classes, *RESERVED_DATA_KEYS, "default"})
    sources: set[str] = set()
    for name, cls in dataset_classes.items():
        try:
            section = data_cfg.get(name) if hasattr(data_cfg, "get") else data_cfg[name]
        except KeyError:
            continue
        if section is None or not hasattr(section, "get"):
            continue
        if any(not _is_skip(section.get(split)) for split in _CONCEPT_SPLITS):
            sources.add(cls.SOURCE_NAME)
    return sources


def _with_transform(
    name: str, split: str, kwargs: dict[str, Any], input_size: Sequence[int]
) -> dict[str, Any]:
    """Return ``kwargs`` with the split's compiled ``transform``.

    Train compiles its YAML spec, or an empty one, so it always ends in Normalize. Val and test take the package eval transform
    and reject a YAML ``transform``.
    """
    if split == "train":
        return {**kwargs, "transform": compile_transform(kwargs.get("transform", {}))}
    if "transform" in kwargs:
        raise ValueError(
            f"{name}.{split}: 'transform' is not configurable on val/test splits; "
            "they resize to model.input_size and normalize"
        )
    return {**kwargs, "transform": build_eval_transform(input_size)}


def build_datasets(
    data_cfg: Any,
    padding: int | None = None,
    *,
    input_size: Sequence[int],
    dataset_classes: dict[str, type[Any]] | None = None,
    names: list[str] | None = None,
    splits: list[str] | None = None,
    verbose: bool = True,
) -> dict[tuple[str, str], Any]:
    """Instantiate every configured ``(dataset, split)`` from ``data_cfg``.

    Args:
        data_cfg: The ``data`` section of the run config (a mapping keyed by
            dataset name, each mapping split name -> split kwargs). A ``repeat``
            key in a split is read by the trainer and never reaches the dataset. A train
            ``transform`` spec reaches it compiled by :func:`compile_transform`; val and test
            splits take no ``transform`` and get :func:`build_eval_transform`.
        padding: Point-annotation padding passed to point-based datasets. Ignored
            for datasets in :data:`NO_PADDING_DATASETS`.
        input_size: ``(height, width)`` of the val/test resize (``model.input_size``).
        dataset_classes: Override the ``name -> class`` registry (used in tests).
        names: If given, only build these dataset names.
        splits: If given, only build these split names (e.g. ``["train"]``).
        verbose: Print a one-line size summary per built dataset.

    Returns:
        Mapping ``(name, split) -> Dataset`` for every non-skipped split.

    Raises:
        ValueError: If a key is neither a dataset nor a reserved key, or ``repeat`` is set on
            a split other than train, or a val/test split sets ``transform``.
    """
    dataset_classes = dataset_classes or DATASET_CLASSES
    _check_data_keys(data_cfg, {*dataset_classes, *RESERVED_DATA_KEYS})
    result: dict[tuple[str, str], Any] = {}

    for name, cls in dataset_classes.items():
        if names is not None and name not in names:
            continue
        section = data_cfg.get(name) if hasattr(data_cfg, "get") else data_cfg[name]
        if section is None:
            continue
        for split, split_cfg in section.items():
            if splits is not None and split not in splits:
                continue
            if _is_skip(split_cfg):
                continue
            if _REPEAT_KEY in split_cfg and split != "train":
                raise ValueError(
                    f"'{_REPEAT_KEY}' is only valid on the train split; found on {name}.{split}"
                )
            dataset_kwargs = {k: v for k, v in split_cfg.items() if k != _REPEAT_KEY}
            dataset_kwargs = _with_transform(name, split, dataset_kwargs, input_size)
            if name in NO_PADDING_DATASETS:
                ds = cls(**dataset_kwargs)
            else:
                ds = cls(**dataset_kwargs, padding=padding)
            result[(name, split)] = ds
            if verbose:
                print(f"{name:>24s} - {split:<5s}: {len(ds):>7d} samples")

    return result
