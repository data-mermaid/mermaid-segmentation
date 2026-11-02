"""Shared dataset construction used by training and prefetch scripts.

Keeping the ``(name, split) -> Dataset`` build loop in one place guarantees the
training job and the cache-prefetch job agree on exactly which source objects a
run needs, so an offline (AWS-free) training run never hits an unwarmed key.
"""

from __future__ import annotations

import logging
from typing import Any

from mermaidseg.datasets.benthos_yuval import BenthosYuvalCoralsDataset
from mermaidseg.datasets.catlin_seaview import CatlinSeaviewDataset
from mermaidseg.datasets.coralnet import CoralNetDataset
from mermaidseg.datasets.coralscapes_v2 import CoralscapesV2Dataset
from mermaidseg.datasets.mermaid import MermaidDataset
from mermaidseg.datasets.moorea_labeled_corals import MooreaLabeledCoralsDataset
from mermaidseg.datasets.pacific_labeled_corals import PacificLabeledCoralsDataset
from mermaidseg.datasets.ucsd_mosaics import UCSDMosaicsDataset

logger = logging.getLogger(__name__)

# Canonical dataset registry (name -> class). Order is deterministic so runs and
# prefetch produce identical iteration order.
DATASET_CLASSES: dict[str, type] = {
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
    """True when a split config means "skip this split".

    ``data_config.yaml`` uses a literal ``None`` which PyYAML may read as the
    Python ``None`` or (in some contexts) the string ``"None"``; treat both as a
    skip.
    """
    return split_cfg is None or split_cfg == "None"


def build_datasets(
    data_cfg: Any,
    padding: int | None = None,
    *,
    dataset_classes: dict[str, type] | None = None,
    names: list[str] | None = None,
    splits: list[str] | None = None,
    verbose: bool = True,
) -> dict[tuple[str, str], Any]:
    """Instantiate every configured ``(dataset, split)`` from ``data_cfg``.

    Args:
        data_cfg: The ``data`` section of the run config (a mapping keyed by
            dataset name, each mapping split name -> split kwargs).
        padding: Point-annotation padding passed to point-based datasets. Ignored
            for datasets in :data:`NO_PADDING_DATASETS`.
        dataset_classes: Override the ``name -> class`` registry (used in tests).
        names: If given, only build these dataset names.
        splits: If given, only build these split names (e.g. ``["train"]``).
        verbose: Print a one-line size summary per built dataset.

    Returns:
        Mapping ``(name, split) -> Dataset`` for every non-skipped split.
    """
    dataset_classes = dataset_classes or DATASET_CLASSES
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
            if name in NO_PADDING_DATASETS:
                ds = cls(**split_cfg)
            else:
                ds = cls(**split_cfg, padding=padding)
            result[(name, split)] = ds
            if verbose:
                print(f"{name:>24s} - {split:<5s}: {len(ds):>7d} samples")

    return result
