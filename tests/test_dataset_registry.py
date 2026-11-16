"""Guards for the DATASET_REGISTRY single-source-of-truth and the uniform dataset
constructor interface that lets the training entrypoint instantiate every dataset the
same way (`cls(**split_cfg, padding=...)`), with no per-dataset special-casing."""

from __future__ import annotations

import inspect

from mermaidseg import datasets as ds_pkg
from mermaidseg.datasets import DATASET_REGISTRY

# The canonical order — must match the pre-registry hardcoded DATASET_CLASSES order in
# scripts/train.py, because iteration order flows into ConcatDataset member order and thus
# the seeded DataLoader's shuffled sample sequence. Reordering silently changes what a given
# seed trains on, breaking baseline reproducibility — so it is asserted, not just the mapping.
_EXPECTED_ORDER = [
    ("pacific_labeled_corals", "PacificLabeledCoralsDataset"),
    ("moorea_labeled_corals", "MooreaLabeledCoralsDataset"),
    ("catlin_seaview", "CatlinSeaviewDataset"),
    ("mermaid", "MermaidDataset"),
    ("coralnet", "CoralNetDataset"),
    ("coralscapes", "CoralscapesDataset"),
    ("coralscapes_v2", "CoralscapesV2Dataset"),
    ("benthos_yuval", "BenthosYuvalCoralsDataset"),
]


def test_registry_maps_expected_names_to_classes():
    assert {name: cls.__name__ for name, cls in DATASET_REGISTRY.items()} == dict(_EXPECTED_ORDER)


def test_registry_preserves_canonical_order():
    """Order is behavior-affecting (ConcatDataset order → seeded shuffle sequence), so
    it must match the pre-registry DATASET_CLASSES order for seed-reproducible runs."""
    assert [(name, cls.__name__) for name, cls in DATASET_REGISTRY.items()] == _EXPECTED_ORDER


def test_registry_excludes_unwired_ucsd_mosaics():
    """UCSDMosaicsDataset is exported but not wired into training — keep it out of the
    registry until it is (documents the intentional omission)."""
    assert "ucsd_mosaics" not in DATASET_REGISTRY
    assert hasattr(ds_pkg, "UCSDMosaicsDataset")  # still exported


def test_every_registered_dataset_accepts_padding():
    """The entrypoint calls ``cls(**split_cfg, padding=...)`` for every dataset, so each
    constructor must accept ``padding`` — either explicitly or via ``**kwargs``
    forwarded to BaseCoralDataset.

    Signature-only (no network/S3 construction).
    """
    for name, cls in DATASET_REGISTRY.items():
        params = inspect.signature(cls.__init__).parameters.values()
        accepts_padding = any(
            p.name == "padding" or p.kind is inspect.Parameter.VAR_KEYWORD for p in params
        )
        assert accepts_padding, f"{name} ({cls.__name__}) constructor cannot accept padding"
