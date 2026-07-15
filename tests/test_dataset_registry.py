"""Guards for the DATASET_REGISTRY single-source-of-truth and the uniform dataset
constructor interface that lets the training entrypoint instantiate every dataset the
same way (`cls(**split_cfg, padding=...)`), with no per-dataset special-casing."""

from __future__ import annotations

import inspect

from mermaidseg import datasets as ds_pkg
from mermaidseg.datasets import DATASET_REGISTRY

_EXPECTED = {
    "coralnet": "CoralNetDataset",
    "mermaid": "MermaidDataset",
    "catlin_seaview": "CatlinSeaviewDataset",
    "moorea_labeled_corals": "MooreaLabeledCoralsDataset",
    "pacific_labeled_corals": "PacificLabeledCoralsDataset",
    "benthos_yuval": "BenthosYuvalCoralsDataset",
    "coralscapes": "CoralscapesDataset",
    "coralscapes_v2": "CoralscapesV2Dataset",
}


def test_registry_maps_expected_names_to_classes():
    assert {name: cls.__name__ for name, cls in DATASET_REGISTRY.items()} == _EXPECTED


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
