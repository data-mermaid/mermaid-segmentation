"""Source -> MERMAID label maps and the source-to-target lookup."""

from __future__ import annotations

import pytest
import torch

from mermaidseg.dataset_reconciliation import label_mapping as lm
from mermaidseg.datasets.coralscapes_v2.coralscapes_v2_dataset import CORALSCAPES_V2_ID2NAME

STATIC_MAPS = [
    lm.fetch_catlin_seaview_to_mermaid,
    lm.fetch_moorea_labeled_corals_to_mermaid,
    lm.fetch_pacific_labeled_corals_to_mermaid,
    lm.fetch_ucsd_mosaics_to_mermaid,
    lm.fetch_benthos_yuval_to_mermaid,
    lm.coralscapes_v2_to_mermaid,
]


@pytest.mark.parametrize("build", STATIC_MAPS, ids=lambda f: f.__name__)
def test_static_maps_are_lowercase_nonempty_name_to_name(build):
    mapping = build()

    assert mapping
    for source, target in mapping.items():
        assert source == source.lower() and source.strip() == source
        assert isinstance(target, str) and target == target.lower() and target.strip()


@pytest.mark.parametrize(
    ("build", "id2name"),
    [
        (lm.coralscapes_v2_to_mermaid, CORALSCAPES_V2_ID2NAME),
    ],
)
def test_coralscapes_maps_cover_exactly_the_native_class_names(build, id2name):
    assert set(build()) == set(id2name.values())


@pytest.mark.parametrize(
    ("build", "source", "target"),
    [
        (lm.fetch_catlin_seaview_to_mermaid, "cca", "crustose coralline algae"),
        (lm.fetch_moorea_labeled_corals_to_mermaid, "p mass", "porites"),
        (lm.fetch_pacific_labeled_corals_to_mermaid, "unclear", "background"),
        (lm.fetch_ucsd_mosaics_to_mermaid, "favia matthai", "dipsastraea matthaii"),
        (lm.fetch_benthos_yuval_to_mermaid, "rock", "bare substrate"),
        (lm.coralscapes_v2_to_mermaid, "acropora bleached", "bleached coral"),
    ],
)
def test_static_maps_pin_a_known_entry(build, source, target):
    assert build()[source] == target


def test_coralnet_map_reads_the_local_config_with_optional_targets():
    mapping = lm.fetch_coralnet_to_mermaid()

    assert mapping
    assert all(isinstance(key, str) and key for key in mapping)
    assert all(value is None or isinstance(value, str) for value in mapping.values())
    assert any(value is not None for value in mapping.values())


def test_source_labels_to_target_labels_indexes_the_lookup_for_a_non_square_batch():
    lookup = torch.tensor([0, 5, 7, 5])
    labels = torch.tensor([[[1, 0, 2], [3, 2, 0]]])  # (B=1, H=2, W=3)

    out = lm.source_labels_to_target_labels(labels, lookup)

    assert out.tolist() == [[[5, 0, 7], [5, 7, 0]]]
    assert out.dtype == torch.long


def test_source_labels_to_target_labels_accepts_non_long_integer_labels():
    lookup = torch.tensor([0, 5, 7])
    labels = torch.tensor([[[2, 1]]], dtype=torch.uint8)

    assert lm.source_labels_to_target_labels(labels, lookup).tolist() == [[[7, 5]]]
