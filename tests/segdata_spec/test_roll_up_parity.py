"""segdata_spec.taxonomy.roll_up must agree with the registry's roll_up_label."""

from __future__ import annotations

import itertools

import pytest

from mermaidseg.dataset_reconciliation.registry import roll_up_label
from segdata_spec.taxonomy import roll_up

HIERARCHY = {
    "acropora": "hard coral",
    "hard coral": None,
    "massive": "hard coral",
    "porites": "massive",
    "bare": None,
    "sand": "bare",
}
LABELS = [*HIERARCHY, "unknown", None]
SUBSETS = [
    set(),
    {"hard coral"},
    {"acropora", "hard coral"},
    {"massive", "hard coral"},
    {"sand"},
]


@pytest.mark.parametrize(("label", "subset"), itertools.product(LABELS, range(len(SUBSETS))))
def test_roll_up_matches_registry(label, subset):
    assert roll_up(label, HIERARCHY, SUBSETS[subset]) == roll_up_label(
        label, HIERARCHY, SUBSETS[subset]
    )
