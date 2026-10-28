"""Expected taxonomic-distance loss on a toy order -> family -> genus taxonomy."""

from __future__ import annotations

import pandas as pd
import pytest
import torch

from mermaidseg.dataset_reconciliation import ConceptSchema
from mermaidseg.dataset_reconciliation.concepts import (
    DEFAULT_CLASS_TO_CONCEPTS_CSV,
    TAXONOMIC_CONCEPTS,
)
from mermaidseg.dataset_reconciliation.taxonomy_tree import TaxonomyTree, build_taxonomy_tree
from mermaidseg.model.distance_loss import TaxonomicDistanceLoss

ORDERS = ["scleractinia", "alcyonacea"]
FAMILIES = ["acroporidae", "poritidae", "alcyoniidae"]
GENERA = ["acropora", "montipora", "isopora", "porites", "goniopora", "sarcophyton", "sinularia"]
PATHS = [
    ("scleractinia", "acroporidae", "acropora"),
    ("scleractinia", "acroporidae", "montipora"),
    ("scleractinia", "acroporidae", "isopora"),
    ("scleractinia", "poritidae", "porites"),
    ("scleractinia", "poritidae", "goniopora"),
    ("alcyonacea", "alcyoniidae", "sarcophyton"),
    ("alcyonacea", "alcyoniidae", "sinularia"),
]
RANKS = {"order": ORDERS, "family": FAMILIES, "genus": GENERA}
ORDER_PROBS = [0.80, 0.20]
FAMILY_PROBS = [0.60, 0.25, 0.15]
GENUS_PROBS = [0.08, 0.25, 0.04, 0.33, 0.12, 0.10, 0.08]
NUM_RANKS = len(TAXONOMIC_CONCEPTS)

# Expected normalised distances for a pixel labelled scleractinia / acroporidae / acropora.
ORDER_EXPECTED = 0.20  # alcyonacea: lineages never meet, 1 / 1
FAMILY_EXPECTED = (0.25 * 1 + 0.15 * 2) / 2  # poritidae: siblings; alcyoniidae: never meet
GENUS_EXPECTED = (0.25 + 0.04) * 1 / 3 + (0.33 + 0.12) * 2 / 3 + (0.10 + 0.08) * 3 / 3


def _encode(classes: list[str], value: str | None) -> list[int]:
    """Taxonomic channel encoding: 2 for the class, 1 for its siblings, all 0 if not given."""
    if value is None:
        return [0] * len(classes)
    return [2 if c == value else 1 for c in classes]


def _concept_value2id(ranks: dict[str, list[str]]) -> dict[str, dict[str, list[int]]]:
    return {
        rank: {"not_given": _encode(classes, None), **{c: _encode(classes, c) for c in classes}}
        for rank, classes in ranks.items()
    }


def _tree(ranks: dict[str, list[str]], paths: list[tuple[str | None, ...]]) -> TaxonomyTree:
    """Taxonomy tree from ``class_to_concepts.csv``-style rows; ``None`` is ``not_given``."""
    rows = [
        {"source_dataset_source": "toy", **dict(zip(ranks, path, strict=True))} for path in paths
    ]
    return build_taxonomy_tree(ranks=tuple(ranks), df=pd.DataFrame(rows).fillna("not_given"))


def _loss(ranks=RANKS, paths=PATHS, **kwargs) -> TaxonomicDistanceLoss:
    return TaxonomicDistanceLoss(_tree(ranks, paths), _concept_value2id(ranks), **kwargs)


def _pixels(paths: list[tuple[str | None, ...]], ranks=RANKS, probs=None):
    """One image with one pixel per path; logits are log of the spec probabilities."""
    labels = torch.tensor(
        [[v for classes, value in zip(ranks.values(), path, strict=True) for v in _encode(classes, value)]
         for path in paths]
    ).T  # (C, W)
    probs = torch.tensor(probs or ORDER_PROBS + FAMILY_PROBS + GENUS_PROBS)
    logits = probs.log()[:, None].expand(-1, len(paths)).clone()
    return logits[None, :, None, :], labels[None, :, None, :]


ACROPORA = ("scleractinia", "acroporidae", "acropora")
FAMILY_ONLY = ("scleractinia", "acroporidae", None)
VOID = (None, None, None)


def test_distance_matrix_is_normalised_tree_distance_in_model_channel_order():
    # ORDERS is not in the tree's sorted channel order, so this also checks the name mapping.
    loss = _loss()
    assert loss.distance_genus[0].tolist() == pytest.approx([0, 1 / 3, 1 / 3, 2 / 3, 2 / 3, 1, 1])
    torch.testing.assert_close(
        loss.distance_family, torch.tensor([[0, 0.5, 1], [0.5, 0, 1], [1, 1, 0]])
    )
    assert loss.distance_order.tolist() == [[0, 1], [1, 0]]


def test_expected_distance_values():
    total, components = _loss()(*_pixels([ACROPORA]))
    assert components["distance/order"] == pytest.approx(ORDER_EXPECTED, abs=1e-6)
    assert components["distance/family"] == pytest.approx(FAMILY_EXPECTED, abs=1e-6)
    assert components["distance/genus"] == pytest.approx(GENUS_EXPECTED, abs=1e-6)
    expected = (ORDER_EXPECTED + FAMILY_EXPECTED + GENUS_EXPECTED) / NUM_RANKS
    assert components["distance"] == pytest.approx(expected, abs=1e-6)
    assert total.item() == pytest.approx(expected, abs=1e-6)


def test_weight_scales_total_not_logged_value():
    total, components = _loss(weight=0.5)(*_pixels([ACROPORA]))
    expected = (ORDER_EXPECTED + FAMILY_EXPECTED + GENUS_EXPECTED) / NUM_RANKS
    assert components["distance"] == pytest.approx(expected, abs=1e-6)
    assert total.item() == pytest.approx(0.5 * expected, abs=1e-6)


def test_near_mistakes_cost_less_than_far_ones():
    loss = _loss()
    _, labels = _pixels([ACROPORA])
    costs = {}
    for name, genus in (("correct", "acropora"), ("sibling", "montipora"), ("same order", "porites"),
                        ("other order", "sarcophyton")):
        logits = torch.full((1, 12, 1, 1), -20.0)
        logits[0, [0, 2 + 0, 5 + GENERA.index(genus)]] = 20.0  # order/family right, genus varies
        costs[name] = loss(logits, labels)[1]["distance/genus"]
    assert costs["correct"] == pytest.approx(0.0, abs=1e-6)
    assert costs["sibling"] == pytest.approx(1 / 3, abs=1e-5)
    assert costs["same order"] == pytest.approx(2 / 3, abs=1e-5)
    assert costs["other order"] == pytest.approx(1.0, abs=1e-5)


def test_unlabelled_ranks_and_void_pixels_contribute_nothing():
    loss = _loss()
    _, components = loss(*_pixels([FAMILY_ONLY]))
    assert components["distance/genus"] == 0.0
    assert components["distance/family"] == pytest.approx(FAMILY_EXPECTED, abs=1e-6)
    with_void, _ = loss(*_pixels([ACROPORA, VOID]))
    without_void, _ = loss(*_pixels([ACROPORA]))
    assert with_void.item() == pytest.approx(without_void.item(), abs=1e-6)
    total, components = loss(*_pixels([VOID]))
    assert total.item() == 0.0 and components["distance"] == 0.0




def test_gradient_is_probability_times_distance_excess():
    logits, labels = _pixels([ACROPORA])
    logits.requires_grad_(True)
    _loss()(logits, labels)[0].backward()
    distance = torch.tensor([0, 1 / 3, 1 / 3, 2 / 3, 2 / 3, 1, 1])
    p = torch.tensor(GENUS_PROBS)
    expected = p * (distance - GENUS_EXPECTED) / NUM_RANKS
    torch.testing.assert_close(logits.grad[0, 5:, 0, 0], expected, atol=1e-6, rtol=0)
    assert logits.grad[0, 5 + 5 :, 0, 0].gt(0).all()  # far mistakes are pushed down
    assert logits.grad[0, 5, 0, 0] < 0  # the label is pushed up


def test_none_is_at_maximum_distance():
    ranks = {"order": ["scleractinia", "none"], "family": ["acroporidae", "poritidae", "none"]}
    paths = [("scleractinia", "acroporidae"), ("scleractinia", "poritidae"), ("none", "none")]
    loss = _loss(ranks, paths)
    torch.testing.assert_close(
        loss.distance_family, torch.tensor([[0, 0.5, 1], [0.5, 0, 1], [1, 1, 0]])
    )
    assert loss.distance_order.tolist() == [[0, 1], [1, 0]]


def test_gradcheck():
    _, labels = _pixels([ACROPORA, FAMILY_ONLY, VOID])
    logits = torch.randn(1, 12, 1, 3, dtype=torch.float64, requires_grad=True)
    loss = _loss()
    assert torch.autograd.gradcheck(lambda x: loss(x, labels)[0], (logits,))


def test_channels_missing_from_tree_raise():
    with pytest.raises(ValueError, match="missing from the taxonomy tree"):
        _loss(paths=PATHS[:-1])  # no sinularia in the tree


def test_real_class_to_concepts():
    sources = {"benthos_yuval", "catlin_seaview", "coralnet", "coralscapes_v2", "mermaid"}
    value2id = ConceptSchema.from_csv(DEFAULT_CLASS_TO_CONCEPTS_CSV, sources=sources).concept_value2id
    loss = TaxonomicDistanceLoss(build_taxonomy_tree(DEFAULT_CLASS_TO_CONCEPTS_CSV), value2id)
    genera = [v for v in value2id["genus"] if v != "not_given"]
    distance = loss.distance_genus
    assert distance.shape == (len(genera), len(genera))
    assert torch.equal(distance, distance.T) and distance.diagonal().eq(0).all()
    assert distance.min() >= 0 and distance.max() == 1
    g = genera.index
    assert distance[g("acropora"), g("montipora")] == pytest.approx(1 / 6)  # same family
    assert distance[g("acropora"), g("none")] == 1
    # Coral Turbinaria is a sibling of Tubastraea (Dendrophylliidae).
    assert distance[g("turbinaria_coral"), g("tubastraea")] == pytest.approx(1 / 6)
