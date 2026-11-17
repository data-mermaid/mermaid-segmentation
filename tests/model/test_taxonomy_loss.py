"""KL coherence loss on a toy order -> family -> genus taxonomy."""

from __future__ import annotations

import logging
import math

import pandas as pd
import pytest
import torch

from mermaidseg.dataset_reconciliation import ConceptSchema
from mermaidseg.dataset_reconciliation.concepts import (
    DEFAULT_CLASS_TO_CONCEPTS_CSV,
    TAXONOMIC_CONCEPTS,
)
from mermaidseg.dataset_reconciliation.taxonomy_tree import TaxonomyTree, build_taxonomy_tree
from mermaidseg.model.taxonomy_loss import (
    TaxonomicCoherenceLoss,
    TaxonomicLoss,
    build_parent_matrices,
)

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
ORDER_PROBS = [0.80, 0.20]
FAMILY_PROBS = [0.60, 0.25, 0.15]
GENUS_PROBS = [0.08, 0.25, 0.04, 0.33, 0.12, 0.10, 0.08]


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


def _concept_row(ranks: dict[str, list[str]], path: tuple[str | None, ...]) -> list[int]:
    return [
        v
        for classes, value in zip(ranks.values(), path, strict=True)
        for v in _encode(classes, value)
    ]


def _tree(ranks: dict[str, list[str]], paths: list[tuple[str | None, ...]]) -> TaxonomyTree:
    """Taxonomy tree from ``class_to_concepts.csv``-style rows; ``None`` is ``not_given``."""
    rows = [
        {"source_dataset_source": "toy", **dict(zip(ranks, path, strict=True))} for path in paths
    ]
    return build_taxonomy_tree(ranks=tuple(ranks), df=pd.DataFrame(rows).fillna("not_given"))


RANKS = {"order": ORDERS, "family": FAMILIES, "genus": GENERA}


def _pixels(paths: list[tuple[str | None, ...]], dtype=torch.float32):
    """One image with one pixel per path; logits are log of the spec probabilities."""
    labels = torch.tensor([_concept_row(RANKS, path) for path in paths]).T  # (C, W)
    probs = torch.tensor(ORDER_PROBS + FAMILY_PROBS + GENUS_PROBS, dtype=dtype)
    logits = probs.log()[:, None].expand(-1, len(paths)).clone()
    return logits[None, :, None, :], labels[None, :, None, :]


PIXEL_A = ("scleractinia", "acroporidae", "montipora")
PIXEL_B = ("scleractinia", "acroporidae", None)
PIXEL_C = ("scleractinia", None, None)
VOID = (None, None, None)


def test_ancestor_matrix_family_to_genus():
    ancestor = build_parent_matrices(_tree(RANKS, PATHS), _concept_value2id(RANKS))
    expected = torch.tensor(
        [
            [1, 1, 1, 0, 0, 0, 0],
            [0, 0, 0, 1, 1, 0, 0],
            [0, 0, 0, 0, 0, 1, 1],
        ],
        dtype=torch.float32,
    )
    torch.testing.assert_close(ancestor[(1, 2)], expected)
    assert set(ancestor) == {(0, 1), (1, 2)}  # adjacent ranks only


def test_none_is_not_a_descendant_of_concrete_taxa():
    ranks = {"order": ["alcyonacea", "none"], "family": ["alcyoniidae", "none"]}
    paths = [("alcyonacea", "alcyoniidae"), ("alcyonacea", "none"), ("none", "none")]
    ancestor = build_parent_matrices(_tree(ranks, paths), _concept_value2id(ranks))
    assert ancestor[(0, 1)].tolist() == [[1, 0], [0, 1]]


def test_parent_matrices_from_class_to_concepts():
    """The real CSV maps onto the run's concept channels, with Turbinaria split by lineage."""
    sources = {"benthos_yuval", "catlin_seaview", "coralnet", "coralscapes_v2", "mermaid"}
    value2id = ConceptSchema.from_csv(DEFAULT_CLASS_TO_CONCEPTS_CSV, sources=sources).concept_value2id
    matrices = build_parent_matrices(build_taxonomy_tree(DEFAULT_CLASS_TO_CONCEPTS_CSV), value2id)

    families = [v for v in value2id["family"] if v != "not_given"]
    genera = [v for v in value2id["genus"] if v != "not_given"]
    family_genus = matrices[(4, 5)]
    assert family_genus.shape == (len(families), len(genera))
    assert family_genus[families.index("acroporidae"), genera.index("acropora")] == 1
    assert family_genus.sum(dim=0).max() == 1  # every genus has one family
    coral_parents = {
        families[i] for i in family_genus[:, genera.index("turbinaria_coral")].nonzero().flatten()
    }
    algae_parents = {
        families[i] for i in family_genus[:, genera.index("turbinaria_algae")].nonzero().flatten()
    }
    assert coral_parents == {"dendrophylliidae"}
    assert algae_parents == {"sargassaceae"}
    assert family_genus[families.index("none"), genera.index("none")] == 1
    assert family_genus[:, genera.index("none")].sum() == 1  # never a child of a real taxon


def test_channels_missing_from_tree_raise():
    with pytest.raises(ValueError, match="missing from the taxonomy tree"):
        build_parent_matrices(_tree(RANKS, PATHS[:-1]), _concept_value2id(RANKS))  # no sinularia


# --- KL coherence loss -------------------------------------------------------------------------


def _coherence(**kwargs) -> TaxonomicCoherenceLoss:
    return TaxonomicCoherenceLoss(_tree(RANKS, PATHS), _concept_value2id(RANKS), **kwargs)


def _logits(order, family, genus, num_pixels: int = 1, dtype=torch.float32) -> torch.Tensor:
    probs = torch.tensor(order + family + genus, dtype=dtype)
    return probs.log()[None, :, None, None].expand(1, -1, 1, num_pixels).clone()


def _no_labels(num_pixels: int = 1) -> torch.Tensor:
    return torch.zeros(1, 1, num_pixels, dtype=torch.long)


KL_ORDER_FAMILY = 0.8 * math.log(0.8 / 0.85) + 0.2 * math.log(0.2 / 0.15)
KL_FAMILY_GENUS = sum(
    p * math.log(p / q) for p, q in zip(FAMILY_PROBS, [0.37, 0.45, 0.18], strict=True)
)


def test_kl_spec_example_values():
    logits = _logits(ORDER_PROBS, FAMILY_PROBS, GENUS_PROBS)
    total, components = _coherence()(logits, _no_labels())
    assert pytest.approx(0.0090, abs=1e-4) == KL_ORDER_FAMILY
    assert pytest.approx(0.1158, abs=1e-4) == KL_FAMILY_GENUS
    assert components["kl/order->family"] == pytest.approx(KL_ORDER_FAMILY, abs=1e-6)
    assert components["kl/family->genus"] == pytest.approx(KL_FAMILY_GENUS, abs=1e-6)
    assert total.item() == pytest.approx(0.1248, abs=1e-4)


def test_kl_is_independent_of_labels():
    loss = _coherence()
    logits, _ = _pixels([PIXEL_A, PIXEL_B, PIXEL_C, VOID])
    target_labels = torch.tensor([[[3, 1, 1, 0]]])
    _, components = loss(logits, target_labels)
    per_pixel = [loss(logits[..., i : i + 1], _no_labels())[0].item() for i in range(4)]
    assert per_pixel == pytest.approx([KL_ORDER_FAMILY + KL_FAMILY_GENUS] * 4, abs=1e-6)
    assert components["kl/family->genus"] == pytest.approx(KL_FAMILY_GENUS, abs=1e-6)


def test_kl_gradients_follow_stop_grad():
    logits = _logits(ORDER_PROBS, FAMILY_PROBS, GENUS_PROBS).requires_grad_(True)
    _coherence()(logits, _no_labels())[0].backward()
    grad = logits.grad[0, :, 0, 0]
    assert torch.all(grad[:2] == 0)
    torch.testing.assert_close(
        grad[2:5], torch.tensor([0.0353, 0.0147, -0.0500]), atol=1e-4, rtol=0
    )
    parent_ratio = [0.6 / 0.37] * 3 + [0.25 / 0.45] * 2 + [0.15 / 0.18] * 2
    expected = torch.tensor(
        [p * (1 - ratio) for p, ratio in zip(GENUS_PROBS, parent_ratio, strict=True)]
    )
    torch.testing.assert_close(grad[5:], expected, atol=1e-6, rtol=0)
    torch.testing.assert_close(
        grad[5:],
        torch.tensor([-0.0497, -0.1554, -0.0249, 0.1467, 0.0533, 0.0167, 0.0133]),
        atol=1e-4,
        rtol=0,
    )


def test_kl_zero_for_coherent_heads():
    logits = _logits([0.82, 0.18], [0.37, 0.45, 0.18], GENUS_PROBS)
    _, components = _coherence()(logits, _no_labels())
    assert components["kl/order->family"] == pytest.approx(0.0, abs=1e-6)
    assert components["kl/family->genus"] == pytest.approx(0.0, abs=1e-6)


def test_kl_matches_naive_probability_sum():
    torch.manual_seed(0)
    logits = torch.randn(1, 12, 1, 3)
    ancestor = build_parent_matrices(_tree(RANKS, PATHS), _concept_value2id(RANKS))
    order, family, genus = (
        logits[0, s, 0].softmax(dim=0) for s in (slice(0, 2), slice(2, 5), slice(5, 12))
    )
    expected = 0.0
    for p, fine, parents in [(order, family, ancestor[(0, 1)]), (family, genus, ancestor[(1, 2)])]:
        q = parents @ fine
        expected += (p * (p.log() - q.log())).sum(dim=0).mean()
    total, _ = _coherence()(logits, _no_labels(3))
    assert total.item() == pytest.approx(expected.item(), abs=1e-5)


def test_kl_gradcheck_without_stop_grad():
    loss = _coherence(stop_grad_coarse=False)
    labels = _no_labels(3)
    logits = torch.randn(1, 12, 1, 3, dtype=torch.float64, requires_grad=True)
    assert torch.autograd.gradcheck(lambda x: loss(x, labels)[0], (logits,))


def test_coherence_handles_taxonomy_defects(caplog):
    ranks = {
        "family": ["acroporidae", "fungiidae", "lithophyllaceae", "corallinaceae"],
        "genus": ["acropora", "lithophyllum", "orphan"],
    }
    paths = [
        ("acroporidae", "acropora"),
        ("fungiidae", None),
        ("lithophyllaceae", "lithophyllum"),
        ("corallinaceae", "lithophyllum"),
        (None, "orphan"),
    ]
    with caplog.at_level(logging.WARNING):
        loss = TaxonomicCoherenceLoss(_tree(ranks, paths), _concept_value2id(ranks))
    for name in ("fungiidae", "orphan", "lithophyllum"):
        assert name in caplog.text
    torch.testing.assert_close(loss.groups_0, torch.tensor([[1.0, 0.0], [0.0, 0.5], [0.0, 0.5]]))
    # Restricted & renormalised: p over {acroporidae, lithophyllaceae, corallinaceae} = [.5, .25, .25],
    # q = [p(acropora), p(lithophyllum)/2, p(lithophyllum)/2] = [.5, .25, .25] -> coherent.
    logits = torch.tensor([0.3, 0.4, 0.15, 0.15, 0.4, 0.4, 0.2]).log()[None, :, None, None]
    total, _ = loss(logits, torch.zeros(1, 1, 1, dtype=torch.long))
    assert total.item() == pytest.approx(0.0, abs=1e-6)


def test_kl_weight_is_constant():
    concept_logits = _logits(ORDER_PROBS, FAMILY_PROBS, GENUS_PROBS)
    target_labels = _no_labels()
    loss = TaxonomicLoss(_tree(RANKS, PATHS), _concept_value2id(RANKS), kl_weight=0.5)
    total, components = loss(concept_logits, target_labels)
    assert "kl_weight" not in components
    assert components["kl"] > 0.0
    assert total.item() == pytest.approx(0.5 * components["kl"], abs=1e-6)


def test_taxonomic_loss_scales_kl_term():
    concept_logits = _logits(ORDER_PROBS, FAMILY_PROBS, GENUS_PROBS)
    target_labels = _no_labels()
    loss = TaxonomicLoss(_tree(RANKS, PATHS), _concept_value2id(RANKS), kl_weight=0.5)
    total, components = loss(concept_logits, target_labels)

    kl = (KL_ORDER_FAMILY + KL_FAMILY_GENUS) / len(TAXONOMIC_CONCEPTS)
    assert components["kl"] == pytest.approx(kl, abs=1e-6)
    assert total.item() == pytest.approx(0.5 * components["kl"], abs=1e-6)


def test_taxonomic_loss_disabled_is_zero():
    concept_logits, _ = _pixels([PIXEL_A, PIXEL_B, PIXEL_C])
    loss = TaxonomicLoss(_tree(RANKS, PATHS), _concept_value2id(RANKS))
    total, components = loss(concept_logits, torch.tensor([[[1, 2, 2]]]))

    assert loss.coherence_loss is None
    assert total.item() == 0.0 and components == {}
