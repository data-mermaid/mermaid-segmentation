"""MetaModel builds the taxonomic and distance losses from config and adds them to the CBM loss."""

from __future__ import annotations

import copy
import math

import pandas as pd
import pytest
import torch

from mermaidseg.dataset_reconciliation.concepts import TAXONOMIC_CONCEPTS
from tests.conftest import IMAGE_SIZE

# Toy six-rank taxonomy. Channels per rank are sorted, like the concept schema.
LINEAGES = {
    "acropora": ("animalia", "cnidaria", "anthozoa", "scleractinia", "acroporidae", "acropora"),
    "porites": ("animalia", "cnidaria", "anthozoa", "scleractinia", "poritidae", "porites"),
    "lithophyllum": (
        "plantae",
        "rhodophyta",
        "florideophyceae",
        "corallinales",
        "corallinaceae",
        "lithophyllum",
    ),
}
CHANNELS = {
    rank: sorted({lineage[i] for lineage in LINEAGES.values()})
    for i, rank in enumerate(TAXONOMIC_CONCEPTS)
}
# Per-rank head probabilities, in channel order.
PROBS = {
    "kingdom": [0.6, 0.4],  # animalia, plantae
    "phylum": [0.7, 0.3],  # cnidaria, rhodophyta
    "class": [0.6, 0.4],  # anthozoa, florideophyceae
    "order": [0.1, 0.9],  # corallinales, scleractinia
    "family": [0.5, 0.2, 0.3],  # acroporidae, corallinaceae, poritidae
    "genus": [0.2, 0.3, 0.5],  # acropora, lithophyllum, porites
}
NUM_CONCEPTS = sum(len(c) for c in CHANNELS.values())
NUM_CLASSES = 3

# Partial loss of an acropora pixel with no uniform mix, summed over adjacent rank pairs:
# cnidaria, anthozoa, scleractinia, {acroporidae, poritidae}, acropora.
ACROPORA_MARGINAL = -math.log(0.7 * 0.6 * 0.9 * 0.8 * 0.2) / len(TAXONOMIC_CONCEPTS)


def _encode(classes: list[str], value: str) -> list[int]:
    return [2 if c == value else 1 for c in classes]


def _concept_value2id() -> dict[str, dict[str, list[int]]]:
    return {
        rank: {"not_given": [0] * len(classes), **{c: _encode(classes, c) for c in classes}}
        for rank, classes in CHANNELS.items()
    }


def _concept_vector(genus: str) -> list[int]:
    lineage = LINEAGES[genus]
    return [
        v for i, rank in enumerate(TAXONOMIC_CONCEPTS) for v in _encode(CHANNELS[rank], lineage[i])
    ]


@pytest.fixture
def mapping_csv(tmp_path):
    rows = [
        {"source_dataset_source": "toy", **dict(zip(TAXONOMIC_CONCEPTS, lineage, strict=True))}
        for lineage in LINEAGES.values()
    ]
    path = tmp_path / "class_to_concepts.csv"
    pd.DataFrame(rows).to_csv(path, index=False)
    return path


def _cfg(minimal_config, mapping_csv, *, mode="concept-bottleneck", loss=None, **training):
    cfg = copy.deepcopy(minimal_config)
    cfg.model.name = "ConceptBottleneckDINOv3"
    cfg.training.training_mode = mode
    cfg.training.concept_mapping_path = str(mapping_csv)
    cfg.training.loss = {"type": "ConceptBottleneckLoss", "ignore_index": 0, **(loss or {})}
    for key, value in training.items():
        cfg.training[key] = value
    return cfg


def _meta(make_meta_model, cfg):
    return make_meta_model(
        cfg,
        num_classes=NUM_CLASSES,
        num_concepts=NUM_CONCEPTS,
        concept_value2id=_concept_value2id(),
    )


def _pin_concept_head(meta) -> None:
    """Zero encoder features leave every pixel's concept logits at log(PROBS)."""
    conv = meta.model.concept_head.classifier
    log_probs = [math.log(p) for rank in TAXONOMIC_CONCEPTS for p in PROBS[rank]]
    with torch.no_grad():
        conv.weight.zero_()
        conv.bias.copy_(torch.tensor(log_probs))


def _batch(genera: list[str]) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """One image per genus, every pixel labelled with that genus and target class 1 or 2."""
    images = torch.randn(len(genera), 3, *IMAGE_SIZE)
    concepts = torch.tensor([_concept_vector(g) for g in genera], dtype=torch.float32)
    concepts = concepts[:, :, None, None].expand(-1, -1, *IMAGE_SIZE).contiguous()
    targets = torch.tensor([1 + i % 2 for i in range(len(genera))])
    targets = targets[:, None, None].expand(-1, *IMAGE_SIZE).contiguous()
    return images, targets, concepts


@pytest.mark.parametrize(
    "training",
    [
        {"taxonomic_loss": {"partial_weight": 1.0}},
        {"distance_loss": {"weight": 1.0}},
    ],
    ids=["taxonomic", "distance"],
)
def test_taxonomy_losses_require_concept_bottleneck(
    minimal_config, make_meta_model, mapping_csv, training
):
    cfg = _cfg(minimal_config, mapping_csv, mode="standard", **training)
    cfg.model.name = "LinearDINOv3"
    cfg.training.loss = {"type": "CrossEntropyLoss"}
    with pytest.raises(ValueError, match="requires training_mode 'concept-bottleneck'"):
        make_meta_model(cfg)


@pytest.mark.parametrize(
    ("section", "inner"),
    [("taxonomic_loss", "partial_loss"), ("distance_loss", None)],
)
@pytest.mark.parametrize(
    ("explicit", "expected"),
    [({}, (0.25, 0.75)), ({"per_pixel_loss_weight": 0.5}, (0.5, 0.75))],
    ids=["inherited", "explicit"],
)
def test_reduction_weights_default_to_the_concept_loss(
    minimal_config, make_meta_model, mapping_csv, section, inner, explicit, expected
):
    weight_key = "partial_weight" if section == "taxonomic_loss" else "weight"
    cfg = _cfg(
        minimal_config,
        mapping_csv,
        loss={"per_pixel_loss_weight": 0.25, "per_image_loss_weight": 0.75},
        **{section: {weight_key: 1.0, **explicit}},
    )
    meta = _meta(make_meta_model, cfg)
    built = getattr(meta, section)
    if inner is not None:
        built = getattr(built, inner)
    assert (built.per_pixel_loss_weight, built.per_image_loss_weight) == expected


@pytest.mark.parametrize("distance_cfg", [{"weight": 0.0}, {}], ids=["zero", "absent"])
def test_distance_loss_is_built_only_for_positive_weight(
    minimal_config, make_meta_model, mapping_csv, distance_cfg
):
    cfg = _cfg(minimal_config, mapping_csv, distance_loss=distance_cfg)
    assert _meta(make_meta_model, cfg).distance_loss is None


def test_total_loss_is_base_plus_taxonomic_plus_distance(
    minimal_config, make_meta_model, mapping_csv
):
    cfg = _cfg(
        minimal_config,
        mapping_csv,
        taxonomic_loss={"partial_weight": 1.0, "kl_weight": 0.5},
        distance_loss={"weight": 0.7},
    )
    meta = _meta(make_meta_model, cfg)
    torch.manual_seed(0)
    images, targets, concepts = _batch(["acropora", "lithophyllum"])

    loss, _, _, _ = meta.batch_predict_loss(images, targets, concepts)

    with torch.no_grad():
        out = meta.model(images)
        base, _ = meta.loss(out.logits, targets, out.concept_logits, concepts)
        taxonomic, _ = meta.taxonomic_loss(out.concept_logits, concepts, targets)
        distance, _ = meta.distance_loss(out.concept_logits, concepts)
    assert min(base.item(), taxonomic.item(), distance.item()) > 0
    assert loss.item() == pytest.approx(base.item() + taxonomic.item() + distance.item(), rel=1e-6)


def test_taxonomic_loss_anneals_with_global_step(minimal_config, make_meta_model, mapping_csv):
    cfg = _cfg(
        minimal_config,
        mapping_csv,
        taxonomic_loss={"partial_weight": 1.0, "partial_uniform_anneal_iters": 10},
    )
    meta = _meta(make_meta_model, cfg)
    _pin_concept_head(meta)
    meta.global_step = 10  # anneal finished: the partial loss is purely marginal

    _, _, _, components = meta.batch_predict_loss(*_batch(["acropora", "acropora"]))

    assert components["partial"] == pytest.approx(ACROPORA_MARGINAL, abs=1e-5)
