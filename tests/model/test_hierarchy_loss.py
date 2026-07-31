"""Unit tests for hierarchy distance matrices and level remaps."""

from __future__ import annotations

import torch

from mermaidseg.model.hierarchy_loss import (
    build_distance_matrix,
    build_level_remap,
    discover_level_names,
    path_symmetric_difference_distance,
)

TINY_HIERARCHY = {
    "acropora": "acroporidae",
    "montipora": "acroporidae",
    "acroporidae": "hard coral",
    "porites": "poritidae",
    "poritidae": "hard coral",
    "hard coral": None,
    "sand": None,
}


def test_path_symmetric_difference_identical_is_zero():
    path = ["acropora", "acroporidae", "hard coral"]
    assert path_symmetric_difference_distance(path, path) == 0.0


def test_sibling_distance_less_than_distant():
    id2label = {
        0: "ignore",
        1: "Acropora",
        2: "Montipora",
        3: "Porites",
        4: "Sand",
        5: "Hard coral",
    }
    dist = build_distance_matrix(id2label, TINY_HIERARCHY, ignore_index=0)
    sibling = dist[1, 2].item()
    distant = dist[1, 4].item()
    assert sibling < distant
    assert dist[1, 1].item() == 0.0


def test_level_remap_rolls_children_to_hard_coral():
    id2label = {
        0: "ignore",
        1: "Acropora",
        2: "Porites",
        3: "Sand",
        4: "Hard coral",
    }
    remap = build_level_remap(id2label, TINY_HIERARCHY, "Hard coral", ignore_index=0)
    assert remap[1].item() == 4
    assert remap[2].item() == 4
    assert remap[4].item() == 4
    assert remap[3].item() == 0  # sand does not roll up to hard coral


def test_discover_level_names_finds_vocab_ancestors():
    id2label = {
        0: "ignore",
        1: "Acropora",
        2: "Hard coral",
        3: "Sand",
    }
    names = discover_level_names(id2label, TINY_HIERARCHY, ignore_index=0)
    assert "hard coral" in names
    assert torch.is_tensor(build_distance_matrix(id2label, TINY_HIERARCHY)[1, 2])
