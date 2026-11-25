"""Concept helpers: label map to per-pixel concept vectors."""

from __future__ import annotations

import torch

from mermaidseg.dataset_reconciliation.concepts import source_labels_to_concepts

# Row i is the concept vector of label i: label 0 (background) is all zeros.
LOOKUP = torch.tensor([[0.0, 0.0, 0.0], [1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])


def test_non_square_label_map_becomes_channels_first_concepts():
    labels = torch.tensor([[[1, 0, 2], [2, 2, 1]]])  # (B=1, H=2, W=3)

    out = source_labels_to_concepts(labels, LOOKUP)

    assert out.shape == (1, 3, 2, 3)
    assert out[0, :, 0, 0].tolist() == [1.0, 2.0, 3.0]  # label 1 at (0, 0)
    assert out[0, :, 0, 1].tolist() == [0.0, 0.0, 0.0]  # background at (0, 1)
    assert out[0, :, 1, 0].tolist() == [4.0, 5.0, 6.0]  # label 2 at (1, 0)
    assert out[0, :, 1, 2].tolist() == [1.0, 2.0, 3.0]  # label 1 at (1, 2)
