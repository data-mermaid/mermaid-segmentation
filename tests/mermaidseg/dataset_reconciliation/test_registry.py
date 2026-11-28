"""SourceLabelRegistry: the run's lookups and label maps from a label space."""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

from mermaidseg.dataset_reconciliation.registry import (
    SourceLabelRegistry,
    source_labels_to_target_labels,
)
from tests.mermaidseg.conftest import label_space_reader


def _datasets(run: SimpleNamespace) -> list:
    return list(run.readers.values())


def test_registry_reads_classes_and_concepts_from_the_label_space(label_space_run) -> None:
    registry = SourceLabelRegistry(_datasets(label_space_run), label_space_run.space)

    assert registry.target_id2label == {1: "acropora", 2: "sand"}
    assert registry.target_label2id == {"acropora": 1, "sand": 2}
    assert registry.num_target_classes == 3
    assert registry.concept_id2name[1] == "kingdom__animalia"
    assert registry.concept_id2name[17] == "bleached"
    assert len(registry.concept_id2name) == registry.num_concepts == 17
    assert registry.dataset_offsets == {"coralscapes_v2": 0, "pacific_labeled_corals": 2}
    assert registry.global_id2source == {
        1: ("coralscapes_v2", "acropora"),
        2: ("coralscapes_v2", "sand"),
        3: ("pacific_labeled_corals", "porites"),
    }
    assert registry.num_global_source_classes == 3


def test_duplicate_source_name_raises(label_space_run, tmp_path: Path) -> None:
    again = label_space_reader(tmp_path / "again", "coralscapes_v2", {1: ("acropora", "Acropora")})
    with pytest.raises(ValueError, match="Duplicate source_name 'coralscapes_v2'"):
        SourceLabelRegistry([*_datasets(label_space_run), again], label_space_run.space)


# ---------- source_labels_to_target_labels ----------


def test_source_labels_to_target_labels_indexes_the_lookup_for_a_non_square_batch_of_any_integer_dtype():
    lookup = torch.tensor([0, 5, 7, 5])
    labels = torch.tensor([[[1, 0, 2], [3, 2, 0]]])  # (B=1, H=2, W=3)

    out = source_labels_to_target_labels(labels, lookup)

    assert out.tolist() == [[[5, 0, 7], [5, 7, 0]]]
    assert out.dtype == torch.long

    as_bytes = source_labels_to_target_labels(labels.to(torch.uint8), lookup)
    assert as_bytes.tolist() == [[[5, 0, 7], [5, 7, 0]]]
