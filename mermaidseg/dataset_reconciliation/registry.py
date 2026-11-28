"""Joint source-label registry of a run's source datasets, over its label space.

The :class:`SourceLabelRegistry` takes each registered reader's global offset, so the global
source ID of a label is ``offset + label_id`` (with ``0`` reserved for background), and builds
the GPU lookup tensors used at training time from the run's
:class:`~mermaidseg.dataset_reconciliation.label_space.LabelSpace`:

- ``source_to_target``: ``LongTensor`` of shape ``(N+1,)`` mapping global
  source IDs to target class IDs.
- ``source_to_concepts``: ``FloatTensor`` of shape ``(N+1, C)`` mapping global
  source IDs to the per-concept row.

The readers already emit masks in this space; see
:func:`mermaidseg.data.factory.source_offsets`.
"""

from __future__ import annotations

import torch

from mermaidseg.data.base import VersionDataset
from mermaidseg.dataset_reconciliation.concept_schema import ConceptSchema
from mermaidseg.dataset_reconciliation.label_space import LabelSpace


def source_labels_to_target_labels(
    source_labels: torch.Tensor,
    lookup: torch.Tensor,
) -> torch.Tensor:
    """Map a global source-label segmentation map to target MERMAID label IDs.

    Pure GPU index op; ``lookup`` must live on the same device as
    ``source_labels``.

    Args:
        source_labels: Integer source-label map of shape ``(B, H, W)`` with
            values in ``[0, N]``.
        lookup: 1-D long tensor of shape ``(N+1,)`` mapping each global source
            label (0 = background) to its MERMAID target label ID
            (0 = background / unmapped).

    Returns:
        Long tensor of shape ``(B, H, W)`` containing target MERMAID label IDs.
    """
    if source_labels.dtype != torch.long:
        source_labels = source_labels.long()
    return lookup[source_labels]


class SourceLabelRegistry:
    """The lookups and label maps of a run's source datasets, read from its label space.

    Only the label space rows of the registered datasets count; the label space may hold others.
    """

    datasets: list[VersionDataset]
    target_label2id: dict[str, int]
    target_id2label: dict[int, str]
    num_target_classes: int
    source_to_target: torch.Tensor
    source_to_concepts: torch.Tensor
    global_id2source: dict[int, tuple[str, str]]
    dataset_offsets: dict[str, int]
    concept_value2id: dict[str, dict]
    concept_id2name: dict[int, str]
    concept_name2id: dict[str, int]

    def __init__(self, datasets: list[VersionDataset], space: LabelSpace):
        """Build the source-to-target and source-to-concepts lookups over the readers' global ids.

        Args:
            datasets: One reader per source; each label gets global id ``global_offset + label_id``.
            space: The run's label space; it holds a row for every label of ``datasets``.

        Raises:
            ValueError: If there are no datasets or two datasets share a ``source_name``.
        """
        if not datasets:
            raise ValueError("SourceLabelRegistry requires at least one dataset")
        seen_sources: set[str] = set()
        for ds in datasets:
            if ds.source_name in seen_sources:
                raise ValueError(
                    f"Duplicate source_name '{ds.source_name}' across registered datasets"
                )
            seen_sources.add(ds.source_name)

        self.datasets = list(datasets)
        self.dataset_offsets = {ds.source_name: ds.global_offset for ds in self.datasets}
        self.source_to_target, self.source_to_concepts = space.lookups(self.dataset_offsets)
        run_labels = space.source_labels[space.source_labels["dataset"].isin(seen_sources)]
        self.global_id2source = {
            self.dataset_offsets[dataset] + int(label_id): (dataset, native_name)
            for dataset, label_id, native_name in zip(
                run_labels["dataset"],
                run_labels["label_id"],
                run_labels["native_name"],
                strict=True,
            )
        }

        classes = space.classes[space.classes["class_id"] != 0]
        self.target_id2label = {
            int(class_id): name
            for class_id, name in zip(classes["class_id"], classes["name"], strict=True)
        }
        self.target_label2id = {v: k for k, v in self.target_id2label.items()}
        self.num_target_classes = len(self.target_id2label) + 1  # +1 for background

        self.concept_value2id = ConceptSchema.from_label_space(space).concept_value2id
        self.concept_id2name = {
            int(channel_id): name
            for channel_id, name in zip(
                space.concepts["channel_id"], space.concepts["name"], strict=True
            )
        }
        self.concept_name2id = {v: k for k, v in self.concept_id2name.items()}

    @property
    def num_concepts(self) -> int:
        """Number of concept channels."""
        return int(self.source_to_concepts.shape[1])

    @property
    def num_global_source_classes(self) -> int:
        """Number of global source classes (excluding background)."""
        return int(self.source_to_target.shape[0]) - 1

    def to(self, device: torch.device | str) -> SourceLabelRegistry:
        """Move the lookup tensors onto ``device`` (in-place) and return self."""
        self.source_to_target = self.source_to_target.to(device)
        self.source_to_concepts = self.source_to_concepts.to(device)
        return self
