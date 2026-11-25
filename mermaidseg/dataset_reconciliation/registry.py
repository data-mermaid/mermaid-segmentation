"""Joint source-label registry across multiple source datasets.

The :class:`SourceLabelRegistry` reads each registered reader's global offset
and ``labels.parquet``, so the global source ID of a label is ``offset +
label_id`` (with ``0`` reserved for background), and pre-computes the GPU
lookup tensors used at training time:

- ``source_to_target``: ``LongTensor`` of shape ``(N+1,)`` mapping global
  source IDs to MERMAID benthic-attribute target IDs.
- ``source_to_concepts``: ``FloatTensor`` of shape ``(N+1, C)`` mapping global
  source IDs to the per-concept multi-hot row (computed against the MERMAID
  benthic-attribute hierarchy).

The readers already emit masks in this space; see
:func:`mermaidseg.data.factory.source_offsets`.
"""

from __future__ import annotations

import logging
from collections.abc import Mapping
from typing import cast

import numpy as np
import torch

from mermaidseg.data.base import VersionDataset
from mermaidseg.dataset_reconciliation.concept_schema import (
    ConceptSchema,
    build_source_to_concepts,
    concept_label_key,
)
from segdata_spec.taxonomy import roll_up

logger = logging.getLogger(__name__)


def _source_rows(ds: VersionDataset) -> list[tuple[int, str, str | None]]:
    """Return ``(global id, label name, lower-cased target or None)`` per ``labels.parquet`` row."""
    labels = ds.version.labels
    return [
        (ds.global_offset + label_id, name, attribute.lower() if attribute else None)
        for label_id, name, attribute in zip(
            labels.column("label_id").to_pylist(),
            labels.column("name").to_pylist(),
            labels.column("mermaid_benthic_attribute").to_pylist(),
            strict=True,
        )
    ]


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
    """Builds a joint global source-label index across a list of source datasets.

    Args:
        datasets: One reader per source; each label of its ``labels.parquet``
            gets global ID ``global_offset + label_id``.
        target_labels: Optional ordered list of canonical MERMAID
            benthic-attribute target label names (1..M). If ``None``, defaults
            are derived from the union of mapped target names across all
            datasets (sorted alphabetically).
        concept_schema: Concept schema that ``source_to_concepts`` is built from.
        target_label_subset: Optional list/set of MERMAID target label names
            to retain. Source labels mapped to a target outside the subset
            collapse to background (``0``) in ``source_to_target``.
    """

    datasets: list[VersionDataset]
    target_label2id: dict[str, int]
    target_id2label: dict[int, str]
    num_target_classes: int
    source_to_target: torch.Tensor
    source_to_concepts: torch.Tensor
    global_id2source: dict[int, tuple[str, str]]
    dataset_offsets: dict[str, int]
    concept_value2id: dict[str, dict[str, list[int]]]
    concept_id2name: dict[int, str]
    concept_name2id: dict[str, int]

    def __init__(
        self,
        datasets: list[VersionDataset],
        concept_schema: ConceptSchema,
        target_labels: list[str] | None = None,
        label_roll_up: bool = False,
        target_label_subset: list[str] | set[str] | None = None,
        benthic_hierarchy: Mapping[str, str | None] | None = None,
    ):
        """Build the source-to-target (and concept) lookups over the readers' global IDs.

        Args:
            datasets: One reader per source.
            target_labels: Ordered target label names, derived from the datasets if ``None``.
            label_roll_up: Whether unmapped targets roll up to the nearest ancestor in
                ``target_label_subset``.
            target_label_subset: Target label names to retain.
            concept_schema: Concept schema that ``source_to_concepts`` is built from.
            benthic_hierarchy: Lower-cased benthic-attribute name -> parent name (see
                ``segdata_spec.taxonomy.hierarchy_from_table``), required with ``label_roll_up``.

        Raises:
            ValueError: If there are no datasets, two datasets share a ``source_name``,
                ``label_roll_up`` has no ``target_label_subset`` or no ``benthic_hierarchy``.
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
        rows = {ds.source_name: _source_rows(ds) for ds in self.datasets}
        self.dataset_offsets = {ds.source_name: ds.global_offset for ds in self.datasets}
        self.global_id2source = {
            global_id: (source, name)
            for source, source_rows in rows.items()
            for global_id, name, _ in source_rows
        }
        max_global_id = max(self.global_id2source, default=0)

        # Default target vocabulary: the alphabetical union of every non-null
        # benthic attribute across the registered labels.
        if target_labels is None:
            target_labels = sorted(
                {target for source_rows in rows.values() for _, _, target in source_rows if target}
            )

        target_labels = list(dict.fromkeys(t.lower() for t in target_labels))

        # Optional caller-provided whitelist; targets outside `subset` collapse
        # to background in `source_to_target` below.
        if target_label_subset is not None:
            subset = {t.lower() for t in target_label_subset}
            target_labels = [t for t in target_labels if t in subset]
        else:
            subset = None

        self.target_id2label = dict(enumerate(target_labels, start=1))
        self.target_label2id = {v: k for k, v in self.target_id2label.items()}
        self.num_target_classes = len(self.target_id2label) + 1  # +1 for background

        if label_roll_up:
            if subset is None:
                raise ValueError("label_roll_up=True requires target_label_subset to be set")
            if benthic_hierarchy is None:
                raise ValueError("label_roll_up=True requires benthic_hierarchy")
            hierarchy = dict(benthic_hierarchy)

        # Dense lookup `global_source_id -> target_class_id`; index 0 stays
        # background, and any label without a target (or filtered out by
        # `subset`) either has an attempted roll up to a parent class or
        # is left as 0 so it collapses to background at train time.
        source_to_target_np = np.zeros(max_global_id + 1, dtype=np.int64)
        for source_rows in rows.values():
            for global_id, _, target_name in source_rows:
                if label_roll_up:
                    target_name = roll_up(target_name, hierarchy, cast("set[str]", subset))
                if target_name is None:
                    continue
                if subset is not None and target_name not in subset:
                    continue
                source_to_target_np[global_id] = self.target_label2id.get(target_name, 0)
        self._log_target_coverage(rows, source_to_target_np)
        self.source_to_target = torch.from_numpy(source_to_target_np).long()
        self._build_concepts(concept_schema)

    @staticmethod
    def _log_target_coverage(
        rows: Mapping[str, list[tuple[int, str, str | None]]], source_to_target_np: np.ndarray
    ) -> None:
        """Log, per source, how many of its ``labels.parquet`` labels reach a target class.

        Warns when none do: a label-key mismatch otherwise sends every pixel of that source to
        background with no error.
        """
        for source, source_rows in rows.items():
            if not source_rows:
                continue
            n_mapped = sum(source_to_target_np[global_id] != 0 for global_id, _, _ in source_rows)
            msg = f"{source}: {n_mapped} of {len(source_rows)} source labels map to a target class"
            if n_mapped:
                logger.info(msg)
            else:
                logger.warning(msg + "; check source label keys against the mapping.")

    def _build_concepts(self, concept_schema: ConceptSchema) -> None:
        self.concept_value2id = concept_schema.concept_value2id
        self.concept_id2name = concept_schema.channel_id2name()
        self.concept_name2id = {v: k for k, v in self.concept_id2name.items()}
        targets = {ds.source_name: ds.label_targets for ds in self.datasets}
        concept_keys = {
            gid: (source, concept_label_key(source, name, targets[source].get(name)))
            for gid, (source, name) in self.global_id2source.items()
        }
        self.source_to_concepts = build_source_to_concepts(concept_keys, concept_schema)

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
