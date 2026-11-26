"""Ground-truth label mapping for the evaluation suite.

Two lookups are built per source dataset:

- ``ClassLookup``: source label name -> MERMAID target class id (0 = ignore).
  This replicates training's ``source_to_target`` with ``label_roll_up=True``:
  the source label is mapped to its ``mermaid_benthic_attribute`` in the dataset
  version's ``labels.parquet`` (``VersionDataset.label_targets``), then rolled up
  through the pinned taxonomy's benthic-attribute hierarchy until it lands on a
  class present in the model's ``id2label`` vocabulary.

- ``ConceptLookup``: source label name -> length-K concept row in the model's
  concept channel order (values ``0=not_given, 1=False, 2=True``). Rows are
  encoded with ``encode_concept_row`` directly from ``class_to_concepts.csv``,
  independent of the global schema width, so they always align to the model's
  ``concept_id2name`` layout.
"""

from __future__ import annotations

import logging
from collections.abc import Mapping
from pathlib import Path

import numpy as np
import pandas as pd
from numpy.typing import NDArray

from mermaidseg.dataset_reconciliation.concepts import (
    encode_concept_row,
    parse_concept_cell,
    read_class_to_concepts,
)
from segdata_spec.taxonomy import roll_up

logger = logging.getLogger(__name__)


class ClassLookup:
    """Maps source label names to MERMAID target class ids (0 = ignore)."""

    def __init__(self, name_to_class_id: dict[str, int], id2label: dict[int, str]):
        """Store the lookup tables.

        Args:
            name_to_class_id: Lower-cased source label name to target class id.
            id2label: Target class id to label name.
        """
        self.name_to_class_id = name_to_class_id
        self.id2label = id2label
        self.unmapped: set[str] = set()

    def class_id(self, source_label_name: str) -> int:
        """Return the target class id of a source label; unmapped labels give 0 (recorded)."""
        key = str(source_label_name).lower()
        cid = self.name_to_class_id.get(key)
        if cid is None:
            self.unmapped.add(key)
            return 0
        return cid


def build_class_lookup(
    label_targets: Mapping[str, str | None],
    source_label_names: list[str],
    id2label: dict[int, str],
    hierarchy: Mapping[str, str | None] | None = None,
) -> ClassLookup:
    """Build a source-label -> MERMAID class-id lookup (training-style, with roll-up).

    Args:
        label_targets: Label name -> ``mermaid_benthic_attribute`` of the dataset version
            (``VersionDataset.label_targets``); ``None`` maps to no target.
        source_label_names: The label names to look up.
        id2label: Target class id to label name; id 0 is ignored.
        hierarchy: Lower-cased benthic-attribute name -> parent name of the pinned taxonomy
            (``mermaidseg.data.factory.load_benthic_hierarchy``), or ``None`` to disable roll-up.
    """
    src_to_tgt = {
        str(name).lower(): (str(target).lower() if target else None)
        for name, target in label_targets.items()
    }

    # id2label is the frozen target vocabulary (0 = ignore/background).
    target_label2id = {str(v).lower(): int(k) for k, v in id2label.items() if int(k) != 0}
    subset = set(target_label2id.keys())
    tree = dict(hierarchy) if hierarchy is not None else {}

    name_to_class_id: dict[str, int] = {}
    for src_name in source_label_names:
        key = str(src_name).lower()
        target = src_to_tgt.get(key)
        rolled = roll_up(target, tree, subset) if target is not None else None
        name_to_class_id[key] = target_label2id.get(rolled, 0) if rolled is not None else 0

    return ClassLookup(name_to_class_id, id2label)


# Phyla whose ``live`` column decides whether a point is living (dead coral keeps its
# coral taxonomy). Every other taxon is living whenever its taxonomy is not ``none``.
LIVE_COLUMN_PHYLA = frozenset({"cnidaria", "echinodermata"})
# Live hard and soft corals: scleractinians, black corals, octocorals, and hydrocorals.
CORAL_ORDERS = frozenset({"scleractinia", "antipatharia"})
CORAL_CLASSES = frozenset({"octocorallia"})
CORAL_FAMILIES = frozenset({"milleporidae", "stylasteridae"})


def _scope_cell(row: pd.Series, column: str, *, rank: bool) -> str:
    """Parsed cell of ``column``; a column the table lacks is ``not_given``."""
    return parse_concept_cell(row[column], rank=rank) if column in row else "not_given"


def point_scopes(row: pd.Series) -> tuple[bool, bool]:
    """``(live, coral)`` flags of one concept CSV row.

    ``live`` is combined per rank with "GT is not ``rank__none``" by
    :class:`~mermaidseg.evaluation.metrics.TaxonomicRankAccuracy`, so it only has to
    reject dead corals / echinoderms. ``coral`` marks live corals in
    ``CORAL_ORDERS`` / ``CORAL_CLASSES`` / ``CORAL_FAMILIES``.
    """
    live_column = _scope_cell(row, "live", rank=False) == "true"
    live = live_column if _scope_cell(row, "phylum", rank=True) in LIVE_COLUMN_PHYLA else True
    coral = live_column and (
        _scope_cell(row, "order", rank=True) in CORAL_ORDERS
        or _scope_cell(row, "class", rank=True) in CORAL_CLASSES
        or _scope_cell(row, "family", rank=True) in CORAL_FAMILIES
    )
    return live, coral


class ConceptLookup:
    """Maps source label names to length-K concept rows in model channel order.

    ``name_to_scope`` holds the per-label ``(live, coral)`` flags of :func:`point_scopes`.
    Labels without an entry are ``(True, False)``: living decided by taxonomy alone.
    """

    def __init__(
        self,
        name_to_row: dict[str, NDArray[np.int8]],
        num_channels: int,
        name_to_scope: dict[str, tuple[bool, bool]] | None = None,
    ):
        """Store the lookup tables.

        Args:
            name_to_row: Lower-cased source label name to concept row.
            num_channels: Number of concept channels in a row.
            name_to_scope: Lower-cased source label name to ``(live, coral)`` flags.
        """
        self.name_to_row = name_to_row
        self.num_channels = num_channels
        self.name_to_scope = name_to_scope or {}
        self._zero = np.zeros(num_channels, dtype=np.int8)
        self.unmapped: set[str] = set()

    def row(self, source_label_name: str) -> NDArray[np.int8]:
        """Return the concept row of a source label; unmapped labels give zeros (recorded)."""
        key = str(source_label_name).lower()
        r = self.name_to_row.get(key)
        if r is None:
            self.unmapped.add(key)
            return self._zero
        return r

    def scopes(self, source_label_name: str) -> tuple[bool, bool]:
        """Return the ``(live, coral)`` flags of a source label."""
        return self.name_to_scope.get(str(source_label_name).lower(), (True, False))


def build_concept_lookup(
    source_name: str,
    source_label_names: list[str],
    concept_names: list[str],
    csv_path: str | Path,
) -> ConceptLookup:
    """Build a source-label -> concept-row lookup aligned to the model channels."""
    df = read_class_to_concepts(csv_path)
    has_live = "live" in df.columns
    if not has_live:
        logger.warning(
            "%s has no 'live' column: living points are decided by taxonomy alone "
            "and the live-coral scope is empty.",
            csv_path,
        )
    df_src = df[df["source_dataset_source"] == source_name.lower()]

    # First row wins for duplicate (source, label) keys.
    row_by_label: dict[str, pd.Series] = {}
    for _, row in df_src.iterrows():
        label = row["source_label_class_name"]
        if label not in row_by_label:
            row_by_label[label] = row

    name_to_row: dict[str, NDArray[np.int8]] = {}
    name_to_scope: dict[str, tuple[bool, bool]] = {}
    for src_name in source_label_names:
        key = str(src_name).lower()
        row = row_by_label.get(key)
        if row is None:
            continue  # -> zeros (ignored) via ConceptLookup.row
        name_to_row[key] = encode_concept_row(row, concept_names)
        if has_live:
            name_to_scope[key] = point_scopes(row)

    return ConceptLookup(name_to_row, len(concept_names), name_to_scope)
