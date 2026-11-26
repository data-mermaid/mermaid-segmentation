"""Frozen run-scoped concept channel layout from class_to_concepts.csv."""

from __future__ import annotations

import logging
from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch

from mermaidseg.dataset_reconciliation.concept_ranks import (
    TAXONOMIC_CONCEPTS,
    binary_offset,
    rank_slices,
)
from mermaidseg.dataset_reconciliation.concepts import (
    DEFAULT_CLASS_TO_CONCEPTS_CSV,
    encode_concept_channels_from_df,
    read_class_to_concepts,
)

logger = logging.getLogger(__name__)

# MERMAID label names are "BA::GF", but class_to_concepts.csv keys their rows by the bare BA.
TARGET_KEYED_CONCEPT_DATASETS: frozenset[str] = frozenset({"mermaid"})


def concept_label_key(dataset: str, label_name: str, target: str | None) -> str:
    """Return the ``class_to_concepts.csv`` label key of one label of ``dataset``.

    Datasets in ``TARGET_KEYED_CONCEPT_DATASETS`` key by ``target`` (the label's
    ``mermaid_benthic_attribute``), so labels sharing a benthic attribute share a row; every other
    dataset keys by ``label_name``. A target-keyed label without a target keeps its name.
    """
    if dataset.lower() in TARGET_KEYED_CONCEPT_DATASETS and target:
        return target
    return label_name


@dataclass(frozen=True)
class ConceptSchema:
    """Run-scoped frozen concept channel layout.

    Channel widths are derived from all CSV rows for the requested sources, independent of which
    labels appear in train vs val splits.
    """

    channel_names: tuple[str, ...]
    concept_value2id: dict[str, dict]
    num_channels: int
    row_by_key: dict[tuple[str, str], np.ndarray]

    @classmethod
    def from_csv(
        cls,
        path: str | Path = DEFAULT_CLASS_TO_CONCEPTS_CSV,
        *,
        sources: set[str],
        removed_concepts: Iterable[str] = (),
    ) -> ConceptSchema:
        """Build a schema from CSV rows whose ``source_dataset_source`` is in ``sources``.

        ``removed_concepts`` (channel names, ``training.concept_removal``) are left out: a removed
        taxonomic value becomes ``not_given`` at its rank, a removed binary concept loses its column.
        Exactly those channels must disappear; anything else is an error.
        """
        df = read_class_to_concepts(path)
        sources_lower = {s.lower() for s in sources}
        df_run = df[df["source_dataset_source"].isin(sources_lower)].reset_index(drop=True)
        removed_concepts = list(dict.fromkeys(removed_concepts))
        if removed_concepts:
            from mermaidseg.dataset_reconciliation.concept_removal import apply_concept_removal

            _, full_channels, _ = encode_concept_channels_from_df(df_run)
            not_channels = sorted(set(removed_concepts) - set(full_channels))
            if not_channels:
                raise ValueError(
                    f"removed concepts that are not channels of this run: {not_channels}"
                )
            df_run = apply_concept_removal(df_run, removed_concepts)

        encoded_df, channel_names, concept_value2id = encode_concept_channels_from_df(df_run)

        if removed_concepts:
            expected = [c for c in full_channels if c not in set(removed_concepts)]
            if list(channel_names) != expected:
                raise RuntimeError(
                    "concept removal changed more than the removed channels: "
                    f"{sorted(set(expected) ^ set(channel_names))}"
                )

        taxonomic_keys = [k for k in concept_value2id if k in TAXONOMIC_CONCEPTS]
        if taxonomic_keys != [k for k in TAXONOMIC_CONCEPTS if k in concept_value2id]:
            raise ValueError(
                f"Taxonomic concepts must follow TAXONOMIC_CONCEPTS order. Got: {taxonomic_keys}"
            )

        row_by_key: dict[tuple[str, str], np.ndarray] = {}
        for _, row in encoded_df.iterrows():
            key = (
                str(row["source_dataset_source"]).lower(),
                str(row["source_label_class_name"]).lower(),
            )
            row_by_key[key] = row[list(channel_names)].to_numpy(dtype=np.float32)

        return cls(
            channel_names=tuple(channel_names),
            concept_value2id=concept_value2id,
            num_channels=len(channel_names),
            row_by_key=row_by_key,
        )

    def row_for(self, source: str, label_name: str) -> np.ndarray:
        """Return the length-``C`` concept vector for ``(source, label_name)``."""
        key = (source.lower(), label_name.lower())
        return self.row_by_key.get(key, np.zeros(self.num_channels, dtype=np.float32))

    def rank_slices(self) -> list[tuple[str, int, int]]:
        """Return ``(rank, offset, width)`` of each taxonomic rank, in channel order."""
        return rank_slices(self.concept_value2id)

    @property
    def binary_offset(self) -> int:
        """Channel index where the binary tail starts."""
        return binary_offset(self.concept_value2id)

    def binary_names(self) -> list[str]:
        """Binary concept names in channel order; each is its own single channel."""
        names = [k for k in self.concept_value2id if k not in TAXONOMIC_CONCEPTS]
        missing = [n for n in names if n not in self.channel_names]
        if missing:
            raise ValueError(f"binary concepts without a channel of the same name: {missing}")
        return names

    def channel_id2name(self) -> dict[int, str]:
        """Map 1-indexed channel id to channel name (e.g. ``kingdom__animalia``)."""
        return {i + 1: name for i, name in enumerate(self.channel_names)}


def build_source_to_concepts(
    global_id2source: dict[int, tuple[str, str]],
    schema: ConceptSchema,
) -> torch.Tensor:
    """Build ``(N+1, C)`` lookup tensor from global source ids and a frozen schema."""
    max_gid = max(global_id2source) if global_id2source else 0
    table = np.zeros((max_gid + 1, schema.num_channels), dtype=np.float32)
    warned: set[tuple[str, str]] = set()
    for gid, (source, label) in global_id2source.items():
        key = (source.lower(), label.lower())
        if key not in schema.row_by_key and key not in warned:
            warned.add(key)
            logger.warning(
                "unmapped source label (concept mapping): source_dataset=%r label=%r",
                key[0],
                key[1],
            )
        table[gid] = schema.row_for(source, label)
    return torch.from_numpy(table)
