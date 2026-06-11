"""Frozen run-scoped concept channel layout from class_to_concepts.csv."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from mermaidseg.dataset_reconciliation.concepts import (
    DEFAULT_CLASS_TO_CONCEPTS_CSV,
    TAXONOMIC_CONCEPTS,
    encode_concept_channels_from_df,
)


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
        binary_taxonomy_encoding: bool = True,
    ) -> ConceptSchema:
        """Build a schema from CSV rows whose ``source_dataset_source`` is in ``sources``."""
        df = pd.read_csv(path)
        df = df.copy()
        df["source_label_class_name"] = df["source_label_class_name"].astype(str).str.lower()
        df["source_dataset_source"] = df["source_dataset_source"].astype(str).str.lower()
        sources_lower = {s.lower() for s in sources}
        df_run = df[df["source_dataset_source"].isin(sources_lower)].reset_index(drop=True)

        encoded_df, channel_names, concept_value2id = encode_concept_channels_from_df(
            df_run, binary_taxonomy_encoding=binary_taxonomy_encoding
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
    for gid, (source, label) in global_id2source.items():
        table[gid] = schema.row_for(source, label)
    return torch.from_numpy(table)
