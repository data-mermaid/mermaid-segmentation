"""Growth-form / morphology targets for the dual-head (non-CBM) path.

Morphology is a many-to-many axis with taxonomy: channels use the same ``0=unknown,
1=False, 2=True`` encoding as multi-hot concept losses so unknown growth forms
contribute no gradient.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from mermaidseg.dataset_reconciliation.concepts import (
    DEFAULT_CLASS_TO_CONCEPTS_CSV,
    MORPHOLOGIC_CONCEPTS,
)

# Curated short list for the dual morph head (subset of MORPHOLOGIC_CONCEPTS).
DEFAULT_MORPHOLOGY_CHANNELS: tuple[str, ...] = (
    "plating",
    "branching",
    "massive",
    "encrusting",
    "tabular",
)

# CoralNet / MERMAID growth_form_name → morphology channel (lowercase keys).
GROWTH_FORM_TO_CHANNEL: dict[str, str] = {
    "plating": "plating",
    "tabular": "tabular",
    "table": "tabular",
    "branching": "branching",
    "branching coral": "branching",
    "massive": "massive",
    "encrusting": "encrusting",
    "encrusting coral": "encrusting",
}


def _encode_bool_cell(value: object) -> int:
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return 0
    text = str(value).strip().upper()
    if text in {"", "NOT_GIVEN", "NONE", "NAN"}:
        return 0
    if text in {"TRUE", "1", "YES"}:
        return 2
    if text in {"FALSE", "0", "NO"}:
        return 1
    return 0


def morphology_channels_from_config(
    channels: Sequence[str] | None = None,
) -> list[str]:
    """Return the ordered morph channel list, validating against known concepts."""
    chosen = list(channels) if channels is not None else list(DEFAULT_MORPHOLOGY_CHANNELS)
    known = set(MORPHOLOGIC_CONCEPTS)
    unknown = [c for c in chosen if c not in known]
    if unknown:
        raise ValueError(f"Unknown morphology channels: {unknown}")
    return chosen


def build_source_to_morphology_from_concept_csv(
    global_id2source: Mapping[int, tuple[str, str]],
    *,
    channels: Sequence[str] | None = None,
    concept_csv: str | Path = DEFAULT_CLASS_TO_CONCEPTS_CSV,
) -> tuple[torch.Tensor, list[str]]:
    """Build ``(N+1, M)`` morph targets from ``class_to_concepts.csv`` rows.

    Row 0 is all-zeros (background). Source labels without a CSV row stay unknown (0).
    """
    channel_names = morphology_channels_from_config(channels)
    df = pd.read_csv(concept_csv)
    df = df.copy()
    df["source_dataset_source"] = df["source_dataset_source"].astype(str).str.lower()
    df["source_label_class_name"] = df["source_label_class_name"].astype(str).str.lower()
    keyed = df.set_index(["source_dataset_source", "source_label_class_name"], drop=False)

    num_global = max(global_id2source.keys(), default=0)
    table = np.zeros((num_global + 1, len(channel_names)), dtype=np.float32)

    for global_id, (source_name, source_label) in global_id2source.items():
        key = (source_name.lower(), source_label.lower())
        if key not in keyed.index:
            continue
        row = keyed.loc[key]
        if isinstance(row, pd.DataFrame):
            row = row.iloc[0]
        for m_idx, channel in enumerate(channel_names):
            if channel not in row.index:
                continue
            table[global_id, m_idx] = _encode_bool_cell(row[channel])

    return torch.from_numpy(table), channel_names


def growth_form_to_morphology_vector(
    growth_form_name: str | None,
    channel_names: Sequence[str],
) -> list[int]:
    """Map a single growth-form string to a multi-hot morph vector (unknown → all 0)."""
    vector = [0] * len(channel_names)
    if growth_form_name is None:
        return vector
    key = str(growth_form_name).strip().lower()
    if not key or key in {"none", "nan", "null", "not_given"}:
        return vector
    channel = GROWTH_FORM_TO_CHANNEL.get(key)
    if channel is None or channel not in channel_names:
        return vector
    idx = list(channel_names).index(channel)
    for i in range(len(vector)):
        vector[i] = 1
    vector[idx] = 2
    return vector


def source_labels_to_morphology(
    source_labels: torch.Tensor,
    source_to_morphology: torch.Tensor,
) -> torch.Tensor:
    """Gather morph targets for a ``(B, H, W)`` source-label map → ``(B, M, H, W)``."""
    lookup = source_to_morphology.to(device=source_labels.device)
    flat = source_labels.long().reshape(-1).clamp(0, lookup.size(0) - 1)
    gathered = lookup[flat]  # (N, M)
    b, h, w = source_labels.shape
    return gathered.view(b, h, w, -1).permute(0, 3, 1, 2).contiguous()
