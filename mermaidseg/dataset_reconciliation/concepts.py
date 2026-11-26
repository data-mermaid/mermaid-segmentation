"""Benthic-attribute concept hierarchy and source-label-to-concept mapping.

Exposes the helpers that build a label/concept matrix from a list of label names, and a
GPU-friendly ``source_labels_to_concepts`` lookup driven by a precomputed
``(N+1, C)`` float tensor (with row 0 as the all-zeros background row).
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping, Sequence
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from numpy.typing import NDArray

from mermaidseg.dataset_reconciliation.concept_ranks import TAXONOMIC_CONCEPTS

DEFAULT_CLASS_TO_CONCEPTS_CSV = (
    Path(__file__).resolve().parents[2] / "configs" / "class_to_concepts.csv"
)


MORPHOLOGIC_CONCEPTS = [
    "oval",
    "arborescent",
    "encrusting",
    "digitate",
    "meandroid",
    "columnar",
    "free_living",
    "plating",
    "fleshy",
    "submassive",
    "round",
    "massive",
    "tubular",
    "bushy",
    "external_polyps",
    "foliose",
    "solitary",
    "brain",
    "phaceloid",
    "branching",
    "tabular",
    "corymbose",
    "lobed_brain",
    "cup_coral",
]

HEALTH_CONCEPTS = ["live", "bleached"]

NONCORAL_CONCEPTS = [
    "algae",
    "background",
    "anthropogenic",
    "trash",
    "transect",
    "transect_line",
    "macroalgae",
    "dark",
    "human",
    "sand",
    "hard_substrate",
    "rubble",
    "calcifying",
    "fish",
    "water surface",
    "water column",
]


BINARY_CELLS = frozenset({"true", "false", "not_given"})
_RANK_PLACEHOLDERS = frozenset({"", "nan", "not given"})
_BINARY_CODES = {"not_given": 0, "false": 1, "true": 2}


def read_class_to_concepts(path: str | Path = DEFAULT_CLASS_TO_CONCEPTS_CSV) -> pd.DataFrame:
    """Read ``class_to_concepts.csv`` with every cell a string and the label keys lower-cased.

    ``dtype=str, keep_default_na=False`` keeps pandas from turning a column into bools or NaN.
    """
    df = pd.read_csv(path, dtype=str, keep_default_na=False)
    for key in ("source_dataset_source", "source_label_class_name"):
        df[key] = df[key].str.lower()
    return df


def parse_concept_cell(value: object, *, rank: bool) -> str:
    """Return one concept cell as its lower-cased value.

    A binary cell is ``"true"``, ``"false"`` or ``"not_given"``. A rank cell is ``"not_given"`` or
    a taxon, which may be ``"none"``. Case and surrounding whitespace are ignored. Any other cell,
    including ``""``, ``"nan"`` and ``"not given"``, raises ``ValueError``.
    """
    cell = value.strip().lower() if isinstance(value, str) else None
    if cell is None or (cell in _RANK_PLACEHOLDERS if rank else cell not in BINARY_CELLS):
        kind = "taxonomic" if rank else "binary"
        raise ValueError(f"invalid {kind} concept cell {value!r}")
    return cell


def encode_concept_row(row: Mapping[str, object], channel_names: Sequence[str]) -> NDArray[np.int8]:
    """Encode one concept-table row into ``channel_names`` order (0 not_given, 1 False, 2 True).

    A channel ``rank__value`` is 2 when the row has ``value`` at ``rank`` and 1 when it has another
    value that has a channel. ``not_given``, a value without a channel and a missing column give 0
    at every channel of the rank. A binary channel reads the column of the same name; a missing
    column gives 0.
    """
    channels = set(channel_names)
    out = np.zeros(len(channel_names), dtype=np.int8)
    for i, channel in enumerate(channel_names):
        rank, sep, value = channel.partition("__")
        if sep and rank in TAXONOMIC_CONCEPTS:
            if rank not in row:
                continue
            cell = parse_concept_cell(row[rank], rank=True)
            if cell == value:
                out[i] = 2
            elif cell != "not_given" and f"{rank}__{cell}" in channels:
                out[i] = 1
        elif channel in row:
            out[i] = _BINARY_CODES[parse_concept_cell(row[channel], rank=False)]
    return out


def generate_taxonomic_binary_id_mapping(
    df: pd.DataFrame, taxonomic_col: str
) -> dict[str, list[int]] | None:
    """Generate binary-style encodings for a taxonomic rank.

    Every value for a rank except ``not_given`` becomes one column; ``none`` is a value like any
    other. ``not_given`` maps to all zeros, and each value maps to 2 in its own column and 1 in the
    others.
    """
    concrete_values = sorted(
        {parse_concept_cell(value, rank=True) for value in df[taxonomic_col].dropna().unique()}
        - {"not_given"}
    )
    if not concrete_values:
        return None

    width = len(concrete_values)
    mapping: dict[str, list[int]] = {"not_given": [0] * width}
    for index, value in enumerate(concrete_values):
        encoded = [1] * width
        encoded[index] = 2
        mapping[value] = encoded
    return mapping


def generate_binary_concept_id_mapping(
    df: pd.DataFrame, taxonomic_col: str
) -> dict[str, int] | None:
    """Generate the ``not_given`` / ``FALSE`` / ``TRUE`` codes of a binary concept column.

    A column holding both true and false values maps ``not_given`` to 0, false to 1 and true to 2.
    Any other column is not a meaningful concept and has no mapping.
    """
    cells = {parse_concept_cell(value, rank=False) for value in df[taxonomic_col].dropna().unique()}
    if {"true", "false"} <= cells:
        return {"not_given": 0, "FALSE": 1, "TRUE": 2}
    return None


def encode_concept_channels_from_df(
    df_mapping: pd.DataFrame,
) -> tuple[pd.DataFrame, list[str], dict[str, dict]]:
    """Encode concept channels from a mapping DataFrame.

    Channel widths are derived from ``df_mapping`` alone (not from a registry
    subset), so the same DataFrame always yields the same layout.

    Returns:
        encoded_df: meta columns plus one column per concept channel.
        channel_names: ordered channel column names.
        concept_value2id: structure expected by loss/model slicing.
    """
    meta = df_mapping[["source_dataset_source", "source_label_class_name"]].copy()
    channel_names: list[str] = []
    concept_value2id: dict[str, dict] = {}

    for col in TAXONOMIC_CONCEPTS:
        taxonomic_binary_mapping = generate_taxonomic_binary_id_mapping(df_mapping, col)
        if taxonomic_binary_mapping is None:
            continue
        concept_value2id[col] = taxonomic_binary_mapping
        channel_names.extend(
            f"{col}__{value}" for value in taxonomic_binary_mapping if value != "not_given"
        )

    for col in MORPHOLOGIC_CONCEPTS + HEALTH_CONCEPTS + NONCORAL_CONCEPTS:
        if col not in df_mapping.columns:
            continue
        binary_mapping = generate_binary_concept_id_mapping(df_mapping, col)
        if binary_mapping is not None:
            concept_value2id[col] = binary_mapping
            channel_names.append(col)

    rows = [encode_concept_row(row, channel_names) for row in df_mapping.to_dict("records")]
    codes = np.stack(rows) if rows else np.zeros((0, len(channel_names)), dtype=np.int8)
    encoded_df = pd.concat(
        [meta, pd.DataFrame(codes, columns=channel_names, index=df_mapping.index)], axis=1
    )
    return encoded_df, channel_names, concept_value2id


def source_labels_to_concepts(
    source_labels: torch.Tensor,
    lookup: torch.Tensor,
) -> torch.Tensor:
    """Map an integer source-label segmentation map to a per-pixel concept vector.

    Pure GPU index op; ``lookup`` is expected to live on the same device as
    ``source_labels``.

    Args:
        source_labels: Integer label map with shape ``(B, H, W)``, values in
            ``[0, N]`` where 0 is background. Must be a long tensor.
        lookup: Concept lookup of shape ``(N+1, C)``; row 0 is the all-zeros
            background row.

    Returns:
        Concept map with shape ``(B, C, H, W)`` on the same device as inputs.
    """
    if source_labels.dtype != torch.long:
        source_labels = source_labels.long()
    mapped = lookup[source_labels]  # (B, H, W, C)
    return mapped.permute(0, 3, 1, 2).contiguous()


NONE_TAXON = "none"


def concept_kind(channel: str) -> str:
    """``genus__acropora`` -> ``genus``; a binary concept -> ``binary``."""
    rank = channel.split("__", 1)[0] if "__" in channel else None
    return rank if rank in TAXONOMIC_CONCEPTS else "binary"


def apply_concept_removal(df_mapping: pd.DataFrame, concepts: Iterable[str]) -> pd.DataFrame:
    """Concept table with ``concepts`` removed, before channel encoding.

    ``rank__value`` sets that rank to ``not_given`` where it equals ``value`` (coarser ranks stay);
    a binary concept's column is dropped. ``rank__none`` cannot be removed: ``none`` marks
    non-organisms and is not a taxon.
    """
    df = df_mapping.copy()
    concepts = list(dict.fromkeys(concepts))
    for concept in concepts:
        kind = concept_kind(concept)
        if kind == "binary":
            if concept not in df.columns:
                raise ValueError(f"cannot remove {concept!r}: not a column of the concept table")
            df = df.drop(columns=concept)
            continue
        value = concept.split("__", 1)[1]
        if value == NONE_TAXON:
            raise ValueError(f"cannot remove {concept!r}: 'none' is not a taxon")
        hit = df[kind].astype(str) == value
        if not hit.any():
            raise ValueError(
                f"cannot remove {concept!r}: no {kind} == {value!r} in the concept table"
            )
        df.loc[hit, kind] = "not_given"
    return df
