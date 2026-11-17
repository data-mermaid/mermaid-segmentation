"""Benthic-attribute concept hierarchy and source-label-to-concept mapping.

Exposes the helpers that fetch the MERMAID benthic-attribute hierarchy, build a label/concept matrix
from a list of label names, and a GPU-friendly ``source_labels_to_concepts`` lookup driven by a
precomputed ``(N+1, C)`` float tensor (with row 0 as the all-zeros background row).
"""

from __future__ import annotations

from pathlib import Path
from typing import cast

import pandas as pd
import requests
import torch

DEFAULT_CLASS_TO_CONCEPTS_CSV = (
    Path(__file__).resolve().parents[2] / "configs" / "class_to_concepts.csv"
)

# Ordered taxonomic ranks used throughout the codebase.
# Keep this centralized so callers can import the canonical ordering.
TAXONOMIC_CONCEPTS = [
    "kingdom",
    "phylum",
    "class",
    "order",
    "family",
    "genus",
    # "species",
]

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


def parse_concept_rank(name: str) -> tuple[str | None, str]:
    """Split a concept name into ``(rank, value)`` if it has a ``<rank>__<value>`` form.

    Morphology / health / auxiliary concepts (``oval``, ``bleached``, ``algae`` …) have no ``__``
    separator; for those the rank is ``None`` and ``value`` is the raw name. Only the first ``__``
    is treated as a separator.
    """
    if "__" not in name:
        return None, name
    rank, value = name.split("__", 1)
    return rank, value


def initialize_benthic_hierarchy(
    hierarchy_json_url: str = "https://api.datamermaid.org/v1/benthicattributes/",
) -> dict[str, str | None]:
    """Fetch and build a benthic-attribute name -> parent-name hierarchy.

    See the original implementation in ``mermaidseg.datasets.concepts`` (now moved here) for full
    docstring.
    """
    response = requests.get(hierarchy_json_url, timeout=30)
    response.raise_for_status()
    data = response.json()
    benthic_attributes = data["results"]

    while data["next"]:
        response = requests.get(data["next"], timeout=30)
        response.raise_for_status()
        data = response.json()
        benthic_attributes.extend(data["results"])

    hierarchy_id_dict: dict[int, int | None] = {}
    id2name_dict: dict[int | None, str] = {}
    for attr in benthic_attributes:
        node_id = attr["id"]
        parent_id = attr["parent"]
        hierarchy_id_dict[node_id] = parent_id
        id2name_dict[node_id] = attr["name"]
    return {
        cast("str", id2name_dict.get(node_id)): id2name_dict.get(parent_id)
        for node_id, parent_id in hierarchy_id_dict.items()
    }


def generate_taxonomic_binary_id_mapping(
    df: pd.DataFrame, taxonomic_col: str
) -> dict[str, list[int]] | None:
    """Generate binary-style encodings for a taxonomic rank.

    The concrete values for a rank become one column each. ``not_given`` maps to all zeros, ``none``
    maps to all ones, and each concrete value maps to 2 in its own column and 1 in the others.
    """
    concrete_values = sorted(
        value for value in df[taxonomic_col].dropna().unique() if value not in ["not_given"]
    )
    if not concrete_values:
        return None

    width = len(concrete_values)
    mapping: dict[str, list[int]] = {
        "not_given": [0] * width,
        # "none": [1] * width,
    }
    for index, value in enumerate(concrete_values):
        encoded = [1] * width
        encoded[index] = 2
        mapping[value] = encoded
    return mapping


def generate_binary_concept_id_mapping(
    df: pd.DataFrame, taxonomic_col: str
) -> dict[str | bool, int] | None:
    """Generate a mapping for one hot (not really) encoded concepts.

    "not given" is always mapped to 0.
    If the concept contains both true and false values, then map true to 2 and false to 1.
    Otherwise, map all values to 0 since they don't represent a meaningful concept.
    TODO: update FALSE, False once concept mapping is fixed
    """
    unique_values = df[taxonomic_col].dropna().unique()

    if (False in unique_values or "FALSE" in unique_values) and (
        True in unique_values or "TRUE" in unique_values
    ):
        return {"not_given": 0, False: 1, "FALSE": 1, True: 2, "TRUE": 2}
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
    meta_cols = ["source_dataset_source", "source_label_class_name"]
    if "global_id" in df_mapping.columns:
        meta_cols = ["global_id", *meta_cols]

    encoded_parts: list[pd.DataFrame] = [df_mapping[meta_cols].copy()]
    channel_names: list[str] = []
    concept_value2id: dict[str, dict] = {}

    for col in TAXONOMIC_CONCEPTS:
        taxonomic_binary_mapping = generate_taxonomic_binary_id_mapping(df_mapping, col)
        if taxonomic_binary_mapping is None:
            continue

        concept_value2id[col] = taxonomic_binary_mapping
        concrete_values = [
            value for value in taxonomic_binary_mapping if value not in ["not_given"]
        ]

        mapped = df_mapping[col].map(taxonomic_binary_mapping)
        mapped = mapped.apply(
            lambda v, mapping=taxonomic_binary_mapping: (
                v if isinstance(v, list) else mapping["not_given"]
            )
        )
        encoded_columns = pd.DataFrame(
            mapped.tolist(),
            columns=[f"{col}__{value}" for value in concrete_values],
            index=df_mapping.index,
        )
        encoded_parts.append(encoded_columns)
        channel_names.extend(encoded_columns.columns.tolist())

    binary_concept_columns = list(MORPHOLOGIC_CONCEPTS + HEALTH_CONCEPTS + NONCORAL_CONCEPTS)
    binary_encoded: dict[str, pd.Series] = {}
    for col in binary_concept_columns:
        if col not in df_mapping.columns:
            continue
        binary_mapping = generate_binary_concept_id_mapping(df_mapping, col)
        if binary_mapping is not None:
            concept_value2id[col] = binary_mapping
            channel_names.append(col)
            binary_encoded[col] = df_mapping[col].map(binary_mapping).fillna(0).astype(int)

    if binary_encoded:
        encoded_parts.append(pd.DataFrame(binary_encoded, index=df_mapping.index))

    encoded_df = pd.concat(encoded_parts, axis=1)
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
