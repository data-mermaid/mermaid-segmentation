"""Benthic-attribute concept hierarchy and source-label-to-concept mapping.

Exposes the helpers that fetch the MERMAID benthic-attribute hierarchy, build
a label/concept matrix from a list of label names, and a GPU-friendly
``source_labels_to_concepts`` lookup driven by a precomputed ``(N+1, C)``
float tensor (with row 0 as the all-zeros background row).
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
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
    "species",
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

HEALTH_CONCEPTS = [
    "dead", 
    "bleached"
]

NONCORAL_CONCEPTS = [
    "algae",
    "background",
    "anthropogenic",
    "trash",
    "transect",
    "macroalgae",
    "dark",
    "human",
    "sand",
    "hard_substrate"
]

def initialize_benthic_hierarchy(
    hierarchy_json_url: str = "https://api.datamermaid.org/v1/benthicattributes/",
) -> dict[str, str | None]:
    """Fetch and build a benthic-attribute name -> parent-name hierarchy.

    See the original implementation in ``mermaidseg.datasets.concepts`` (now
    moved here) for full docstring.
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
    id2name_dict: dict[int, str] = {}
    for attr in benthic_attributes:
        node_id = attr["id"]
        parent_id = attr["parent"]
        hierarchy_id_dict[node_id] = parent_id
        id2name_dict[node_id] = attr["name"]
    return {
        id2name_dict.get(node_id): id2name_dict.get(parent_id)
        for node_id, parent_id in hierarchy_id_dict.items()
    }


def get_hierarchy_level(
    concept_list: list[str], hierarchy_dict: dict[str, str]
) -> dict[str, int | None]:
    """Determine the hierarchy level for each concept in a list."""
    levels: dict[str, int | None] = {}
    for col in concept_list:
        level: int | None = 0
        current = col
        visited = {current}
        while True:
            parent = hierarchy_dict.get(current)
            if parent is None:
                break
            if parent in visited:
                level = None
                break
            level += 1
            visited.add(parent)
            if parent not in hierarchy_dict:
                break
            current = parent
        levels[col] = level
    return levels


def generate_hierarchy_path(label: str, hierarchy_dict: dict[str, str]) -> list[str]:
    """Generate the upward hierarchy path for a given label."""
    path = [label]
    label_tmp = label
    while label_tmp in hierarchy_dict and hierarchy_dict[label_tmp] is not None:
        label_tmp = hierarchy_dict[label_tmp]
        path.append(label_tmp)
    return path


def generate_taxonomic_id_mapping(df: pd.DataFrame, taxonomic_col: str):
    """Generate a mapping of values at different taxonomic ranks to unique integer IDs.

    "not given" is always mapped to 0, "none" (if exists) is mapped to 1, and the rest of the values
    are mapped to integers starting from 2, sorted alphabetically.
    """
    sorted_values = sorted(df[taxonomic_col].dropna().unique())
    sorted_values = (
        ["not_given"]
        + ["none" if "none" in sorted_values else []]
        + [v for v in sorted_values if v not in ["not_given", "none"]]
    )
    return {value: idx for idx, value in enumerate(sorted_values)}


def generate_taxonomic_binary_id_mapping(
    df: pd.DataFrame, taxonomic_col: str
) -> dict[str, list[int]] | None:
    """Generate binary-style encodings for a taxonomic rank.

    The concrete values for a rank become one column each. ``not_given`` maps
    to all zeros, ``none`` maps to all ones, and each concrete value maps to 2
    in its own column and 1 in the others.
    """
    concrete_values = sorted(
        value for value in df[taxonomic_col].dropna().unique() if value not in ["not_given", "none"]
    )
    if not concrete_values:
        return None

    width = len(concrete_values)
    mapping: dict[str, list[int]] = {
        "not_given": [0] * width,
        "none": [1] * width,
    }
    for index, value in enumerate(concrete_values):
        encoded = [1] * width
        encoded[index] = 2
        mapping[value] = encoded
    return mapping


def generate_binary_concept_id_mapping(
    df: pd.DataFrame, taxonomic_col: str
) -> dict[str | int] | None:
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


def initialize_taxonomic_concept_mapping(
    df_mapping: pd.DataFrame, num_global_source: int
) -> tuple[torch.Tensor, dict[str, dict]]:
    """Initialize taxonomic concept mapping by generating a mapping from source labels to taxonomic
    concepts and a dictionary of names to ID mappings for each taxonomic rank."""

    df_mapping_taxonomic = df_mapping[
        ["global_id", "source_label_class_name", "source_dataset_source"]
    ].copy()
    taxonomic_name2id = {}
    for col in TAXONOMIC_CONCEPTS:
        taxonomic_name2id[col] = generate_taxonomic_id_mapping(df_mapping, col)
        df_mapping_taxonomic[col] = (
            df_mapping[col].map(taxonomic_name2id[col]).fillna(0).astype(int)
        )
    df_mapping_taxonomic = df_mapping_taxonomic.sort_values("global_id").reset_index(drop=True)
    source_to_taxonomic_concepts_np = np.zeros(
        (num_global_source + 1, len(TAXONOMIC_CONCEPTS)), dtype=np.int64
    )
    source_to_taxonomic_concepts_np[df_mapping_taxonomic["global_id"].to_numpy()] = (
        df_mapping_taxonomic[TAXONOMIC_CONCEPTS].to_numpy()
    )
    source_to_taxonomic_concepts = torch.from_numpy(source_to_taxonomic_concepts_np).long()
    return source_to_taxonomic_concepts, taxonomic_name2id


def initialize_taxonomic_binary_concept_mapping(
    df_mapping: pd.DataFrame,
    num_global_source: int,
) -> tuple[torch.Tensor, dict[str, dict[str, list[int]]]]:
    """Initialize a binary-style taxonomic concept mapping.

    Each concrete value within a taxonomic rank gets its own output column.
    ``not_given`` is encoded as all zeros and ``none`` is encoded as all ones.
    Concrete values are encoded as 2 at their own column and 1 everywhere else.
    """

    df_mapping_taxonomic = df_mapping[
        ["global_id", "source_label_class_name", "source_dataset_source"]
    ].copy()
    taxonomic_name2id: dict[str, dict[str, list[int]]] = {}
    binary_concept_columns: list[str] = []

    for col in TAXONOMIC_CONCEPTS:
        taxonomic_binary_mapping = generate_taxonomic_binary_id_mapping(df_mapping, col)
        if taxonomic_binary_mapping is None:
            continue

        taxonomic_name2id[col] = taxonomic_binary_mapping
        concrete_values = [
            value for value in taxonomic_binary_mapping if value not in ["not_given", "none"]
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
        df_mapping_taxonomic = pd.concat([df_mapping_taxonomic, encoded_columns], axis=1)
        binary_concept_columns.extend(encoded_columns.columns.tolist())

    df_mapping_taxonomic = df_mapping_taxonomic.sort_values("global_id").reset_index(drop=True)
    source_to_taxonomic_concepts_np = np.zeros(
        (num_global_source + 1, len(binary_concept_columns)), dtype=np.int64
    )
    if binary_concept_columns:
        source_to_taxonomic_concepts_np[df_mapping_taxonomic["global_id"].to_numpy()] = (
            df_mapping_taxonomic[binary_concept_columns].to_numpy()
        )
    source_to_taxonomic_concepts = torch.from_numpy(source_to_taxonomic_concepts_np).long()
    return source_to_taxonomic_concepts, taxonomic_name2id


def initialize_binary_concept_mapping(
    df_mapping: pd.DataFrame, num_global_source: int
) -> tuple[torch.Tensor, dict[str, dict]]:
    """Initialize binary concept mapping by generating a mapping from source labels to the binary
    concepts and a dictionary of names to ID mappings for each taxonomic rank."""

    binary_concept_columns = list(
        MORPHOLOGIC_CONCEPTS + HEALTH_CONCEPTS + NONCORAL_CONCEPTS
    )

    df_mapping_binary = df_mapping[
        ["global_id", "source_label_class_name", "source_dataset_source"]
    ].copy()
    binary_name2id = {}
    valid_binary_concept_columns = []
    for col in binary_concept_columns:
        binary_mapping = generate_binary_concept_id_mapping(df_mapping, col)
        if binary_mapping is not None:
            binary_name2id[col] = binary_mapping
            valid_binary_concept_columns.append(col)
            df_mapping_binary[col] = df_mapping[col].map(binary_name2id[col]).fillna(0).astype(int)

    df_mapping_binary = df_mapping_binary.sort_values("global_id").reset_index(drop=True)
    source_to_binary_concepts_np = np.zeros(
        (num_global_source + 1, len(valid_binary_concept_columns)), dtype=np.int64
    )
    source_to_binary_concepts_np[df_mapping_binary["global_id"].to_numpy()] = df_mapping_binary[
        valid_binary_concept_columns
    ].to_numpy()
    source_to_binary_concepts = torch.from_numpy(source_to_binary_concepts_np).long()
    return source_to_binary_concepts, binary_name2id


def initialize_benthic_concepts(
    mapping_location: str | Path = DEFAULT_CLASS_TO_CONCEPTS_CSV,
    global_id2source: dict | None = None,
    global_idmask: dict | None = None,
    binary_taxonomy_encoding: bool = True,
):
    """The function initializes benthic concepts by loading a concept mapping file, subsetting it to
    the source label registry, and generating mappings for taxonomic and one-hot encoded concepts.

    It returns the processed concept mapping DataFrame, tensors for source to concept mappings, and
    dictionaries for taxonomic and one-hot concept ID mappings.
    """
    df_mapping = pd.read_csv(mapping_location)
    num_global_source = len(global_id2source) if global_id2source is not None else len(df_mapping)

    if global_id2source is not None:
        # if global_idmask is not None:
        # global_id2source = {
        #     global_id: source
        #     for global_id, source in global_id2source.items()
        #     if global_idmask.get(global_id, False)
        # }
        df_id2source = pd.DataFrame(
            global_id2source.values(), columns=["source_dataset_source", "source_label_class_name"]
        )
        df_id2source["global_id"] = global_id2source.keys()
        df_id2source = df_id2source[
            ["global_id", "source_dataset_source", "source_label_class_name"]
        ]

        df_id2source["source_label_class_name"] = df_id2source["source_label_class_name"].apply(
            lambda s: s.lower()
        )

        df_mapping = df_mapping.merge(
            df_id2source, on=["source_label_class_name", "source_dataset_source"], how="right"
        )

        ### TODO: FIX BUG - This will be resolved once the class_to_concepts.csv file is fully updated
        # if df_mapping.shape[0] != df_id2source.shape[0]:
        #     raise ValueError(
        #         "The concept map has a different number of rows than the source label registry after merging. "
        #         "Please check the mapping and registry for consistency as some source labels in the registry may not have a corresponding mapping entry or vice versa."
        #     )

    if binary_taxonomy_encoding:
        source_to_taxonomic_concepts, taxonomic_mapping_dictionary = (
            initialize_taxonomic_binary_concept_mapping(
                df_mapping=df_mapping, num_global_source=num_global_source
            )
        )
    else:
        source_to_taxonomic_concepts, taxonomic_mapping_dictionary = (
            initialize_taxonomic_concept_mapping(
                df_mapping=df_mapping, num_global_source=num_global_source
            )
        )

    source_to_binary_concepts, binary_mapping_dictionary = initialize_binary_concept_mapping(
        df_mapping=df_mapping, num_global_source=num_global_source
    )

    source_to_concepts = torch.cat([source_to_taxonomic_concepts, source_to_binary_concepts], dim=1)
    concept_value2id = {**taxonomic_mapping_dictionary, **binary_mapping_dictionary}

    if list(concept_value2id.keys())[:7] != TAXONOMIC_CONCEPTS:
        raise ValueError(
            f"The first 7 concepts in the concept_value2id mapping must be the taxonomic concepts in the following order: {TAXONOMIC_CONCEPTS}. "
            f"Currently, the first 7 concepts are: {list(concept_value2id.keys())[:7]}. "
            f"Please check the concept mapping initialization to ensure that the taxonomic concepts are correctly identified and ordered at the beginning of the mapping."
        )
    return df_mapping, source_to_concepts, concept_value2id


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


def postprocess_predicted_concepts(
    pixel_probs: np.ndarray,
    concept_matrix: pd.DataFrame,
    conceptid2labelid: dict[int, int] | None = None,
    threshold: float = 0.5,
) -> torch.Tensor:
    """Vectorized hierarchical concept prediction for a batch.

    Same behaviour as the legacy implementation in
    ``mermaidseg.datasets.concepts`` (kept verbatim modulo the new module
    location).
    """
    original_shape = pixel_probs.shape
    if pixel_probs.ndim == 3:
        pixel_probs = pixel_probs[np.newaxis, ...]

    B, C, H, W = pixel_probs.shape

    flat_probs = pixel_probs.reshape(B * H * W, C)
    concept_levels = concept_matrix.columns.get_level_values("level").to_numpy()

    predictions = np.full(B * H * W, -1, dtype=np.int32)

    for level in [4, 3, 2, 1, 0]:
        level_mask = concept_levels == level
        if not level_mask.any():
            continue

        level_probs = flat_probs[:, level_mask]

        unassigned = predictions == -1
        if not unassigned.any():
            break
        max_prob_at_level = level_probs[unassigned].max(axis=1)
        above_threshold = max_prob_at_level > threshold
        if above_threshold.any():
            best_concept_in_level = level_probs[unassigned].argmax(axis=1)
            level_indices = np.where(level_mask)[0]
            global_indices = level_indices[best_concept_in_level]
            unassigned_indices = np.where(unassigned)[0]
            pixels_to_assign = unassigned_indices[above_threshold]
            predictions[pixels_to_assign] = global_indices[above_threshold]

    predictions = predictions.reshape(B, H, W)
    if len(original_shape) == 3:
        predictions = predictions[0]

    return torch.from_numpy(predictions).long()
