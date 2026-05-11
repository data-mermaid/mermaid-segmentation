"""Benthic-attribute concept hierarchy and source-label-to-concept mapping.

Exposes the helpers that fetch the MERMAID benthic-attribute hierarchy, build a label/concept matrix
from a list of label names, and a GPU-friendly ``source_labels_to_concepts`` lookup driven by a
precomputed ``(N+1, C)`` float tensor (with row 0 as the all-zeros background row).
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import requests
import torch


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


def initialize_benthic_concepts(
    labelset_benthic: list[str], hierarchy_dict: dict[str, str]
) -> tuple[list[str], pd.DataFrame]:
    """Create a sorted list of unique benthic concepts and the (label, concept) DataFrame."""
    benthic_concept_set: set[str] = set()
    for label in labelset_benthic:
        benthic_path = generate_hierarchy_path(label, hierarchy_dict)
        for concept in benthic_path:
            benthic_concept_set.add(concept)
    benthic_concept_list = sorted(benthic_concept_set)

    levels = get_hierarchy_level(benthic_concept_list, hierarchy_dict)
    tuples = [(col, levels.get(col)) for col in benthic_concept_list]
    benthic_concept_matrix = pd.DataFrame(0, index=labelset_benthic, columns=benthic_concept_list)
    benthic_concept_matrix.columns = pd.MultiIndex.from_tuples(tuples, names=["concept", "level"])

    for label in labelset_benthic:
        benthic_path = generate_hierarchy_path(label, hierarchy_dict)
        for concept in benthic_path:
            benthic_concept_matrix.loc[label, concept] = 1
    return benthic_concept_list, benthic_concept_matrix


def map_benthic_to_concept(
    benthic_label: str | int, benthic_concept_matrix: pd.DataFrame
) -> np.ndarray:
    """Map a benthic class label to its corresponding concept one-hot vector."""
    if isinstance(benthic_label, str) and benthic_label in benthic_concept_matrix.index:
        return benthic_concept_matrix.loc[benthic_label, :].to_numpy()
    if isinstance(benthic_label, int) and benthic_label <= len(benthic_concept_matrix.index):
        return benthic_concept_matrix.iloc[benthic_label - 1, :].to_numpy()
    raise ValueError(f"Benthic label '{benthic_label}' not found in concept matrix index.")


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
