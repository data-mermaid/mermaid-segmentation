"""
title: mermaidseg.datasets.concepts
abstract: Module that contains concept functionality.
author: Viktor Domazetoski
date: 19-11-2025

functions:
- initialize_benthic_hierarchy: Fetch and build a benthic attribute name hierarchy from a paginated JSON API.
- generate_hierarchy_path: Generate the upward hierarchy path for a given label
- initialize_benthic_concepts: Create a sorted list of unique benthic concepts derived from a set of labels.
- map_benthic_to_concept: Map a benthic class label to its corresponding concept one-hot vector.
"""

from typing import Union

import numpy as np
import pandas as pd
import requests
import torch


def initialize_benthic_hierarchy(
    hierarchy_json_url: str = "https://api.datamermaid.org/v1/benthicattributes/",
):
    """
    Fetch and build a benthic attribute name hierarchy from a paginated JSON API.
    This function retrieves benthic attribute records from a REST endpoint (default:
    "https://api.datamermaid.org/v1/benthicattributes/"), following pagination using
    the "next" field in the JSON response until all pages are collected. Each record
    is expected to contain at least the keys "id", "parent", and "name". The function
    constructs and returns a dictionary mapping each attribute name to its parent
    attribute name (or None if the attribute has no parent or the parent ID is not
    present in the retrieved results).
    Parameters
    ----------
    hierarchy_json_url : str, optional
        Base URL of the benthic attributes API to fetch. Defaults to
        "https://api.datamermaid.org/v1/benthicattributes/". The function will issue
        GET requests to this URL and to any subsequent URLs found in the "next"
        field of the JSON responses.
    Returns
    -------
    dict[str, Optional[str]]
        A dictionary where keys are attribute names (str) and values are the parent
        attribute names (str) or None.
    """

    response = requests.get(hierarchy_json_url)
    data = response.json()
    benthic_attributes = data["results"]

    # Keep fetching next pages while there is a 'next' URL
    while data["next"]:
        response = requests.get(data["next"])
        data = response.json()
        benthic_attributes.extend(data["results"])

    hierarchy_id_dict = {}
    id2name_dict = {}
    for attr in benthic_attributes:
        node_id = attr["id"]
        parent_id = attr["parent"]
        hierarchy_id_dict[node_id] = parent_id
        id2name_dict[node_id] = attr["name"]
    hierarchy_dict = {
        id2name_dict.get(node_id, None): id2name_dict.get(parent_id, None)
        for node_id, parent_id in hierarchy_id_dict.items()
    }
    return hierarchy_dict


def generate_hierarchy_path(label: str, hierarchy_dict: dict[str, str]) -> list[str]:
    """
    Generate the upward hierarchy path for a given label using a parent mapping.
    The function walks from the provided label to its parent, then that parent's
    parent, and so on, appending each visited node to a list until a label is
    not present in the mapping or its mapped value is None.
    Args:
        label: The starting label (any hashable object used as a key in hierarchy_dict).
        hierarchy_dict: A mapping (e.g., dict) where each key is a label and its value
            is the parent label or None to indicate the top of the hierarchy.
    Returns:
        list: A list of labels representing the path from the starting label up to
        the top ancestor. The first element is the original label, followed by its
        parent, the parent's parent, etc.
    """

    path = [label]
    label_tmp = label
    while label_tmp in hierarchy_dict and hierarchy_dict[label_tmp] is not None:
        label_tmp = hierarchy_dict[label_tmp]
        path.append(label_tmp)
    return path


def initialize_benthic_concepts(
    labelset_benthic: list[str], hierarchy_dict: dict[str, str]
) -> tuple[list[str], pd.DataFrame]:
    """
    Create a sorted list of unique benthic concepts derived from a set of labels.
    This function iterates over each label in `labelset_benthic`, uses
    `generate_hierarchy_path(label, hierarchy_dict)` to obtain the hierarchical
    concept path for that label, collects all concepts into a set to ensure
    uniqueness, and returns the concepts as an alphabetically sorted list.
    Args:
        labelset_benthic (Iterable[str]): An iterable of label identifiers (e.g.,
            class names or IDs) for which hierarchical concept paths should be
            generated.
        hierarchy_dict (Mapping): A mapping or dictionary that encodes the
            hierarchical relationships used by `generate_hierarchy_path` to build
            a concept path for a given label.
    Returns:
        benthic_concept_set (List[str]): A sorted list of unique concept names (strings) found across
        all hierarchy paths for the provided labels.
        benthic_concept_matrix (pd.DataFrame): A DataFrame with rows indexed by the original labels and columns by the unique concepts,
        initialized with zeros.
    """

    benthic_concept_set = set()
    for label in labelset_benthic:
        benthic_path = generate_hierarchy_path(label, hierarchy_dict)
        for concept in benthic_path:
            benthic_concept_set.add(concept)
    benthic_concept_set = list(benthic_concept_set)
    benthic_concept_set.sort()

    benthic_concept_matrix = pd.DataFrame(
        0, index=labelset_benthic, columns=benthic_concept_set
    )

    for label in labelset_benthic:
        benthic_path = generate_hierarchy_path(label, hierarchy_dict)
        for concept in benthic_path:
            benthic_concept_matrix.at[label, concept] = 1
    return benthic_concept_set, benthic_concept_matrix


def map_benthic_to_concept(
    benthic_label: Union[str, int], benthic_concept_matrix: pd.DataFrame
) -> np.ndarray:
    """
    Map a benthic class label to its corresponding concept one-hot vector.
    Parameters
    ----------
    benthic_label : str
        The benthic class label to look up (commonly a string that matches
        the DataFrame index of `benthic_concept_matrix`).
    benthic_concept_matrix : pd.DataFrame
        A DataFrame whose index contains benthic labels and whose columns
        represent concept dimensions. Each row should encode the mapping
        from a benthic label to a concept vector (e.g., one-hot or multi-hot).
    Returns
    -------
    np.ndarray
        A 1-D NumPy array containing the concept vector for the provided
        `benthic_label`. Shape will be (n_concepts,).
    """

    if isinstance(benthic_label, str) and benthic_label in benthic_concept_matrix.index:
        benthic_one_hot = benthic_concept_matrix.loc[benthic_label, :].values
        return benthic_one_hot
    elif isinstance(benthic_label, int) and benthic_label <= len(
        benthic_concept_matrix.index
    ):
        benthic_one_hot = benthic_concept_matrix.iloc[benthic_label - 1, :].values
        return benthic_one_hot
    else:
        raise ValueError(
            f"Benthic label '{benthic_label}' not found in concept matrix index."
        )


def labels_to_concepts(
    labels: Union[torch.Tensor, np.ndarray], benthic_concept_matrix: pd.DataFrame
) -> Union[torch.Tensor, np.ndarray]:
    """
    labels: torch.Tensor or np.ndarray with shape (B, H, W), dtype int (values 0..N)
    benthic_concept_matrix: pd.DataFrame returned by initialize_benthic_concepts
    Returns: torch.Tensor (B, C, H, W) if input was torch.Tensor, otherwise np.ndarray (B, C, H, W)
    Background label 0 -> all zeros vector.
    """
    # build lookup: row 0 = zeros, rows 1..N = mapping of class i -> concept vector
    vals = benthic_concept_matrix.values.astype(
        np.float32
    )  # shape (n_labels, n_concepts)
    n_labels, n_concepts = vals.shape
    lookup = np.zeros((n_labels + 1, n_concepts), dtype=np.float32)
    lookup[1:] = (
        vals  # assumes integer label i corresponds to row i-1 (map_benthic_to_concept uses that convention)
    )

    # handle torch input
    if isinstance(labels, torch.Tensor):
        device = labels.device
        labels_np = labels.detach().cpu().numpy().astype(np.int64)
        mapped = lookup[labels_np]  # shape (B, H, W, C)
        mapped = np.transpose(mapped, (0, 3, 1, 2))  # (B, C, H, W)
        out = torch.from_numpy(mapped).to(device)
        return out

    # numpy input
    labels = labels.astype(np.int64)
    mapped = lookup[labels]
    return np.transpose(mapped, (0, 3, 1, 2))
