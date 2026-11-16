"""Hierarchy helpers for taxonomical class loss and metrics.

Builds a class-to-class tree-distance matrix and coarse level remaps from a MERMAID
benthic-attribute ``{child: parent}`` dict and an ``id2label`` map.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence

import torch

from mermaidseg.dataset_reconciliation.concepts import (
    generate_hierarchy_path,
    normalize_benthic_hierarchy,
)


def _normalize_name(name: str) -> str:
    return name.strip().lower()


def _label_paths(
    id2label: Mapping[int, str],
    hierarchy: Mapping[str, str | None],
    *,
    ignore_index: int = 0,
) -> dict[int, list[str]]:
    """Return lowercase ancestor paths (leaf → root) for each non-ignore class id."""
    hier = normalize_benthic_hierarchy(dict(hierarchy))
    paths: dict[int, list[str]] = {}
    for class_id, name in id2label.items():
        if class_id == ignore_index:
            continue
        key = _normalize_name(str(name))
        path = [_normalize_name(p) for p in generate_hierarchy_path(key, hier)]
        paths[class_id] = path
    return paths


def path_symmetric_difference_distance(path_a: Sequence[str], path_b: Sequence[str]) -> float:
    """Symmetric-difference size of two ancestor paths (0 when identical)."""
    set_a = set(path_a)
    set_b = set(path_b)
    return float(len(set_a.symmetric_difference(set_b)))


def build_distance_matrix(
    id2label: Mapping[int, str],
    hierarchy: Mapping[str, str | None],
    *,
    ignore_index: int = 0,
    num_classes: int | None = None,
) -> torch.Tensor:
    """Build a dense ``[C, C]`` pairwise tree-distance matrix.

    Distance is the symmetric difference of hierarchy paths. Classes missing from the
    hierarchy keep a path of ``[name]`` only. Pairs involving ``ignore_index`` are set
    to 0 (masked out of the loss by the valid-pixel mask). When either label is unknown
    to ``id2label``, distance falls back to the matrix max after an initial pass (or
    ``1.0`` if all finite distances are zero).
    """
    if num_classes is None:
        num_classes = max(id2label.keys(), default=-1) + 1
    if num_classes <= 0:
        return torch.zeros(0, 0)

    paths = _label_paths(id2label, hierarchy, ignore_index=ignore_index)
    distance = torch.zeros(num_classes, num_classes, dtype=torch.float32)

    known_ids = sorted(paths)
    for i in known_ids:
        for j in known_ids:
            distance[i, j] = path_symmetric_difference_distance(paths[i], paths[j])

    finite_max = float(distance.max().item()) if known_ids else 0.0
    fallback = finite_max if finite_max > 0.0 else 1.0

    for i in range(num_classes):
        if i == ignore_index:
            continue
        if i not in paths:
            distance[i, :] = fallback
            distance[:, i] = fallback
            distance[i, i] = 0.0
            continue
        for j in range(num_classes):
            if j == ignore_index or j in paths:
                continue
            distance[i, j] = fallback
            distance[j, i] = fallback

    distance[ignore_index, :] = 0.0
    distance[:, ignore_index] = 0.0
    return distance


def _name_to_id(id2label: Mapping[int, str], *, ignore_index: int = 0) -> dict[str, int]:
    out: dict[str, int] = {}
    for class_id, name in id2label.items():
        if class_id == ignore_index:
            continue
        out[_normalize_name(str(name))] = int(class_id)
    return out


def discover_level_names(
    id2label: Mapping[int, str],
    hierarchy: Mapping[str, str | None],
    *,
    ignore_index: int = 0,
) -> list[str]:
    """Ancestor names that appear in ``id2label`` (excluding a label's own name)."""
    hier = normalize_benthic_hierarchy(dict(hierarchy))
    name_to_id = _name_to_id(id2label, ignore_index=ignore_index)
    discovered: set[str] = set()
    for class_id, name in id2label.items():
        if class_id == ignore_index:
            continue
        path = generate_hierarchy_path(_normalize_name(str(name)), hier)
        for ancestor in path[1:]:
            key = _normalize_name(ancestor)
            if key in name_to_id:
                discovered.add(key)
    return sorted(discovered)


def build_level_remap(
    id2label: Mapping[int, str],
    hierarchy: Mapping[str, str | None],
    level_name: str,
    *,
    ignore_index: int = 0,
    num_classes: int | None = None,
) -> torch.Tensor:
    """Map each class id to the id of ``level_name`` when that ancestor is on its path.

    Classes that do not roll up to ``level_name`` (and are not the level itself) map to
    ``ignore_index``.
    """
    if num_classes is None:
        num_classes = max(id2label.keys(), default=-1) + 1
    remap = torch.full((num_classes,), ignore_index, dtype=torch.long)
    hier = normalize_benthic_hierarchy(dict(hierarchy))
    name_to_id = _name_to_id(id2label, ignore_index=ignore_index)
    level_key = _normalize_name(level_name)
    if level_key not in name_to_id:
        return remap
    level_id = name_to_id[level_key]
    remap[level_id] = level_id

    for class_id, name in id2label.items():
        if class_id == ignore_index:
            continue
        path = [
            _normalize_name(p) for p in generate_hierarchy_path(_normalize_name(str(name)), hier)
        ]
        if level_key in path:
            remap[class_id] = level_id
    return remap


def build_level_remaps(
    id2label: Mapping[int, str],
    hierarchy: Mapping[str, str | None],
    level_names: Sequence[str] | None,
    *,
    ignore_index: int = 0,
    num_classes: int | None = None,
) -> dict[str, torch.Tensor]:
    """Build remaps for explicit ``level_names``, or auto-discover them when
    ``None``."""
    names = (
        list(level_names)
        if level_names is not None
        else discover_level_names(id2label, hierarchy, ignore_index=ignore_index)
    )
    return {
        name: build_level_remap(
            id2label,
            hierarchy,
            name,
            ignore_index=ignore_index,
            num_classes=num_classes,
        )
        for name in names
    }
