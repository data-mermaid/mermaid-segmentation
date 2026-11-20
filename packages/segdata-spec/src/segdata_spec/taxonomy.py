"""Benthic-attribute hierarchy helpers shared by the publisher and the trainer."""

from __future__ import annotations

import pyarrow as pa


def roll_up(
    label: str | None, benthic_hierarchy: dict[str, str | None], subset: set[str]
) -> str | None:
    """Return ``label`` if it is in ``subset``, else its nearest ancestor in ``subset``.

    Args:
        label: Target label name, or ``None`` for an unmapped label.
        benthic_hierarchy: Lower-cased name -> parent-name mapping, ``None`` at the roots.
        subset: Names that may be returned.

    Returns:
        The label or ancestor in ``subset``, or ``None`` if there is none.
    """
    if label in subset:
        return label
    if label in benthic_hierarchy:
        parent = benthic_hierarchy[label]
        while parent is not None:
            if parent in subset:
                return parent
            parent = benthic_hierarchy[parent]
    return None


def hierarchy_from_table(benthic_attributes: pa.Table) -> dict[str, str | None]:
    """Build a lower-cased name -> parent-name mapping from ``benthic_attributes.parquet``.

    The parent is ``None`` at roots and when ``parent_id`` matches no row.
    """
    id_to_name = dict(
        zip(
            benthic_attributes.column("id").to_pylist(),
            benthic_attributes.column("name").to_pylist(),
            strict=True,
        )
    )
    hierarchy: dict[str, str | None] = {}
    for name, parent_id in zip(
        benthic_attributes.column("name").to_pylist(),
        benthic_attributes.column("parent_id").to_pylist(),
        strict=True,
    ):
        parent = id_to_name.get(parent_id)
        hierarchy[name.lower()] = parent.lower() if parent is not None else None
    return hierarchy
