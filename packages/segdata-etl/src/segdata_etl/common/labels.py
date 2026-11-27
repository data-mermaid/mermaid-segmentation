"""Class-id and palette construction, and benthic-attribute name lookup in a taxonomy."""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Iterable, Mapping

import numpy as np
import pyarrow as pa

TaxonomyNameIndex = Mapping[str, list[tuple[str, str]]]


class AmbiguousTaxonomyName(ValueError):
    """A name matches more than one taxonomy benthic attribute, ignoring case."""

    def __init__(self, name: str, matches: list[str]) -> None:
        """Record ``name`` and name every attribute spelling it matches."""
        super().__init__(
            f"{name!r} matches several taxonomy benthic attributes (ignoring case): {matches}"
        )


def build_classes_and_colors(
    names: Iterable[str], seed: int = 1337
) -> tuple[dict[str, int], dict[str, list[int]]]:
    """Return ``classes.json`` and ``colors.json`` content.

    Names are deduplicated and numbered alphabetically from 1; ``unlabeled`` is 0 and black.
    Colors are drawn in alphabetical order from a seeded generator, so they are reproducible.
    """
    ordered = sorted(set(names))
    classes = {"unlabeled": 0}
    classes.update({name: i for i, name in enumerate(ordered, start=1)})
    rng = np.random.default_rng(seed)
    colors: dict[str, list[int]] = {"unlabeled": [0, 0, 0]}
    for name in ordered:
        colors[name] = [int(x) for x in rng.integers(0, 256, size=3)]
    return classes, colors


def canonicalize(labels: Iterable[str]) -> dict[str, str]:
    """Case-merge label spellings; prefer mixed case over all-caps, then lexicographic order."""
    groups: dict[str, list[str]] = defaultdict(list)
    for label in labels:
        groups[label.lower()].append(label)
    out: dict[str, str] = {}
    for variants in groups.values():
        mixed = [v for v in variants if v != v.upper()]
        canonical = sorted(mixed)[0] if mixed else sorted(variants)[0]
        for v in variants:
            out[v] = canonical
    return out


def taxonomy_name_index(attributes: pa.Table) -> dict[str, list[tuple[str, str]]]:
    """Index a ``benthic_attributes`` table's ``(id, name)`` rows by lowercased name."""
    index: dict[str, list[tuple[str, str]]] = {}
    for attr_id, name in zip(
        attributes.column("id").to_pylist(), attributes.column("name").to_pylist(), strict=True
    ):
        index.setdefault(name.lower(), []).append((attr_id, name))
    return index


def resolve_taxonomy_name(name: str, index: TaxonomyNameIndex) -> tuple[str, str] | None:
    """Return the taxonomy ``(id, canonical name)`` of ``name`` ignoring case, or None if absent.

    Raises:
        AmbiguousTaxonomyName: ``name`` matches more than one taxonomy row.
    """
    matches = index.get(name.lower(), [])
    if len(matches) > 1:
        raise AmbiguousTaxonomyName(name, [match_name for _, match_name in matches])
    return matches[0] if matches else None
