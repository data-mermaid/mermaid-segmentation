"""Taxonomic concept ranks and concept-name parsing (stdlib only)."""

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
