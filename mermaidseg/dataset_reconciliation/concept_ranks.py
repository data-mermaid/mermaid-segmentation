"""Taxonomic concept ranks and the channel slices they define (stdlib only)."""

from collections.abc import Mapping

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


def rank_slices(concept_value2id: Mapping[str, Mapping]) -> list[tuple[str, int, int]]:
    """Return ``(rank, offset, width)`` of each taxonomic rank present, in channel order.

    ``width`` counts the values of the rank other than ``not_given``. Only the taxonomic keys are
    read, so a mapping with stringified binary keys slices the same.
    """
    slices = []
    offset = 0
    for rank in TAXONOMIC_CONCEPTS:
        if rank not in concept_value2id:
            continue
        width = sum(1 for value in concept_value2id[rank] if value != "not_given")
        slices.append((rank, offset, width))
        offset += width
    return slices


def binary_offset(concept_value2id: Mapping[str, Mapping]) -> int:
    """Return the channel index where the binary tail starts."""
    return sum(width for _, _, width in rank_slices(concept_value2id))
