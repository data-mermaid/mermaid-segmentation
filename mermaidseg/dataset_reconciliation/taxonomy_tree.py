"""Parent-child taxonomy over the taxonomic ranks, built from ``class_to_concepts.csv``.

Provides the rank-incidence and tree-distance matrices used by the hierarchical metrics.

Matrices are indexed in the model's channel order, which
``concepts.generate_taxonomic_binary_id_mapping`` defines as the sorted unique values of a rank
without ``not_given``.

``not_given`` (rank unknown) is not a channel. ``none`` (not an organism, e.g. "Rock") is a channel
but not a taxon, and has no parent-child edges.

Channels are keyed on the value alone, so one name used for two taxa would share a channel and
have two parents. The CSV keeps those lineages apart: ``turbinaria_coral`` (Dendrophylliidae) and
``turbinaria_algae`` (Sargassaceae). See ``AmbiguityPolicy``.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal

import numpy as np
import pandas as pd

from mermaidseg.dataset_reconciliation.concepts import (
    DEFAULT_CLASS_TO_CONCEPTS_CSV,
    TAXONOMIC_CONCEPTS,
)

#: Rank order, coarse -> fine. Mirrors the model's active taxonomic channels.
#: NOTE: ``species`` is currently commented out of ``TAXONOMIC_CONCEPTS``, so only six
#: ranks are active even though the CSV carries a seventh.
RANKS: tuple[str, ...] = tuple(TAXONOMIC_CONCEPTS)

#: Rank is unknown for this label. Not a channel; never a node.
NOT_GIVEN = "not_given"

#: Label is not an organism. Is a channel, but is not a taxon.
NONE_TAXON = "none"

#: How to treat a fine-rank taxon that has more than one parent:
#:
#: * ``permissive`` (default): keep every edge; a prediction is consistent if it matches any
#:   parent. A mixed lineage is still caught at the rank pair where it diverges.
#: * ``exclude``: drop ambiguous taxa; they are listed in ``TaxonomyTree.excluded``.
#: * ``strict``: raise ``ValueError``.
AmbiguityPolicy = Literal["permissive", "exclude", "strict"]


@dataclass(frozen=True)
class AmbiguityReport:
    """One fine-rank taxon found to have more than one parent."""

    coarse_rank: str
    fine_rank: str
    taxon: str
    parents: tuple[str, ...]
    #: Source datasets contributing the conflicting rows, for triage.
    sources: tuple[str, ...]

    def __str__(self) -> str:
        return (
            f"{self.fine_rank}='{self.taxon}' has {len(self.parents)} "
            f"{self.coarse_rank} parents {self.parents} (sources: {', '.join(self.sources)})"
        )


@dataclass
class TaxonomyTree:
    """Audited parent-child structure over the taxonomic ranks.

    Attributes:
        ranks: Rank names, coarse -> fine.
        channels: ``rank -> list of channel values``, in the model's channel order.
        edges: ``(coarse, fine) -> {fine_value: set of coarse_values}``.
        ambiguities: Every multi-parent taxon found, regardless of policy.
        policy: The ``AmbiguityPolicy`` this structure was built under.
        excluded: Taxa dropped, when ``policy="exclude"``.
    """

    ranks: tuple[str, ...]
    channels: dict[str, list[str]]
    edges: dict[tuple[str, str], dict[str, set[str]]]
    ambiguities: list[AmbiguityReport] = field(default_factory=list)
    policy: AmbiguityPolicy = "permissive"
    excluded: dict[str, set[str]] = field(default_factory=dict)

    def index_of(self, rank: str) -> dict[str, int]:
        """Map ``value -> channel index`` for ``rank`` (matches the model's channels)."""
        return {v: i for i, v in enumerate(self.channels[rank])}

    def is_taxon(self, rank: str, value: str) -> bool:
        """True if ``value`` is a real taxon (not a sentinel) at ``rank``."""
        return value not in (NOT_GIVEN, NONE_TAXON) and value in self.index_of(rank)


def load_taxonomy_frame(
    mapping_location: str | Path = DEFAULT_CLASS_TO_CONCEPTS_CSV,
) -> pd.DataFrame:
    """Load the concept-mapping CSV with the same normalisation the registry applies."""
    df = pd.read_csv(mapping_location).copy()
    for col in ("source_label_class_name", "source_dataset_source"):
        if col in df.columns:
            df[col] = df[col].astype(str).str.lower()
    return df


def channel_values(df: pd.DataFrame, rank: str) -> list[str]:
    """Channel values for ``rank``, in the model's channel order.

    Mirrors ``concepts.generate_taxonomic_binary_id_mapping`` exactly: sorted unique values with
    ``not_given`` removed. ``none`` is retained -- it is a real channel.
    """
    return sorted(v for v in df[rank].dropna().unique() if v != NOT_GIVEN)


def audit_taxonomy(
    df: pd.DataFrame,
    ranks: tuple[str, ...] = RANKS,
) -> list[AmbiguityReport]:
    """Find every fine-rank taxon with more than one parent.

    Sentinels are excluded: ``not_given`` carries no lineage, and ``none`` is not a taxon,
    so neither can meaningfully have a parent.

    Returns:
        One :class:`AmbiguityReport` per offending taxon. Empty means the structure is a tree.
    """
    reports: list[AmbiguityReport] = []
    for coarse, fine in zip(ranks, ranks[1:], strict=False):
        sub = df[[coarse, fine, "source_dataset_source"]].dropna(subset=[coarse, fine])
        sub = sub[
            ~sub[coarse].isin((NOT_GIVEN, NONE_TAXON)) & ~sub[fine].isin((NOT_GIVEN, NONE_TAXON))
        ]
        for taxon, grp in sub.groupby(fine):
            parents = sorted(grp[coarse].unique())
            if len(parents) > 1:
                reports.append(
                    AmbiguityReport(
                        coarse_rank=coarse,
                        fine_rank=fine,
                        taxon=str(taxon),
                        parents=tuple(str(p) for p in parents),
                        sources=tuple(sorted(grp["source_dataset_source"].unique())),
                    )
                )
    return reports


def build_taxonomy_tree(
    mapping_location: str | Path = DEFAULT_CLASS_TO_CONCEPTS_CSV,
    ranks: tuple[str, ...] = RANKS,
    ambiguity_policy: AmbiguityPolicy = "permissive",
    df: pd.DataFrame | None = None,
) -> TaxonomyTree:
    """Build the audited parent-child structure.

    Args:
        mapping_location: Concept-mapping CSV. Ignored when ``df`` is supplied.
        ranks: Ranks to include, coarse -> fine.
        ambiguity_policy: See :data:`AmbiguityPolicy`.
        df: Pre-loaded frame, to avoid re-reading the CSV.

    Raises:
        ValueError: If ``ambiguity_policy="strict"`` and any taxon has multiple parents.
    """
    if df is None:
        df = load_taxonomy_frame(mapping_location)

    ambiguities = audit_taxonomy(df, ranks)
    if ambiguity_policy == "strict" and ambiguities:
        detail = "\n  ".join(str(a) for a in ambiguities)
        raise ValueError(
            f"Taxonomy is not a tree under strict policy; {len(ambiguities)} "
            f"taxa have multiple parents:\n  {detail}"
        )

    excluded: dict[str, set[str]] = {}
    if ambiguity_policy == "exclude":
        for amb in ambiguities:
            excluded.setdefault(amb.fine_rank, set()).add(amb.taxon)

    channels = {rank: channel_values(df, rank) for rank in ranks}

    edges: dict[tuple[str, str], dict[str, set[str]]] = {}
    for coarse, fine in zip(ranks, ranks[1:], strict=False):
        sub = df[[coarse, fine]].dropna()
        sub = sub[
            ~sub[coarse].isin((NOT_GIVEN, NONE_TAXON)) & ~sub[fine].isin((NOT_GIVEN, NONE_TAXON))
        ]
        mapping: dict[str, set[str]] = {}
        for taxon, grp in sub.groupby(fine):
            taxon = str(taxon)
            if taxon in excluded.get(fine, set()):
                continue
            mapping[taxon] = {str(p) for p in grp[coarse].unique()}
        edges[(coarse, fine)] = mapping

    return TaxonomyTree(
        ranks=tuple(ranks),
        channels=channels,
        edges=edges,
        ambiguities=ambiguities,
        policy=ambiguity_policy,
        excluded=excluded,
    )


def build_rank_incidence(tree: TaxonomyTree, coarse: str, fine: str) -> np.ndarray:
    """Binary incidence matrix ``M`` between two adjacent ranks, in channel order.

    ``M[i, j] == 1`` iff fine-rank taxon ``i`` sits inside coarse-rank taxon ``j``. Under
    ``policy="permissive"`` a homonym's row has several ones; a row of zeros means the taxon
    has no recorded parent (sentinel, or excluded).

    Returns:
        ``np.ndarray`` of shape ``(len(channels[fine]), len(channels[coarse]))``, uint8.
    """
    if (coarse, fine) not in tree.edges:
        raise ValueError(
            f"({coarse!r}, {fine!r}) is not an adjacent rank pair. "
            f"Adjacent pairs: {sorted(tree.edges)}"
        )
    fine_index = tree.index_of(fine)
    coarse_index = tree.index_of(coarse)
    matrix = np.zeros((len(fine_index), len(coarse_index)), dtype=np.uint8)
    for child, parents in tree.edges[(coarse, fine)].items():
        row = fine_index.get(child)
        if row is None:
            continue
        for parent in parents:
            col = coarse_index.get(parent)
            if col is not None:
                matrix[row, col] = 1
    return matrix


def build_all_incidences(tree: TaxonomyTree) -> dict[tuple[str, str], np.ndarray]:
    """Incidence matrices for every adjacent rank pair, coarse -> fine."""
    return {
        (coarse, fine): build_rank_incidence(tree, coarse, fine)
        for coarse, fine in zip(tree.ranks, tree.ranks[1:], strict=False)
    }


def build_ancestor_sets(tree: TaxonomyTree, rank: str) -> list[dict[str, set[str]]]:
    """Ancestors of each taxon at ``rank``, one mapping per coarser rank.

    Values are sets because a homonym under ``policy="permissive"`` has several lineages.

    Returns:
        ``list`` of length ``depth(rank)``; element ``i`` is
        ``{channel_value: set of ancestors at tree.ranks[i]}``.
    """
    depth = tree.ranks.index(rank)
    # ancestors[i] maps a taxon at `rank` to its possible ancestors at tree.ranks[i].
    ancestors: list[dict[str, set[str]]] = [{} for _ in range(depth)]
    if depth == 0:
        return ancestors

    # Seed with the immediate parents, then walk upward one rank at a time.
    current: dict[str, set[str]] = {
        taxon: set(parents) for taxon, parents in tree.edges[(tree.ranks[depth - 1], rank)].items()
    }
    ancestors[depth - 1] = {k: set(v) for k, v in current.items()}

    for level in range(depth - 2, -1, -1):
        edge = tree.edges[(tree.ranks[level], tree.ranks[level + 1])]
        nxt: dict[str, set[str]] = {}
        for taxon, mids in current.items():
            grand: set[str] = set()
            for mid in mids:
                grand |= edge.get(mid, set())
            if grand:
                nxt[taxon] = grand
        ancestors[level] = nxt
        current = nxt
    return ancestors


def rank_distance_matrix(tree: TaxonomyTree, rank: str) -> np.ndarray:
    """Pairwise taxonomic distance between the channels at ``rank``.

    Distance is the number of ranks to ascend before the two lineages meet (lowest common
    ancestor, as in Bertinetto et al., "Making Better Mistakes"): 0 for the same taxon, 1
    for siblings, up to ``depth + 1`` when the lineages never meet. Homonyms take the
    minimum over their lineages. ``none`` has no lineage, so it is at the maximum distance
    from every other channel; mask it with ``taxon_channel_mask`` to exclude it.

    Returns:
        ``np.ndarray`` of shape ``(C, C)``, dtype int16, indexed in channel order.
    """
    values = tree.channels[rank]
    size = len(values)
    depth = tree.ranks.index(rank)
    max_distance = depth + 1

    distance = np.full((size, size), max_distance, dtype=np.int16)
    np.fill_diagonal(distance, 0)
    if depth == 0:
        return distance

    ancestors = build_ancestor_sets(tree, rank)
    # Coarse levels first, so the finest (smallest) distance overwrites them.
    for level in range(0, depth):
        steps = depth - level  # levels ascended to reach tree.ranks[level]
        level_sets = [ancestors[level].get(v, set()) for v in values]
        for i in range(size):
            a = level_sets[i]
            if not a:
                continue
            for j in range(i + 1, size):
                b = level_sets[j]
                if b and not a.isdisjoint(b):
                    distance[i, j] = steps
                    distance[j, i] = steps
    np.fill_diagonal(distance, 0)
    return distance


def taxon_channel_mask(tree: TaxonomyTree, rank: str) -> np.ndarray:
    """Boolean mask over ``rank``'s channels, true for real taxa.

    False for the ``none`` sentinel, which is a channel but not an organism, and for any taxon
    dropped under ``policy="exclude"``.
    """
    excluded = tree.excluded.get(rank, set())
    return np.array(
        [v != NONE_TAXON and v not in excluded for v in tree.channels[rank]],
        dtype=bool,
    )


def summarise(tree: TaxonomyTree) -> str:
    """Human-readable audit summary, for notebooks and run logs."""
    lines = [
        f"TaxonomyTree  ranks={' > '.join(tree.ranks)}  policy={tree.policy}",
        "",
        f"{'rank':10s} {'channels':>9s} {'with parent':>12s} {'orphans':>8s}",
    ]
    for i, rank in enumerate(tree.ranks):
        n = len(tree.channels[rank])
        if i == 0:
            lines.append(f"{rank:10s} {n:9d} {'-':>12s} {'-':>8s}")
            continue
        mapping = tree.edges[(tree.ranks[i - 1], rank)]
        real = [v for v in tree.channels[rank] if v not in (NOT_GIVEN, NONE_TAXON)]
        with_parent = sum(1 for v in real if v in mapping)
        lines.append(f"{rank:10s} {n:9d} {with_parent:12d} {len(real) - with_parent:8d}")

    lines.append("")
    if tree.ambiguities:
        lines.append(f"AMBIGUOUS TAXA ({len(tree.ambiguities)}):")
        for amb in tree.ambiguities:
            lines.append(f"  - {amb}")
    else:
        lines.append("No ambiguous taxa: the structure is a genuine tree.")
    if tree.excluded:
        total = sum(len(v) for v in tree.excluded.values())
        lines.append(f"EXCLUDED under policy='exclude': {total} taxa {dict(tree.excluded)}")
    return "\n".join(lines)
