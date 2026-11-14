"""Per-source masking of binary concepts that a source never annotates as True.

CoralNet sources use different label sets. When no label used by a source carries a binary concept
(e.g. the source has no "bleached" labels), every point of that source is a False target for it,
even where the concept is visible but simply not annotated. Masking sets those targets to 0
(``not_given``), which the concept losses ignore.

The masked ``(source_id, concept)`` pairs are computed once by ``scripts/build_concept_source_mask.py``
and written to a CSV next to a ``.meta.yaml`` sidecar; training reads the file instead of
recomputing it.
"""

from __future__ import annotations

import hashlib
from pathlib import Path

import numpy as np
import pandas as pd

from mermaidseg.dataset_reconciliation.concept_schema import ConceptSchema
from mermaidseg.dataset_reconciliation.concepts import TAXONOMIC_CONCEPTS

# Concept channel encoding (see ``encode_concept_channels_from_df``).
NOT_GIVEN, FALSE, TRUE = 0, 1, 2

MASK_COLUMNS = ["source_id", "concept", "n_true", "n_false", "n_not_given"]


def binary_concepts(schema: ConceptSchema) -> list[str]:
    """Binary concept names in channel order; each is its own single channel."""
    names = [k for k in schema.concept_value2id if k not in TAXONOMIC_CONCEPTS]
    missing = [n for n in names if n not in schema.channel_names]
    if missing:
        raise ValueError(f"binary concepts without a channel of the same name: {missing}")
    return names


def source_concept_counts(
    df_annotations: pd.DataFrame,
    schema: ConceptSchema,
    dataset_source: str = "coralnet",
) -> pd.DataFrame:
    """Count True / False / not_given points per ``(source_id, binary concept)``.

    Each point's concept values come from ``schema.row_for(dataset_source, source_label_name)``,
    the lookup training uses; a label missing from the schema counts as not_given.

    Returns:
        One row per source and binary concept with columns ``MASK_COLUMNS``.
    """
    concepts = binary_concepts(schema)
    channel = [schema.channel_names.index(c) for c in concepts]
    per_label = (
        df_annotations.groupby(["source_id", "source_label_name"], observed=True)
        .size()
        .rename("n")
        .reset_index()
    )
    labels = per_label["source_label_name"].unique()
    values = pd.DataFrame(
        np.stack([schema.row_for(dataset_source, str(name))[channel] for name in labels]).astype(
            np.int8
        ),
        index=pd.Index(labels, name="source_label_name"),
        columns=concepts,
    )
    long = per_label.join(values, on="source_label_name").melt(
        id_vars=["source_id", "n"], value_vars=concepts, var_name="concept", value_name="value"
    )
    counts = (
        long.pivot_table(
            index=["source_id", "concept"], columns="value", values="n", aggfunc="sum", fill_value=0
        )
        .reindex(columns=[NOT_GIVEN, FALSE, TRUE], fill_value=0)
        .rename(columns={NOT_GIVEN: "n_not_given", FALSE: "n_false", TRUE: "n_true"})
        .reset_index()
    )
    counts.columns.name = None
    counts["concept"] = pd.Categorical(counts["concept"], categories=concepts, ordered=True)
    counts = counts.sort_values(["source_id", "concept"]).reset_index(drop=True)
    counts["concept"] = counts["concept"].astype(str)
    return counts[MASK_COLUMNS].astype({c: np.int64 for c in MASK_COLUMNS if c.startswith("n_")})


def only_false_pairs(counts: pd.DataFrame) -> pd.DataFrame:
    """Rows of ``counts`` with False points and no True points: the pairs to mask."""
    return counts[(counts["n_false"] > 0) & (counts["n_true"] == 0)].reset_index(drop=True)


def file_sha1(path: str | Path) -> str:
    return hashlib.sha1(Path(path).read_bytes()).hexdigest()


def meta_path(mask_csv: str | Path) -> Path:
    """Sidecar path: ``coralnet_only_false.csv`` -> ``coralnet_only_false.meta.yaml``."""
    mask_csv = Path(mask_csv)
    return mask_csv.with_name(mask_csv.stem + ".meta.yaml")
