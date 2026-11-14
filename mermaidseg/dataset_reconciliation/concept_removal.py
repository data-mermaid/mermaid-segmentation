"""Remove concept channels with too little training support (``training.concept_removal``).

A concept is removed when it fails any of the coverage rules (counted on True annotations):

- images: fewer than ``min_images`` images in total and fewer than ``min_noncoralnet_images``
  images outside CoralNet;
- annotations: fewer than ``min_annotations``, where each dense mask (Coralscapes v2, UCSD mosaics,
  Benthos) that contains the concept counts as ``dense_annotations_per_mask``;
- sources: no non-CoralNet dataset has it and fewer than ``min_coralnet_sources`` CoralNet sources do.

The removed concepts are computed once by ``scripts/build_concept_removal.py`` and written to a CSV
with a ``.meta.yaml`` sidecar; training reads the file. A removed taxonomic value is set to
``not_given`` in the concept table before the channels are built, so its points keep their coarser
ranks (a removed genus keeps its family). A removed binary concept loses its channel.
"""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import scipy.sparse as sp
import yaml

from mermaidseg.dataset_reconciliation.concept_schema import ConceptSchema
from mermaidseg.dataset_reconciliation.concept_source_mask import file_sha1, meta_path
from mermaidseg.dataset_reconciliation.concepts import TAXONOMIC_CONCEPTS

DENSE_DATASETS = frozenset({"coralscapes_v2", "ucsd_mosaics", "benthos_yuval"})
CORALNET = "coralnet"
NONE_TAXON = "none"

REMOVAL_COLUMNS = [
    "concept",
    "kind",
    "n_annotations",
    "n_images",
    "n_noncoralnet_images",
    "n_coralnet_sources",
    "n_other_datasets",
    "fails_images",
    "fails_annotations",
    "fails_sources",
]


@dataclass(frozen=True)
class CoverageRule:
    min_images: int = 100
    min_noncoralnet_images: int = 50
    min_annotations: int = 500
    dense_annotations_per_mask: int = 50
    min_coralnet_sources: int = 3

    def as_dict(self) -> dict[str, int]:
        return asdict(self)


def concept_kind(channel: str) -> str:
    """``genus__acropora`` -> ``genus``; a binary concept -> ``binary``."""
    rank = channel.split("__", 1)[0] if "__" in channel else None
    return rank if rank in TAXONOMIC_CONCEPTS else "binary"


def concept_coverage(
    counts: pd.DataFrame,
    schema: ConceptSchema,
    rule: CoverageRule | None = None,
    dense_datasets: Iterable[str] = DENSE_DATASETS,
) -> pd.DataFrame:
    """Per-channel support and rule outcome from per-image label counts.

    Args:
        counts: One row per ``(dataset, image, label)`` with columns ``dataset``, ``image_id``,
            ``label`` (lowercased source label), ``n`` (points; pixels for dense datasets) and
            ``source`` (the CoralNet ``source_id`` for CoralNet rows, ignored otherwise), as
            written by ``scripts/train_label_counts.py`` plus a source column.
        schema: The run's concept schema; a point is True for a channel where its row is 2.
        rule: Coverage thresholds.
        dense_datasets: Datasets whose ``n`` is pixels; they count per mask instead.

    Returns:
        One row per channel (index: channel name) with the ``REMOVAL_COLUMNS`` counts and flags
        plus ``removed``.
    """
    rule = rule or CoverageRule()
    dense_datasets = set(dense_datasets)
    counts = counts.reset_index(drop=True)
    keys = pd.MultiIndex.from_frame(counts[["dataset", "label"]].astype(str))
    key_codes, key_values = pd.factorize(keys)
    # Key x channel: 1 where the (dataset, label) row of the schema is True (2).
    true_rows = np.stack(
        [schema.row_for(ds, label) == 2 for ds, label in key_values]
        or [np.zeros(schema.num_channels, bool)]
    )
    K = sp.csr_matrix(true_rows.astype(np.float64))
    n_keys = K.shape[0]

    dense = counts["dataset"].isin(dense_datasets).to_numpy()
    is_cn = (counts["dataset"] == CORALNET).to_numpy()
    n = counts["n"].to_numpy(dtype=np.float64)
    image = (counts["dataset"].astype(str) + "/" + counts["image_id"].astype(str)).to_numpy()
    source = np.where(is_cn, "coralnet:" + counts["source"].astype(str), "").astype(object)

    def groups_with_channel(groups: np.ndarray, rows: np.ndarray) -> np.ndarray:
        """Number of distinct ``groups`` (among ``rows``) with at least one True point per channel."""
        if not rows.any():
            return np.zeros(schema.num_channels, dtype=np.int64)
        g = pd.factorize(groups[rows])[0]
        m = sp.csr_matrix(
            (np.ones(int(rows.sum())), (g, key_codes[rows])), shape=(g.max() + 1, n_keys)
        )
        return np.asarray(((m @ K) > 0).sum(axis=0)).ravel().astype(np.int64)

    sparse_points = np.bincount(key_codes, weights=np.where(dense, 0.0, n), minlength=n_keys)
    n_annotations = K.T @ sparse_points + rule.dense_annotations_per_mask * groups_with_channel(
        image, dense
    )
    out = pd.DataFrame(index=pd.Index(schema.channel_names, name="concept"))
    out["kind"] = [concept_kind(c) for c in out.index]
    out["n_annotations"] = np.asarray(n_annotations).ravel().astype(np.int64)
    all_rows = np.ones(len(counts), dtype=bool)
    out["n_images"] = groups_with_channel(image, all_rows)
    out["n_noncoralnet_images"] = groups_with_channel(image, ~is_cn)
    out["n_coralnet_sources"] = groups_with_channel(source, is_cn)
    out["n_other_datasets"] = groups_with_channel(counts["dataset"].to_numpy(), ~is_cn)

    out["fails_images"] = ~(
        (out["n_images"] >= rule.min_images)
        | (out["n_noncoralnet_images"] >= rule.min_noncoralnet_images)
    )
    out["fails_annotations"] = out["n_annotations"] < rule.min_annotations
    out["fails_sources"] = ~(
        (out["n_other_datasets"] >= 1) | (out["n_coralnet_sources"] >= rule.min_coralnet_sources)
    )
    out["removed"] = out[["fails_images", "fails_annotations", "fails_sources"]].any(axis=1)
    return out


def removed_concepts(coverage: pd.DataFrame) -> pd.DataFrame:
    """The removed rows of :func:`concept_coverage`, as written to the removal CSV."""
    removed = coverage[coverage["removed"]].reset_index()
    return removed[REMOVAL_COLUMNS]


def load_concept_removal(path: str | Path, *, concepts_csv: str | Path | None = None) -> list[str]:
    """Concept channel names listed in a removal CSV (``scripts/build_concept_removal.py``).

    With ``concepts_csv``, its sha1 must match the one in the ``.meta.yaml`` sidecar.
    """
    path = Path(path)
    meta_file = meta_path(path)
    if not meta_file.is_file():
        raise FileNotFoundError(f"concept removal {path} has no sidecar {meta_file}")
    meta = yaml.safe_load(meta_file.read_text())
    if concepts_csv is not None and meta.get("concepts_csv_sha1") != file_sha1(concepts_csv):
        raise ValueError(
            f"{path} was built for a concept table with sha1 {meta.get('concepts_csv_sha1')}, "
            f"but this run uses {concepts_csv} (sha1 {file_sha1(concepts_csv)}). "
            "Rebuild it with scripts/build_concept_removal.py."
        )
    df = pd.read_csv(path, dtype={"concept": str})
    if "concept" not in df.columns:
        raise ValueError(f"{path} has no 'concept' column")
    return df["concept"].tolist()


def apply_concept_removal(df_mapping: pd.DataFrame, concepts: Iterable[str]) -> pd.DataFrame:
    """Concept table with ``concepts`` removed, before channel encoding.

    ``rank__value`` sets that rank to ``not_given`` where it equals ``value`` (coarser ranks stay);
    a binary concept's column is dropped. ``rank__none`` cannot be removed: ``none`` marks
    non-organisms and is not a taxon.
    """
    df = df_mapping.copy()
    concepts = list(dict.fromkeys(concepts))
    for concept in concepts:
        kind = concept_kind(concept)
        if kind == "binary":
            if concept not in df.columns:
                raise ValueError(f"cannot remove {concept!r}: not a column of the concept table")
            df = df.drop(columns=concept)
            continue
        value = concept.split("__", 1)[1]
        if value == NONE_TAXON:
            raise ValueError(f"cannot remove {concept!r}: 'none' is not a taxon")
        hit = df[kind].astype(str) == value
        if not hit.any():
            raise ValueError(f"cannot remove {concept!r}: no {kind} == {value!r} in the concept table")
        df.loc[hit, kind] = "not_given"
    return df
