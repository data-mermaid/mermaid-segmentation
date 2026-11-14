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
import logging
from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
import yaml
from torch.utils.data import Dataset

from mermaidseg.dataset_reconciliation.concept_schema import ConceptSchema
from mermaidseg.dataset_reconciliation.concepts import TAXONOMIC_CONCEPTS

logger = logging.getLogger(__name__)

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


# ---------- training side ----------


@dataclass(frozen=True)
class ConceptSourceMask:
    """Keep table for one run: row 0 keeps every channel, row ``i`` belongs to ``source_ids[i - 1]``.

    ``keep[row, channel]`` is False where that source's targets for that channel are set to 0.
    Datasets other than ``dataset`` (and its sources missing from the file) use row 0.
    """

    dataset: str
    source_ids: tuple[str, ...]
    keep: torch.Tensor  # (len(source_ids) + 1, num_channels), bool
    path: str

    @property
    def num_masked(self) -> int:
        return int((~self.keep).sum())

    def row_by_source(self) -> dict[str, int]:
        return {source: i + 1 for i, source in enumerate(self.source_ids)}

    def rows_for(self, source_ids: Iterable[Any]) -> np.ndarray:
        """Keep-table rows for a sequence of source ids; unknown sources get row 0."""
        lookup = self.row_by_source()
        return np.array([lookup.get(str(s), 0) for s in source_ids], dtype=np.int64)


def load_concept_source_mask(
    path: str | Path,
    schema: ConceptSchema,
    *,
    concepts_csv: str | Path | None = None,
    removed_concepts: Iterable[str] = (),
) -> ConceptSourceMask:
    """Read a mask CSV written by ``scripts/build_concept_source_mask.py`` into a keep table.

    Checks, before training starts:
    - every concept is a binary channel of ``schema``; concepts in ``removed_concepts`` (removed
      from the run by ``training.concept_removal``) are skipped, anything else is an error;
    - with ``concepts_csv``, its sha1 matches the one recorded in the ``.meta.yaml`` sidecar, so a
      mask built for another concept table is refused.
    """
    path = Path(path)
    meta_file = meta_path(path)
    if not meta_file.is_file():
        raise FileNotFoundError(f"concept source mask {path} has no sidecar {meta_file}")
    meta = yaml.safe_load(meta_file.read_text())
    if concepts_csv is not None and meta.get("concepts_csv_sha1") != file_sha1(concepts_csv):
        raise ValueError(
            f"{path} was built for a concept table with sha1 {meta.get('concepts_csv_sha1')}, "
            f"but this run uses {concepts_csv} (sha1 {file_sha1(concepts_csv)}). "
            "Rebuild it with scripts/build_concept_source_mask.py."
        )

    df = pd.read_csv(path, dtype={"source_id": str, "concept": str})
    missing_cols = {"source_id", "concept"} - set(df.columns)
    if missing_cols:
        raise ValueError(f"{path} is missing columns {sorted(missing_cols)}")

    binary = set(binary_concepts(schema))
    removed = set(removed_concepts)
    unknown = sorted(set(df["concept"]) - binary - removed)
    if unknown:
        raise ValueError(
            f"{path} lists concepts that are not binary channels of this run: {unknown}"
        )
    skipped = df["concept"].isin(removed) & ~df["concept"].isin(binary)
    if skipped.any():
        logger.info(
            "concept source mask: skipping %d rows of removed concepts %s",
            int(skipped.sum()),
            sorted(set(df.loc[skipped, "concept"])),
        )
    df = df[~skipped]

    source_ids = tuple(sorted(df["source_id"].unique(), key=_source_sort_key))
    row = {source: i + 1 for i, source in enumerate(source_ids)}
    channel = {name: i for i, name in enumerate(schema.channel_names)}
    keep = torch.ones(len(source_ids) + 1, schema.num_channels, dtype=torch.bool)
    if len(df):
        keep[
            torch.as_tensor(df["source_id"].map(row).to_numpy()),
            torch.as_tensor(df["concept"].map(channel).to_numpy()),
        ] = False
    return ConceptSourceMask(
        dataset=str(meta.get("dataset", "coralnet")),
        source_ids=source_ids,
        keep=keep,
        path=str(path),
    )


def _source_sort_key(source: str) -> tuple[int, int | str]:
    return (0, int(source)) if source.isdigit() else (1, source)


def apply_concept_source_mask(
    target_concepts: torch.Tensor,
    keep: torch.Tensor,
    source_rows: torch.Tensor,
) -> torch.Tensor:
    """Set masked channels of each sample's concept targets to 0 (not_given).

    Args:
        target_concepts: ``(B, C, H, W)`` concept targets.
        keep: ``(S + 1, C)`` bool keep table (see :class:`ConceptSourceMask`).
        source_rows: ``(B,)`` keep-table row of each sample.
    """
    keep_b = keep.to(target_concepts.device)[source_rows.to(target_concepts.device).long()]
    return target_concepts * keep_b.to(target_concepts.dtype)[:, :, None, None]


class _ConstantSourceRow(Dataset):
    """Adds a constant keep-table row to a dataset that does not subclass ``BaseCoralDataset``."""

    def __init__(self, dataset: Dataset, row: int = 0) -> None:
        self.dataset = dataset
        self.row = int(row)

    def __len__(self) -> int:
        return len(self.dataset)  # type: ignore[arg-type]

    def __getitem__(self, idx: int) -> tuple[Any, ...]:
        return (*self.dataset[idx], self.row)

    def __getattr__(self, name: str) -> Any:
        # Forward attribute lookups (SOURCE_NAME, num_load_failures, ...) to the wrapped dataset.
        if name == "dataset":
            raise AttributeError(name)
        return getattr(self.dataset, name)


def attach_source_rows(datasets: Sequence[Dataset], mask: ConceptSourceMask) -> list[Dataset]:
    """Make every training dataset return ``(image, labels, source_row)``.

    ``BaseCoralDataset`` subclasses return the row from ``_safe_getitem``, so a retry on another
    image returns that image's row. Datasets of ``mask.dataset`` map each image's ``source_id``;
    the others, and datasets outside the base class (wrapped), always return row 0.
    """
    from mermaidseg.datasets.base_dataset import BaseCoralDataset

    out: list[Dataset] = []
    wrapped: dict[int, Dataset] = {}
    for ds in datasets:
        if isinstance(ds, BaseCoralDataset):
            if getattr(ds, "SOURCE_NAME", None) == mask.dataset:
                rows = mask.rows_for(ds.df_images["source_id"])
                ds.set_source_rows(pd.Series(rows, index=ds.df_images.index))
            else:
                ds.set_source_rows(None)
            out.append(ds)
        else:
            # The same dataset object can appear several times (repeated coralscapes_v2).
            out.append(wrapped.setdefault(id(ds), _ConstantSourceRow(ds, 0)))
    return out
