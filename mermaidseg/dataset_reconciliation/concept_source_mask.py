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

import logging
from collections.abc import Collection, Iterable, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
import yaml

from mermaidseg.data.base import VersionDataset
from mermaidseg.data.point_dataset import select_points
from mermaidseg.dataset_reconciliation.artifact_meta import (
    file_sha1,
    meta_path,
    source_id_of_group,
)
from mermaidseg.dataset_reconciliation.concept_schema import ConceptSchema, concept_keys

logger = logging.getLogger(__name__)

# Concept channel encoding (see ``encode_concept_channels_from_df``).
NOT_GIVEN, FALSE, TRUE = 0, 1, 2

MASK_COLUMNS = ["source_id", "concept", "n_true", "n_false", "n_not_given"]


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
    concepts = schema.binary_names()
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
    source_order = {
        source: i
        for i, source in enumerate(sorted(counts["source_id"].unique(), key=_source_sort_key))
    }
    counts = counts.assign(_order=counts["source_id"].map(source_order))
    counts = counts.sort_values(["_order", "concept"]).reset_index(drop=True)
    counts["concept"] = counts["concept"].astype(str)
    return counts[MASK_COLUMNS].astype({c: np.int64 for c in MASK_COLUMNS if c.startswith("n_")})


def point_annotations(
    dataset: str,
    points: pd.DataFrame,
    images: pd.DataFrame,
    labels: pd.DataFrame,
    groups: Collection[str] | None,
    annotators: Sequence[str] | None,
) -> pd.DataFrame:
    """Return one row per point of the images in ``groups``: ``source_id``, ``source_label_name``.

    Points are selected by annotator as training selects them (``select_points``), and
    ``source_label_name`` is each point's concept label key (:func:`concept_keys`).

    Args:
        dataset: Dataset name.
        points: ``points.parquet`` rows (``image_id``, ``row``, ``col``, ``label_id``,
            ``annotator``).
        images: ``images.parquet`` rows (``image_id``, ``group``, ``height``, ``width``).
        labels: ``labels.parquet`` rows (``label_id``, ``name``, ``mermaid_benthic_attribute``).
        groups: Groups to keep; ``None`` keeps all.
        annotators: The split block's annotators in priority order; ``None`` requires one.

    Raises:
        ValueError: A group is not in ``images``, a group is not ``s<source id>``, the annotator
            selection fails, or a point's label id is missing from ``labels``.
    """
    if groups is not None:
        unknown = sorted(set(groups) - set(images["group"]))
        if unknown:
            raise ValueError(f"groups not in the dataset version: {unknown}")
        images = images[images["group"].isin(list(groups))]
    kept = select_points(points, images, annotators)
    keys = kept["label_id"].map(concept_keys(dataset, labels))
    unnamed = sorted(set(kept.loc[keys.isna(), "label_id"]))
    if unnamed:
        raise ValueError(f"{dataset}: label ids missing from labels.parquet: {unnamed}")
    group_of = images.set_index("image_id")["group"]
    return pd.DataFrame(
        {
            "source_id": [source_id_of_group(g) for g in kept["image_id"].map(group_of)],
            "source_label_name": keys.to_numpy(),
        }
    )


def only_false_pairs(counts: pd.DataFrame) -> pd.DataFrame:
    """Rows of ``counts`` with False points and no True points: the pairs to mask."""
    return counts[(counts["n_false"] > 0) & (counts["n_true"] == 0)].reset_index(drop=True)


# ---------- training side ----------


@dataclass(frozen=True)
class ConceptSourceMask:
    """Keep table for one run: row 0 keeps every channel, row ``i`` belongs to ``source_ids[i - 1]``.

    ``keep[row, channel]`` is False where that source's targets for that channel are set to 0.
    Datasets other than ``dataset``, and its counted sources with nothing masked, use row 0.
    ``covered_sources`` are the sources the builder counted; :func:`attach_source_rows` refuses a
    ``dataset`` source outside them.
    """

    dataset: str
    source_ids: tuple[str, ...]
    keep: torch.Tensor  # (len(source_ids) + 1, num_channels), bool
    path: str
    covered_sources: frozenset[str]
    data_config: str | None

    @property
    def num_masked(self) -> int:
        """Number of masked source x channel entries."""
        return int((~self.keep).sum())

    def row_by_source(self) -> dict[str, int]:
        """Map each source id to its keep-table row."""
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
    """Read a mask CSV from ``scripts/build_concept_source_mask.py`` into a keep table.

    Checks, before training starts:
    - every concept is a binary channel of ``schema``; concepts in ``removed_concepts`` (removed
      from the run by ``training.concept_removal``) are skipped, anything else is an error;
    - with ``concepts_csv``, its sha1 matches the one recorded in the ``.meta.yaml`` sidecar, so a
      mask built for another concept table is refused;
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

    binary = set(schema.binary_names())
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
        covered_sources=frozenset(str(s) for s in meta["source_ids"]),
        data_config=meta.get("data_config"),
    )


def _source_sort_key(source: Any) -> tuple[int, int | str]:
    source = str(source)
    return (0, int(source)) if source.isdigit() else (1, source)


def _check_covered(source_ids: Iterable[Any], mask: ConceptSourceMask) -> None:
    """Raise if any of ``source_ids`` is outside the sources ``mask`` was built from."""
    missing = sorted({str(s) for s in source_ids} - mask.covered_sources, key=_source_sort_key)
    if missing:
        raise ValueError(
            f"{len(missing)} {mask.dataset} sources are not in concept source mask {mask.path} "
            f"(built from {mask.data_config}): {', '.join(missing[:10])}"
            f"{' ...' if len(missing) > 10 else ''}. Rebuild it with "
            "scripts/build_concept_source_mask.py --data-config <this run's data config>."
        )


def attach_source_rows(
    datasets: Sequence[VersionDataset], mask: ConceptSourceMask
) -> list[VersionDataset]:
    """Make every training dataset return ``(image, labels, source_row)``.

    Readers of ``mask.dataset`` map each image's group ``s<id>`` to source ``<id>`` and its
    keep-table row; the others return row 0. The reader returns the row of the image it loads, so
    a train-split retry on another image returns that image's row. A dataset listed several times
    (a repeated train block) is one object and gets the same rows each time.

    Returns:
        ``datasets``, as a list.

    Raises:
        ValueError: A ``mask.dataset`` image comes from a source the mask was not built from. Its
            never-True concepts would silently train as False, or an image group is not
            ``s<source id>``.
    """
    for ds in datasets:
        if mask.dataset == ds.source_name:
            sources = [source_id_of_group(group) for group in ds.images["group"]]
            _check_covered(sources, mask)
            ds.set_source_rows(mask.rows_for(sources))
        else:
            ds.set_source_rows(np.zeros(len(ds.images), dtype=np.int64))
    return list(datasets)
