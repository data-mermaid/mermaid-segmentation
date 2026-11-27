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

from collections.abc import Collection, Iterable
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import scipy.sparse as sp

from mermaidseg.data.config import DataSection, entry_split_groups
from mermaidseg.data.factory import open_dataset_version
from mermaidseg.dataset_reconciliation.artifact_meta import (
    check_label_keys,
    file_sha1,
    read_sidecar,
    select_groups,
    source_id_of_group,
)
from mermaidseg.dataset_reconciliation.concept_schema import ConceptSchema, concept_keys
from mermaidseg.dataset_reconciliation.concepts import concept_kind
from mermaidseg.io import REPO_ROOT

DENSE_DATASETS = frozenset({"coralscapes_v2", "ucsd_mosaics", "benthos_yuval"})
CORALNET = "coralnet"

LABEL_COUNT_COLUMNS = ["dataset", "image_id", "label", "n", "source"]
# Labels that mark unannotated points or pixels, never a class.
IGNORE_LABEL_NAMES: frozenset[str] = frozenset({"unlabeled", "unlabeled / unidentified"})

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
    """Thresholds a concept channel must meet to be kept."""

    min_images: int = 100
    min_noncoralnet_images: int = 50
    min_annotations: int = 500
    dense_annotations_per_mask: int = 50
    min_coralnet_sources: int = 3

    def as_dict(self) -> dict[str, int]:
        """Return the thresholds as a dict."""
        return asdict(self)


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
            ``source`` (the CoralNet ``source_id`` for CoralNet rows, ignored otherwise); built
            by ``scripts/build_concept_removal.py`` from each version's
            ``stats/label_counts.parquet`` joined to its ``labels.parquet`` names.
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
        """Distinct ``groups`` (among ``rows``) with at least one True point, per channel."""
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
    meta = read_sidecar(path, artifact="concept removal")
    if concepts_csv is not None and meta.get("concepts_csv_sha1") != file_sha1(concepts_csv):
        raise ValueError(
            f"{path} was built for a concept table with sha1 {meta.get('concepts_csv_sha1')}, "
            f"but this run uses {concepts_csv} (sha1 {file_sha1(concepts_csv)}). "
            "Rebuild it with scripts/build_concept_removal.py."
        )
    df = pd.read_csv(path, dtype={"concept": str})
    return df["concept"].tolist()


def train_label_counts(
    dataset: str,
    label_counts: pd.DataFrame,
    images: pd.DataFrame,
    labels: pd.DataFrame,
    groups: Collection[str] | None,
) -> pd.DataFrame:
    """Return one dataset's per-image label counts over ``groups``, in ``LABEL_COUNT_COLUMNS``.

    Args:
        dataset: Dataset name, written to ``dataset`` (and to ``source`` outside CoralNet).
        label_counts: ``stats/label_counts.parquet`` rows (``image_id``, ``label_id``, ``n``).
        images: ``images.parquet`` rows (``image_id``, ``group``).
        labels: ``labels.parquet`` rows (``label_id``, ``name``).
        groups: Groups to count; ``None`` counts all.

    Raises:
        ValueError: A group is not in ``images``, a counted label id has no name, or a CoralNet
            group is not ``s<source id>``.
    """
    images = select_groups(dataset, images, groups)
    group_of = images.set_index("image_id")["group"]
    counts = label_counts[
        label_counts["image_id"].isin(group_of.index) & (label_counts["label_id"] != 0)
    ]
    label = counts["label_id"].map(concept_keys(dataset, labels))
    check_label_keys(dataset, counts["label_id"], label)
    names = counts["label_id"].map(labels.set_index("label_id")["name"])
    ignored = names.astype(str).str.strip().str.lower().isin(IGNORE_LABEL_NAMES).to_numpy()
    out = pd.DataFrame(
        {
            "dataset": dataset,
            "image_id": counts["image_id"].astype(str).to_numpy(),
            "label": label.astype(str).str.strip().str.lower().to_numpy(),
            "n": counts["n"].to_numpy(),
        }
    )
    out["source"] = (
        [source_id_of_group(g) for g in counts["image_id"].map(group_of)]
        if dataset == CORALNET
        else dataset
    )
    return out[~ignored].reset_index(drop=True)[LABEL_COUNT_COLUMNS]


def require_single_annotator(dataset: str, annotators: Collection[str]) -> None:
    """Raise if ``annotators`` (a points dataset's ``annotator`` column) holds several names.

    ``stats/label_counts.parquet`` counts every annotator's labels and cannot be filtered by
    annotator, so a train block, whether or not it names annotators, would count each position
    once per annotator.
    """
    found = sorted(set(annotators))
    if len(found) > 1:
        raise ValueError(
            f"{dataset}: several annotators {found} in points.parquet; "
            "label counts cannot be filtered by annotator"
        )


def section_train_label_counts(section: DataSection, cache_dir: Path) -> pd.DataFrame:
    """Concatenate :func:`train_label_counts` over every dataset with a train block.

    Raises:
        ValueError: No dataset has a train block, or a points dataset has several annotators.
    """
    parts = []
    for name, entry in section.datasets.items():
        if entry.train is None:
            continue
        version = open_dataset_version(section, name, cache_dir)
        if version.points is not None:
            require_single_annotator(name, version.points.column("annotator").to_pylist())
        parts.append(
            train_label_counts(
                name,
                version.label_counts.to_pandas(),
                version.images.to_pandas(),
                version.labels.to_pandas(),
                entry_split_groups(entry, "train", REPO_ROOT),
            )
        )
    if not parts:
        raise ValueError("no dataset has a train block to count")
    return pd.concat(parts, ignore_index=True)
