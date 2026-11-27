"""The resolved label space of a model: its target classes, concept channels and source labels.

A label space records, for every ``labels.parquet`` row of every dataset, the target class it
trains and the concept channels it sets, with the reason when it drops to background. It is
written to a directory of CSVs (see :func:`write_label_space`) and committed, and the training
lookups are built from it (see :meth:`LabelSpace.lookups`).
"""

from __future__ import annotations

import logging
from collections.abc import Collection, Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import pyarrow as pa
import torch
import yaml

from mermaidseg.dataset_reconciliation.artifact_meta import select_groups
from mermaidseg.dataset_reconciliation.concept_ranks import TAXONOMIC_CONCEPTS
from mermaidseg.dataset_reconciliation.concept_schema import ConceptSchema, concept_label_key
from segdata_spec.read import VersionTables
from segdata_spec.taxonomy import hierarchy_from_table, roll_up

logger = logging.getLogger(__name__)

CLASSES_CSV = "classes.csv"
CONCEPTS_CSV = "concepts.csv"
SOURCE_LABELS_CSV = "source_labels.csv"
META_YAML = "label_space.meta.yaml"

CLASS_COLUMNS = ["class_id", "name", "kind", "mermaid_benthic_attribute_id"]
CONCEPT_COLUMNS = ["channel_id", "name", "rank"]
SOURCE_LABEL_COLUMNS = [
    "dataset",
    "version",
    "label_id",
    "native_name",
    "mermaid_benthic_attribute_id",
    "target_name",
    "rolled_up_to",
    "class_id",
    "drop_reason",
    "n",
    "concept_key",
    "concept_status",
    "concepts",
    "false_concepts",
]
_INT_COLUMNS = {
    CLASSES_CSV: ["class_id"],
    CONCEPTS_CSV: ["channel_id"],
    SOURCE_LABELS_CSV: ["label_id", "class_id", "n"],
}
ACK_COLUMNS = ["dataset", "native_name", "kind", "reason"]
ACK_KINDS = frozenset({"missing_concepts", "rank_disagreement"})

_TRUE = 2
_FALSE = 1


def _split_names(cell: str) -> list[str]:
    return cell.split(";") if cell else []


@dataclass(frozen=True)
class LabelSpace:
    """A model's target classes, concept channels and per-source-label resolution.

    Attributes:
        classes: ``CLASS_COLUMNS``; ``class_id`` 0 is ``ignore``, 1..M the sorted ``class_subset``.
        concepts: ``CONCEPT_COLUMNS``; ``channel_id`` 1..C in channel order, ``rank`` empty for
            a binary concept.
        source_labels: ``SOURCE_LABEL_COLUMNS``, one row per ``labels.parquet`` row.
            ``concepts`` names the channels coded true and ``false_concepts`` the binary
            channels coded false, each ``;``-joined in channel order.
        meta: Provenance written to ``label_space.meta.yaml``.
    """

    classes: pd.DataFrame
    concepts: pd.DataFrame
    source_labels: pd.DataFrame
    meta: dict

    def lookups(self, offsets: Mapping[str, int]) -> tuple[torch.Tensor, torch.Tensor]:
        """Build ``source_to_target`` ``(N+1,)`` and ``source_to_concepts`` ``(N+1, C)``.

        The global id of a label is ``offsets[dataset] + label_id``; row 0 is background. A
        taxonomic channel is 1 when another channel of its rank is true, as
        ``encode_concept_row`` codes it.

        Args:
            offsets: Global offset of each dataset in ``source_labels``.

        Raises:
            ValueError: Two datasets produce the same global id.
        """
        _check_global_ids(self.source_labels, offsets)
        labels = self.source_labels
        global_ids = [
            offsets[d] + int(i) for d, i in zip(labels["dataset"], labels["label_id"], strict=True)
        ]
        size = max(global_ids, default=0) + 1
        channel = {name: i for i, name in enumerate(self.concepts["name"])}
        rank_channels: dict[str, list[int]] = {}
        for name, rank in zip(self.concepts["name"], self.concepts["rank"], strict=True):
            if rank:
                rank_channels.setdefault(rank, []).append(channel[name])
        rank_of = dict(zip(self.concepts["name"], self.concepts["rank"], strict=True))

        targets = np.zeros(size, dtype=np.int64)
        concepts = np.zeros((size, len(channel)), dtype=np.float32)
        for gid, class_id, true, false in zip(
            global_ids,
            labels["class_id"],
            labels["concepts"],
            labels["false_concepts"],
            strict=True,
        ):
            targets[gid] = class_id
            true_names = _split_names(true)
            for rank in {rank_of[name] for name in true_names} - {""}:
                concepts[gid, rank_channels[rank]] = _FALSE
            for name in true_names:
                concepts[gid, channel[name]] = _TRUE
            for name in _split_names(false):
                concepts[gid, channel[name]] = _FALSE
        return torch.from_numpy(targets), torch.from_numpy(concepts)


def _resolve_target(
    target: str | None,
    class_id: Mapping[str, int],
    hierarchy: Mapping[str, str | None],
    label_roll_up: bool,
) -> tuple[int, str, str]:
    """Return ``(class_id, rolled_up_to, drop_reason)`` of one lower-cased target."""
    if target is None:
        return 0, "", "unmapped"
    if target in class_id:
        return class_id[target], "", ""
    if not label_roll_up or target not in hierarchy:
        return 0, "", "not_in_subset"
    ancestor = roll_up(target, dict(hierarchy), set(class_id))
    if ancestor is None:
        return 0, "", "no_ancestor_in_subset"
    return class_id[ancestor], ancestor, ""


def _classes_table(names: list[str], benthic_attributes: pa.Table) -> pd.DataFrame:
    """Class 0 ``ignore``, then ``names``; a name of the taxonomy is ``mermaid`` with its id."""
    ids: dict[str, str] = {}
    for attribute_id, name in zip(
        benthic_attributes.column("id").to_pylist(),
        benthic_attributes.column("name").to_pylist(),
        strict=True,
    ):
        ids.setdefault(name.lower(), str(attribute_id))
    rows = [(0, "ignore", "ignore", "")]
    rows += [
        (i, name, "mermaid" if name in ids else "extra", ids.get(name, ""))
        for i, name in enumerate(names, start=1)
    ]
    return pd.DataFrame(rows, columns=CLASS_COLUMNS)


def _concepts_table(schema: ConceptSchema) -> pd.DataFrame:
    rows = []
    for i, name in enumerate(schema.channel_names, start=1):
        rank, sep, _ = name.partition("__")
        rows.append((i, name, rank if sep and rank in TAXONOMIC_CONCEPTS else ""))
    return pd.DataFrame(rows, columns=CONCEPT_COLUMNS)


def _concept_cells(row: np.ndarray, channel_names: list[str], ranks: list[str]) -> tuple[str, str]:
    """``(concepts, false_concepts)`` of one concept row: true channels, false binary channels."""
    true = [name for name, code in zip(channel_names, row, strict=True) if code == _TRUE]
    false = [
        name
        for name, rank, code in zip(channel_names, ranks, row, strict=True)
        if code == _FALSE and not rank
    ]
    return ";".join(true), ";".join(false)


def _train_counts(
    dataset: str, version: VersionTables, groups: Collection[str] | None
) -> dict[int, int]:
    """Sum ``stats/label_counts.parquet`` ``n`` per label over the images of ``groups``."""
    images = select_groups(dataset, version.images.to_pandas(), groups)
    counts = version.label_counts.to_pandas()
    counts = counts[counts["image_id"].isin(images["image_id"])]
    return {int(k): int(v) for k, v in counts.groupby("label_id")["n"].sum().items()}


def _dataset_rows(
    dataset: str,
    version: VersionTables,
    counts: Mapping[int, int],
    class_id: Mapping[str, int],
    hierarchy: Mapping[str, str | None],
    label_roll_up: bool,
    schema: ConceptSchema,
    ranks: list[str],
) -> list[dict[str, Any]]:
    """One ``SOURCE_LABEL_COLUMNS`` row per ``labels.parquet`` row of ``dataset``."""
    channel_names = list(schema.channel_names)
    rows = []
    for label in version.labels.to_pylist():
        attribute = label["mermaid_benthic_attribute"] or None
        target = attribute.lower() if attribute else None
        cid, rolled_up_to, drop_reason = _resolve_target(target, class_id, hierarchy, label_roll_up)
        key = concept_label_key(dataset, label["name"], attribute)
        found = (dataset.lower(), key.lower()) in schema.row_by_key
        cells = (
            _concept_cells(schema.row_for(dataset, key), channel_names, ranks)
            if found
            else ("", "")
        )
        rows.append(
            {
                "dataset": dataset,
                "version": version.manifest.version,
                "label_id": int(label["label_id"]),
                "native_name": label["name"],
                "mermaid_benthic_attribute_id": label["mermaid_benthic_attribute_id"] or "",
                "target_name": target or "",
                "rolled_up_to": rolled_up_to,
                "class_id": cid,
                "drop_reason": drop_reason,
                "n": counts.get(int(label["label_id"]), 0),
                "concept_key": key,
                "concept_status": "ok" if found else "missing",
                "concepts": cells[0],
                "false_concepts": cells[1],
            }
        )
    return rows


def _check_global_ids(source_labels: pd.DataFrame, offsets: Mapping[str, int]) -> None:
    """Raise when two datasets produce the same global id ``offset + label_id``."""
    owner: dict[int, str] = {}
    for dataset, label_id in zip(source_labels["dataset"], source_labels["label_id"], strict=True):
        gid = offsets[dataset] + int(label_id)
        if gid in owner and owner[gid] != dataset:
            raise ValueError(
                f"global id {gid} is produced by both '{owner[gid]}' and '{dataset}'; "
                "give the sources disjoint offsets"
            )
        owner[gid] = dataset


def _log_target_coverage(source_labels: pd.DataFrame) -> None:
    """Log, per source, how many of its labels reach a target class; warn when none do.

    A label-key mismatch otherwise sends every pixel of that source to background with no error.
    """
    for dataset, rows in source_labels.groupby("dataset", sort=False):
        n_mapped = int((rows["class_id"] != 0).sum())
        msg = f"{dataset}: {n_mapped} of {len(rows)} source labels map to a target class"
        if n_mapped:
            logger.info(msg)
        else:
            logger.warning(msg + "; check source label keys against the mapping.")


def resolve_label_space(
    tables: Mapping[str, VersionTables],
    offsets: Mapping[str, int],
    benthic_attributes: pa.Table,
    training_cfg: Any,
    schema: ConceptSchema,
    *,
    train_groups: Mapping[str, Collection[str] | None],
) -> LabelSpace:
    """Resolve every ``labels.parquet`` row of ``tables`` to a target class and concept row.

    Targets (``mermaid_benthic_attribute``, lower-cased) outside ``training.class_subset`` drop to
    class 0, or roll up to their nearest ancestor in it when ``training.label_roll_up`` is set.

    Args:
        tables: Dataset name -> its synced tables.
        offsets: Global offset of each dataset (see ``mermaidseg.data.factory.source_offsets``).
        benthic_attributes: ``benthic_attributes.parquet`` of the pinned taxonomy.
        training_cfg: The ``training`` section (``class_subset``, ``label_roll_up``).
        schema: The frozen concept channels, after concept removal.
        train_groups: Groups of each dataset's train split; ``None`` is every group. A dataset
            missing here has no train split and counts ``n = 0`` for every label.

    Returns:
        The label space, with an empty ``meta``.

    Raises:
        ValueError: ``tables`` is empty, ``class_subset`` is unset, a train group is not in its
            dataset, or two datasets produce the same global id.
    """
    if not tables:
        raise ValueError("a label space requires at least one dataset")
    if training_cfg.class_subset is None:
        raise ValueError("a label space requires training.class_subset")
    names = sorted(dict.fromkeys(str(label).lower() for label in training_cfg.class_subset))
    class_id = {name: i for i, name in enumerate(names, start=1)}
    hierarchy = hierarchy_from_table(benthic_attributes)
    concepts = _concepts_table(schema)
    ranks = concepts["rank"].tolist()

    rows: list[dict[str, Any]] = []
    for dataset, version in tables.items():
        counts = (
            _train_counts(dataset, version, train_groups[dataset])
            if dataset in train_groups
            else {}
        )
        rows += _dataset_rows(
            dataset,
            version,
            counts,
            class_id,
            hierarchy,
            bool(training_cfg.label_roll_up),
            schema,
            ranks,
        )
    source_labels = pd.DataFrame(rows, columns=SOURCE_LABEL_COLUMNS)
    _check_global_ids(source_labels, offsets)
    _log_target_coverage(source_labels)
    return LabelSpace(
        classes=_classes_table(names, benthic_attributes),
        concepts=concepts,
        source_labels=source_labels,
        meta={},
    )


def write_label_space(directory: Path, space: LabelSpace) -> None:
    """Write ``classes.csv``, ``concepts.csv``, ``source_labels.csv`` and the meta YAML."""
    directory.mkdir(parents=True, exist_ok=True)
    space.classes.to_csv(directory / CLASSES_CSV, index=False)
    space.concepts.to_csv(directory / CONCEPTS_CSV, index=False)
    space.source_labels.to_csv(directory / SOURCE_LABELS_CSV, index=False)
    (directory / META_YAML).write_text(yaml.safe_dump(space.meta, sort_keys=False))


def _read_table(directory: Path, file: str, columns: list[str]) -> pd.DataFrame:
    """Read one CSV with empty cells as ``''``; its integer columns become ``int64``.

    Raises:
        ValueError: A column of ``columns`` is missing.
    """
    path = directory / file
    df = pd.read_csv(path, dtype=str, keep_default_na=False)
    missing = [c for c in columns if c not in df.columns]
    if missing:
        raise ValueError(f"{path} is missing columns {missing}")
    df = df[columns]
    return df.astype(dict.fromkeys(_INT_COLUMNS[file], "int64"))


def _check_source_labels(
    path: Path, source_labels: pd.DataFrame, classes: pd.DataFrame, concepts: pd.DataFrame
) -> None:
    """Raise when ``source_labels`` repeats a label or names an unknown class or concept."""
    duplicated = source_labels.duplicated(["dataset", "version", "label_id"], keep=False)
    if duplicated.any():
        repeated = sorted(
            set(
                source_labels.loc[duplicated, ["dataset", "version", "label_id"]].itertuples(
                    index=False, name=None
                )
            )
        )
        raise ValueError(f"{path} repeats (dataset, version, label_id) {repeated}")
    unknown_classes = sorted(set(source_labels["class_id"]) - set(classes["class_id"]))
    if unknown_classes:
        raise ValueError(f"{path} names class ids not in {CLASSES_CSV}: {unknown_classes}")
    named = {
        name
        for column in ("concepts", "false_concepts")
        for cell in source_labels[column]
        for name in _split_names(cell)
    }
    unknown_concepts = sorted(named - set(concepts["name"]))
    if unknown_concepts:
        raise ValueError(f"{path} names concepts not in {CONCEPTS_CSV}: {unknown_concepts}")


def read_label_space(directory: Path) -> LabelSpace:
    """Read a label space written by :func:`write_label_space`.

    Raises:
        ValueError: A table misses a column, ``source_labels.csv`` repeats a
            ``(dataset, version, label_id)``, or names a class or concept that its table lacks.
    """
    classes = _read_table(directory, CLASSES_CSV, CLASS_COLUMNS)
    concepts = _read_table(directory, CONCEPTS_CSV, CONCEPT_COLUMNS)
    source_labels = _read_table(directory, SOURCE_LABELS_CSV, SOURCE_LABEL_COLUMNS)
    _check_source_labels(directory / SOURCE_LABELS_CSV, source_labels, classes, concepts)
    meta = yaml.safe_load((directory / META_YAML).read_text())
    return LabelSpace(classes=classes, concepts=concepts, source_labels=source_labels, meta=meta)


def _lineage(target: str, hierarchy: Mapping[str, str | None]) -> set[str]:
    """``target`` and its ancestors in ``hierarchy``; a cycle stops the walk."""
    lineage = {target}
    parent = hierarchy.get(target)
    while parent is not None and parent not in lineage:
        lineage.add(parent)
        parent = hierarchy.get(parent)
    return lineage


def _rank_disagreements(
    concepts: str, target: str, rank_of: Mapping[str, str], hierarchy: Mapping[str, str | None]
) -> list[str]:
    """``rank 'value'`` of each true rank channel naming a MERMAID attribute off ``target``'s line."""
    lineage = _lineage(target, hierarchy)
    found = []
    for name in _split_names(concepts):
        rank = rank_of[name]
        value = name.partition("__")[2]
        if rank and value in hierarchy and value not in lineage:
            found.append(f"{rank} {value!r}")
    return found


def _cases(
    space: LabelSpace, benthic_hierarchy: Mapping[str, str | None]
) -> dict[tuple[str, str, str], str]:
    """``(dataset, native_name, kind) -> detail`` of every case that needs an acknowledgement."""
    rank_of = dict(zip(space.concepts["name"], space.concepts["rank"], strict=True))
    cases: dict[tuple[str, str, str], str] = {}
    for row in space.source_labels.itertuples(index=False):
        if row.concept_status == "missing" and not row.drop_reason:
            cases[(row.dataset, row.native_name, "missing_concepts")] = (
                f"no concept row for key {row.concept_key!r}"
            )
        if row.target_name:
            disagreements = _rank_disagreements(
                row.concepts, row.target_name, rank_of, benthic_hierarchy
            )
            if disagreements:
                cases[(row.dataset, row.native_name, "rank_disagreement")] = (
                    f"{', '.join(disagreements)} is not {row.target_name!r} or an ancestor"
                )
    return cases


def _check_acknowledged(acknowledged: pd.DataFrame) -> None:
    """Raise when an acknowledgement has an unknown kind or an empty reason."""
    bad = acknowledged[
        ~acknowledged["kind"].isin(list(ACK_KINDS))
        | (acknowledged["reason"].fillna("").astype(str).str.strip() == "")
    ]
    if len(bad):
        rows = bad[ACK_COLUMNS].to_dict("records")
        raise ValueError(
            f"acknowledgements need a kind in {sorted(ACK_KINDS)} and a reason: {rows}"
        )


def check_label_space(
    space: LabelSpace, acknowledged: pd.DataFrame, benthic_hierarchy: Mapping[str, str | None]
) -> None:
    """Refuse a label space with unacknowledged cases, or acknowledgements that match none.

    Cases are a label that trains a class but has no concept row (``missing_concepts``), and a label whose true rank
    channel names a MERMAID benthic attribute that is neither its target nor an ancestor of it
    (``rank_disagreement``). A rank value naming no attribute, and a label without a target, are
    not checked. A label that drops to background is never a case, so an acknowledgement of one is
    stale.

    Args:
        space: The label space.
        acknowledged: ``ACK_COLUMNS`` rows; ``kind`` is a case kind, ``reason`` is required.
        benthic_hierarchy: Lower-cased name -> parent name (``hierarchy_from_table``).

    Raises:
        ValueError: An acknowledgement is malformed, or, listing every one, a case is not
            acknowledged or an acknowledgement is stale.
    """
    _check_acknowledged(acknowledged)
    cases = _cases(space, benthic_hierarchy)
    acks = set(acknowledged[["dataset", "native_name", "kind"]].itertuples(index=False, name=None))
    problems = [
        f"dataset={d} native_name={n} kind={k}: {detail}"
        for (d, n, k), detail in cases.items()
        if (d, n, k) not in acks
    ]
    problems += [
        f"stale acknowledgement: dataset={d} native_name={n} kind={k} matches no case"
        for d, n, k in sorted(acks - set(cases))
    ]
    if problems:
        raise ValueError("label space check failed:\n" + "\n".join(problems))
