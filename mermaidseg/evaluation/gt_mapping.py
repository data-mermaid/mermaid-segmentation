"""Ground-truth label mapping for the evaluation suite.

Two lookups are built per source dataset:

- ``ClassLookup``: source label name -> MERMAID target class id (0 = ignore).
  This replicates training's ``source_to_target`` with ``label_roll_up=True``:
  the source label is mapped to its MERMAID benthic-attribute name, then rolled
  up through the benthic-attribute hierarchy until it lands on a class present in
  the model's ``id2label`` vocabulary.

- ``ConceptLookup``: source label name -> length-K concept row in the model's
  concept channel order (values ``0=not_given, 1=False, 2=True``). This
  reproduces ``encode_concept_channels_from_df`` per-row semantics directly from
  ``class_to_concepts.csv``, independent of the global schema width, so it always
  aligns to the model's ``concept_id2name`` layout.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path

import numpy as np
import pandas as pd
from numpy.typing import NDArray

from mermaidseg.dataset_reconciliation.concepts import (
    TAXONOMIC_CONCEPTS,
    initialize_benthic_hierarchy,
)
from mermaidseg.dataset_reconciliation.label_mapping import (
    fetch_benthos_yuval_to_mermaid,
    fetch_catlin_seaview_to_mermaid,
    fetch_coralnet_to_mermaid,
    fetch_moorea_labeled_corals_to_mermaid,
    fetch_pacific_labeled_corals_to_mermaid,
    fetch_ucsd_mosaics_to_mermaid,
)
from mermaidseg.dataset_reconciliation.registry import roll_up_label

logger = logging.getLogger(__name__)

_PLACEHOLDER_TAXA = {"not_given", "not given", "", "nan"}


def _source_to_target_map(source_name: str) -> dict[str, str]:
    """Return the static/config ``source_label -> MERMAID target`` map for a source."""
    source_name = source_name.lower()
    if source_name == "coralnet":
        raw = fetch_coralnet_to_mermaid()
    elif source_name == "pacific_labeled_corals":
        raw = fetch_pacific_labeled_corals_to_mermaid()
    elif source_name == "benthos_yuval":
        raw = fetch_benthos_yuval_to_mermaid()
    elif source_name == "catlin_seaview":
        raw = fetch_catlin_seaview_to_mermaid()
    elif source_name == "moorea_labeled_corals":
        raw = fetch_moorea_labeled_corals_to_mermaid()
    elif source_name == "ucsd_mosaics":
        raw = fetch_ucsd_mosaics_to_mermaid()
    else:
        raise ValueError(f"No source->target mapping wired for source {source_name!r}")
    return {str(k).lower(): (str(v).lower() if v is not None else None) for k, v in raw.items()}


def load_benthic_hierarchy(
    hierarchy_json: str | Path | None = None,
    output_dir: str | Path | None = None,
    fetch_remote: bool = True,
) -> dict[str, str | None] | None:
    """Load (and cache) the benthic-attribute name->parent hierarchy.

    Precedence: explicit ``hierarchy_json`` file, then a cached
    ``benthic_hierarchy.json`` in ``output_dir``, then a remote fetch (cached to
    ``output_dir`` when possible). Returns ``None`` if unavailable (roll-up
    disabled with a warning).
    """
    if hierarchy_json is not None:
        with Path(hierarchy_json).open() as f:
            raw = json.load(f)
        return {k.lower(): (v.lower() if v else None) for k, v in raw.items()}

    cache_path = Path(output_dir) / "benthic_hierarchy.json" if output_dir else None
    if cache_path is not None and cache_path.is_file():
        logger.info("Loading cached benthic hierarchy from %s", cache_path)
        with cache_path.open() as f:
            raw = json.load(f)
        return {k.lower(): (v.lower() if v else None) for k, v in raw.items()}

    if not fetch_remote:
        logger.warning("Benthic hierarchy unavailable (no cache, fetch_remote=False); roll-up OFF.")
        return None

    try:
        logger.info("Fetching benthic-attribute hierarchy from MERMAID API ...")
        hierarchy = initialize_benthic_hierarchy()
    except Exception as e:  # noqa: BLE001
        logger.warning("Could not fetch benthic hierarchy (%s); roll-up disabled.", e)
        return None

    hierarchy = {
        str(k).lower(): (str(v).lower() if v is not None else None) for k, v in hierarchy.items()
    }
    if cache_path is not None:
        try:
            cache_path.parent.mkdir(parents=True, exist_ok=True)
            with cache_path.open("w") as f:
                json.dump(hierarchy, f)
            logger.info("Cached benthic hierarchy to %s", cache_path)
        except OSError as e:  # noqa: BLE001
            logger.warning("Failed to cache benthic hierarchy: %s", e)
    return hierarchy


class ClassLookup:
    """Maps source label names to MERMAID target class ids (0 = ignore)."""

    def __init__(self, name_to_class_id: dict[str, int], id2label: dict[int, str]):
        self.name_to_class_id = name_to_class_id
        self.id2label = id2label
        self.unmapped: set[str] = set()

    def class_id(self, source_label_name: str) -> int:
        key = str(source_label_name).lower()
        cid = self.name_to_class_id.get(key)
        if cid is None:
            self.unmapped.add(key)
            return 0
        return cid


def build_class_lookup(
    source_name: str,
    source_label_names: list[str],
    id2label: dict[int, str],
    hierarchy: dict[str, str | None] | None = None,
) -> ClassLookup:
    """Build a source-label -> MERMAID class-id lookup (training-style, with roll-up)."""
    src_to_tgt = _source_to_target_map(source_name)

    # id2label is the frozen target vocabulary (0 = ignore/background).
    target_label2id = {str(v).lower(): int(k) for k, v in id2label.items() if int(k) != 0}
    subset = set(target_label2id.keys())

    name_to_class_id: dict[str, int] = {}
    for src_name in source_label_names:
        key = str(src_name).lower()
        target = src_to_tgt.get(key)
        if target is None:
            name_to_class_id[key] = 0
            continue
        if hierarchy is not None:
            rolled = roll_up_label(target, hierarchy, subset)
        else:
            rolled = target if target in subset else None
        name_to_class_id[key] = target_label2id.get(rolled, 0) if rolled is not None else 0

    return ClassLookup(name_to_class_id, id2label)


def _cell_str(value: object) -> str:
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return "not_given"
    return str(value).strip().lower()


def _encode_concept_row(
    row: pd.Series,
    concept_names: list[str],
    ranks: set[str],
) -> NDArray[np.int8]:
    """Encode one CSV row into the model concept-channel order (0/1/2 values)."""
    out = np.zeros(len(concept_names), dtype=np.int8)
    for i, channel in enumerate(concept_names):
        if "__" in channel:
            rank, value = channel.split("__", 1)
            if rank not in ranks:
                continue
            cell = _cell_str(row.get(rank))
            if cell in _PLACEHOLDER_TAXA:
                out[i] = 0  # not_given -> all zeros for this rank
            elif cell == value:
                out[i] = 2  # active class at this rank
            else:
                out[i] = 1  # inactive sibling
        else:
            cell = _cell_str(row.get(channel))
            if cell == "true":
                out[i] = 2
            elif cell == "false":
                out[i] = 1
            else:
                out[i] = 0  # not_given / none / missing
    return out


class ConceptLookup:
    """Maps source label names to length-K concept rows in model channel order."""

    def __init__(self, name_to_row: dict[str, NDArray[np.int8]], num_channels: int):
        self.name_to_row = name_to_row
        self.num_channels = num_channels
        self._zero = np.zeros(num_channels, dtype=np.int8)
        self.unmapped: set[str] = set()

    def row(self, source_label_name: str) -> NDArray[np.int8]:
        key = str(source_label_name).lower()
        r = self.name_to_row.get(key)
        if r is None:
            self.unmapped.add(key)
            return self._zero
        return r


def build_concept_lookup(
    source_name: str,
    source_label_names: list[str],
    concept_names: list[str],
    csv_path: str | Path,
) -> ConceptLookup:
    """Build a source-label -> concept-row lookup aligned to the model channels."""
    df = pd.read_csv(csv_path)
    df["source_label_class_name"] = df["source_label_class_name"].astype(str).str.lower()
    df["source_dataset_source"] = df["source_dataset_source"].astype(str).str.lower()
    df_src = df[df["source_dataset_source"] == source_name.lower()]

    # First row wins for duplicate (source, label) keys.
    row_by_label: dict[str, pd.Series] = {}
    for _, row in df_src.iterrows():
        label = row["source_label_class_name"]
        if label not in row_by_label:
            row_by_label[label] = row

    ranks = set(TAXONOMIC_CONCEPTS)
    name_to_row: dict[str, NDArray[np.int8]] = {}
    for src_name in source_label_names:
        key = str(src_name).lower()
        row = row_by_label.get(key)
        if row is None:
            continue  # -> zeros (ignored) via ConceptLookup.row
        name_to_row[key] = _encode_concept_row(row, concept_names, ranks)

    return ConceptLookup(name_to_row, len(concept_names))
