"""Build the label space of a run from a tables-only cache, and summarize it."""

from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path
from typing import Any

import pandas as pd

from mermaidseg.data.config import TAXONOMY_NAME, DataSection, entry_split_groups
from mermaidseg.dataset_reconciliation.artifact_meta import (
    concept_sidecar_meta,
    file_sha1,
    repo_relative,
)
from mermaidseg.dataset_reconciliation.concept_schema import ConceptSchema
from mermaidseg.dataset_reconciliation.label_space import (
    ACK_COLUMNS,
    LabelSpace,
    check_label_space,
    label_keys_sha1,
    resolve_label_space,
)
from mermaidseg.io import REPO_ROOT, ConfigDict, load_section, resolve_repo_path
from mermaidseg.lineage import data_version
from mermaidseg.run_setup import concept_mapping_path, concept_removal_list
from segdata_spec import DATASET_BUCKET
from segdata_spec.manifest import Pin
from segdata_spec.read import open_tables, open_taxonomy
from segdata_spec.taxonomy import hierarchy_from_table


def _read_acknowledged(path: Path) -> pd.DataFrame:
    """``ACK_COLUMNS`` rows of ``path``; a missing file has none."""
    if not path.is_file():
        return pd.DataFrame(columns=ACK_COLUMNS)
    return pd.read_csv(path, dtype=str, keep_default_na=False)


def _file_fields(name: str, path: Path | None) -> dict[str, str | None]:
    """``{name: repo-relative path, name_sha1: sha1}`` of a file, or both null without one."""
    if path is None:
        return {name: None, f"{name}_sha1": None}
    return {name: repo_relative(path), f"{name}_sha1": file_sha1(path)}


def _optional_path(value: str) -> Path | None:
    return resolve_repo_path(value) if value else None


def _count_meta(space: LabelSpace) -> dict[str, Any]:
    """Class, extra class and source label counts, and source labels per drop reason."""
    dropped = space.source_labels["drop_reason"]
    return {
        "n_classes": len(space.classes) - 1,
        "n_extra_classes": int((space.classes["kind"] == "extra").sum()),
        "n_source_labels": len(space.source_labels),
        "drops": {
            str(r): int(n) for r, n in dropped[dropped != ""].value_counts().sort_index().items()
        },
    }


def build_label_space(
    data_config: Path,
    training_config: Path,
    schema_config: Path,
    cache_dir: Path,
    acknowledged: Path,
    git: Mapping[str, object],
) -> LabelSpace:
    """Resolve and check the label space of a run from a tables-only cache, with its meta filled.

    Reads the tables and taxonomy that ``data_config`` pins from ``cache_dir`` (filled by
    ``segdata_spec.sync.sync_tables``); no image is read. A dataset without a train block counts
    ``n = 0``. Writes nothing.

    Args:
        data_config: Data config of the run; its datasets set the global offsets in order.
        training_config: Config with the ``training`` section.
        schema_config: Data config whose datasets freeze the concept channels.
        cache_dir: Tables-only cache.
        acknowledged: ``ACK_COLUMNS`` CSV of acknowledged cases; a missing file has none.
        git: ``mermaidseg.lineage.git_state()``, read before anything is written.

    Raises:
        ValueError: The label space is refused (see :func:`check_label_space`).
    """
    section = DataSection.model_validate(load_section(data_config, "data"))
    training_cfg = ConfigDict(load_section(training_config, "training"))
    tables = {
        name: open_tables(
            cache_dir,
            DATASET_BUCKET,
            Pin(name=name, version=entry.version, manifest_sha256=entry.manifest_sha256),
        )
        for name, entry in section.datasets.items()
    }
    train_groups = {
        name: entry_split_groups(entry, "train", REPO_ROOT)
        for name, entry in section.datasets.items()
        if entry.train is not None
    }
    mapping_path = concept_mapping_path(training_cfg)
    schema_sources = set(load_section(schema_config, "data")["datasets"])
    schema = ConceptSchema.from_csv(
        mapping_path,
        sources=schema_sources,
        removed_concepts=concept_removal_list(training_cfg, mapping_path),
    )
    taxonomy = open_taxonomy(
        cache_dir, DATASET_BUCKET, Pin(name=TAXONOMY_NAME, **section.taxonomy.model_dump())
    )
    space = resolve_label_space(
        tables,
        taxonomy.benthic_attributes,
        training_cfg,
        schema,
        train_groups=train_groups,
    )
    acknowledged_table = _read_acknowledged(acknowledged)
    check_label_space(space, acknowledged_table, hierarchy_from_table(taxonomy.benthic_attributes))
    space.meta.update(
        {
            "data_version": data_version(section),
            "label_keys_sha1": label_keys_sha1(training_cfg),
            **concept_sidecar_meta(data_config, Path(mapping_path), schema_config, git),
            "training_config": repo_relative(training_config),
            **_file_fields("concept_removal", _optional_path(training_cfg["concept_removal"])),
            **_file_fields(
                "concept_source_mask", _optional_path(training_cfg["concept_source_mask"])
            ),
            **_file_fields("acknowledged", acknowledged if acknowledged.is_file() else None),
            **_count_meta(space),
        }
    )
    return space


def summarize_label_space(space: LabelSpace) -> list[str]:
    """Lines naming, per dataset, its labels, those mapped to a class, and drops per reason."""
    lines = []
    for dataset, rows in space.source_labels.groupby("dataset", sort=False):
        lines.append(f"{dataset}: {len(rows)} labels, {int((rows['class_id'] != 0).sum())} mapped")
        dropped = rows[rows["drop_reason"] != ""]
        for reason, group in dropped.groupby("drop_reason", sort=True):
            lines.append(f"  {reason}: {len(group)} labels, n={int(group['n'].sum())}")
    return lines
