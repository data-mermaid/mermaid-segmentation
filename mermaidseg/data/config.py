"""The ``data`` section of a data config, and the sync plan derived from it."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

import pandas as pd
from pydantic import BaseModel, ConfigDict, model_validator

from segdata_spec.manifest import Pin
from segdata_spec.sync import MissingMode

DATASET_BUCKET = "mermaid-seg-datasets"
TAXONOMY_NAME = "mermaid-benthic"

Split = Literal["train", "val", "test"]


class _Strict(BaseModel):
    model_config = ConfigDict(extra="forbid")


class PinRef(_Strict):
    """A version and the sha256 of its manifest; the pin's name comes from where it is used."""

    version: str
    manifest_sha256: str


class SplitSelection(_Strict):
    """Which groups feed one split: the splits of ``split_file`` (``use``) or named groups."""

    use: list[Split] | None = None
    groups: Literal["all"] | list[str] | None = None
    annotator: list[str] | None = None
    repeat: int | None = None
    transform: Any = None

    @model_validator(mode="after")
    def _exactly_one_selector(self) -> SplitSelection:
        if (self.use is None) == (self.groups is None):
            raise ValueError("a split block needs exactly one of 'use' or 'groups'")
        return self


class DatasetEntry(_Strict):
    """One pinned dataset and its per-split selections."""

    version: str
    manifest_sha256: str
    split_file: Path | None = None
    on_missing_images: MissingMode = "fail"
    train: SplitSelection | None = None
    val: SplitSelection | None = None
    test: SplitSelection | None = None

    @model_validator(mode="after")
    def _use_needs_split_file(self) -> DatasetEntry:
        if self.split_file is None and any(b.use is not None for b in self.blocks()):
            raise ValueError("a split block with 'use' needs 'split_file'")
        return self

    def blocks(self) -> list[SplitSelection]:
        """The split blocks the entry defines."""
        return [b for b in (self.train, self.val, self.test) if b is not None]


class DataSection(_Strict):
    """The ``data`` section of a data config."""

    taxonomy: PinRef | None = None
    datasets: dict[str, DatasetEntry]
    default: Any = None


@dataclass(frozen=True)
class SyncPlan:
    """The arguments ``segdata_spec.sync.sync`` takes, derived from a data config."""

    pins: dict[str, Pin]
    taxonomy: Pin | None
    selection: dict[str, set[str] | None]
    on_missing_images: dict[str, MissingMode]


def _split_groups(split_file: Path, use: list[Split]) -> set[str]:
    """Return the groups of ``split_file`` whose split is in ``use``."""
    table = pd.read_csv(split_file, dtype=str, keep_default_na=False)
    missing = {"group", "split"} - set(table.columns)
    if missing:
        raise ValueError(f"split file {split_file} lacks columns {sorted(missing)}")
    return set(table.loc[table["split"].isin(use), "group"])


def _block_groups(block: SplitSelection, split_file: Path | None) -> set[str] | None:
    """Return a block's groups, or None for all of them."""
    if block.groups == "all":
        return None
    if block.groups is not None:
        return set(block.groups)
    assert block.use is not None and split_file is not None
    return _split_groups(split_file, block.use)


def _entry_selection(entry: DatasetEntry, repo_root: Path) -> set[str] | None:
    """Return the union of the entry's block groups, or None if any block selects all."""
    split_file = repo_root / entry.split_file if entry.split_file is not None else None
    selected: set[str] = set()
    for block in entry.blocks():
        groups = _block_groups(block, split_file)
        if groups is None:
            return None
        selected |= groups
    return selected


def sync_plan(data_cfg: Mapping[str, Any], repo_root: Path) -> SyncPlan:
    """Validate the ``data`` section and derive the sync arguments from it.

    Args:
        data_cfg: The ``data`` section of a data config.
        repo_root: The directory ``split_file`` paths are relative to.

    Raises:
        pydantic.ValidationError: The section has an unknown key or an invalid block.
        ValueError: A split file lacks the ``group`` or ``split`` column.
    """
    section = DataSection.model_validate(data_cfg)
    taxonomy = (
        Pin(name=TAXONOMY_NAME, **section.taxonomy.model_dump())
        if section.taxonomy is not None
        else None
    )
    return SyncPlan(
        pins={
            name: Pin(name=name, version=e.version, manifest_sha256=e.manifest_sha256)
            for name, e in section.datasets.items()
        },
        taxonomy=taxonomy,
        selection={name: _entry_selection(e, repo_root) for name, e in section.datasets.items()},
        on_missing_images={name: e.on_missing_images for name, e in section.datasets.items()},
    )
