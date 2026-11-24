"""Load a dataset's curated label map from the packaged CSV."""

from __future__ import annotations

import hashlib
from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from importlib import resources
from importlib.resources.abc import Traversable
from pathlib import Path
from typing import Any

import pyarrow as pa
import pyarrow.csv as pacsv

CURATED_SCHEMA = pa.schema(
    [
        ("label_id", pa.int32()),
        ("name", pa.string()),
        ("mermaid_benthic_attribute", pa.string()),
    ]
)

_REPO_DIR = "packages/segdata-etl/src/segdata_etl/label_maps/curated"


@dataclass(frozen=True)
class CuratedSource:
    """Provenance of a curated map: repo-relative CSV path and sha256 of the CSV bytes."""

    path: str
    sha256: str


def _directory(root: Path | None) -> Path | Traversable:
    return root if root is not None else resources.files(__package__) / "curated"


def _row_key(dataset: str, row: Mapping[str, Any]) -> str:
    """Return the row's key: ``label_id`` when set, else its name, which must be stripped lowercase.

    Raises:
        ValueError: The row has neither key, or its name key is not stripped lowercase.
    """
    if row["label_id"] is not None:
        return f"label_id {row['label_id']}"
    name = row["name"]
    if name is None:
        raise ValueError(f"curated map {dataset}: a row has neither label_id nor name")
    if name != name.strip().lower():
        raise ValueError(f"curated map {dataset}: name {name!r} is not normalized")
    return f"name {name!r}"


def _validate(dataset: str, rows: Iterable[Mapping[str, Any]]) -> None:
    """Refuse a row without a valid key, or a key that repeats; a null target is unmapped."""
    seen: set[str] = set()
    for row in rows:
        key = _row_key(dataset, row)
        if key in seen:
            raise ValueError(f"curated map {dataset}: duplicate {key}")
        seen.add(key)


def load_curated(dataset: str, *, root: Path | None = None) -> pa.Table:
    """Read ``<dataset>.csv`` from the packaged ``curated`` directory, or from ``root`` when given.

    An empty cell is null; a null ``mermaid_benthic_attribute`` leaves the label unmapped. A row is
    keyed by ``label_id`` when set, otherwise by ``name``.

    Raises:
        ValueError: A key repeats, a name key is not stripped lowercase, or a row has neither
            ``label_id`` nor ``name``.
    """
    with resources.as_file(_directory(root) / f"{dataset}.csv") as path:
        table = pacsv.read_csv(
            path,
            convert_options=pacsv.ConvertOptions(
                column_types=CURATED_SCHEMA, strings_can_be_null=True
            ),
        )
    _validate(dataset, table.to_pylist())
    return table


def curated_source(dataset: str, *, root: Path | None = None) -> CuratedSource:
    """Return the CSV path of ``dataset``'s curated map and the sha256 of its raw bytes.

    The path is repo-relative for the packaged map, and under ``root`` when ``root`` is given.
    """
    raw = (_directory(root) / f"{dataset}.csv").read_bytes()
    path = f"{_REPO_DIR}/{dataset}.csv" if root is None else str(root / f"{dataset}.csv")
    return CuratedSource(path=path, sha256=hashlib.sha256(raw).hexdigest())
