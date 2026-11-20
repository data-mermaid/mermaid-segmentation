"""The split-file format shared by the dataset publisher and the trainer, and its coverage check.

A split file is a CSV with required columns ``group`` and ``split`` (train, val or test) and
optional ``source_id`` and ``split_group`` columns. Other columns are allowed.
"""

from __future__ import annotations

import csv
from collections import Counter
from collections.abc import Callable, Mapping
from pathlib import Path

import pyarrow as pa

from segdata_spec.errors import SplitCoverageError

REQUIRED_COLUMNS = ("group", "split")
SPLITS = frozenset({"train", "val", "test"})


def _read_columns(path: Path) -> dict[str, list[str]]:
    """Return the file's columns as lists of strings, keyed by header name."""
    with path.open(newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        names = list(reader.fieldnames or [])
        columns: dict[str, list[str]] = {name: [] for name in names}
        for row in reader:
            for name in names:
                columns[name].append(row[name] or "")
    return columns


def _check_columns(path: Path, columns: Mapping[str, list[str]]) -> None:
    missing = [c for c in REQUIRED_COLUMNS if c not in columns]
    if missing:
        raise ValueError(f"split file {path.name} lacks required columns {missing}")


def _check_splits(path: Path, splits: list[str]) -> None:
    bad = sorted(set(splits) - SPLITS)
    if bad:
        raise ValueError(f"split file {path.name} has split values {bad}, expected train/val/test")


def _check_unique_groups(path: Path, groups: list[str]) -> None:
    duplicated = sorted(g for g, n in Counter(groups).items() if n > 1)
    if duplicated:
        raise ValueError(f"split file {path.name} lists groups more than once: {duplicated}")


def _check_split_groups(path: Path, splits: list[str], split_groups: list[str]) -> None:
    seen: dict[str, set[str]] = {}
    for split_group, split in zip(split_groups, splits, strict=True):
        seen.setdefault(split_group, set()).add(split)
    straddling = sorted(g for g, found in seen.items() if len(found) > 1)
    if straddling:
        raise ValueError(f"split file {path.name} puts split_group {straddling} in several splits")


def read_split_file(path: Path) -> pa.Table:
    """Read a split file with every column as a string.

    Raises:
        ValueError: A required column is missing, a split value is not train/val/test, a group
            appears twice, or (when ``split_group`` is present) a split_group spans two splits.
    """
    columns = _read_columns(path)
    _check_columns(path, columns)
    _check_splits(path, columns["split"])
    _check_unique_groups(path, columns["group"])
    if "split_group" in columns:
        _check_split_groups(path, columns["split"], columns["split_group"])
    return pa.table({name: pa.array(values, pa.string()) for name, values in columns.items()})


def check_coverage(split_file: Path, dataset: str, groups: set[str]) -> None:
    """Fail when ``groups`` holds a group the split file does not list.

    Groups the file lists but ``groups`` lacks are allowed.

    Raises:
        SplitCoverageError: Names ``dataset`` and the sorted unlisted groups.
    """
    listed = set(read_split_file(split_file).column("group").to_pylist())
    unlisted = sorted(groups - listed)
    if unlisted:
        raise SplitCoverageError(
            f"dataset {dataset!r}: split file {split_file.name} does not list groups {unlisted}"
        )


def coverage_checker(split_files: Mapping[str, Path]) -> Callable[[str, set[str]], None]:
    """Return a ``sync(check_split=...)`` hook that checks the mapped datasets and skips others."""

    def check(dataset: str, groups: set[str]) -> None:
        if dataset in split_files:
            check_coverage(split_files[dataset], dataset, groups)

    return check
