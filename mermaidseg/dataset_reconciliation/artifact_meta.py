"""Helpers shared by the derived concept artifacts (removal list, source mask) and their sidecars."""

from __future__ import annotations

import hashlib
import re
from pathlib import Path

from mermaidseg.io import REPO_ROOT

_GROUP = re.compile(r"s(\d+)")


def file_sha1(path: str | Path) -> str:
    """Return the hex SHA-1 digest of a file's bytes."""
    return hashlib.sha1(Path(path).read_bytes()).hexdigest()


def meta_path(mask_csv: str | Path) -> Path:
    """Sidecar path: ``coralnet_only_false.csv`` -> ``coralnet_only_false.meta.yaml``."""
    mask_csv = Path(mask_csv)
    return mask_csv.with_name(mask_csv.stem + ".meta.yaml")


def source_id_of_group(group: str) -> str:
    """Map a CoralNet image group ``s<digits>`` to its source id, digits as written.

    Raises:
        ValueError: ``group`` is not ``s`` followed by digits.
    """
    match = _GROUP.fullmatch(str(group))
    if match is None:
        raise ValueError(f"CoralNet group is not s<source id>: {group!r}")
    return match.group(1)


def repo_relative(path: Path) -> str:
    """``path`` relative to the repository root, or absolute when it lies outside it."""
    path = path.resolve()
    return str(path.relative_to(REPO_ROOT)) if path.is_relative_to(REPO_ROOT) else str(path)
