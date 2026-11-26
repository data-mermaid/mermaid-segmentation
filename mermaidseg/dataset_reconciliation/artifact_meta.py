"""Helpers shared by the derived concept artifacts (removal list, source mask) and their sidecars."""

from __future__ import annotations

import datetime
import hashlib
import re
from collections.abc import Iterable, Mapping
from pathlib import Path

import yaml

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


def concept_sidecar_meta(
    data_config: Path, concepts_csv: Path, schema_config: Path, git: Mapping[str, object]
) -> dict[str, object]:
    """Sidecar fields shared by the derived concept artifacts: input paths, sha1s, time, git state."""
    return {
        "data_config": repo_relative(data_config),
        "data_config_sha1": file_sha1(data_config),
        "concepts_csv": repo_relative(concepts_csv),
        "concepts_csv_sha1": file_sha1(concepts_csv),
        "schema_config": repo_relative(schema_config),
        "created": datetime.datetime.now().isoformat(timespec="seconds"),
        **git,
    }


def read_sidecar(
    csv_path: str | Path, *, artifact: str, builder: str, required: Iterable[str] = ()
) -> dict:
    """Load the ``.meta.yaml`` sidecar of ``csv_path``; an empty or non-mapping file reads as ``{}``.

    Raises:
        FileNotFoundError: The sidecar does not exist.
        ValueError: The sidecar lacks a key in ``required``.
    """
    meta_file = meta_path(csv_path)
    if not meta_file.is_file():
        raise FileNotFoundError(f"{artifact} {csv_path} has no sidecar {meta_file}")
    meta = yaml.safe_load(meta_file.read_text())
    if not isinstance(meta, dict):
        meta = {}
    missing = sorted(set(required) - set(meta))
    if missing:
        raise ValueError(f"{meta_file} is missing {missing}. Rebuild it with scripts/{builder}.py.")
    return meta
