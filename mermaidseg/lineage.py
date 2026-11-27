"""Lineage of a training run: code revision, environment lock, config, seed, data."""

from __future__ import annotations

import hashlib
import os
import subprocess
from importlib.metadata import version
from pathlib import Path

from mermaidseg.data.config import DataSection
from mermaidseg.io import REPO_ROOT

GIT_TIMEOUT_SEC = 10
# CI builds container images from a committed tree and passes the commit here; the image has no .git.
BUILD_GIT_SHA_ENV = "MERMAIDSEG_GIT_SHA"
UV_LOCK = "uv.lock"
VERSIONED_PACKAGES = ("torch", "transformers", "peft")


def _git_output(*args: str) -> str | None:
    """Stdout of ``git -C <repo> <args>``; None when git is missing, fails, or hangs."""
    try:
        result = subprocess.run(
            ["git", "-C", str(REPO_ROOT), *args],
            capture_output=True,
            text=True,
            check=True,
            timeout=GIT_TIMEOUT_SEC,
        )
    except (OSError, subprocess.SubprocessError):
        return None
    return result.stdout


def _git_sha() -> str | None:
    out = _git_output("rev-parse", "HEAD")
    return out.strip() if out is not None else None


def _git_dirty() -> bool:
    """Whether tracked files differ from HEAD, or git cannot be read; untracked files do not count."""
    out = _git_output("status", "--porcelain", "--untracked-files=no")
    return True if out is None else bool(out.strip())


def _uv_lock_sha256() -> str | None:
    lock = Path(REPO_ROOT) / UV_LOCK
    if not lock.is_file():
        return None
    return hashlib.sha256(lock.read_bytes()).hexdigest()


def data_version(section: DataSection) -> dict:
    """The pinned dataset and taxonomy versions of a run.

    Args:
        section: The validated ``data`` section of the run.

    Returns:
        ``{"datasets": {name: {"version", "manifest_sha256", "resize"}},
        "taxonomy": {"version", "manifest_sha256"}}``. ``resize`` is the entry's resize spec
        as a dict, or None.
    """
    return {
        "datasets": {
            name: {
                "version": entry.version,
                "manifest_sha256": entry.manifest_sha256,
                "resize": entry.resize.model_dump() if entry.resize is not None else None,
            }
            for name, entry in section.datasets.items()
        },
        "taxonomy": section.taxonomy.model_dump(),
    }


def collect_lineage(*, resolved_config: dict, seed: int, data_version: dict) -> dict:
    """Everything needed to trace a run back to its code, environment, config and data.

    ``git_dirty`` counts tracked changes only; unreadable git counts as dirty.
    Inside a checkout (``REPO_ROOT/.git`` exists as a directory or a worktree file) git gives the
    revision and dirty state. Without ``.git``, a non-empty ``MERMAIDSEG_GIT_SHA`` is the revision of
    a CI-built image: it is recorded as ``git_sha`` with ``git_dirty`` False (the tree is committed by
    construction) and git is not run.
    """
    build_sha = os.environ.get(BUILD_GIT_SHA_ENV)
    if build_sha and not (Path(REPO_ROOT) / ".git").exists():
        git_sha: str | None = build_sha
        git_dirty = False
    else:
        git_sha, git_dirty = _git_sha(), _git_dirty()
    return {
        "git_sha": git_sha,
        "git_dirty": git_dirty,
        "uv_lock_sha256": _uv_lock_sha256(),
        "seed": seed,
        "data_version": data_version,
        "resolved_config": resolved_config,
        "versions": {name: version(name) for name in VERSIONED_PACKAGES},
    }


def _pin_differences(label: str, new: dict, old: dict, fields: tuple[str, ...]) -> list[str]:
    """One line per field of ``fields`` that differs between two pins, naming ``label``."""
    return [
        f"{label}: {field} {old.get(field)!r} in the checkpoint, {new.get(field)!r} now"
        for field in fields
        if new.get(field) != old.get(field)
    ]


def _data_differences(new: dict, old: dict) -> list[str]:
    """Every difference between two ``data_version`` dicts, datasets sorted by name."""
    new_datasets, old_datasets = new.get("datasets") or {}, old.get("datasets") or {}
    lines = []
    for name in sorted(new_datasets.keys() | old_datasets.keys()):
        if name not in old_datasets:
            lines.append(f"dataset {name!r}: added, not in the checkpoint")
        elif name not in new_datasets:
            lines.append(f"dataset {name!r}: removed, pinned in the checkpoint")
        else:
            lines += _pin_differences(
                f"dataset {name!r}",
                new_datasets[name],
                old_datasets[name],
                ("version", "manifest_sha256", "resize"),
            )
    lines += _pin_differences(
        "taxonomy", new["taxonomy"], old["taxonomy"], ("version", "manifest_sha256")
    )
    return lines


def check_same_data(data_version: dict, source: dict) -> None:
    """Refuse a resume whose data pins differ from those of the checkpoint being resumed.

    Args:
        data_version: The pins of the new session, as returned by ``data_version()``.
        source: Lineage stored in the checkpoint; its ``data_version`` is the reference.

    Raises:
        ValueError: Naming every dataset or taxonomy pin (version, manifest_sha256, resize)
            that differs, and every dataset added or removed.
    """
    differences = _data_differences(data_version, source["data_version"])
    if differences:
        raise ValueError(
            "Resuming a run cannot change its data; the pins differ from the checkpoint's:\n"
            + "\n".join(differences)
            + "\nUse --init-weights to start a new run from these weights on other data."
        )


def inherit_lineage(lineage: dict, source: dict) -> dict:
    """Lineage of a session that started from a checkpoint with lineage ``source``.

    The source's ``git_dirty`` is already its own inherited verdict, so a dirty ancestor
    propagates across any number of resumes.

    Args:
        lineage: Lineage of the new session.
        source: Lineage stored in the checkpoint.

    Returns:
        A new dict with the fields of ``lineage`` plus ``parent`` and a combined ``git_dirty``.
        ``parent`` is a dict holding only the source's ``git_sha``, ``git_dirty`` and
        ``data_version``. ``git_dirty`` is True if either side is True.
    """
    git_dirty = lineage["git_dirty"] or source["git_dirty"]
    parent = {
        "git_sha": source.get("git_sha"),
        "git_dirty": source["git_dirty"],
        "data_version": source.get("data_version"),
    }
    return {**lineage, "git_dirty": git_dirty, "parent": parent}
