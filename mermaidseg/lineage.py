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


def _git_dirty() -> bool | None:
    """Whether tracked files differ from HEAD; untracked files do not count."""
    out = _git_output("status", "--porcelain", "--untracked-files=no")
    return None if out is None else bool(out.strip())


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
        "taxonomy": {"version", "manifest_sha256"} | None}``. ``resize`` is the entry's resize spec
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
        "taxonomy": section.taxonomy.model_dump() if section.taxonomy is not None else None,
    }


def collect_lineage(*, resolved_config: dict, seed: int, data_version: dict) -> dict:
    """Everything needed to trace a run back to its code, environment, config and data.

    ``git_dirty`` counts tracked changes only; None (git unavailable) is as untrusted as True.
    Inside a checkout (``REPO_ROOT/.git`` exists as a directory or a worktree file) git gives the
    revision and dirty state. Without ``.git``, a non-empty ``MERMAIDSEG_GIT_SHA`` is the revision of
    a CI-built image: it is recorded as ``git_sha`` with ``git_dirty`` False (the tree is committed by
    construction) and git is not run.
    """
    build_sha = os.environ.get(BUILD_GIT_SHA_ENV)
    if build_sha and not (Path(REPO_ROOT) / ".git").exists():
        git_sha: str | None = build_sha
        git_dirty: bool | None = False
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


def inherit_lineage(lineage: dict, source: dict | None) -> dict:
    """Lineage of a session that started from a checkpoint with lineage ``source``.

    The source's ``git_dirty`` is already its own inherited verdict, so a dirty ancestor
    propagates across any number of resumes.

    Args:
        lineage: Lineage of the new session.
        source: Lineage stored in the checkpoint, or None when the checkpoint has none.

    Returns:
        A new dict with the fields of ``lineage`` plus ``parent`` and a combined ``git_dirty``.
        ``parent`` is None when ``source`` is None, else a dict holding only the source's
        ``git_sha`` and ``git_dirty``. ``git_dirty`` is True if either side is True, else None
        if either side is unknown (including a missing ``source``), else False.
    """
    own_dirty = lineage.get("git_dirty")
    source_dirty = None if source is None else source.get("git_dirty")
    if own_dirty is True or source_dirty is True:
        git_dirty = True
    elif own_dirty is None or source_dirty is None:
        git_dirty = None
    else:
        git_dirty = False
    parent = (
        None
        if source is None
        else {"git_sha": source.get("git_sha"), "git_dirty": source.get("git_dirty")}
    )
    return {**lineage, "git_dirty": git_dirty, "parent": parent}
