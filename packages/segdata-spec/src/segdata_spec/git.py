"""Git state of the ETL repository at stage and publish time, and the refusal rules."""

from __future__ import annotations

import hashlib
import subprocess
from dataclasses import dataclass
from pathlib import Path

from segdata_spec.errors import PublishRefused


@dataclass(frozen=True)
class GitState:
    """The state of the ETL repository at publish time."""

    sha: str
    dirty: bool
    pushed: bool


def _git(repo_dir: Path, *args: str) -> str:
    result = subprocess.run(  # noqa: S603
        ["git", "-C", str(repo_dir), *args],  # noqa: S607
        capture_output=True,
        text=True,
        check=True,
    )
    return result.stdout.strip()


def probe_git(repo_dir: Path) -> GitState:
    """Read HEAD, tracked-file dirtiness and whether a remote branch contains HEAD."""
    sha = _git(repo_dir, "rev-parse", "HEAD")
    dirty = bool(_git(repo_dir, "status", "--porcelain", "--untracked-files=no"))
    pushed = bool(_git(repo_dir, "branch", "-r", "--contains", "HEAD"))
    return GitState(sha=sha, dirty=dirty, pushed=pushed)


def uv_lock_sha256(repo_dir: Path) -> str | None:
    """Return the hex sha256 of ``uv.lock`` in ``repo_dir``, or None when it is absent."""
    lock = repo_dir / "uv.lock"
    if not lock.is_file():
        return None
    return hashlib.sha256(lock.read_bytes()).hexdigest()


def refuse_unpublishable(git: GitState, repo_dir: Path) -> str:
    """Return the sha256 of ``repo_dir``'s ``uv.lock`` for a publishable tree.

    Raises:
        PublishRefused: The tree has uncommitted tracked changes, HEAD is on no remote branch, or
            ``repo_dir`` has no ``uv.lock``.
    """
    if git.dirty:
        raise PublishRefused("working tree has uncommitted changes to tracked files")
    if not git.pushed:
        raise PublishRefused(f"HEAD {git.sha} is not contained in any remote branch")
    lock = uv_lock_sha256(repo_dir)
    if lock is None:
        raise PublishRefused(f"no uv.lock in {repo_dir}")
    return lock
