"""Git state of the ETL repository at stage and publish time, and the refusal rules."""

from __future__ import annotations

import hashlib
import subprocess
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

from segdata_spec.errors import DirtyTree, PublishRefused


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


def on_remote(repo_dir: Path, sha: str) -> bool:
    """Whether some remote-tracking branch contains ``sha``; False when git cannot tell.

    The attached ``=`` form keeps ``sha`` from ever being parsed as an option.
    """
    result = subprocess.run(  # noqa: S603
        ["git", "-C", str(repo_dir), "branch", "-r", f"--contains={sha}"],  # noqa: S607
        capture_output=True,
        text=True,
        check=False,
    )
    return result.returncode == 0 and bool(result.stdout.strip())


def probe_git(repo_dir: Path) -> GitState:
    """Read HEAD, tracked-file dirtiness and whether a remote branch contains HEAD."""
    sha = _git(repo_dir, "rev-parse", "HEAD")
    dirty = bool(_git(repo_dir, "status", "--porcelain", "--untracked-files=no"))
    pushed = on_remote(repo_dir, sha)
    return GitState(sha=sha, dirty=dirty, pushed=pushed)


def uv_lock_sha256(repo_dir: Path) -> str | None:
    """Return the hex sha256 of ``uv.lock`` in ``repo_dir``, or None when it is absent."""
    lock = repo_dir / "uv.lock"
    if not lock.is_file():
        return None
    return hashlib.sha256(lock.read_bytes()).hexdigest()


def _refuse_unpublishable(git: GitState, repo_dir: Path) -> str:
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


def find_repo_root(start: Path) -> Path:
    """Return the nearest directory at or above ``start`` holding both ``.git`` and ``uv.lock``.

    Raises:
        FileNotFoundError: No such directory exists.
    """
    for directory in (start, *start.parents):
        if (directory / ".git").exists() and (directory / "uv.lock").is_file():
            return directory
    raise FileNotFoundError(f"no git root holding uv.lock at or above {start}")


def is_tracked(repo_dir: Path, path: Path) -> bool:
    """Whether ``path`` is a file tracked by the repo at ``repo_dir``; False outside it.

    Args:
        repo_dir: A directory inside the repo to ask.
        path: The file to look up; relative paths resolve against the current directory.
    """
    result = subprocess.run(  # noqa: S603
        ["git", "-C", str(repo_dir), "ls-files", "--error-unmatch", "--", str(path.resolve())],  # noqa: S607
        capture_output=True,
        text=True,
        check=False,
    )
    return result.returncode == 0


def clean_git_state(start: Path, *, inputs: Sequence[Path] = ()) -> GitState:
    """Probe the repo root above ``start``; require tracked files to match HEAD and ``inputs`` tracked.

    Untracked files do not make the tree dirty, so an input file the recorded sha must pin is
    checked to be tracked in the repo. Relative ``inputs`` resolve against the current directory.

    Raises:
        FileNotFoundError: No repo root holding ``uv.lock`` is found.
        DirtyTree: Tracked files differ from HEAD, or an input is not a tracked file of the repo
            (untracked, or outside it).
    """
    root = find_repo_root(start)
    git = probe_git(root)
    if git.dirty:
        raise DirtyTree("working tree has uncommitted changes to tracked files")
    for path in inputs:
        if not is_tracked(root, path):
            raise DirtyTree(f"{path} is not a git-tracked file of {root}")
    return git


@dataclass(frozen=True)
class Publishable:
    """A repo whose tree was checked clean, pushed and locked; holding one is the proof."""

    root: Path
    git: GitState
    uv_lock_sha256: str


def publishable_repo(start: Path) -> Publishable:
    """Find the repo root above ``start``, probe its git state and refuse an unpublishable tree.

    Raises:
        FileNotFoundError: No repo root holding ``uv.lock`` is found.
        PublishRefused: The git tree is dirty or unpushed, or ``uv.lock`` is missing.
    """
    root = find_repo_root(start)
    git = probe_git(root)
    return Publishable(root=root, git=git, uv_lock_sha256=_refuse_unpublishable(git, root))
