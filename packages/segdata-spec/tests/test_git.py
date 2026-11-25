import hashlib
import subprocess
from pathlib import Path

import pytest

from segdata_spec.errors import PublishRefused
from segdata_spec.git import GitState, probe_git, refuse_unpublishable


def _run(cwd: Path, *args: str) -> None:
    subprocess.run(["git", *args], cwd=cwd, check=True, capture_output=True)  # noqa: S603, S607


def test_probe_git_reports_dirty_and_pushed(tmp_path):
    origin = tmp_path / "origin.git"
    work = tmp_path / "work"
    work.mkdir()
    _run(tmp_path, "init", "--bare", "-b", "main", str(origin))
    _run(work, "init", "-b", "main")
    _run(work, "config", "user.name", "t")
    _run(work, "config", "user.email", "t@example.com")
    _run(work, "remote", "add", "origin", str(origin))
    tracked = work / "f.txt"
    tracked.write_text("one")
    _run(work, "add", "f.txt")
    _run(work, "commit", "-m", "one")
    _run(work, "push", "-u", "origin", "main")

    state = probe_git(work)
    assert (state.dirty, state.pushed) == (False, True)
    assert len(state.sha) == 40

    (work / "untracked.txt").write_text("x")
    assert probe_git(work).dirty is False

    tracked.write_text("two")
    assert probe_git(work).dirty is True

    _run(work, "commit", "-am", "two")
    local = probe_git(work)
    assert (local.dirty, local.pushed) == (False, False)


def test_refuse_unpublishable_refuses_a_dirty_tree(tmp_path):
    (tmp_path / "uv.lock").write_text("lock")
    with pytest.raises(PublishRefused, match="uncommitted changes to tracked files"):
        refuse_unpublishable(GitState(sha="abc", dirty=True, pushed=True), tmp_path)


def test_refuse_unpublishable_refuses_an_unpushed_head(tmp_path):
    (tmp_path / "uv.lock").write_text("lock")
    with pytest.raises(PublishRefused, match="HEAD abc is not contained in any remote branch"):
        refuse_unpublishable(GitState(sha="abc", dirty=False, pushed=False), tmp_path)


def test_refuse_unpublishable_refuses_a_tree_without_uv_lock(tmp_path):
    with pytest.raises(PublishRefused, match="no uv.lock"):
        refuse_unpublishable(GitState(sha="abc", dirty=False, pushed=True), tmp_path)


def test_refuse_unpublishable_returns_the_uv_lock_hash_for_a_clean_pushed_tree(tmp_path):
    (tmp_path / "uv.lock").write_bytes(b"lock-bytes")
    lock = refuse_unpublishable(GitState(sha="abc", dirty=False, pushed=True), tmp_path)
    assert lock == hashlib.sha256(b"lock-bytes").hexdigest()
