import hashlib
import subprocess
from pathlib import Path

import pytest

from segdata_spec.errors import PublishRefused
from segdata_spec.git import GitState, on_remote, probe_git, refuse_unpublishable


def _run(cwd: Path, *args: str) -> None:
    subprocess.run(["git", *args], cwd=cwd, check=True, capture_output=True)  # noqa: S603, S607


def _head(repo: Path) -> str:
    result = subprocess.run(  # noqa: S603
        ["git", "rev-parse", "HEAD"],  # noqa: S607
        cwd=repo,
        check=True,
        capture_output=True,
        text=True,
    )
    return result.stdout.strip()


def _repo_with_remote(tmp_path: Path) -> tuple[Path, str]:
    """A repo whose ``origin/main`` holds the returned sha."""
    origin = tmp_path / "origin.git"
    work = tmp_path / "work"
    work.mkdir()
    _run(tmp_path, "init", "--bare", "-b", "main", str(origin))
    _run(work, "init", "-b", "main")
    _run(work, "config", "user.name", "t")
    _run(work, "config", "user.email", "t@example.com")
    _run(work, "remote", "add", "origin", str(origin))
    (work / "f.txt").write_text("one")
    _run(work, "add", "f.txt")
    _run(work, "commit", "-m", "one")
    _run(work, "push", "-u", "origin", "main")
    return work, _head(work)


def test_on_remote_is_true_for_a_sha_on_a_remote_branch(tmp_path):
    work, pushed = _repo_with_remote(tmp_path)
    assert on_remote(work, pushed) is True


def test_on_remote_is_false_for_a_sha_only_on_a_local_branch(tmp_path):
    work, _ = _repo_with_remote(tmp_path)
    (work / "g.txt").write_text("two")
    _run(work, "add", "g.txt")
    _run(work, "commit", "-m", "two")
    assert on_remote(work, _head(work)) is False


def test_on_remote_is_false_for_a_sha_the_repository_does_not_know(tmp_path):
    work, _ = _repo_with_remote(tmp_path)
    assert on_remote(work, "0123456789abcdef0123456789abcdef01234567") is False


def test_on_remote_is_false_outside_a_git_checkout(tmp_path):
    plain = tmp_path / "plain"
    plain.mkdir()
    assert on_remote(plain, "0123456789abcdef0123456789abcdef01234567") is False


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


@pytest.mark.parametrize(
    ("dirty", "pushed", "with_lock", "keyword"),
    [
        (True, True, True, "uncommitted"),
        (False, False, True, "not contained in any remote branch"),
        (False, True, False, "no uv.lock"),
    ],
    ids=["dirty", "unpushed", "no-uv-lock"],
)
def test_refuse_unpublishable_refuses_an_unpublishable_tree(
    tmp_path, dirty, pushed, with_lock, keyword
):
    if with_lock:
        (tmp_path / "uv.lock").write_text("lock")
    with pytest.raises(PublishRefused, match=keyword):
        refuse_unpublishable(GitState(sha="abc", dirty=dirty, pushed=pushed), tmp_path)


def test_refuse_unpublishable_returns_the_uv_lock_hash_for_a_clean_pushed_tree(tmp_path):
    (tmp_path / "uv.lock").write_bytes(b"lock-bytes")
    lock = refuse_unpublishable(GitState(sha="abc", dirty=False, pushed=True), tmp_path)
    assert lock == hashlib.sha256(b"lock-bytes").hexdigest()
