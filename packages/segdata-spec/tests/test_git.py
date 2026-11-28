import hashlib
import subprocess
from pathlib import Path

import pytest

from segdata_spec.errors import PublishRefused
from segdata_spec.git import Publishable, on_remote, probe_git, publishable_repo


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


def _repo_with_lock(tmp_path: Path) -> tuple[Path, str]:
    """A pushed repo holding a committed ``uv.lock``; returns it and the lock's sha256."""
    work, _ = _repo_with_remote(tmp_path)
    (work / "uv.lock").write_bytes(b"lock-bytes")
    _run(work, "add", "uv.lock")
    _run(work, "commit", "-m", "lock")
    _run(work, "push")
    return work, hashlib.sha256(b"lock-bytes").hexdigest()


def test_publishable_repo_returns_root_git_state_and_lock_hash_for_a_clean_pushed_repo(tmp_path):
    work, lock = _repo_with_lock(tmp_path)
    nested = work / "pkg" / "src"
    nested.mkdir(parents=True)

    repo = publishable_repo(nested)

    assert isinstance(repo, Publishable)
    assert repo.root == work
    assert (repo.git.dirty, repo.git.pushed) == (False, True)
    assert repo.git.sha == _head(work)
    assert repo.uv_lock_sha256 == lock


def _make_dirty(work: Path) -> None:
    (work / "f.txt").write_text("changed")


def _make_unpushed(work: Path) -> None:
    (work / "g.txt").write_text("two")
    _run(work, "add", "g.txt")
    _run(work, "commit", "-m", "two")


@pytest.mark.parametrize(
    ("spoil", "keyword"),
    [
        (_make_dirty, "uncommitted"),
        (_make_unpushed, "not contained in any remote branch"),
    ],
    ids=["dirty", "unpushed"],
)
def test_publishable_repo_refuses_an_unpublishable_tree(tmp_path, spoil, keyword):
    work, _ = _repo_with_lock(tmp_path)
    spoil(work)

    with pytest.raises(PublishRefused, match=keyword):
        publishable_repo(work)


def test_publishable_repo_refuses_a_missing_uv_lock(tmp_path, monkeypatch):
    work, _ = _repo_with_remote(tmp_path)
    (work / "uv.lock").write_bytes(b"lock-bytes")
    # find_repo_root requires the lock, so a lock deleted after the root is found models the gap.
    monkeypatch.setattr("segdata_spec.git.find_repo_root", lambda _start: work)
    (work / "uv.lock").unlink()

    with pytest.raises(PublishRefused, match="no uv.lock"):
        publishable_repo(work)


def test_publishable_repo_raises_when_no_root_holds_git_and_uv_lock(tmp_path):
    with pytest.raises(FileNotFoundError, match="uv.lock"):
        publishable_repo(tmp_path)
