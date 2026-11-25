"""Run lineage: git state, lock hash, data version."""

from __future__ import annotations

import copy
import shutil
import subprocess
from pathlib import Path

import pytest

from mermaidseg import lineage
from mermaidseg.data.config import DataSection
from mermaidseg.lineage import collect_lineage, data_version, inherit_lineage


def _collect(**overrides):
    kwargs = {"resolved_config": {"a": 1}, "seed": 7, "data_version": {"x": 1}}
    return collect_lineage(**{**kwargs, **overrides})


def _git(repo: Path, *args: str) -> str:
    return subprocess.run(
        ["git", "-C", str(repo), *args], check=True, capture_output=True, text=True
    ).stdout


@pytest.fixture()
def git_repo(monkeypatch, tmp_path: Path) -> Path:
    if shutil.which("git") is None:
        pytest.skip("git is not on PATH")
    _git(tmp_path, "init", "-q")
    _git(tmp_path, "config", "user.name", "Test")
    _git(tmp_path, "config", "user.email", "test@example.com")
    _git(tmp_path, "config", "commit.gpgsign", "false")
    (tmp_path / "tracked.txt").write_text("one\n")
    _git(tmp_path, "add", "tracked.txt")
    _git(tmp_path, "commit", "-q", "-m", "init")
    monkeypatch.setattr(lineage, "REPO_ROOT", tmp_path)
    return tmp_path


def test_untracked_file_alone_is_not_dirty(git_repo):
    (git_repo / "new.txt").write_text("x\n")
    assert _collect()["git_dirty"] is False


@pytest.mark.parametrize(
    "failure",
    [
        FileNotFoundError("git"),
        subprocess.CalledProcessError(128, "git"),
        subprocess.TimeoutExpired("git", 5),
    ],
)
def test_unavailable_git_leaves_sha_and_dirty_none(monkeypatch, failure):
    def run(*_args, **_kwargs):
        raise failure

    monkeypatch.setattr(lineage.subprocess, "run", run)
    result = _collect()
    assert result["git_sha"] is None
    assert result["git_dirty"] is None


@pytest.fixture(autouse=True)
def _no_build_sha(monkeypatch):
    monkeypatch.delenv("MERMAIDSEG_GIT_SHA", raising=False)


def test_build_sha_is_recorded_clean_without_calling_git(monkeypatch, tmp_path: Path):
    def run(*_args, **_kwargs):
        raise AssertionError("git must not be invoked in a checkout without .git")

    monkeypatch.setattr(lineage, "REPO_ROOT", tmp_path)
    monkeypatch.setenv("MERMAIDSEG_GIT_SHA", "abc123")
    monkeypatch.setattr(lineage.subprocess, "run", run)
    result = _collect()
    assert result["git_sha"] == "abc123"
    assert result["git_dirty"] is False


def test_build_sha_is_ignored_in_a_git_checkout(monkeypatch, git_repo):
    monkeypatch.setenv("MERMAIDSEG_GIT_SHA", "abc123")
    (git_repo / "tracked.txt").write_text("two\n")
    result = _collect()
    assert result["git_sha"] == _git(git_repo, "rev-parse", "HEAD").strip()
    assert result["git_dirty"] is True


def test_build_sha_is_ignored_in_a_git_worktree(monkeypatch, git_repo, tmp_path_factory):
    worktree = tmp_path_factory.mktemp("worktrees") / "checkout"
    _git(git_repo, "worktree", "add", "--detach", "-q", str(worktree))
    assert (worktree / ".git").is_file()
    monkeypatch.setattr(lineage, "REPO_ROOT", worktree)
    monkeypatch.setenv("MERMAIDSEG_GIT_SHA", "abc123")
    (worktree / "tracked.txt").write_text("two\n")
    result = _collect()
    assert result["git_sha"] == _git(worktree, "rev-parse", "HEAD").strip()
    assert result["git_dirty"] is True


def test_empty_build_sha_falls_back_to_git(monkeypatch, git_repo):
    monkeypatch.setenv("MERMAIDSEG_GIT_SHA", "")
    (git_repo / "tracked.txt").write_text("two\n")
    result = _collect()
    assert result["git_sha"] == _git(git_repo, "rev-parse", "HEAD").strip()
    assert result["git_dirty"] is True


def test_uv_lock_sha256_hashes_the_lock_file(monkeypatch, tmp_path: Path):
    (tmp_path / "uv.lock").write_bytes(b"locked\n")
    monkeypatch.setattr(lineage, "REPO_ROOT", tmp_path)
    assert _collect()["uv_lock_sha256"] == (
        "3a52732e0c98263090a2cd2509e7d2244d7194bd65f78b29e6ef6448e8143666"
    )


def test_missing_uv_lock_gives_none(monkeypatch, tmp_path: Path):
    monkeypatch.setattr(lineage, "REPO_ROOT", tmp_path)
    assert _collect()["uv_lock_sha256"] is None


def test_collect_lineage_carries_the_run_inputs_and_library_versions(monkeypatch):
    result = _collect()
    assert result["seed"] == 7
    assert result["resolved_config"] == {"a": 1}
    assert result["data_version"] == {"x": 1}
    assert set(result["versions"]) == {"torch", "transformers", "peft"}
    assert all(isinstance(v, str) for v in result["versions"].values())


def test_data_version_pins_each_dataset_and_the_taxonomy():
    section = DataSection.model_validate(
        {
            "taxonomy": {"version": "t1", "manifest_sha256": "tt"},
            "datasets": {
                "coralnet": {
                    "version": "v1",
                    "manifest_sha256": "aa",
                    "resize": {"max_edge": 2048},
                    "train": {"groups": ["s1"]},
                    "test": {"groups": ["s2"]},
                },
                "mermaid": {"version": "v2", "manifest_sha256": "bb", "train": {"groups": "all"}},
            },
        }
    )
    assert data_version(section) == {
        "datasets": {
            "coralnet": {
                "version": "v1",
                "manifest_sha256": "aa",
                "resize": {"max_edge": 2048, "filter": "lanczos", "jpeg_quality": 95},
            },
            "mermaid": {"version": "v2", "manifest_sha256": "bb", "resize": None},
        },
        "taxonomy": {"version": "t1", "manifest_sha256": "tt"},
    }


def test_data_version_without_a_taxonomy_pin_records_none():
    section = DataSection.model_validate(
        {"datasets": {"m": {"version": "v", "manifest_sha256": "s", "val": {"groups": "all"}}}}
    )

    assert data_version(section)["taxonomy"] is None


@pytest.mark.parametrize(
    ("own", "source", "expected"),
    [
        (False, {"git_dirty": False}, False),
        (False, {"git_dirty": True}, True),
        (True, {"git_dirty": False}, True),
        (False, {"git_dirty": None}, None),
        (None, {"git_dirty": False}, None),
        (False, {}, None),
        (False, None, None),
        (True, None, True),
        (None, {"git_dirty": True}, True),
    ],
)
def test_inherit_lineage_combines_git_dirty_of_own_and_source(own, source, expected):
    assert inherit_lineage({"git_dirty": own}, source)["git_dirty"] is expected


def test_inherit_lineage_keeps_only_the_source_sha_and_dirty_flag_without_mutating():
    own = {"git_dirty": False, "git_sha": "abc", "seed": 7, "resolved_config": {"a": 1}}
    source = {
        "git_dirty": False,
        "git_sha": "old",
        "parent": {"git_dirty": True},
        "resolved_config": {"b": 2},
    }
    own_before, source_before = copy.deepcopy(own), copy.deepcopy(source)

    result = inherit_lineage(own, source)

    assert result["parent"] == {"git_sha": "old", "git_dirty": False}
    assert {k: result[k] for k in ("git_sha", "seed", "resolved_config")} == {
        "git_sha": "abc",
        "seed": 7,
        "resolved_config": {"a": 1},
    }
    assert own == own_before
    assert source == source_before


def test_inherit_lineage_records_a_missing_source_as_a_null_parent():
    result = inherit_lineage({"git_dirty": False}, None)
    assert "parent" in result and result["parent"] is None
