"""Run lineage: git state, lock hash, data version."""

from __future__ import annotations

import copy
import shutil
import subprocess
from pathlib import Path

import pytest

from mermaidseg import lineage
from mermaidseg.data.config import DataSection
from mermaidseg.lineage import check_same_data, collect_lineage, data_version, inherit_lineage


def _collect(**overrides):
    kwargs = {"resolved_config": {"a": 1}, "seed": 7, "data_version": {"x": 1}, "inputs": []}
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


def test_an_untracked_input_makes_a_clean_repo_dirty(git_repo):
    (git_repo / "cfg.yaml").write_text("x: 1\n")
    inputs = [git_repo / "tracked.txt", git_repo / "cfg.yaml"]
    assert _collect(inputs=inputs)["git_dirty"] is True


def test_tracked_inputs_leave_a_clean_repo_clean(git_repo):
    assert _collect(inputs=[git_repo / "tracked.txt"])["git_dirty"] is False


def test_build_sha_image_ignores_inputs(monkeypatch, tmp_path: Path):
    monkeypatch.setattr(lineage, "REPO_ROOT", tmp_path)
    monkeypatch.setenv("MERMAIDSEG_GIT_SHA", "abc123")
    assert _collect(inputs=[tmp_path / "missing.yaml"])["git_dirty"] is False


@pytest.mark.parametrize(
    "failure",
    [
        FileNotFoundError("git"),
        subprocess.CalledProcessError(128, "git"),
    ],
)
def test_unavailable_git_leaves_sha_none_and_counts_as_dirty(monkeypatch, failure):
    def run(*_args, **_kwargs):
        raise failure

    monkeypatch.setattr(lineage.subprocess, "run", run)
    result = _collect()
    assert result["git_sha"] is None
    assert result["git_dirty"] is True


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


@pytest.mark.parametrize(
    ("own", "source", "expected"),
    [
        (False, {"git_dirty": False}, False),
        (True, {"git_dirty": False}, True),
        (False, {"git_dirty": True}, True),
    ],
)
def test_inherit_lineage_combines_git_dirty_of_own_and_source(own, source, expected):
    assert inherit_lineage({"git_dirty": own}, source)["git_dirty"] is expected


def test_inherit_lineage_keeps_only_the_source_sha_dirty_flag_and_data_without_mutating():
    own = {"git_dirty": False, "git_sha": "abc", "seed": 7, "resolved_config": {"a": 1}}
    source = {
        "git_dirty": False,
        "git_sha": "old",
        "data_version": {"datasets": {"d": {"version": "v1"}}, "taxonomy": None},
        "parent": {"git_dirty": True},
        "resolved_config": {"b": 2},
    }
    own_before, source_before = copy.deepcopy(own), copy.deepcopy(source)

    result = inherit_lineage(own, source)

    assert result["parent"] == {
        "git_sha": "old",
        "git_dirty": False,
        "data_version": {"datasets": {"d": {"version": "v1"}}, "taxonomy": None},
    }
    assert {k: result[k] for k in ("git_sha", "seed", "resolved_config")} == {
        "git_sha": "abc",
        "seed": 7,
        "resolved_config": {"a": 1},
    }
    assert own == own_before
    assert source == source_before


def _pins() -> dict:
    return {
        "datasets": {
            "coralnet": {"version": "v1", "manifest_sha256": "a" * 64, "resize": 2048},
        },
        "taxonomy": {"version": "t1", "manifest_sha256": "b" * 64},
    }


def test_check_same_data_accepts_identical_pins():
    assert check_same_data(_pins(), {"data_version": _pins()}) is None


def _with_dataset(pins: dict, name: str, **fields) -> dict:
    out = copy.deepcopy(pins)
    out["datasets"][name] = {**out["datasets"].get("coralnet", {}), **fields}
    return out


def _changed(**fields) -> dict:
    return _with_dataset(_pins(), "coralnet", **fields)


def _added() -> dict:
    return _with_dataset(_pins(), "coralscapes")


def _taxonomy_changed() -> dict:
    out = _pins()
    out["taxonomy"] = {"version": "t2", "manifest_sha256": "b" * 64}
    return out


def _two_datasets() -> dict:
    return _with_dataset(
        _pins(), "coralscapes", version="s1", manifest_sha256="d" * 64, resize=None
    )


def _all_changed() -> dict:
    out = _two_datasets()
    out["datasets"]["coralnet"]["version"] = "v2"
    out["datasets"]["coralscapes"]["version"] = "s2"
    out["taxonomy"] = {"version": "t2", "manifest_sha256": "b" * 64}
    return out


@pytest.mark.parametrize(
    ("new", "old", "needles"),
    [
        pytest.param(_changed(version="v2"), _pins(), ("coralnet", "v1", "v2"), id="version"),
        pytest.param(
            _changed(manifest_sha256="c" * 64),
            _pins(),
            ("coralnet", "manifest_sha256"),
            id="manifest",
        ),
        pytest.param(_changed(resize=None), _pins(), ("coralnet", "resize"), id="resize"),
        pytest.param(_added(), _pins(), ("coralscapes",), id="dataset-added"),
        pytest.param(_pins(), _added(), ("coralscapes",), id="dataset-removed"),
        pytest.param(_taxonomy_changed(), _pins(), ("taxonomy", "t1", "t2"), id="taxonomy"),
        pytest.param(
            _all_changed(),
            _two_datasets(),
            ("coralnet", "coralscapes", "taxonomy"),
            id="several-at-once",
        ),
    ],
)
def test_check_same_data_names_every_differing_pin(new, old, needles):
    with pytest.raises(ValueError) as raised:
        check_same_data(new, {"data_version": old})
    for needle in needles:
        assert needle in str(raised.value)
