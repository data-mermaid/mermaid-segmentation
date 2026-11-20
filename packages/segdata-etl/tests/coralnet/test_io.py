"""Tests for the I/O helpers: parquet writes, S3 URIs, the ibis CSV reader and version tags."""

from __future__ import annotations

import re
import subprocess

import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from segdata_etl.datasets.coralnet import io as dataset_io
from segdata_etl.datasets.coralnet.io import (
    _git_short_sha,
    compute_version_tag,
    parse_s3_uri,
    read_csv_via_ibis,
    upload_to_s3,
    write_parquet_deterministic,
)

_PAIR_SCHEMA = pa.schema([pa.field("k", pa.int32()), pa.field("v", pa.string())])


def test_write_parquet_deterministic_stores_rows_in_primary_key_order(tmp_path):
    out = tmp_path / "nested" / "dir" / "pairs.parquet"
    df = pd.DataFrame({"v": ["a", "b", "c", "d"], "k": [2, 1, 2, 1]})

    write_parquet_deterministic(df, out, _PAIR_SCHEMA, ("k",))

    # Ties keep their input order (a before c, b before d).
    assert pq.read_table(out).to_pylist() == [
        {"k": 1, "v": "b"},
        {"k": 1, "v": "d"},
        {"k": 2, "v": "a"},
        {"k": 2, "v": "c"},
    ]


@pytest.mark.parametrize(
    ("uri", "expected"),
    [
        ("s3://b/k/x.parquet", ("b", "k/x.parquet")),
        ("s3://b/k", ("b", "k")),
        ("s3://b/", ("b", "")),
        ("s3://b", ("b", "")),
    ],
)
def test_parse_s3_uri(uri, expected):
    assert parse_s3_uri(uri) == expected


@pytest.mark.parametrize("uri", ["http://b/k", "b/k", ""])
def test_parse_s3_uri_rejects_a_non_s3_uri(uri):
    with pytest.raises(ValueError, match="Expected s3:// URI"):
        parse_s3_uri(uri)


@pytest.mark.parametrize(
    ("uri", "message"),
    [
        ("s3://b", "S3 URI missing key"),
        ("s3://b/", "S3 URI missing key"),
        ("gs://b/k", "Expected s3:// URI"),
    ],
)
def test_upload_to_s3_rejects_a_bad_uri_before_uploading(fake_s3, tmp_path, uri, message):
    local = tmp_path / "f.bin"
    local.write_bytes(b"data")

    with pytest.raises(ValueError, match=message):
        upload_to_s3(fake_s3, local, uri)

    assert fake_s3.objects == {}


class _StubTable:
    def __init__(self, df: pd.DataFrame):
        self._df = df

    def execute(self) -> pd.DataFrame:
        return self._df


class _StubCon:
    """Stands in for the ibis connection: returns a table or raises on ``read_csv``."""

    def __init__(self, result: pd.DataFrame | Exception):
        self._result = result
        self.calls: list[tuple[str, dict]] = []

    def read_csv(self, uri: str, **kwargs):
        self.calls.append((uri, kwargs))
        if isinstance(self._result, Exception):
            raise self._result
        return _StubTable(self._result)


def test_read_csv_via_ibis_reads_the_s3_uri_with_the_messy_csv_options():
    df = pd.DataFrame({"a": ["1"]})
    con = _StubCon(df)

    result = read_csv_via_ibis(con, "bkt", "p/s1/annotations.csv")

    assert result is df
    assert con.calls == [
        (
            "s3://bkt/p/s1/annotations.csv",
            {"null_padding": True, "all_varchar": True, "quote": '"'},
        )
    ]


@pytest.mark.parametrize(
    "message",
    [
        "HTTP Error: HTTP GET error on 's3://b/k' (HTTP 404)",
        "IO Error: Not Found",
        "NoSuchKey: the key does not exist",
    ],
)
def test_read_csv_via_ibis_maps_a_missing_object_to_none(message):
    assert read_csv_via_ibis(_StubCon(RuntimeError(message)), "b", "k") is None


def test_read_csv_via_ibis_reraises_other_errors():
    with pytest.raises(RuntimeError, match="HTTP 403"):
        read_csv_via_ibis(_StubCon(RuntimeError("HTTP 403 Forbidden")), "b", "k")


def _git(repo, *args: str) -> str:
    return subprocess.run(
        ["git", "-c", "user.name=t", "-c", "user.email=t@example.com", *args],
        cwd=repo,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()


@pytest.fixture
def git_repo(tmp_path, monkeypatch):
    """A one-commit git repo, entered as the working directory."""
    for var in ("GIT_DIR", "GIT_WORK_TREE", "GIT_INDEX_FILE"):
        monkeypatch.delenv(var, raising=False)
    repo = tmp_path / "repo"
    repo.mkdir()
    _git(repo, "init", "-q")
    _git(repo, "commit", "-q", "--allow-empty", "-m", "first", "--no-gpg-sign")
    monkeypatch.chdir(repo)
    return repo


@pytest.fixture
def outside_git(tmp_path, monkeypatch):
    """A directory no git repository contains, entered as the working directory."""
    monkeypatch.setenv("GIT_CEILING_DIRECTORIES", str(tmp_path))
    monkeypatch.delenv("GIT_DIR", raising=False)
    work = tmp_path / "plain"
    work.mkdir()
    monkeypatch.chdir(work)
    return work


def test_git_short_sha_returns_the_head_prefix(git_repo):
    sha = _git_short_sha()

    assert re.fullmatch(r"[0-9a-f]{7,}", sha)
    assert _git(git_repo, "rev-parse", "HEAD").startswith(sha)


def test_git_short_sha_is_none_outside_a_repository(outside_git):
    assert _git_short_sha() is None


def test_git_short_sha_is_none_when_git_is_not_installed(outside_git, monkeypatch):
    monkeypatch.setenv("PATH", str(outside_git))

    assert _git_short_sha() is None


def test_git_short_sha_is_none_when_git_times_out(monkeypatch):
    def _timeout(*_args, **_kwargs):
        raise subprocess.TimeoutExpired(cmd="git", timeout=5)

    monkeypatch.setattr(dataset_io.subprocess, "run", _timeout)

    assert _git_short_sha() is None


def test_compute_version_tag_returns_the_override_unchanged():
    assert compute_version_tag("my-tag") == "my-tag"


def test_compute_version_tag_is_the_date_and_the_head_sha(git_repo):
    tag = compute_version_tag()

    date, sha = tag.split("_")
    assert re.fullmatch(r"\d{8}", date)
    assert _git(git_repo, "rev-parse", "HEAD").startswith(sha)


def test_compute_version_tag_ends_in_nogit_outside_a_repository(outside_git):
    assert re.fullmatch(r"\d{8}_nogit", compute_version_tag())
