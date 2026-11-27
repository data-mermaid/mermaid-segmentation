"""``fetch_taxonomy`` and ``run_taxonomy``, against a fake HTTP session and a fake S3."""

from __future__ import annotations

import argparse
import hashlib
from pathlib import Path
from typing import Any

import pytest
import requests
from s3_fake import FakeS3

from segdata_etl import cli, taxonomy
from segdata_spec import DATASET_BUCKET
from segdata_spec.errors import PublishRefused
from segdata_spec.git import GitState
from segdata_spec.manifest import Pin
from segdata_spec.read import open_taxonomy

BASE = "https://api.test/v1"
CLEAN_GIT = GitState(sha="abcdef1234567", dirty=False, pushed=True)


class _Response:
    def __init__(self, payload: Any, status: int = 200):
        self._payload = payload
        self.status_code = status

    def raise_for_status(self) -> None:
        if self.status_code >= 400:
            raise requests.HTTPError(f"{self.status_code}")

    def json(self) -> Any:
        return self._payload


class FakeSession:
    """Serves ``routes`` by URL and records each call with its keyword arguments."""

    def __init__(self, routes: dict[str, _Response]):
        self.routes = routes
        self.calls: list[tuple[str, dict[str, Any]]] = []

    def get(self, url: str, **kwargs: Any) -> _Response:
        self.calls.append((url, kwargs))
        return self.routes[url]


def _routes() -> dict[str, _Response]:
    page2 = f"{BASE}/benthicattributes/?page=2"
    return {
        f"{BASE}/benthicattributes/": _Response(
            {
                "count": 2,
                "next": page2,
                "results": [{"id": "b2", "name": "Acropora", "parent": "a1"}],
            }
        ),
        page2: _Response(
            {
                "count": 2,
                "next": None,
                "results": [{"id": "a1", "name": "Hard coral", "parent": None}],
            }
        ),
        f"{BASE}/choices/": _Response(
            [
                {"name": "other", "data": [{"id": "zz", "name": "ignored"}]},
                {
                    "name": "growthforms",
                    "data": [
                        {"id": "g2", "name": "Massive", "extra": 1},
                        {"id": "g1", "name": "Branching"},
                    ],
                },
            ]
        ),
    }


def test_fetch_follows_next_and_maps_parent_and_growth_forms() -> None:
    session = FakeSession(_routes())
    tables = taxonomy.fetch_taxonomy(session, BASE)  # type: ignore[arg-type]
    assert list(tables) == ["benthic_attributes.parquet", "growth_forms.parquet"]
    assert tables["benthic_attributes.parquet"].to_pylist() == [
        {"id": "a1", "name": "Hard coral", "parent_id": None},
        {"id": "b2", "name": "Acropora", "parent_id": "a1"},
    ]
    assert tables["growth_forms.parquet"].to_pylist() == [
        {"id": "g1", "name": "Branching"},
        {"id": "g2", "name": "Massive"},
    ]
    assert all(kwargs.get("timeout") for _, kwargs in session.calls)


def test_fetch_raises_on_a_non_2xx_response() -> None:
    routes = _routes()
    routes[f"{BASE}/choices/"] = _Response({}, status=503)
    with pytest.raises(requests.HTTPError):
        taxonomy.fetch_taxonomy(FakeSession(routes), BASE)  # type: ignore[arg-type]


def test_fetch_raises_when_choices_has_no_growthforms_entry() -> None:
    routes = _routes()
    routes[f"{BASE}/choices/"] = _Response([{"name": "other", "data": []}])
    with pytest.raises(ValueError, match="growthforms"):
        taxonomy.fetch_taxonomy(FakeSession(routes), BASE)  # type: ignore[arg-type]


def _bad_attribute_routes(case: str) -> dict[str, _Response]:
    routes = _routes()
    first = f"{BASE}/benthicattributes/"
    page2 = f"{first}?page=2"
    if case == "empty page":
        routes[first] = _Response({"count": 2, "next": None, "results": []})
    elif case == "unknown parent":
        routes[page2] = _Response(
            {"count": 2, "next": None, "results": [{"id": "a1", "name": "Hard", "parent": "zz"}]}
        )
    elif case == "next repeats":
        routes[first] = _Response(
            {"count": 9, "next": first, "results": [{"id": "b2", "name": "A", "parent": None}]}
        )
    elif case == "more rows than count":
        routes[first] = _Response(
            {"count": 1, "next": page2, "results": [{"id": "b2", "name": "A", "parent": None}]}
        )
    elif case == "fewer rows than count":
        routes[page2] = _Response({"count": 2, "next": None, "results": []})
    return routes


@pytest.mark.parametrize(
    ("case", "exc", "match"),
    [
        ("empty page", ValueError, "0 rows fetched"),
        ("unknown parent", ValueError, "unknown parents"),
        ("next repeats", ValueError, "repeats"),
        ("more rows than count", ValueError, "2 rows fetched, API count is 1"),
        ("fewer rows than count", ValueError, "1 rows fetched, API count is 2"),
    ],
)
def test_fetch_refuses_a_partial_duplicated_or_flattened_fetch(
    case: str, exc: type[Exception], match: str
) -> None:
    session = FakeSession(_bad_attribute_routes(case))
    with pytest.raises(exc, match=match):
        taxonomy.fetch_taxonomy(session, BASE)  # type: ignore[arg-type]


def test_fetch_refuses_duplicate_growth_form_ids() -> None:
    routes = _routes()
    routes[f"{BASE}/choices/"] = _Response(
        [
            {
                "name": "growthforms",
                "data": [{"id": "g1", "name": "Massive"}, {"id": "g1", "name": "Branching"}],
            }
        ]
    )
    with pytest.raises(ValueError, match="growth forms hold duplicate ids"):
        taxonomy.fetch_taxonomy(FakeSession(routes), BASE)  # type: ignore[arg-type]


def test_fetch_refuses_an_empty_growthforms_entry() -> None:
    routes = _routes()
    routes[f"{BASE}/choices/"] = _Response([{"name": "growthforms", "data": []}])
    with pytest.raises(ValueError, match="growthforms.*empty"):
        taxonomy.fetch_taxonomy(FakeSession(routes), BASE)  # type: ignore[arg-type]


@pytest.fixture
def repo(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """A fake repo root, returned by ``find_repo_root``, with a clean pushed git state."""
    root = tmp_path / "repo"
    root.mkdir()
    (root / "uv.lock").write_text("lock")
    monkeypatch.setattr("segdata_etl.common.stage.find_repo_root", lambda _start: root)
    monkeypatch.setattr("segdata_etl.common.stage.probe_git", lambda _dir: CLEAN_GIT)
    return root


def _args(**overrides: Any) -> argparse.Namespace:
    values = {
        "stage_dir": None,
        "base_url": BASE,
    }
    return argparse.Namespace(**{**values, **overrides})


def _run(
    args: argparse.Namespace,
    session: Any,
    capsys: pytest.CaptureFixture[str],
    s3: FakeS3 | None = None,
) -> str:
    assert taxonomy.run_taxonomy(args, source_client=s3 or FakeS3(), session=session) == 0
    return capsys.readouterr().out


def _staged_dir(stage: Path) -> Path:
    (version_dir,) = (stage / DATASET_BUCKET / "taxonomy" / "mermaid-benthic").iterdir()
    return version_dir


def _stage(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> Path:
    _run(_args(stage_dir=tmp_path / "staged"), FakeSession(_routes()), capsys)
    return _staged_dir(tmp_path / "staged")


def test_stage_writes_a_readable_version_under_the_stage_dir_and_nothing_to_s3(
    repo: Path, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    stage, s3 = tmp_path / "stage", FakeS3()
    out = _run(_args(stage_dir=stage), FakeSession(_routes()), capsys, s3)
    version_dir = _staged_dir(stage)
    assert version_dir.name.endswith("-abcdef1")
    sha = hashlib.sha256((version_dir / "manifest.json").read_bytes()).hexdigest()
    assert f"version: {version_dir.name}\nmanifest_sha256: {sha}" in out
    assert s3.puts == []
    pin = Pin(name="mermaid-benthic", version=version_dir.name, manifest_sha256=sha)
    tax = open_taxonomy(stage, DATASET_BUCKET, pin)
    assert tax.benthic_attributes.num_rows == 2
    assert tax.manifest.source.uri == BASE
    assert tax.manifest.etl.command == f"segdata taxonomy --stage-dir {stage} --base-url {BASE}"


def test_taxonomy_subcommand_stages_the_taxonomy_through_cli_main(
    repo: Path,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    monkeypatch.setattr(cli, "make_s3_client", lambda **_kw: FakeS3())
    monkeypatch.setattr(taxonomy.requests, "Session", lambda: FakeSession(_routes()))
    stage = tmp_path / "stage"

    assert cli.main(["taxonomy", "--stage-dir", str(stage), "--base-url", BASE]) == 0

    out = capsys.readouterr().out
    assert f"version: {_staged_dir(stage).name}\n" in out
    assert (_staged_dir(stage) / "manifest.json").is_file()


@pytest.mark.parametrize("state", [GitState("abcdef1", True, True)])
def test_a_dirty_or_unpushed_tree_is_refused_before_any_fetch(
    repo: Path, monkeypatch: pytest.MonkeyPatch, tmp_path: Path, state: GitState
) -> None:
    monkeypatch.setattr("segdata_etl.common.stage.probe_git", lambda _dir: state)
    session = FakeSession({})
    with pytest.raises(PublishRefused):
        taxonomy.run_taxonomy(
            _args(stage_dir=tmp_path / "stage"),
            source_client=FakeS3(),
            session=session,  # type: ignore[arg-type]
        )
    assert session.calls == []
