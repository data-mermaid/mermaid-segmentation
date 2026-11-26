"""``segdata promote --taxonomy``: a staged taxonomy uploaded write-once, tables then manifest."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pyarrow as pa
import pytest
from s3_fake import FakeS3

from segdata_etl import cli, promote
from segdata_spec import DATASET_BUCKET
from segdata_spec.errors import ChecksumMismatch, PublishRefused
from segdata_spec.git import GitState
from segdata_spec.manifest import SourceInfo
from segdata_spec.stage import stage_taxonomy

NAME = "mermaid-benthic"
VERSION = "20261125-abcdef1"
PREFIX = f"taxonomy/{NAME}/{VERSION}/"
TABLES = ("benthic_attributes.parquet", "growth_forms.parquet")
CLEAN_GIT = GitState(sha="abcdef1234567", dirty=False, pushed=True)


@pytest.fixture
def stage(tmp_path: Path) -> Path:
    repo = tmp_path / "repo"
    repo.mkdir()
    (repo / "uv.lock").write_text("lock")
    stage_taxonomy(
        tmp_path / "stage",
        bucket=DATASET_BUCKET,
        name=NAME,
        version=VERSION,
        tables={
            "benthic_attributes.parquet": pa.table(
                {"id": ["1", "2"], "name": ["Hard coral", "Acropora"], "parent_id": [None, "1"]}
            ),
            "growth_forms.parquet": pa.table({"id": ["9"], "name": ["Branching"]}),
        },
        etl_package="segdata-etl",
        etl_command="segdata taxonomy",
        source=SourceInfo(type="test", uri="https://api.test", revision="r1"),
        repo_dir=repo,
        git=CLEAN_GIT,
    )
    return tmp_path / "stage"


def _staged(stage: Path) -> Path:
    return stage / DATASET_BUCKET / "taxonomy" / NAME / VERSION


def _promote(client: FakeS3, stage: Path, version: str = VERSION) -> None:
    promote.promote_taxonomy(client, stage_dir=stage, name=NAME, version=version, workers=2)


def _writes(client: FakeS3) -> list[str]:
    return [key for op, _, key, _ in client.calls if op == "put"]


def test_tables_then_the_manifest_are_put_write_once(stage: Path) -> None:
    fake = FakeS3()

    _promote(fake, stage)

    assert sorted(_writes(fake)[:-1]) == [PREFIX + t for t in TABLES]
    assert _writes(fake)[-1] == PREFIX + "manifest.json"
    assert all(kw["IfNoneMatch"] == "*" for op, _, _, kw in fake.calls if op == "put")
    for name in (*TABLES, "manifest.json"):
        assert fake.objects[(DATASET_BUCKET, PREFIX + name)] == (_staged(stage) / name).read_bytes()
    assert fake.ops("copy") == []


def test_an_already_published_version_is_refused_with_no_writes(stage: Path) -> None:
    fake = FakeS3({(DATASET_BUCKET, PREFIX + "manifest.json"): b"{}"})

    with pytest.raises(PublishRefused, match="already published"):
        _promote(fake, stage)

    assert _writes(fake) == []


def test_a_staged_table_altered_after_staging_is_refused_with_no_writes(stage: Path) -> None:
    fake = FakeS3()
    (_staged(stage) / "growth_forms.parquet").write_bytes(b"altered")

    with pytest.raises(ChecksumMismatch, match="growth_forms.parquet"):
        _promote(fake, stage)

    assert fake.calls == []


def test_a_taxonomy_table_edited_after_preflight_is_uploaded_as_checked(
    stage: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    fake = FakeS3()
    table = _staged(stage) / "growth_forms.parquet"
    checked = table.read_bytes()
    real_preflight = promote._taxonomy_preflight

    def edit_after_preflight(*args: Any, **kwargs: Any) -> Any:
        plan = real_preflight(*args, **kwargs)
        table.write_bytes(b"edited after preflight")
        return plan

    monkeypatch.setattr(promote, "_taxonomy_preflight", edit_after_preflight)

    _promote(fake, stage)

    assert fake.objects[(DATASET_BUCKET, PREFIX + "growth_forms.parquet")] == checked


def test_a_staged_manifest_describing_another_version_is_refused(stage: Path) -> None:
    manifest = _staged(stage) / "manifest.json"
    manifest.write_text(manifest.read_text().replace(VERSION, "20261126-abcdef1"))

    with pytest.raises(PublishRefused, match="describes mermaid-benthic 20261126-abcdef1"):
        _promote(FakeS3(), stage)


def test_a_destination_table_holding_other_bytes_is_refused_before_any_write(stage: Path) -> None:
    fake = FakeS3({(DATASET_BUCKET, PREFIX + "growth_forms.parquet"): b"other"})

    with pytest.raises(PublishRefused, match="already hold other content"):
        _promote(fake, stage)

    assert _writes(fake) == []


def test_rerun_after_a_crash_between_the_tables_and_the_manifest_writes_the_manifest(
    stage: Path,
) -> None:
    staged = {name: (_staged(stage) / name).read_bytes() for name in TABLES}
    fake = FakeS3({(DATASET_BUCKET, PREFIX + name): data for name, data in staged.items()})

    _promote(fake, stage)

    assert (
        fake.objects[(DATASET_BUCKET, PREFIX + "manifest.json")]
        == (_staged(stage) / "manifest.json").read_bytes()
    )
    assert _writes(fake)[-1] == PREFIX + "manifest.json"
    for name, data in staged.items():
        assert fake.objects[(DATASET_BUCKET, PREFIX + name)] == data


def test_promote_taxonomy_flag_runs_the_taxonomy_path(
    stage: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    fake = FakeS3()
    monkeypatch.setattr(cli, "make_s3_client", lambda **_kw: fake)

    argv = ["promote", "--stage-dir", str(stage), "--taxonomy", NAME, "--version", VERSION]

    assert cli.main(argv) == 0
    assert (DATASET_BUCKET, PREFIX + "manifest.json") in fake.objects


def test_dataset_and_taxonomy_flags_are_mutually_exclusive() -> None:
    argv = ["promote", "--stage-dir", "s", "--dataset", "d", "--taxonomy", "t", "--version", "v"]

    with pytest.raises(SystemExit):
        cli.build_parser().parse_args(argv)


def test_one_of_dataset_or_taxonomy_is_required() -> None:
    with pytest.raises(SystemExit):
        cli.build_parser().parse_args(["promote", "--stage-dir", "s", "--version", "v"])
