from __future__ import annotations

from datetime import UTC, date, datetime
from pathlib import Path

from s3_fake import FakeS3

from segdata_etl.common import stage
from segdata_etl.common.stage import next_version


def test_next_version_defaults_to_the_utc_date(tmp_path: Path, monkeypatch):
    class _Clock(datetime):
        @classmethod
        def now(cls, tz=None):
            # A naive now() is host-local 23:30, the previous day in UTC; a UTC now() is the next day.
            return (
                datetime(2026, 11, 24, 4, 30, tzinfo=UTC)
                if tz is not None
                else datetime(2026, 11, 23, 23, 30)
            )

    monkeypatch.setattr(stage, "datetime", _Clock)
    assert next_version(tmp_path, "bkt", "ds", git_sha="abc1234def5678") == "20261124-abc1234"


def test_next_version_appends_first_free_counter(tmp_path: Path):
    kwargs = {"today": date(2026, 11, 23), "git_sha": "abc1234def5678"}
    assert next_version(tmp_path, "bkt", "ds", **kwargs) == "20261123-abc1234"
    base = tmp_path / "bkt" / "datasets" / "ds"
    (base / "20261123-abc1234").mkdir(parents=True)
    assert next_version(tmp_path, "bkt", "ds", **kwargs) == "20261123-abc1234-2"
    (base / "20261123-abc1234-2").mkdir()
    assert next_version(tmp_path, "bkt", "ds", **kwargs) == "20261123-abc1234-3"


def test_next_version_counts_a_version_published_in_the_bucket_as_taken(tmp_path: Path):
    client = FakeS3()
    kwargs = {"today": date(2026, 11, 23), "git_sha": "abc1234def5678", "client": client}
    client.put("bkt", "datasets/ds/20261123-abc1234/manifest.json", b"{}")
    assert next_version(tmp_path, "bkt", "ds", **kwargs) == "20261123-abc1234-2"
    client.put("bkt", "datasets/ds/20261123-abc1234-2/manifest.json", b"{}")
    assert next_version(tmp_path, "bkt", "ds", **kwargs) == "20261123-abc1234-3"


def test_next_version_with_a_client_still_honours_the_staged_dir(tmp_path: Path):
    kwargs = {"today": date(2026, 11, 23), "git_sha": "abc1234def5678", "client": FakeS3()}
    (tmp_path / "bkt" / "datasets" / "ds" / "20261123-abc1234").mkdir(parents=True)
    assert next_version(tmp_path, "bkt", "ds", **kwargs) == "20261123-abc1234-2"
