"""next_version and list_versions produce and find ``YYYYMMDD-<sha7>[-N]`` versions."""

from __future__ import annotations

from datetime import date
from pathlib import Path

import pytest

from segdata_etl.common.local_store import LocalStore
from segdata_etl.common.version import list_versions, next_version

TODAY = date(2026, 11, 23)


def test_first_version_of_the_day_is_date_and_seven_char_sha() -> None:
    assert next_version([], git_sha="abc1234def5678", today=TODAY) == "20261123-abc1234"


@pytest.mark.parametrize(
    ("existing", "expected"),
    [
        (["20261123-abc1234"], "20261123-abc1234-2"),
        (["20261123-abc1234", "20261123-abc1234-2"], "20261123-abc1234-3"),
        (["20261123-abc1234", "20261123-abc1234-3"], "20261123-abc1234-2"),
    ],
)
def test_taken_version_gets_the_lowest_free_numeric_suffix(
    existing: list[str], expected: str
) -> None:
    assert next_version(existing, git_sha="abc1234", today=TODAY) == expected


def test_versions_of_another_day_or_sha_do_not_add_a_suffix() -> None:
    existing = ["20261122-abc1234", "20261123-1111111", "20261123-1111111-2"]

    assert next_version(existing, git_sha="abc1234", today=TODAY) == "20261123-abc1234"


def test_list_versions_returns_immediate_version_folders_only(tmp_path: Path) -> None:
    store = LocalStore(tmp_path)
    for key in [
        "datasets/ds/20261123-abc1234/manifest.json",
        "datasets/ds/20261123-abc1234-2/manifest.json",
        "datasets/ds/scratch/notes.txt",
        "datasets/ds/20261123-abc1234/images/x.parquet",
        "datasets/other/20261101-1111111/manifest.json",
    ]:
        store.put_object(Bucket="b", Key=key, Body=b"x")

    assert list_versions(store, "b", "datasets/ds/") == [
        "20261123-abc1234",
        "20261123-abc1234-2",
    ]
