"""Tests for the coralnet-raw pin loader."""

from __future__ import annotations

import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from coralnet_scraper import pin as pin_mod
from coralnet_scraper.pin import RawPin, load_raw_pin
from segdata_spec import DATASET_BUCKET
from segdata_spec.layout import cache_path, dataset_prefix
from segdata_spec.manifest import Pin


def _images(rows: list[tuple[str, str]]) -> pa.Table:
    return pa.table(
        {
            "image_id": [i for i, _ in rows],
            "group": [g for _, g in rows],
            "key": [f"images/{i}.jpg" for i, _ in rows],
        },
        schema=pa.schema([("image_id", pa.string()), ("group", pa.string()), ("key", pa.string())]),
    )


def _patch_tables(monkeypatch, rows: list[tuple[str, str]]) -> dict:
    """Replace the sync call with one that writes the cached images table; return its args."""
    seen: dict = {}

    def fake_sync(client, pins, cache_dir, *, bucket):
        seen["sync"] = (client, dict(pins), cache_dir, bucket)
        (pin,) = pins.values()
        key = f"{dataset_prefix(pin.name, pin.version)}images.parquet"
        path = cache_path(cache_dir, bucket, key)
        path.parent.mkdir(parents=True, exist_ok=True)
        pq.write_table(_images(rows), path)

    monkeypatch.setattr(pin_mod, "sync_tables", fake_sync)
    return seen


def test_the_image_ids_are_grouped_by_source(monkeypatch, tmp_path):
    seen = _patch_tables(monkeypatch, [("10", "s1"), ("11", "s1"), ("20", "s2")])
    client = object()

    raw = load_raw_pin(client, "20261124-abc1234", "deadbeef", tmp_path)

    assert raw == RawPin(
        "20261124-abc1234", "deadbeef", {1: frozenset({10, 11}), 2: frozenset({20})}
    )
    expected = Pin(name="coralnet-raw", version="20261124-abc1234", manifest_sha256="deadbeef")
    assert seen["sync"] == (client, {"coralnet-raw": expected}, tmp_path, DATASET_BUCKET)


def test_ids_for_a_source_the_pin_lacks_is_empty(monkeypatch, tmp_path):
    _patch_tables(monkeypatch, [("10", "s1")])

    raw = load_raw_pin(object(), "20261124-abc1234", "deadbeef", tmp_path)

    assert raw.ids_for(1) == frozenset({10})
    assert raw.ids_for(3) == frozenset()


@pytest.mark.parametrize(
    ("row", "bad"),
    [(("10", "x1"), "x1"), (("abc", "s1"), "abc")],
    ids=["group", "image-id"],
)
def test_an_unparseable_group_or_image_id_raises_naming_it(monkeypatch, tmp_path, row, bad):
    _patch_tables(monkeypatch, [row])

    with pytest.raises(ValueError, match=bad):
        load_raw_pin(object(), "20261124-abc1234", "deadbeef", tmp_path)
