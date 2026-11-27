"""check_cache: the preflight that a synced cache holds what a data config pins."""

from __future__ import annotations

import shutil
from pathlib import Path

import pyarrow as pa
import pytest
import yaml

import scripts.check_cache as script
from segdata_spec._marker import SyncMarker, dump_marker
from segdata_spec.layout import resize_id, synced_marker_dir
from segdata_spec.manifest import Pin
from tests.mermaidseg.data.conftest import (
    BUCKET,
    IMAGE_BUCKET,
    IMAGE_PREFIX,
    VERSION,
    ImageSpec,
    write_taxonomy,
    write_version,
)


def _image_file(cache_dir: Path, name: str, group: str) -> Path:
    return cache_dir / IMAGE_BUCKET / IMAGE_PREFIX / f"{name}_{group}1.jpg"


def _write_dataset(tmp_path: Path, name: str) -> tuple[Path, Pin]:
    # Image ids carry the dataset name so two datasets share a cache without sharing files.
    images = [ImageSpec(f"{name}_a1", group="a"), ImageSpec(f"{name}_b1", group="b")]
    points = [(f"{name}_a1", 0, 0, 1), (f"{name}_b1", 0, 0, 1)]
    cache_dir, _, pin = write_version(
        tmp_path, kind="points", images=images, labels={1: "Sand"}, points=points, name=name
    )
    return cache_dir, pin


def _entry(pin: Pin, train: dict, sha: str | None = None) -> dict:
    return {
        "version": pin.version,
        "manifest_sha256": sha or pin.manifest_sha256,
        "train": train,
    }


def _cached_taxonomy(cache_dir: Path, tmp_path: Path) -> dict:
    """Write a taxonomy into the cache and return its pin as a config block."""
    table = pa.table({"id": ["1"], "name": ["Hard coral"], "parent_id": [None]})
    tax = write_taxonomy(cache_dir, tmp_path, table)
    return {"version": tax.version, "manifest_sha256": tax.manifest_sha256}


def _write_config(tmp_path: Path, entries: dict[str, dict], taxonomy: dict) -> Path:
    split_file = tmp_path / "split.csv"
    split_file.write_text("group,split\na,train\nb,val\n")
    entries = {n: {**e, "split_file": str(split_file)} for n, e in entries.items()}
    section = {"taxonomy": taxonomy, "datasets": entries}
    path = tmp_path / "data_config.yaml"
    path.write_text(yaml.safe_dump({"data": section}))
    return path


def _run(config: Path, cache_dir: Path, capsys) -> tuple[int, list[str]]:
    code = script.main(["--data-config", str(config), "--cache-dir", str(cache_dir)])
    return code, capsys.readouterr().out.splitlines()


def _two_datasets(tmp_path: Path) -> tuple[Path, dict[str, Pin]]:
    cache_dir, pin_x = _write_dataset(tmp_path, "ds_x")
    _, pin_y = _write_dataset(tmp_path, "ds_y")
    return cache_dir, {"ds_x": pin_x, "ds_y": pin_y}


def _config_for(
    tmp_path: Path, cache_dir: Path, pins: dict[str, Pin], wrong_sha: frozenset[str] = frozenset()
) -> Path:
    entries = {
        name: _entry(pin, {"groups": "all"}, "0" * 64 if name in wrong_sha else None)
        for name, pin in pins.items()
    }
    return _write_config(tmp_path, entries, _cached_taxonomy(cache_dir, tmp_path))


def _marker_file(cache_dir: Path, pin: Pin) -> Path:
    directory = synced_marker_dir(cache_dir, resize_id(None), BUCKET, pin.name, pin.version)
    return directory / "marker.json"


def test_synced_datasets_report_one_ok_line_each_and_exit_zero(tmp_path, capsys):
    cache_dir, pins = _two_datasets(tmp_path)
    config = _config_for(tmp_path, cache_dir, pins)

    code, lines = _run(config, cache_dir, capsys)

    assert code == 0
    assert lines == [
        f"ds_x: version {VERSION}, resize original, ok",
        f"ds_y: version {VERSION}, resize original, ok",
        f"taxonomy: version {VERSION}, ok",
    ]


def test_pinned_taxonomy_absent_from_the_cache_fails(tmp_path, capsys):
    cache_dir, pin = _write_dataset(tmp_path, "ds_x")
    config = _write_config(
        tmp_path,
        {"ds_x": _entry(pin, {"groups": "all"})},
        taxonomy={"version": "20260101-abc1234", "manifest_sha256": "0" * 64},
    )

    code, lines = _run(config, cache_dir, capsys)

    assert code == 1
    assert lines[0] == f"ds_x: version {VERSION}, resize original, ok"
    assert lines[1].startswith("taxonomy: ")
    assert "scripts/sync_data.py" in lines[1]


@pytest.mark.parametrize(
    ("synced", "train", "ok", "named"),
    [
        ({"a", "b"}, {"use": ["train"]}, True, None),
        ({"a"}, {"use": ["train", "val"]}, False, "b"),
        ({"a", "b"}, {"groups": "all"}, True, None),
        ({"a"}, {"groups": "all"}, False, "b"),
        ({"a", "b"}, {"groups": ["zzz"]}, False, "zzz"),
    ],
    ids=[
        "subset-selected",
        "missing-group",
        "all-selected-and-every-group-synced",
        "all-selected-but-partial-sync",
        "selected-group-not-in-version",
    ],
)
def test_selected_groups_must_exist_and_be_synced(tmp_path, capsys, synced, train, ok, named):
    cache_dir, pin = _write_dataset(tmp_path, "ds_x")
    _marker_file(cache_dir, pin).write_bytes(dump_marker(SyncMarker(None, frozenset(synced))))
    config = _write_config(
        tmp_path, {"ds_x": _entry(pin, train)}, _cached_taxonomy(cache_dir, tmp_path)
    )

    code, lines = _run(config, cache_dir, capsys)

    assert code == (0 if ok else 1)
    if ok:
        assert lines == [
            f"ds_x: version {VERSION}, resize original, ok",
            f"taxonomy: version {VERSION}, ok",
        ]
    else:
        assert not lines[0].endswith(", ok")
        assert lines[0].startswith("ds_x: ")
        assert "scripts/sync_data.py" in lines[0]
        assert f"['{named}']" in lines[0]


def _break_marker_dir(cache_dir: Path, pin: Pin) -> bool:
    shutil.rmtree(_marker_file(cache_dir, pin).parent)
    return False


def _break_pin(cache_dir: Path, pin: Pin) -> bool:
    return True


def _break_table(cache_dir: Path, pin: Pin) -> bool:
    (cache_dir / BUCKET / "datasets" / pin.name / pin.version / "images.parquet").unlink()
    return False


def _break_image_file(cache_dir: Path, pin: Pin) -> bool:
    _image_file(cache_dir, pin.name, "a").unlink()
    return False


def _break_marker_bytes(cache_dir: Path, pin: Pin) -> bool:
    _marker_file(cache_dir, pin).write_bytes(b"\x00junk")
    return False


@pytest.mark.parametrize(
    "breakage",
    [_break_marker_dir, _break_pin, _break_table, _break_image_file, _break_marker_bytes],
    ids=["not-synced", "pin-mismatch", "missing-table", "missing-image-file", "corrupt-marker"],
)
def test_a_broken_dataset_fails_while_the_good_one_is_still_checked(tmp_path, capsys, breakage):
    cache_dir, pins = _two_datasets(tmp_path)
    wrong_sha = breakage(cache_dir, pins["ds_x"])
    config = _config_for(
        tmp_path, cache_dir, pins, frozenset({"ds_x"}) if wrong_sha else frozenset()
    )

    code, lines = _run(config, cache_dir, capsys)

    assert code == 1
    assert lines[0].startswith("ds_x: ")
    assert not lines[0].endswith(", ok")
    assert "scripts/sync_data.py" in lines[0]
    assert lines[1] == f"ds_y: version {VERSION}, resize original, ok"


def test_an_error_outside_the_known_failures_propagates(tmp_path, capsys, monkeypatch):
    cache_dir, pin = _write_dataset(tmp_path, "ds_x")
    config = _write_config(
        tmp_path, {"ds_x": _entry(pin, {"groups": "all"})}, _cached_taxonomy(cache_dir, tmp_path)
    )

    def boom(*args, **kwargs):
        raise RuntimeError("unexpected")

    monkeypatch.setattr(script, "build_datasets", boom)

    with pytest.raises(RuntimeError):
        _run(config, cache_dir, capsys)


def test_an_invalid_data_config_raises(tmp_path, capsys):
    cache_dir, pin = _write_dataset(tmp_path, "ds_x")
    config = _write_config(
        tmp_path, {"ds_x": _entry(pin, {"groups": "all"})}, _cached_taxonomy(cache_dir, tmp_path)
    )
    doc = yaml.safe_load(config.read_text())
    doc["data"]["unknown_key"] = 1
    config.write_text(yaml.safe_dump(doc))

    with pytest.raises(ValueError, match="unknown_key"):
        _run(config, cache_dir, capsys)
