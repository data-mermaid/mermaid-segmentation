import json

import pytest
from pydantic import ValidationError

from segdata_spec.manifest import Manifest, Pin, dump_manifest

SHA = "a" * 64


def _manifest_dict() -> dict:
    return {
        "schema_version": 1,
        "dataset": "coralnet",
        "version": "20261120-ab12cd3",
        "kind": "points",
        "created_at": "2026-11-20T10:00:00Z",
        "image_store": {"bucket": "seg-bucket", "prefix": "images/coralnet/", "owner": "segdata"},
        "etl": {
            "package": "segdata-etl",
            "command": "segdata build coralnet",
            "git_sha": "ab12cd3",
            "git_dirty": False,
            "uv_lock_sha256": SHA,
        },
        "source": {"type": "coralnet-scrape", "uri": "s3://src/", "revision": "run-1"},
        "parents": [
            {"dataset": "coralnet-raw", "version": "20261101-1234567", "manifest_sha256": SHA}
        ],
        "label_mapping": {
            "source": "cleaned_up_rerun_json",
            "sha256": SHA,
            "n_labels": 10,
            "n_mapped": 4,
            "git_ref": "e0d3876",
            "path": "configs/mapping.json",
            "blob": "f" * 40,
        },
        "files": {"images.parquet": {"sha256": SHA, "bytes": 123, "rows": 2}},
        "stats": {
            "n_images": 2,
            "n_points": 5,
            "n_groups": 1,
            "image_bytes": 999,
            "dropped": {"label_unnamed": 3},
        },
        "diff_from_parent": {
            "images_added": 1,
            "images_removed": 0,
            "images_changed": 0,
            "points_added": 5,
            "points_removed": 0,
        },
    }


def test_dump_round_trips_and_is_byte_stable():
    manifest = Manifest.model_validate(_manifest_dict())

    data = dump_manifest(manifest)

    assert Manifest.model_validate_json(data) == manifest
    assert dump_manifest(Manifest.model_validate_json(data)) == data
    assert data.endswith(b"\n")


@pytest.mark.parametrize(
    ("version", "valid"),
    [
        ("20261120-ab12cd3", True),
        ("20261120-ab12cd3-2", True),
        ("20261120-ab12cd", False),
        ("20261120-AB12CD3", False),
    ],
)
def test_manifest_version_format(version, valid):
    raw = {**_manifest_dict(), "version": version}

    if valid:
        assert Manifest.model_validate(raw).version == version
    else:
        with pytest.raises(ValidationError):
            Manifest.model_validate(raw)


@pytest.mark.parametrize(
    ("prefix", "valid"),
    [("images/ds/", True), ("", True), ("images/ds", False)],
)
def test_image_store_prefix_is_empty_or_ends_with_a_slash(prefix, valid):
    raw = _manifest_dict()
    raw["image_store"]["prefix"] = prefix

    if valid:
        assert Manifest.model_validate(raw).image_store.prefix == prefix
    else:
        with pytest.raises(ValidationError, match="prefix"):
            Manifest.model_validate(raw)


def test_a_manifest_without_a_taxonomy_pin_loads_and_dumps_without_the_key():
    manifest = Manifest.model_validate(_manifest_dict())

    assert manifest.taxonomy is None
    assert "taxonomy" not in json.loads(dump_manifest(manifest))


def test_a_taxonomy_pin_round_trips_in_the_manifest():
    pin = {"name": "mermaid-benthic", "version": "20261124-e6abfad", "manifest_sha256": SHA}
    manifest = Manifest.model_validate({**_manifest_dict(), "taxonomy": pin})

    data = dump_manifest(manifest)

    assert json.loads(data)["taxonomy"] == pin
    assert Manifest.model_validate_json(data).taxonomy == Pin.model_validate(pin)
    assert Manifest.model_validate_json(data) == manifest
