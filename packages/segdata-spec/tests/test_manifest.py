import json

import pytest
from pydantic import ValidationError

from segdata_spec.manifest import Manifest, dump_manifest, manifest_sha256

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
        ("2026-11-20-ab12cd3", False),
        ("20261120-ab12cd", False),
        ("20261120-AB12CD3", False),
        ("20261120-ab12cd3x", False),
        ("latest", False),
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
    "mutate",
    [
        lambda d: d.update(surprise=1),
        lambda d: d.update(kind="mesh"),
        lambda d: d["image_store"].update(owner="other"),
        lambda d: d["label_mapping"].update(source="guess"),
    ],
    ids=["unknown-field", "bad-kind", "bad-owner", "bad-mapping-source"],
)
def test_manifest_rejects_invalid_input(mutate):
    raw = _manifest_dict()
    mutate(raw)

    with pytest.raises(ValidationError):
        Manifest.model_validate_json(json.dumps(raw))


def test_manifest_sha256_hashes_the_given_bytes():
    assert (
        manifest_sha256(b"abc")
        == "ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad"
    )
