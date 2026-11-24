"""Shared helpers of the external dataset version builders."""

from __future__ import annotations

from pathlib import Path

import pyarrow as pa
import pytest
from s3_fake import FakeS3

from segdata_etl.common.legacy import LegacyObject
from segdata_etl.datasets import external
from segdata_etl.label_maps.curated import CURATED_SCHEMA, CuratedSource
from segdata_spec import DATASET_BUCKET
from segdata_spec.errors import PublishRefused

from .conftest import TAXONOMY_VERSION, publish_fixture_taxonomy

ATTRIBUTES = pa.table(
    {
        "id": ["ba-hc", "ba-acro"],
        "name": ["Hard coral", "Acropora"],
        "parent_id": [None, "ba-hc"],
    }
)
CLASSES = {"unlabeled": 0, "Acropora digitate": 5, "Water": 7, "Mystery": 9}
CURATED = pa.table(
    {
        "label_id": [None, None],
        "name": ["acropora digitate", "water"],
        "mermaid_benthic_attribute": ["acropora", "background"],
    },
    schema=CURATED_SCHEMA,
)


def _row(label_id: int, name: str, ba_id: str | None, ba: str | None) -> dict[str, object]:
    return {
        "label_id": label_id,
        "name": name,
        "mermaid_benthic_attribute_id": ba_id,
        "mermaid_benthic_attribute": ba,
        "mermaid_growth_form": None,
        "mapping_source": "curated",
    }


def test_build_labels_resolves_targets_case_insensitively_and_drops_unlabeled() -> None:
    labels = external.build_labels(CLASSES, CURATED, ATTRIBUTES)
    assert labels.to_pylist() == [
        _row(5, "Acropora digitate", "ba-acro", "Acropora"),
        _row(7, "Water", None, "background"),
        _row(9, "Mystery", None, None),
    ]


def test_build_labels_refuses_a_target_matching_two_taxonomy_names() -> None:
    attributes = pa.table(
        {"id": ["a", "b"], "name": ["Acropora", "ACROPORA"], "parent_id": [None, None]}
    )
    with pytest.raises(ValueError, match="acropora"):
        external.build_labels({"Acropora digitate": 5}, CURATED, attributes)


def test_label_mapping_record_counts_mapped_and_unresolved_labels() -> None:
    labels = external.build_labels(CLASSES, CURATED, ATTRIBUTES)
    info = CuratedSource(path="curated/x.csv", sha256="f" * 64)
    record = external.label_mapping_record(labels, info, git_sha="abc123")
    assert (record.n_labels, record.n_mapped, record.n_unresolved) == (3, 2, 1)
    assert (record.source, record.sha256, record.path, record.git_ref) == (
        "curated",
        "f" * 64,
        "curated/x.csv",
        "abc123",
    )


def _points(*rows: tuple[str, int, int, int]) -> pa.Table:
    return pa.table(
        {
            "image_id": [r[0] for r in rows],
            "row": [r[1] for r in rows],
            "col": [r[2] for r in rows],
            "label_id": [r[3] for r in rows],
            "annotator": ["default"] * len(rows),
        }
    )


def _objects(*keys: str) -> dict[str, LegacyObject]:
    return {key: LegacyObject(key, f"etag-{i}", 10) for i, key in enumerate(keys)}


@pytest.mark.parametrize(
    ("key", "identity"),
    [("images/PAC_USA/2001.JPG", ("2001", "PAC_USA")), ("classes.json", None)],
    ids=["image", "not-under-images"],
)
def test_group_folder_identity_reads_stem_and_group(
    key: str, identity: tuple[str, str] | None
) -> None:
    assert external.group_folder_identity(key) == identity


@pytest.mark.parametrize("key", ["images/1001.jpg", "images/ATL/sub/1001.jpg"])
def test_group_folder_identity_refuses_an_image_not_one_folder_deep(key: str) -> None:
    with pytest.raises(ValueError, match=key):
        external.group_folder_identity(key)


def test_legacy_images_skips_keys_identify_returns_none_for() -> None:
    def identify(key: str) -> tuple[str, str] | None:
        return None if key.startswith("skip/") else (key.upper(), "g")

    images = external.legacy_images(_objects("skip/x", "b", "a"), identify)
    assert images == [
        external.LegacyImage(image_id="A", group="g", key="a", etag="etag-2"),
        external.LegacyImage(image_id="B", group="g", key="b", etag="etag-1"),
    ]


def test_legacy_images_refuses_a_duplicate_image_id_naming_both_keys() -> None:
    with pytest.raises(ValueError, match="images/ATL/1001.jpg.*images/PAC_USA/1001.jpg"):
        external.legacy_images(
            _objects("images/ATL/1001.jpg", "images/PAC_USA/1001.jpg"),
            external.group_folder_identity,
        )


def test_image_key_takes_png_extension_from_media_type() -> None:
    image = external.LegacyImage(image_id="7", group="g", key="x/7.png", etag="e")
    assert external.image_key(image, "image/png") == "g/7.png"


SIZES = {"7.jpg": ("image/jpeg", 4, 3)}
IMAGE_7 = external.LegacyImage(image_id="7", group="g", key="7.jpg", etag="e")


def test_drop_points_outside_keeps_the_last_pixel_and_drops_each_edge_past_it() -> None:
    points = _points(
        ("7", 2, 3, 1), ("7", 3, 0, 2), ("7", 0, 4, 3), ("7", -1, 0, 4), ("7", 0, -1, 5)
    )
    kept, dropped = external.drop_points_outside(points, [IMAGE_7], SIZES)
    assert dropped == 4
    assert kept.column("label_id").to_pylist() == [1]


def test_drop_points_outside_keeps_in_bounds_points_in_order_and_counts_the_rest() -> None:
    points = _points(("7", 2, 3, 1), ("7", 1, 4, 2), ("7", 0, 0, 3), ("7", 3, 1, 4), ("7", 1, 1, 5))
    kept, dropped = external.drop_points_outside(points, [IMAGE_7], SIZES)
    assert dropped == 2
    assert kept.column("label_id").to_pylist() == [1, 3, 5]
    assert kept.schema == points.schema


def test_drop_points_outside_refuses_a_point_whose_image_has_no_size() -> None:
    with pytest.raises(ValueError, match=r"no sniffed size.*'8'"):
        external.drop_points_outside(_points(("7", 0, 0, 1), ("8", 0, 0, 1)), [IMAGE_7], SIZES)


def test_drop_points_outside_and_record_merges_the_count_only_when_points_are_dropped() -> None:
    inside = _points(("7", 0, 0, 1))
    assert external.drop_points_outside_and_record(inside, [IMAGE_7], SIZES, {"a": 1}) == (
        inside,
        {"a": 1},
    )
    kept, dropped = external.drop_points_outside_and_record(
        _points(("7", 0, 0, 1), ("7", 3, 0, 2)), [IMAGE_7], SIZES, {"a": 1}
    )
    assert kept.num_rows == 1
    assert dropped == {"a": 1, "points_out_of_bounds": 1}


def test_require_known_labels_refuses_a_label_id_missing_from_labels() -> None:
    labels = external.build_labels(CLASSES, CURATED, ATTRIBUTES)
    external.require_known_labels(_points(("7", 0, 0, 5), ("7", 0, 0, 9)), labels)
    with pytest.raises(ValueError, match=r"2 points.*\[0\]"):
        external.require_known_labels(_points(("7", 0, 0, 0), ("7", 0, 0, 0)), labels)


@pytest.mark.parametrize(
    ("found", "expected", "missing"),
    [({"a"}, {"a", "b"}, "b"), ({"a", "c"}, {"a"}, "c")],
    ids=["annotated-without-image", "image-without-annotations"],
)
def test_require_same_ids_names_the_unmatched_id(
    found: set[str], expected: set[str], missing: str
) -> None:
    external.require_same_ids({"a"}, {"a"}, what="images")
    with pytest.raises(ValueError, match=missing):
        external.require_same_ids(found, expected, what="images")


def _published_taxonomy(tmp_path: Path) -> tuple[FakeS3, str]:
    repo = tmp_path / "repo"
    repo.mkdir()
    (repo / "uv.lock").write_text("lock")
    s3 = FakeS3()
    return s3, publish_fixture_taxonomy(s3, repo)


def test_load_taxonomy_refuses_a_pin_that_is_cached_but_not_published(tmp_path: Path) -> None:
    s3, sha = _published_taxonomy(tmp_path)
    cache = tmp_path / "cache"
    external.load_taxonomy(
        s3, DATASET_BUCKET, version=TAXONOMY_VERSION, manifest_sha256=sha, cache_dir=cache
    )
    s3.objects.clear()

    with pytest.raises(PublishRefused, match=f"{TAXONOMY_VERSION} is not published"):
        external.load_taxonomy(
            s3, DATASET_BUCKET, version=TAXONOMY_VERSION, manifest_sha256=sha, cache_dir=cache
        )


def test_load_taxonomy_refuses_a_published_manifest_other_than_the_pin(tmp_path: Path) -> None:
    s3, _ = _published_taxonomy(tmp_path)

    with pytest.raises(PublishRefused, match="does not match its pin"):
        external.load_taxonomy(
            s3,
            DATASET_BUCKET,
            version=TAXONOMY_VERSION,
            manifest_sha256="0" * 64,
            cache_dir=tmp_path / "cache",
        )
