"""Shared helpers of the external dataset version builders."""

from __future__ import annotations

import pandas as pd
import pyarrow as pa
import pytest

from segdata_etl.common.copy import CopiedObject, CopyItem
from segdata_etl.common.legacy import LegacyDataset, LegacyObject
from segdata_etl.datasets import external
from segdata_etl.label_maps.curated import CuratedMapInfo

ATTRIBUTES = pa.table(
    {
        "id": ["ba-hc", "ba-acro"],
        "name": ["Hard coral", "Acropora"],
        "parent_id": [None, "ba-hc"],
    }
)
CLASSES = {"unlabeled": 0, "Acropora digitate": 5, "Water": 7, "Mystery": 9}
CURATED = {"acropora digitate": "acropora", "water": "background"}


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
    info = CuratedMapInfo(path="curated/x.csv", sha256="f" * 64, n_entries=2)
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


LEGACY = LegacyDataset(
    name="benthos",
    bucket="legacy-bucket",
    prefix="legacy/benthos/",
    annotations=pd.DataFrame(),
    classes={},
    manifest={},
    manifest_sha256="0" * 64,
    objects={},
)


def test_copy_items_and_image_rows_carry_the_mask_copy() -> None:
    image = external.LegacyImage(
        image_id="7", group="site1", key="imgs/7.jpg", etag="src-etag", mask_key="masks/7.png"
    )
    sniffed = {"imgs/7.jpg": ("image/jpeg", 40, 30)}

    items = external.copy_items([image], sniffed, legacy=LEGACY, dataset="benthos")

    assert items == [
        CopyItem("legacy-bucket", "legacy/benthos/imgs/7.jpg", "images/benthos/site1/7.jpg"),
        CopyItem("legacy-bucket", "legacy/benthos/masks/7.png", "images/benthos/site1/7_mask.png"),
    ]
    copied = {
        "images/benthos/site1/7.jpg": CopiedObject(etag='"i"', size=100, crc64nvme="ic"),
        "images/benthos/site1/7_mask.png": CopiedObject(etag='"m"', size=20, crc64nvme="mc"),
    }
    row = external.image_rows([image], sniffed, copied, dataset="benthos").to_pylist()[0]
    assert (row["key"], row["etag"], row["size"], row["crc64nvme"]) == (
        "site1/7.jpg",
        '"i"',
        100,
        "ic",
    )
    assert (row["mask_key"], row["mask_etag"], row["mask_size"], row["mask_crc64nvme"]) == (
        "site1/7_mask.png",
        '"m"',
        20,
        "mc",
    )


def test_image_rows_writes_attrs_as_a_sorted_json_object() -> None:
    image = external.LegacyImage(
        image_id="7", group="g", key="7.jpg", etag="e", attrs={"site": "s1", "depth": 5}
    )
    copied = {"images/d/g/7.jpg": CopiedObject(etag='"i"', size=1, crc64nvme=None)}
    rows = external.image_rows(
        [image], {"7.jpg": ("image/jpeg", 4, 3)}, copied, dataset="d"
    ).to_pylist()
    assert rows[0]["attrs"] == '{"depth": 5, "site": "s1"}'


SIZES = {"7.jpg": ("image/jpeg", 4, 3)}
IMAGE_7 = external.LegacyImage(image_id="7", group="g", key="7.jpg", etag="e")


def test_require_points_inside_accepts_the_last_pixel() -> None:
    external.require_points_inside(_points(("7", 0, 0, 1), ("7", 2, 3, 1)), [IMAGE_7], SIZES)


@pytest.mark.parametrize(
    ("row", "col"), [(3, 0), (0, 4), (-1, 0), (0, -1)], ids=["row", "col", "neg-row", "neg-col"]
)
def test_require_points_inside_refuses_a_point_past_an_edge(row: int, col: int) -> None:
    points = _points(("7", 1, 1, 1), ("7", row, col, 1))
    with pytest.raises(ValueError, match=rf"1 points outside.*'7', {row}, {col}"):
        external.require_points_inside(points, [IMAGE_7], SIZES)


def test_drop_points_outside_keeps_in_bounds_points_in_order_and_counts_the_rest() -> None:
    points = _points(("7", 2, 3, 1), ("7", 1, 4, 2), ("7", 0, 0, 3), ("7", 3, 1, 4), ("7", 1, 1, 5))
    kept, dropped = external.drop_points_outside(points, [IMAGE_7], SIZES)
    assert dropped == 2
    assert kept.column("label_id").to_pylist() == [1, 3, 5]
    assert kept.schema == points.schema


def test_drop_points_outside_refuses_a_point_whose_image_has_no_size() -> None:
    with pytest.raises(ValueError, match=r"no sniffed size.*'8'"):
        external.drop_points_outside(_points(("7", 0, 0, 1), ("8", 0, 0, 1)), [IMAGE_7], SIZES)


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
