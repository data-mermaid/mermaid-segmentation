import pyarrow as pa
import pytest

from segdata_spec.errors import SchemaValidationError
from segdata_spec.schema import (
    DATASET_TABLES,
    IMAGES_SCHEMA,
    POINTS_SCHEMA,
    conform,
    validate_dataset_tables,
)


def _images(ids=("a", "b"), mask_key=None) -> pa.Table:
    n = len(ids)
    masks = list(mask_key) if mask_key is not None else [None] * n
    return pa.table(
        {
            "image_id": list(ids),
            "key": [f"images/d/g/{i}.jpg" for i in ids],
            "etag": ["e"] * n,
            "size": [10] * n,
            "crc64nvme": [None] * n,
            "media_type": ["image/jpeg"] * n,
            "width": [4] * n,
            "height": [4] * n,
            "group": ["g"] * n,
            "mask_key": masks,
            "mask_etag": [None] * n,
            "mask_size": [None] * n,
            "mask_crc64nvme": [None] * n,
            "source_fingerprint": ["fp"] * n,
            "attrs": [None] * n,
        }
    )


def _points() -> pa.Table:
    return pa.table(
        {
            "image_id": ["a"],
            "row": [1],
            "col": [2],
            "label_id": [7],
            "annotator": ["default"],
        }
    )


def _labels(ids=(7, 8)) -> pa.Table:
    n = len(ids)
    return pa.table(
        {
            "label_id": list(ids),
            "name": [f"l{i}" for i in ids],
            "mermaid_benthic_attribute_id": [None] * n,
            "mermaid_benthic_attribute": [None] * n,
            "mermaid_growth_form": [None] * n,
            "mapping_source": ["identity"] * n,
        }
    )


def _counts() -> pa.Table:
    return pa.table({"image_id": ["a"], "label_id": [7], "n": [1]})


def _tables(*, points: bool, images=None, labels=None) -> dict[str, pa.Table]:
    tables = {
        "images.parquet": images if images is not None else _images(),
        "labels.parquet": labels if labels is not None else _labels(),
        "stats/label_counts.parquet": _counts(),
    }
    if points:
        tables["points.parquet"] = _points()
    return {name: conform(t, DATASET_TABLES[name]) for name, t in tables.items()}


def test_conform_selects_casts_and_orders_columns():
    table = pa.table(
        {
            "annotator": ["default"],
            "extra": [1],
            "col": pa.array([2], pa.int64()),
            "row": pa.array([1], pa.int64()),
            "label_id": pa.array([7], pa.int64()),
            "image_id": ["a"],
        }
    )

    result = conform(table, POINTS_SCHEMA)

    assert result.schema.equals(POINTS_SCHEMA)
    assert result.schema.field("row").type == pa.int32()
    assert result.to_pylist() == [
        {"image_id": "a", "row": 1, "col": 2, "label_id": 7, "annotator": "default"}
    ]


def test_conform_names_the_missing_column():
    table = _images().drop_columns(["group"])

    with pytest.raises(SchemaValidationError, match="group"):
        conform(table, IMAGES_SCHEMA)


def test_valid_tables_pass():
    validate_dataset_tables("points", _tables(points=True))
    validate_dataset_tables(
        "dense", _tables(points=False, images=_images(mask_key=["m/a.png", "m/b.png"]))
    )


@pytest.mark.parametrize(
    ("kind", "tables"),
    [
        ("points", _tables(points=True, images=_images(ids=("a", "a")))),
        ("points", _tables(points=True, labels=_labels(ids=(7, 7)))),
        ("points", _tables(points=False)),
        ("dense", _tables(points=True, images=_images(mask_key=["m/a.png", "m/b.png"]))),
        ("dense", _tables(points=False, images=_images(mask_key=["m/a.png", None]))),
        ("points", _tables(points=True, images=_images(mask_key=["m/a.png", None]))),
    ],
    ids=[
        "duplicate-image-id",
        "duplicate-label-id",
        "points-kind-without-points",
        "dense-kind-with-points",
        "dense-kind-null-mask-key",
        "points-kind-with-mask-key",
    ],
)
def test_invalid_table_sets_are_rejected(kind, tables):
    with pytest.raises(SchemaValidationError):
        validate_dataset_tables(kind, tables)
