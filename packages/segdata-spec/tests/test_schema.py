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


def _images(
    ids=("a", "b"), mask_key=None, mask_etag=None, mask_size=None, source_name=None
) -> pa.Table:
    n = len(ids)
    masks = list(mask_key) if mask_key is not None else [None] * n
    # Rows with a mask get a full fingerprint unless the caller overrides the column.
    etags = mask_etag if mask_etag is not None else ["me" if m else None for m in masks]
    sizes = mask_size if mask_size is not None else [5 if m else None for m in masks]
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
            "mask_etag": list(etags),
            "mask_size": list(sizes),
            "mask_crc64nvme": [None] * n,
            "source_fingerprint": ["fp"] * n,
            "source_name": source_name if source_name is not None else [None] * n,
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


@pytest.mark.parametrize("source_name", [[None, None], ["s1/a.jpg", "s1/b.jpg"]])
def test_conform_places_nullable_source_name_after_source_fingerprint(source_name):
    result = conform(_images(source_name=source_name), IMAGES_SCHEMA)

    names = result.column_names
    assert names[names.index("source_fingerprint") + 1] == "source_name"
    assert result["source_name"].to_pylist() == source_name


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
        ("points", _tables(points=True, images=_images(mask_key=["m/a.png", None]))),
        (
            "dense",
            _tables(
                points=False,
                images=_images(mask_key=["m/a.png", "m/b.png"], mask_etag=["me", None]),
            ),
        ),
        (
            "dense",
            _tables(
                points=False,
                images=_images(mask_key=["m/a.png", "m/b.png"], mask_size=[5, None]),
            ),
        ),
    ],
    ids=[
        "duplicate-image-id",
        "duplicate-label-id",
        "points-kind-without-points",
        "dense-kind-with-points",
        "points-kind-with-mask-key",
        "dense-kind-null-mask-etag",
        "dense-kind-null-mask-size",
    ],
)
def test_invalid_table_sets_are_rejected(kind, tables):
    with pytest.raises(SchemaValidationError):
        validate_dataset_tables(kind, tables)


def test_dense_kind_rejects_a_null_mask_key_alone():
    images = _images(mask_key=["m/a.png", None], mask_etag=["me", "me"], mask_size=[5, 5])

    with pytest.raises(SchemaValidationError, match="mask_key"):
        validate_dataset_tables("dense", _tables(points=False, images=images))
