"""Tests for the per-source completeness verification."""

from __future__ import annotations

import pytest

from coralnet_scraper.verify import verify_source

_LIST_KEY = "p/s1/image_list.csv"
_ANN_KEY = "p/s1/annotations.csv"

_LIST_ROWS = [
    ("a.jpg - Confirmed", "/image/11/view/"),
    ("b.jpg - Unconfirmed", "/image/12/view/"),
    ("c.jpg - Unclassified", "/image/13/view/"),
]
_ANN_TEXT = "Name,Row,Column,Label ID\na.jpg,1,1,7\na.jpg,2,2,7\nb.jpg,3,3,8\n"


def _list_csv(rows: list[tuple[str, str]]) -> bytes:
    body = "".join(f"{name},{page},\n" for name, page in rows)
    return f"Name,Image Page,Image URL\n{body}".encode()


def _store(fake_s3, *, rows=_LIST_ROWS, ann: str | None = _ANN_TEXT, images=(11, 12, 13)) -> None:
    """Put a source on the fake bucket: list, annotations and one 5-byte object per image id."""
    if rows is not None:
        fake_s3.put("b", _LIST_KEY, _list_csv(rows))
    if ann is not None:
        fake_s3.put("b", _ANN_KEY, ann.encode())
    for image_id in images:
        fake_s3.put("b", f"p/s1/images/{image_id}.jpg", b"JPEG!")


def _verify(fake_s3, expected_total: int = 3):
    return verify_source(fake_s3, "b", "p", 1, expected_total=expected_total)


def test_a_complete_source_passes_with_its_counts(fake_s3):
    _store(fake_s3)

    v = _verify(fake_s3)

    assert v.ok is True
    assert v.failures == []
    assert v.warnings == []
    assert v.stats == {
        "total_images": 3,
        "annotation_rows": 3,
        "annotated_images": 2,
        "image_objects": 3,
    }


@pytest.mark.parametrize(
    ("store_kwargs", "expected_total", "failure"),
    [
        ({"ann": None}, 3, "annotations_missing"),
        ({"rows": None}, 3, "image_list_missing"),
        ({"rows": _LIST_ROWS[:2], "images": (11, 12)}, 3, "image_list_count:2!=3"),
        (
            {"rows": [*_LIST_ROWS[:2], ("c.jpg - Unclassified", "/image/12/view/")]},
            3,
            "image_list_bad_ids",
        ),
        ({"ann": "Row,Column,Label ID\n1,1,7\n"}, 3, "annotations_bad_columns"),
        ({"ann": "Name,<html>,Label ID\na.jpg,1,7\n"}, 3, "annotations_bad_columns"),
        ({"ann": "Name,Row,Column,Label ID\n,1,1,7\n"}, 3, "annotations_null_name"),
        (
            {"ann": "Name,Row,Column,Label ID\na.jpg,1,1,7\nz.jpg,1,1,7\n"},
            3,
            "annotations_name_not_in_list:1",
        ),
        ({"images": (11, 12)}, 3, "image_objects_missing:1"),
    ],
    ids=[
        "no-annotations",
        "no-image-list",
        "list-short",
        "duplicate-id",
        "no-name-column",
        "html-column",
        "null-name",
        "foreign-name",
        "missing-object",
    ],
)
def test_each_failure_blocks_the_source(fake_s3, store_kwargs, expected_total, failure):
    _store(fake_s3, **store_kwargs)

    v = _verify(fake_s3, expected_total)

    assert v.ok is False
    assert failure in v.failures


def test_a_zero_byte_image_object_counts_as_missing(fake_s3):
    _store(fake_s3)
    fake_s3.put("b", "p/s1/images/13.jpg", b"")

    v = _verify(fake_s3)

    assert v.failures == ["image_objects_missing:1"]


@pytest.mark.parametrize(
    ("store_kwargs", "extra_object", "warning"),
    [
        (
            {"ann": "Name,Row,Column,Label ID\nb.jpg,3,3,8\n"},
            None,
            "confirmed_without_annotations:1",
        ),
        ({}, "p/s1/images/99.jpg", "orphan_image_objects:1"),
        (
            {"rows": [*_LIST_ROWS[:2], ("a.jpg - Unclassified", "/image/13/view/")]},
            None,
            "duplicate_names:1",
        ),
    ],
    ids=["confirmed-unannotated", "orphan-object", "duplicate-name"],
)
def test_warnings_are_recorded_without_blocking(fake_s3, store_kwargs, extra_object, warning):
    _store(fake_s3, **store_kwargs)
    if extra_object:
        fake_s3.put("b", extra_object, b"JPEG!")

    v = _verify(fake_s3)

    assert v.ok is True
    assert v.warnings == [warning]
