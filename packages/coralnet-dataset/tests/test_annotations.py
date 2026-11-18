"""Tests for the annotations merge stage."""

from __future__ import annotations

import threading

import pandas as pd
import pyarrow.parquet as pq
import pytest

from coralnet_dataset.annotations import (
    _derive_image_id,
    build_annotations,
    list_source_folders,
)
from coralnet_dataset.io import make_csv_reader_s3, write_parquet_deterministic
from coralnet_dataset.schemas import ANNOTATIONS_PRIMARY_KEY, ANNOTATIONS_SCHEMA

_LABELSET_100S = b"Label ID,Name\n100,a\n101,b\n102,c\n"

# The union of sample_csv_factory's labelset.csv.
_FACTORY_ID2NAME = {
    "1000": "label_0",
    "1001": "label_1",
    "1002": "label_2",
    "1003": "label_3",
    "1004": "label_4",
    "1005": "label_5",
    "1006": "label_6",
}


def test_derive_image_id_strips_url_decorations():
    s = pd.Series(
        ["/image/12345/view/", "/image/9/view/", "https://coralnet.ucsd.edu/image/77/view/"]
    )
    out = _derive_image_id(s).tolist()
    assert out == ["12345", "9", "77"]


def test_build_annotations_merges_per_source(fake_s3, sample_csv_factory, populate):
    bucket, prefix = "b", "p"
    populate(
        fake_s3,
        bucket,
        prefix,
        1,
        sample_csv_factory(source_id=1, n_images=2, n_points_per_image=3),
    )
    populate(
        fake_s3,
        bucket,
        prefix,
        2,
        sample_csv_factory(source_id=2, n_images=1, n_points_per_image=4),
    )

    ann_df = build_annotations(
        [1, 2], bucket=bucket, prefix=prefix, workers=1, s3_client=fake_s3
    ).annotations

    # 2 imgs * 3 pts + 1 img * 4 pts
    assert len(ann_df) == 10
    assert set(ann_df.columns) == {"source_id", "image_id", "row", "col", "coralnet_id", "status"}
    # Status carried through (Confirmed in the fixture).
    assert (ann_df["status"] == "Confirmed").all()

    # image_id matches the URL-derived integer-as-string for each source.
    assert "10000" in set(ann_df.loc[ann_df["source_id"] == 1, "image_id"])
    assert "20000" in set(ann_df.loc[ann_df["source_id"] == 2, "image_id"])


def test_build_annotations_sorts_deterministically(fake_s3, sample_csv_factory, populate):
    populate(
        fake_s3, "b", "p", 5, sample_csv_factory(source_id=5, n_images=2, n_points_per_image=2)
    )
    a = build_annotations([5], bucket="b", prefix="p", workers=1, s3_client=fake_s3).annotations
    b = build_annotations([5], bucket="b", prefix="p", workers=1, s3_client=fake_s3).annotations
    assert a.equals(b)
    keys = list(zip(a["source_id"], a["image_id"], a["row"], a["col"], strict=False))
    assert keys == sorted(keys)


def test_build_annotations_strips_all_status_suffixes(fake_s3, populate):
    """Only Confirmed rows are kept in the merged parquet."""
    annotations = (
        "Name,Row,Column,Label ID,Status\n"
        "a.jpg,1,2,100,Confirmed\n"
        "b.jpg,3,4,101,Unconfirmed\n"
        "c.jpg,5,6,102,Unclassified\n"
    )
    image_list = (
        "Name,Image Page\n"
        "a.jpg - Confirmed,/image/1/view/\n"
        "b.jpg - Unconfirmed,/image/2/view/\n"
        "c.jpg - Unclassified,/image/3/view/\n"
    )
    files = {
        "annotations.csv": annotations.encode(),
        "image_list.csv": image_list.encode(),
        "labelset.csv": _LABELSET_100S,
    }
    populate(fake_s3, "b", "p", 99, files)

    ann = build_annotations([99], bucket="b", prefix="p", workers=1, s3_client=fake_s3).annotations
    assert len(ann) == 1
    assert ann["image_id"].iloc[0] == "1"
    assert ann["status"].iloc[0] == "Confirmed"


def test_build_annotations_drops_unconfirmed_when_status_from_image_list_suffix(fake_s3, populate):
    """Status from image_list suffix: only Confirmed rows are kept."""
    annotations = "Name,Row,Column,Label ID\nok.jpg,1,2,100\nbad.jpg,3,4,101\n"
    image_list = (
        "Name,Image Page\n"
        "ok.jpg - Confirmed,/image/42/view/\n"
        "bad.jpg - Unconfirmed,/image/43/view/\n"
    )
    files = {
        "annotations.csv": annotations.encode(),
        "image_list.csv": image_list.encode(),
        "labelset.csv": _LABELSET_100S,
    }
    populate(fake_s3, "b", "p", 50, files)

    ann = build_annotations([50], bucket="b", prefix="p", workers=1, s3_client=fake_s3).annotations
    assert len(ann) == 1
    assert ann["image_id"].iloc[0] == "42"
    assert ann["status"].iloc[0] == "Confirmed"


def test_list_source_folders_keeps_only_numeric_source_folders(fake_s3):
    bucket, prefix = "b", "coralnet-public-images"
    for folder in ("s12", "s3", "temporary", "resized", "sNotANumber"):
        fake_s3.put(bucket, f"{prefix}/{folder}/annotations.csv", b"a,b\n1,2\n")
    assert list_source_folders(fake_s3, bucket, prefix) == [3, 12]


def test_build_annotations_without_producing_sources_writes_empty_parquet(fake_s3, tmp_path):
    ann = build_annotations([404], bucket="b", prefix="p", workers=1, s3_client=fake_s3).annotations

    assert ann.empty
    assert list(ann.columns) == [f.name for f in ANNOTATIONS_SCHEMA]
    out = tmp_path / "ann.parquet"
    write_parquet_deterministic(ann, out, ANNOTATIONS_SCHEMA, ANNOTATIONS_PRIMARY_KEY)
    assert pq.read_schema(out).names == [f.name for f in ANNOTATIONS_SCHEMA]


def test_build_annotations_unions_labelsets_of_complete_sources(
    fake_s3, sample_csv_factory, populate
):
    # Both sources share the factory labelset, so this also covers a repeated id and name.
    for sid in (1, 2):
        populate(
            fake_s3,
            "b",
            "p",
            sid,
            sample_csv_factory(source_id=sid, n_images=1, n_points_per_image=2),
        )

    build = build_annotations([1, 2], bucket="b", prefix="p", workers=1, s3_client=fake_s3)

    assert sorted(build.annotations["source_id"].unique().tolist()) == [1, 2]
    assert build.id2name == _FACTORY_ID2NAME
    assert build.exclusions == {}


def test_build_annotations_keeps_source_with_zero_confirmed_rows(fake_s3, populate):
    files = {
        "annotations.csv": b"Name,Row,Column,Label ID,Status\na.jpg,1,2,100,Unconfirmed\n",
        "image_list.csv": b"Name,Image Page\na.jpg - Unconfirmed,/image/1/view/\n",
        # A non-numeric Label ID row is dropped, and the source stays included.
        "labelset.csv": _LABELSET_100S + b"abc,junk\n",
    }
    populate(fake_s3, "b", "p", 7, files)

    build = build_annotations([7], bucket="b", prefix="p", workers=1, s3_client=fake_s3)

    assert build.annotations.empty
    assert build.exclusions == {}
    assert build.id2name == {"100": "a", "101": "b", "102": "c"}


def _drop_last_line(data: bytes) -> bytes:
    return b"".join(data.splitlines(keepends=True)[:-1])


_BREAKS = {
    "annotations_missing": lambda f: f.pop("annotations.csv"),
    "image_list_missing": lambda f: f.pop("image_list.csv"),
    "labelset_missing": lambda f: f.pop("labelset.csv"),
    "csv_read_failed": lambda f: None,
    "columns_missing": lambda f: f.update({"labelset.csv": b"Label ID,Label\n1000,x\n"}),
    "annotations_empty": lambda f: f.update(
        {"annotations.csv": b"Name,Row,Column,Label ID,Status\n"}
    ),
    "image_list_incomplete": lambda f: f.update(
        {"image_list.csv": _drop_last_line(f["image_list.csv"])}
    ),
    # Label 5555 is in neither source's labelset.
    "label_unnamed": lambda f: f.update(
        {"annotations.csv": f["annotations.csv"].replace(b",1000,Confirmed", b",5555,Confirmed")}
    ),
}


@pytest.mark.parametrize("kind", list(_BREAKS))
def test_build_annotations_excludes_broken_source(fake_s3, sample_csv_factory, populate, kind):
    populate(
        fake_s3, "b", "p", 1, sample_csv_factory(source_id=1, n_images=1, n_points_per_image=1)
    )
    broken = sample_csv_factory(source_id=2, n_images=2, n_points_per_image=1)
    _BREAKS[kind](broken)
    populate(fake_s3, "b", "p", 2, broken)
    s3_reader = make_csv_reader_s3(fake_s3)

    def reader(bucket, key):
        if kind == "csv_read_failed" and key == "p/s2/labelset.csv":
            raise OSError("read refused")
        return s3_reader(bucket, key)

    build = build_annotations([1, 2], bucket="b", prefix="p", workers=1, csv_reader=reader)

    assert build.annotations["source_id"].unique().tolist() == [1]
    assert build.exclusions == {kind: [2]}


def test_build_annotations_reports_a_missing_csv_over_a_read_failure(
    fake_s3, sample_csv_factory, populate
):
    broken = sample_csv_factory(source_id=2, n_images=1, n_points_per_image=1)
    broken.pop("labelset.csv")
    populate(fake_s3, "b", "p", 2, broken)
    s3_reader = make_csv_reader_s3(fake_s3)

    def reader(bucket, key):
        if key == "p/s2/annotations.csv":
            raise OSError("read refused")
        return s3_reader(bucket, key)

    build = build_annotations([2], bucket="b", prefix="p", workers=1, csv_reader=reader)

    assert build.exclusions == {"labelset_missing": [2]}


def test_build_annotations_unnames_labels_only_an_excluded_source_named(fake_s3, populate):
    image_list = b"Name,Image Page\na.jpg - Confirmed,/image/1/view/\n"
    # Source 1 uses label 7, which only source 2 names; source 2 uses label 9, which nobody names.
    populate(
        fake_s3,
        "b",
        "p",
        1,
        {
            "annotations.csv": b"Name,Row,Column,Label ID,Status\na.jpg,1,2,7,Confirmed\n",
            "image_list.csv": image_list,
            "labelset.csv": b"Label ID,Name\n8,eight\n",
        },
    )
    populate(
        fake_s3,
        "b",
        "p",
        2,
        {
            "annotations.csv": b"Name,Row,Column,Label ID,Status\na.jpg,1,2,9,Confirmed\n",
            "image_list.csv": image_list,
            "labelset.csv": b"Label ID,Name\n7,seven\n",
        },
    )

    build = build_annotations([1, 2], bucket="b", prefix="p", workers=1, s3_client=fake_s3)

    assert build.exclusions == {"label_unnamed": [1, 2]}
    assert build.id2name == {}


def test_build_annotations_raises_on_conflicting_label_names(fake_s3, populate):
    for sid, name in ((1, "Porites"), (2, "Acropora")):
        populate(
            fake_s3,
            "b",
            "p",
            sid,
            {
                "annotations.csv": b"Name,Row,Column,Label ID,Status\na.jpg,1,2,100,Confirmed\n",
                "image_list.csv": b"Name,Image Page\na.jpg - Confirmed,/image/1/view/\n",
                "labelset.csv": f"Label ID,Name\n100,{name}\n".encode(),
            },
        )

    with pytest.raises(ValueError, match=r"100.*Porites.*Acropora|100.*Acropora.*Porites"):
        build_annotations([1, 2], bucket="b", prefix="p", workers=1, s3_client=fake_s3)


def test_build_annotations_stops_queued_sources_after_a_worker_error(monkeypatch):
    from coralnet_dataset import annotations

    release = threading.Event()
    started: list[int] = []

    def fake_outcome(_reader, _bucket, _prefix, source_id):
        started.append(source_id)
        if source_id == 1:
            raise RuntimeError("unexpected")
        # Hold each running worker so the queue still has sources when source 1 fails.
        release.wait(timeout=1)
        return annotations._SourceExcluded("annotations_missing", "stub")

    monkeypatch.setattr(annotations, "_get_thread_reader", lambda: None)
    monkeypatch.setattr(annotations, "_load_source_outcome", fake_outcome)

    with pytest.raises(RuntimeError, match="unexpected"):
        build_annotations([1, 2, 3, 4, 5, 6], bucket="b", prefix="p", workers=2)
    release.set()

    # Two workers run at most source 1, source 2 and one more before the queue is cancelled.
    assert set(started) <= {1, 2, 3}
