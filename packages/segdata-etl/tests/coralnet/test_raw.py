"""``segdata build coralnet raw``: the frozen CoralNet build staged as a ``coralnet-raw`` version."""

from __future__ import annotations

import argparse
import io
import json
import math
import sys
from pathlib import Path
from typing import Any

import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import pytest
from PIL import Image
from s3_fake import FakeS3, crc64nvme, md5_hex, quoted_etag

from segdata_etl import cli, promote
from segdata_etl.datasets.coralnet import raw
from segdata_etl.datasets.coralnet.io import make_csv_reader_s3
from segdata_spec import DATASET_BUCKET
from segdata_spec.errors import PublishRefused
from segdata_spec.manifest import Manifest, ParentRef, manifest_sha256
from segdata_spec.publish import GitState
from segdata_spec.schema import IMAGES_SCHEMA

from .conftest import _make_jpeg_bytes

SRC = "src"
PREFIX = "coralnet-public-images"
GIT = GitState(sha="abcdef1234567", dirty=False, pushed=True)
JPEG = _make_jpeg_bytes(3000, 2000)
JPEG_12 = _make_jpeg_bytes(10, 10) + b"twelve"


def _png(width: int, height: int) -> bytes:
    buf = io.BytesIO()
    Image.new("RGB", (width, height)).save(buf, format="PNG")
    return buf.getvalue()


PNG = _png(640, 480)
DEST_10 = "images/coralnet/s1/10.jpg"
DEST_20 = "images/coralnet/s2/20.png"


class RecordingS3(FakeS3):
    """The test FakeS3, recording every HEAD and GET key and every put.

    ``fail_on_head`` makes the HEAD of that source-image ordinal (1-based) raise, as a crash does.
    ``put_object`` answers with the CRC64NVME, as S3 does for a write that asks for one.
    """

    def __init__(self, objects: dict[tuple[str, str], bytes] | None = None):
        super().__init__(objects=objects if objects is not None else {})
        self.heads: list[str] = []
        self.gets: list[str] = []
        self.fail_on_head: int | None = None

    def head_object(self, *, Bucket: str, Key: str, ChecksumMode: str | None = None) -> Any:
        if "/images/" in Key:
            self.heads.append(Key)
            if self.fail_on_head == len(self.heads):
                raise RuntimeError("interrupted")
        return super().head_object(Bucket=Bucket, Key=Key, ChecksumMode=ChecksumMode)

    def get_object(self, *, Bucket: str, Key: str, Range: str | None = None, **kw: Any) -> Any:
        self.gets.append(Key)
        return super().get_object(Bucket=Bucket, Key=Key, Range=Range, **kw)

    def put_object(self, *, Bucket: str, Key: str, Body: Any, **kw: Any) -> dict[str, Any]:
        response = super().put_object(Bucket=Bucket, Key=Key, Body=Body, **kw)
        return {**response, "ChecksumCRC64NVME": crc64nvme(self.objects[(Bucket, Key)])}

    def list_objects_v2(self, *, MaxKeys: int | None = None, **kw: Any) -> dict[str, Any]:
        del MaxKeys
        return super().list_objects_v2(**kw)

    def image_heads(self) -> list[str]:
        return sorted(self.heads)

    def image_gets(self) -> list[str]:
        return sorted(key for key in self.gets if "/images/" in key)


def _key(sid: int, image_id: str, ext: str = "jpg") -> str:
    return f"{PREFIX}/s{sid}/images/{image_id}.{ext}"


def _images(rows: list[tuple[int, str, str, int | None, int | None]]) -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "source_id": sid,
                "image_id": image_id,
                "s3_key": _key(sid, image_id),
                "width": width,
                "height": height,
                "header_status": status,
            }
            for sid, image_id, status, width, height in rows
        ]
    )


def _annotations(rows: list[tuple[int, str, int, int, int]]) -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "source_id": sid,
                "image_id": image_id,
                "row": row,
                "col": col,
                "coralnet_id": label,
                "status": "Confirmed",
            }
            for sid, image_id, row, col, label in rows
        ]
    )


ID2NAME = {"100": "Acropora", "200": "Sand", "300": "Unused", "2474": math.nan}


def _fixture_build() -> raw.BuildTables:
    images = _images(
        [
            (1, "10", "ok", 3000, 2000),
            (2, "20", "pil_fallback", 640, 480),
            (2, "21", "not_found", None, None),
            (1, "12", "ok", 10, 10),
        ]
    )
    annotations = _annotations(
        [
            (1, "10", 5, 7, 100),
            (1, "10", 8, 9, 100),
            (1, "10", 1500, 2900, 200),
            (2, "20", 1, 2, 200),
            (2, "21", 3, 4, 100),
        ]
    )
    return raw.BuildTables(annotations=annotations, images=images, id2name=ID2NAME)


def _image_list(rows: list[tuple[str, str]]) -> bytes:
    lines = ["Name,Image Page", *(f"{name},/image/{image_id}/view/" for name, image_id in rows)]
    return ("\n".join(lines) + "\n").encode()


@pytest.fixture
def source() -> RecordingS3:
    s3 = RecordingS3()
    s3.put(SRC, _key(1, "10"), JPEG)
    s3.put(SRC, _key(2, "20"), PNG)
    s3.put(SRC, _key(1, "12"), JPEG_12)
    s3.put(SRC, f"{PREFIX}/s1/image_list.csv", _image_list([("a.jpg - Confirmed", "10")]))
    s3.put(
        SRC,
        f"{PREFIX}/s2/image_list.csv",
        _image_list([("b.png - Unconfirmed", "20"), ("c.jpg", "21")]),
    )
    return s3


def _build(
    source: FakeS3,
    tmp_path: Path,
    build: raw.BuildTables | None = None,
    parent_images: pa.Table | None = None,
) -> raw.RawTables:
    return raw.build_raw(
        source,
        build or _fixture_build(),
        source_bucket=SRC,
        prefix=PREFIX,
        workers=1,
        checkpoint=tmp_path / "checkpoint.parquet",
        parent_images=parent_images,
        csv_reader=make_csv_reader_s3(source),
    )


def test_images_keep_annotated_images_with_dimensions(source: RecordingS3, tmp_path: Path) -> None:
    result = _build(source, tmp_path)

    images = result.tables["images.parquet"].to_pylist()
    assert [
        {k: row[k] for k in ("image_id", "group", "key", "media_type", "width", "height")}
        for row in images
    ] == [
        {
            "image_id": "10",
            "group": "s1",
            "key": "s1/10.jpg",
            "media_type": "image/jpeg",
            "width": 3000,
            "height": 2000,
        },
        {
            "image_id": "20",
            "group": "s2",
            "key": "s2/20.png",
            "media_type": "image/png",
            "width": 640,
            "height": 480,
        },
    ]
    assert [row["source_fingerprint"] for row in images] == [md5_hex(JPEG), md5_hex(PNG)]
    assert [row["source_name"] for row in images] == ["a.jpg", "b.png"]
    assert images[0]["attrs"] == (
        '{"source_id": 1, "source_key": "coralnet-public-images/s1/images/10.jpg"}'
    )
    assert all(row["mask_key"] is None for row in images)
    assert result.dropped["no_dimensions"] == 1
    assert result.dropped["no_annotations"] == 1


def test_images_carry_the_probed_objects_fingerprints(source: RecordingS3, tmp_path: Path) -> None:
    result = _build(source, tmp_path)

    images = result.tables["images.parquet"].to_pylist()
    assert [(row["etag"], row["size"], row["crc64nvme"]) for row in images] == [
        (md5_hex(JPEG), len(JPEG), crc64nvme(JPEG)),
        (md5_hex(PNG), len(PNG), crc64nvme(PNG)),
    ]


def test_a_multipart_original_is_read_whole_for_its_single_part_copy_values(
    source: RecordingS3, tmp_path: Path
) -> None:
    source.multipart.add((SRC, _key(1, "10")))

    result = _build(source, tmp_path)

    row = result.tables["images.parquet"].to_pylist()[0]
    assert (row["etag"], row["crc64nvme"], row["source_fingerprint"]) == (
        md5_hex(JPEG),
        crc64nvme(JPEG),
        f"{md5_hex(JPEG)}-2",
    )
    assert _key(1, "10") in source.full_reads


def test_points_keep_original_row_col_of_kept_images(source: RecordingS3, tmp_path: Path) -> None:
    result = _build(source, tmp_path)

    assert result.tables["points.parquet"].to_pylist() == [
        {"image_id": "10", "row": 5, "col": 7, "label_id": 100, "annotator": "default"},
        {"image_id": "10", "row": 8, "col": 9, "label_id": 100, "annotator": "default"},
        {"image_id": "10", "row": 1500, "col": 2900, "label_id": 200, "annotator": "default"},
        {"image_id": "20", "row": 1, "col": 2, "label_id": 200, "annotator": "default"},
    ]


def test_label_counts_count_points_per_image_and_label(source: RecordingS3, tmp_path: Path) -> None:
    result = _build(source, tmp_path)

    assert result.tables["stats/label_counts.parquet"].to_pylist() == [
        {"image_id": "10", "label_id": 100, "n": 2},
        {"image_id": "10", "label_id": 200, "n": 1},
        {"image_id": "20", "label_id": 200, "n": 1},
    ]


def test_labels_hold_every_id2name_id_and_name_none_for_2474() -> None:
    id2name = json.loads('{"2474": NaN, "300": "Unused", "100": "Acropora"}')

    assert raw.labels_table(id2name).to_pylist() == [
        {
            "label_id": label_id,
            "name": name,
            "mermaid_benthic_attribute_id": None,
            "mermaid_benthic_attribute": None,
            "mermaid_growth_form": None,
            "mapping_source": "identity",
        }
        for label_id, name in ((100, "Acropora"), (300, "Unused"), (2474, "None"))
    ]


@pytest.mark.parametrize("id2name", [{"2474": 7}, {"5": math.nan}])
def test_labels_refuse_any_other_non_string_name(id2name: dict[str, object]) -> None:
    with pytest.raises(ValueError, match="non-string"):
        raw.labels_table(id2name)


def _parent(rows: list[tuple[str, str, str]]) -> pa.Table:
    return pa.Table.from_pylist(
        [
            {
                "image_id": image_id,
                "key": key,
                "etag": "parent-etag",
                "size": 99,
                "crc64nvme": "parent-crc",
                "media_type": "image/png" if key.endswith(".png") else "image/jpeg",
                "source_fingerprint": fingerprint,
            }
            for image_id, key, fingerprint in rows
        ]
    )


def test_parent_rows_matching_the_listed_etag_are_reused_without_any_request(
    source: RecordingS3, tmp_path: Path
) -> None:
    parent = _parent([("10", "s1/10.jpg", md5_hex(JPEG)), ("20", "s2/20.png", "stale")])

    result = _build(source, tmp_path, parent_images=parent)

    images = {row["image_id"]: row for row in result.tables["images.parquet"].to_pylist()}
    assert (images["10"]["etag"], images["10"]["size"], images["10"]["crc64nvme"]) == (
        "parent-etag",
        99,
        "parent-crc",
    )
    assert (images["20"]["etag"], images["20"]["crc64nvme"]) == (md5_hex(PNG), crc64nvme(PNG))
    assert source.image_heads() == [_key(2, "20")]
    assert source.image_gets() == [_key(2, "20")]


def test_carried_rows_keep_the_parents_media_type_unsniffed(
    source: RecordingS3, tmp_path: Path
) -> None:
    gif = b"GIF89a\x01\x00"
    source.put(SRC, _key(2, "20"), gif)
    parent = _parent([("20", "s2/20.png", md5_hex(gif))])

    result = _build(source, tmp_path, parent_images=parent)

    images = {row["image_id"]: row for row in result.tables["images.parquet"].to_pylist()}
    assert (images["20"]["media_type"], images["20"]["key"]) == ("image/png", "s2/20.png")
    assert result.dropped["unsupported_format"] == 0


CARRIED_ROW = {
    "image_id": "30",
    "key": "s1/30.jpg",
    "etag": "parent-etag",
    "size": 99,
    "crc64nvme": "parent-crc",
    "media_type": "image/jpeg",
    "width": 3000,
    "height": 2000,
    "group": "s1",
    "mask_key": None,
    "mask_etag": None,
    "mask_size": None,
    "mask_crc64nvme": None,
    "source_fingerprint": "parent-fingerprint",
    "source_name": "parent-name.jpg",
    "attrs": '{"source_id": 1, "source_key": "coralnet-public-images/s1/images/30.jpg"}',
}


def _build_with_unlisted_image(
    status: str = "ok", width: int | None = 3000, height: int | None = 2000
) -> raw.BuildTables:
    """The fixture build plus annotated image 30, which the scrape does not list."""
    build = _fixture_build()
    return raw.BuildTables(
        annotations=pd.concat([build.annotations, _annotations([(1, "30", 4, 5, 200)])]),
        images=pd.concat([build.images, _images([(1, "30", status, width, height)])]),
        id2name=ID2NAME,
    )


def _parent_holding(*rows: dict[str, Any]) -> pa.Table:
    return pa.Table.from_pylist(list(rows), schema=IMAGES_SCHEMA)


def test_an_unlisted_image_the_parent_holds_keeps_the_parents_row_and_points_untouched(
    source: RecordingS3, tmp_path: Path
) -> None:
    parent = _parent_holding(CARRIED_ROW)

    result = _build(source, tmp_path, _build_with_unlisted_image(), parent)

    images = {row["image_id"]: row for row in result.tables["images.parquet"].to_pylist()}
    assert list(images) == ["10", "20", "30"]
    assert images["30"] == CARRIED_ROW
    assert ("30", 4, 5, 200) in [
        (p["image_id"], p["row"], p["col"], p["label_id"])
        for p in result.tables["points.parquet"].to_pylist()
    ]
    assert (result.dropped["carried_from_parent"], result.dropped["source_missing"]) == (1, 0)
    assert result.carried == {"30"}
    assert all("/30." not in key for key in source.heads + source.gets)


def test_a_not_found_build_row_the_parent_holds_is_carried_with_its_points(
    source: RecordingS3, tmp_path: Path
) -> None:
    build = _build_with_unlisted_image("not_found", None, None)
    source.put(SRC, _key(1, "30"), JPEG)

    result = _build(source, tmp_path, build, _parent_holding(CARRIED_ROW))

    images = {row["image_id"]: row for row in result.tables["images.parquet"].to_pylist()}
    assert images["30"] == CARRIED_ROW
    assert ("30", 4, 5, 200) in [
        (p["image_id"], p["row"], p["col"], p["label_id"])
        for p in result.tables["points.parquet"].to_pylist()
    ]
    assert (result.dropped["carried_from_parent"], result.dropped["source_missing"]) == (1, 0)
    assert result.dropped["no_dimensions"] == 1  # image 21, which the parent lacks
    assert result.carried == {"30"}
    assert all("/30." not in key for key in source.heads + source.gets)


def test_a_not_found_build_row_the_parent_lacks_is_dropped_as_no_dimensions(
    source: RecordingS3, tmp_path: Path
) -> None:
    other = {**CARRIED_ROW, "image_id": "31", "key": "s1/31.jpg"}
    build = _build_with_unlisted_image("not_found", None, None)

    result = _build(source, tmp_path, build, _parent_holding(other))

    assert "30" not in result.tables["images.parquet"].column("image_id").to_pylist()
    assert result.dropped["no_dimensions"] == 2
    assert result.dropped["carried_from_parent"] == 0


def test_a_carried_image_whose_build_size_differs_from_the_parents_is_refused(
    source: RecordingS3, tmp_path: Path
) -> None:
    build = _build_with_unlisted_image(width=111, height=222)

    with pytest.raises(ValueError, match=r"1 carried images.*\['30'\]"):
        _build(source, tmp_path, build, _parent_holding(CARRIED_ROW))


def test_a_parent_without_unlisted_images_reports_zero_carried(
    source: RecordingS3, tmp_path: Path
) -> None:
    result = _build(source, tmp_path, parent_images=_parent_holding(CARRIED_ROW))

    assert result.dropped["carried_from_parent"] == 0
    assert result.carried == frozenset()


def test_an_unlisted_image_the_parent_lacks_is_still_source_missing(
    source: RecordingS3, tmp_path: Path
) -> None:
    other = {**CARRIED_ROW, "image_id": "31", "key": "s1/31.jpg"}

    result = _build(source, tmp_path, _build_with_unlisted_image(), _parent_holding(other))

    assert result.tables["images.parquet"].column("image_id").to_pylist() == ["10", "20"]
    assert "30" not in result.tables["points.parquet"].column("image_id").to_pylist()
    assert (result.dropped["carried_from_parent"], result.dropped["source_missing"]) == (0, 1)


def test_the_copy_plan_omits_carried_images_and_keeps_listed_ones(
    source: RecordingS3, tmp_path: Path
) -> None:
    result = _build(source, tmp_path, _build_with_unlisted_image(), _parent_holding(CARRIED_ROW))

    plan = raw._copies(result.tables["images.parquet"], SRC, result.carried)

    assert [line["dst_key"] for line in plan] == [DEST_10, DEST_20]


def test_annotations_without_an_image_row_are_counted_and_dropped(
    source: RecordingS3, tmp_path: Path
) -> None:
    build = _fixture_build()
    build = raw.BuildTables(
        annotations=pd.concat([build.annotations, _annotations([(1, "99", 0, 0, 100)])]),
        images=build.images,
        id2name=ID2NAME,
    )

    result = _build(source, tmp_path, build)

    assert "99" not in result.tables["points.parquet"].column("image_id").to_pylist()
    assert result.dropped["annotations_without_image"] == 1


def test_a_missing_source_original_drops_the_image_and_its_points(
    source: RecordingS3, tmp_path: Path
) -> None:
    del source.objects[(SRC, _key(2, "20"))]

    result = _build(source, tmp_path)

    assert result.tables["images.parquet"].column("image_id").to_pylist() == ["10"]
    assert set(result.tables["points.parquet"].column("image_id").to_pylist()) == {"10"}
    assert result.dropped["source_missing"] == 1


NO_SOF_JPEG = b"\xff\xd8\xff\xe0" + b"\x00" * 64


def test_a_pil_fallback_jpeg_without_a_sof_is_kept_with_the_build_runs_dimensions(
    source: RecordingS3, tmp_path: Path
) -> None:
    build = _fixture_build()
    build.images.loc[build.images["image_id"] == "10", "header_status"] = "pil_fallback"
    source.put(SRC, _key(1, "10"), NO_SOF_JPEG)

    result = _build(source, tmp_path, build)

    row = result.tables["images.parquet"].to_pylist()[0]
    assert (row["image_id"], row["media_type"], row["width"], row["height"]) == (
        "10",
        "image/jpeg",
        3000,
        2000,
    )
    assert (row["etag"], row["size"]) == (md5_hex(NO_SOF_JPEG), len(NO_SOF_JPEG))


@pytest.mark.parametrize("fault", ["multipart", "no_crc"])
def test_a_whole_read_pil_fallback_jpeg_without_a_sof_is_kept_with_facts_from_its_bytes(
    source: RecordingS3, tmp_path: Path, fault: str
) -> None:
    build = _fixture_build()
    build.images.loc[build.images["image_id"] == "10", "header_status"] = "pil_fallback"
    source.put(SRC, _key(1, "10"), NO_SOF_JPEG)
    getattr(source, fault).add((SRC, _key(1, "10")))

    result = _build(source, tmp_path, build)

    row = result.tables["images.parquet"].to_pylist()[0]
    assert (row["image_id"], row["width"], row["height"]) == ("10", 3000, 2000)
    assert (row["etag"], row["size"], row["crc64nvme"]) == (
        md5_hex(NO_SOF_JPEG),
        len(NO_SOF_JPEG),
        crc64nvme(NO_SOF_JPEG),
    )
    assert _key(1, "10") in source.full_reads
    assert "image_unreadable_header" not in result.dropped


def test_a_missing_image_list_leaves_source_names_null_and_is_counted(
    source: RecordingS3, tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    del source.objects[(SRC, f"{PREFIX}/s2/image_list.csv")]

    with caplog.at_level("WARNING", logger=raw.__name__):
        result = _build(source, tmp_path)

    images = result.tables["images.parquet"].to_pylist()
    assert [(row["image_id"], row["source_name"]) for row in images] == [
        ("10", "a.jpg"),
        ("20", None),
    ]
    assert result.dropped["source_name_missing"] == 1
    assert not any(
        "absent from their image_list" in r.getMessage()
        for r in caplog.records
        if r.levelname == "WARNING"
    )


def test_an_image_absent_from_its_image_list_keeps_a_null_name_and_is_counted(
    source: RecordingS3, tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    source.put(SRC, f"{PREFIX}/s2/image_list.csv", _image_list([("c.jpg", "21")]))

    with caplog.at_level("WARNING", logger=raw.__name__):
        result = _build(source, tmp_path)

    images = result.tables["images.parquet"].to_pylist()
    assert [(row["image_id"], row["source_name"]) for row in images] == [
        ("10", "a.jpg"),
        ("20", None),
    ]
    assert result.dropped["source_name_missing"] == 1
    assert any("['20']" in r.getMessage() for r in caplog.records if r.levelname == "WARNING")


@pytest.mark.parametrize(
    ("unsupported", "kept"),
    [((2, "20"), "10"), ((1, "10"), "20")],
    ids=["last_row", "first_row"],
)
def test_an_original_neither_jpeg_nor_png_drops_the_image_and_its_points(
    source: RecordingS3, tmp_path: Path, unsupported: tuple[int, str], kept: str
) -> None:
    build = _fixture_build()
    build.images.loc[build.images["image_id"] == unsupported[1], "header_status"] = "pil_fallback"
    source.put(SRC, _key(*unsupported), b"GIF89a\x01\x00")

    result = _build(source, tmp_path, build)

    assert result.tables["images.parquet"].column("image_id").to_pylist() == [kept]
    assert set(result.tables["points.parquet"].column("image_id").to_pylist()) == {kept}
    assert result.dropped["unsupported_format"] == 1


def test_an_ok_header_status_is_taken_as_jpeg_without_a_sniff(
    source: RecordingS3, tmp_path: Path
) -> None:
    _build(source, tmp_path)

    assert source.image_gets() == [_key(2, "20")]


def test_an_interrupted_probe_resumes_from_its_checkpoint(
    source: RecordingS3, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(raw, "_CHECKPOINT_BATCH", 1)
    monkeypatch.setattr(raw, "_CHECKPOINT_EVERY", 1)
    uninterrupted = _build(RecordingS3(dict(source.objects)), tmp_path / "clean")
    source.fail_on_head = 2

    with pytest.raises(RuntimeError, match="interrupted"):
        _build(source, tmp_path)
    source.fail_on_head = None
    source.heads.clear()
    source.gets.clear()
    rerun = _build(source, tmp_path)

    assert source.image_heads() == [_key(2, "20")]
    assert source.image_gets() == [_key(2, "20")]
    assert rerun.tables == uninterrupted.tables


def test_a_checkpointed_row_whose_source_changed_is_probed_again(
    source: RecordingS3, tmp_path: Path
) -> None:
    _build(source, tmp_path)
    changed = _make_jpeg_bytes(3000, 2000) + b"changed"
    source.put(SRC, _key(1, "10"), changed)
    source.heads.clear()

    result = _build(source, tmp_path)

    assert source.image_heads() == [_key(1, "10")]
    row = result.tables["images.parquet"].to_pylist()[0]
    assert (row["etag"], row["crc64nvme"]) == (md5_hex(changed), crc64nvme(changed))


@pytest.mark.parametrize(("extra", "writes"), [(1, 2), (2, 2)], ids=["3_images", "4_images"])
def test_the_checkpoint_is_written_every_interval_and_once_after_the_last_batch(
    source: RecordingS3,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    extra: int,
    writes: int,
) -> None:
    monkeypatch.setattr(raw, "_CHECKPOINT_BATCH", 1)
    monkeypatch.setattr(raw, "_CHECKPOINT_EVERY", 2)
    written: list[int] = []
    real_write = raw._write_checkpoint

    def record(path: Path, probed: pd.DataFrame) -> None:
        written.append(len(probed))
        real_write(path, probed)

    monkeypatch.setattr(raw, "_write_checkpoint", record)
    build = _fixture_build()
    ids = [str(30 + n) for n in range(extra)]
    for image_id in ids:
        source.put(SRC, _key(1, image_id), _make_jpeg_bytes(10, 10) + image_id.encode())
    build = raw.BuildTables(
        annotations=pd.concat([build.annotations, _annotations([(1, i, 0, 0, 100) for i in ids])]),
        images=pd.concat([build.images, _images([(1, i, "ok", 10, 10) for i in ids])]),
        id2name=ID2NAME,
    )

    _build(source, tmp_path, build)

    assert len(written) == writes
    assert written[-1] == 2 + extra


def test_source_listings_follow_continuation_tokens_across_pages() -> None:
    s3 = FakeS3(page_size=2)
    source_keys = [_key(1, str(n)) for n in range(5)]
    for key in source_keys:
        s3.put(SRC, key, PNG)

    etags = raw._list_source_etags(s3, SRC, PREFIX, 1)

    assert etags == dict.fromkeys(source_keys, quoted_etag(PNG))


def test_image_list_names_parse_as_exact_strings(source: RecordingS3) -> None:
    source.put(
        SRC,
        f"{PREFIX}/s9/image_list.csv",
        (
            b"Name,Image Page\n"
            b'"a, b.jpg - Confirmed",/image/1/view/\n'
            b"NaN,/image/2/view/\n"
            b"None,/image/3/view/\n"
            b"0012.jpg,/image/4/view/\n"
            b"  padded.jpg  ,/image/5/view/\n"
            b"NA - Unconfirmed,/image/6/view/\n"
        ),
    )
    image_list = raw._string_csv_reader(source)(SRC, f"{PREFIX}/s9/image_list.csv")
    assert image_list is not None

    names = raw._source_names(image_list)

    assert dict(zip(names["image_id"], names["source_name"], strict=True)) == {
        "1": "a, b.jpg",
        "2": "NaN",
        "3": "None",
        "4": "0012.jpg",
        "5": "  padded.jpg  ",
        "6": "NA",
    }


def test_trailing_commas_on_every_row_do_not_shift_the_names_into_the_index(
    source: RecordingS3,
) -> None:
    source.put(
        SRC,
        f"{PREFIX}/s8/image_list.csv",
        b"Name,Image Page\nx.jpg,/image/1/view/,\ny.jpg,/image/2/view/,\n",
    )
    image_list = raw._string_csv_reader(source)(SRC, f"{PREFIX}/s8/image_list.csv")
    assert image_list is not None

    names = raw._source_names(image_list)

    assert dict(zip(names["image_id"], names["source_name"], strict=True)) == {
        "1": "x.jpg",
        "2": "y.jpg",
    }


# --- the staging command ------------------------------------------------------------------------


def _put_build(source: FakeS3, run: str) -> None:
    build = _fixture_build()
    base = f"etl-outputs/coralnet/{run}/coralnet_"
    for kind, frame in (("annotations", build.annotations), ("images", build.images)):
        source.put(SRC, f"{base}{kind}_{run}.parquet", frame.to_parquet(index=False))
    source.put(SRC, f"{base}id2name_{run}.json", b'{"100": "Acropora", "200": "Sand"}')


@pytest.fixture
def staging(source: RecordingS3, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> RecordingS3:
    """Point the raw command at ``source`` with a clean, pushed repo holding ``uv.lock``."""
    repo = tmp_path / "repo"
    repo.mkdir()
    (repo / "uv.lock").write_text("lock")
    monkeypatch.setenv("MERMAID_CORALNET_BUCKET", SRC)
    monkeypatch.delenv("MERMAID_CORALNET_PREFIX", raising=False)
    monkeypatch.setattr(raw, "make_s3_client", lambda **_kw: source)
    monkeypatch.setitem(sys.modules, "ibis", None)
    monkeypatch.setitem(sys.modules, "duckdb", None)
    monkeypatch.setattr(raw, "find_repo_root", lambda _start: repo)
    monkeypatch.setattr(raw, "probe_git", lambda _dir: GIT)
    _put_build(source, "run1")
    return source


def _stage_raw(stage: Path, *extra: str) -> None:
    argv = ["build", "coralnet", "raw", "--stage-dir", str(stage), "--build-run", "run1"]
    assert cli.main([*argv, "--workers", "1", *extra]) == 0


def _only_version(stage: Path) -> Path:
    (version_dir,) = (stage / DATASET_BUCKET / "datasets" / "coralnet-raw").iterdir()
    return version_dir


def test_staging_writes_the_tables_and_a_copy_plan_and_nothing_to_s3(
    staging: RecordingS3, tmp_path: Path
) -> None:
    before = dict(staging.objects)

    _stage_raw(tmp_path / "stage")

    version_dir = _only_version(tmp_path / "stage")
    assert version_dir.name.endswith("-abcdef1")
    for name in ("images", "points", "labels", "stats/label_counts"):
        assert (version_dir / f"{name}.parquet").is_file()
    manifest = Manifest.model_validate_json((version_dir / "manifest.json").read_bytes())
    assert (manifest.stats.n_images, manifest.source.revision) == (2, "run1")
    assert manifest.image_store.prefix == "images/coralnet/"
    rows = pq.read_table(version_dir / "images.parquet").to_pylist()
    lines = [json.loads(line) for line in (version_dir / "copies.jsonl").read_text().splitlines()]
    assert lines == [
        {
            "src_bucket": SRC,
            "src_key": _key(sid, image_id, "jpg"),
            "src_etag": row["source_fingerprint"],
            "dst_bucket": DATASET_BUCKET,
            "dst_key": dst,
            "etag": row["etag"],
            "size": row["size"],
            "crc64nvme": row["crc64nvme"],
        }
        for (sid, image_id, dst), row in zip(
            [(1, "10", DEST_10), (2, "20", DEST_20)], rows, strict=True
        )
    ]
    assert [line["src_etag"] for line in lines] == [md5_hex(JPEG), md5_hex(PNG)]
    assert staging.copy_calls == [] and staging.puts == []
    assert staging.objects == before


def test_the_recorded_command_names_the_stage_dir(staging: RecordingS3, tmp_path: Path) -> None:
    stage = tmp_path / "stage"

    _stage_raw(stage, "--source-ids", "1,2")

    manifest = Manifest.model_validate_json((_only_version(stage) / "manifest.json").read_bytes())
    assert manifest.etl.command == (
        f"segdata build coralnet raw --stage-dir {stage} --build-run run1 --source-ids 1,2"
    )


def test_source_ids_narrow_the_staged_version_to_those_sources(
    staging: RecordingS3, tmp_path: Path
) -> None:
    _stage_raw(tmp_path / "stage", "--source-ids", "2")

    rows = pq.read_table(_only_version(tmp_path / "stage") / "images.parquet").to_pylist()
    assert [row["image_id"] for row in rows] == ["20"]


def test_a_rerun_reads_its_checkpoint_under_the_stage_dir(
    staging: RecordingS3, tmp_path: Path
) -> None:
    stage = tmp_path / "stage"
    _stage_raw(stage)
    staging.heads.clear()

    _stage_raw(stage)

    assert (stage / "checkpoints" / "coralnet-raw-run1-abcdef1.parquet").is_file()
    assert staging.image_heads() == []
    assert len(list((stage / DATASET_BUCKET / "datasets" / "coralnet-raw").iterdir())) == 2


def test_a_staged_parent_pin_carries_unchanged_images_forward(
    staging: RecordingS3, tmp_path: Path
) -> None:
    stage = tmp_path / "stage"
    _stage_raw(stage)
    parent = _only_version(stage)
    sha = manifest_sha256((parent / "manifest.json").read_bytes())
    (stage / "checkpoints" / "coralnet-raw-run1-abcdef1.parquet").unlink()
    changed = _png(640, 480)[:-12] + b"changed-tail"
    staging.put(SRC, _key(2, "20"), changed)
    staging.gets.clear()
    staging.heads.clear()

    _stage_raw(stage, "--parent-version", parent.name, "--parent-manifest-sha256", sha)

    (child,) = [p for p in parent.parent.iterdir() if p != parent]
    manifest = Manifest.model_validate_json((child / "manifest.json").read_bytes())
    assert manifest.parents == (
        ParentRef(dataset="coralnet-raw", version=parent.name, manifest_sha256=sha),
    )
    old = pq.read_table(parent / "images.parquet").to_pylist()
    new = pq.read_table(child / "images.parquet").to_pylist()
    assert new[0] == old[0]
    assert (new[1]["etag"], new[1]["source_fingerprint"]) == (md5_hex(changed), md5_hex(changed))
    assert staging.image_heads() == [_key(2, "20")]
    assert staging.image_gets() == [_key(2, "20")]


def test_a_staged_version_promotes_into_intelligent_tiering_with_the_manifest_last(
    staging: RecordingS3, tmp_path: Path
) -> None:
    stage = tmp_path / "stage"
    _stage_raw(stage)
    version = _only_version(stage).name

    promote.promote(staging, stage_dir=stage, dataset="coralnet-raw", version=version, workers=1)

    assert sorted(call["Key"] for call in staging.copy_calls) == [DEST_10, DEST_20]
    assert {call["StorageClass"] for call in staging.copy_calls} == {"INTELLIGENT_TIERING"}
    assert staging.objects[(DATASET_BUCKET, DEST_20)] == PNG
    assert staging.puts[-1] == f"datasets/coralnet-raw/{version}/manifest.json"


@pytest.mark.parametrize(
    ("git", "message"),
    [
        (GitState(sha="abcdef1234567", dirty=False, pushed=False), "not contained in any remote"),
        (GitState(sha="abcdef1234567", dirty=True, pushed=True), "uncommitted changes"),
    ],
)
def test_an_unpublishable_tree_is_refused_before_any_s3_call(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, git: GitState, message: str
) -> None:
    monkeypatch.setenv("MERMAID_CORALNET_BUCKET", SRC)
    monkeypatch.setattr(raw, "find_repo_root", lambda _start: tmp_path)
    monkeypatch.setattr(raw, "probe_git", lambda _dir: git)

    def no_s3(**_kw: object) -> None:
        raise AssertionError("the guard must fire before an S3 client is made")

    monkeypatch.setattr(raw, "make_s3_client", no_s3)
    argv = ["build", "coralnet", "raw", "--stage-dir", str(tmp_path / "stage")]

    with pytest.raises(PublishRefused, match=message):
        cli.main(argv)


@pytest.mark.parametrize(
    ("flags", "message"),
    [
        ([], "--stage-dir"),
        (["--stage-dir", "{tmp}", "--parent-version", "v1"], "--parent-manifest-sha256"),
    ],
    ids=["no_stage_dir", "half_parent_pin"],
)
def test_raw_flag_misuse_is_a_usage_error(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
    flags: list[str],
    message: str,
) -> None:
    def no_aws(**_kw: object) -> None:
        raise AssertionError("a usage error must not reach S3")

    monkeypatch.setattr(raw, "make_s3_client", no_aws)
    argv = ["build", "coralnet", "raw", *(flag.format(tmp=tmp_path) for flag in flags)]

    with pytest.raises(SystemExit) as exit_info:
        cli.main(argv)

    assert exit_info.value.code == 2
    assert message in capsys.readouterr().err


def test_a_missing_bucket_env_var_refuses_before_any_s3_client_is_made(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.delenv("MERMAID_CORALNET_BUCKET", raising=False)
    monkeypatch.setattr(raw, "find_repo_root", lambda _start: tmp_path)
    monkeypatch.setattr(raw, "probe_git", lambda _dir: GIT)

    def no_s3(**_kw: object) -> None:
        raise AssertionError("an unset bucket must be refused before an S3 client is made")

    monkeypatch.setattr(raw, "make_s3_client", no_s3)
    args = argparse.Namespace(
        stage_dir=tmp_path / "stage",
        build_run="run1",
        source_ids=[1],
        parent_version=None,
        parent_manifest_sha256=None,
        workers=1,
    )

    with pytest.raises(ValueError, match="MERMAID_CORALNET_BUCKET"):
        raw.run_raw(args)
