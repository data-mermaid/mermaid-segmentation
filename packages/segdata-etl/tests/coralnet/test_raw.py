"""``segdata build coralnet raw``: the frozen CoralNet build as a ``coralnet-raw`` version."""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path
from typing import Any

import pandas as pd
import pyarrow as pa
import pytest
from s3_fake import FakeS3, quoted_etag

from segdata_etl import cli
from segdata_etl.common import publishing
from segdata_etl.common.local_store import LocalStore
from segdata_etl.datasets.coralnet import raw
from segdata_etl.datasets.coralnet.io import make_csv_reader_s3
from segdata_spec import DATASET_BUCKET
from segdata_spec.errors import ObjectExists, PublishRefused
from segdata_spec.manifest import Manifest
from segdata_spec.publish import GitState

SRC = "src"
DST = "dst"
PREFIX = "coralnet-public-images"
JPEG = b"\xff\xd8\xff\xe0raw-jpeg"
PNG = b"\x89PNG\r\n\x1a\nraw-png"
# MD5 hex and S3's base64 CRC64NVME of JPEG and PNG, computed once by hand.
JPEG_MD5 = "079d0088636f564bd6155ecf97ed218a"
JPEG_CRC = "WGQed5e816A="
PNG_MD5 = "e2fd60f691f0c1aff8e668983033eeac"
PNG_CRC = "2PyFTsw6muE="
DEST_10 = "images/coralnet/s1/10.jpg"
MULTIPART_10 = f'"{JPEG_MD5}-2"'


def _key(sid: int, image_id: str) -> str:
    return f"{PREFIX}/s{sid}/images/{image_id}.jpg"


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
def source(fake_s3: FakeS3) -> FakeS3:
    fake_s3.put(SRC, _key(1, "10"), JPEG)
    fake_s3.put(SRC, _key(2, "20"), PNG)
    fake_s3.put(SRC, _key(1, "12"), JPEG)
    fake_s3.put(SRC, f"{PREFIX}/s1/image_list.csv", _image_list([("a.jpg - Confirmed", "10")]))
    fake_s3.put(
        SRC,
        f"{PREFIX}/s2/image_list.csv",
        _image_list([("b.png - Unconfirmed", "20"), ("c.jpg", "21")]),
    )
    return fake_s3


class RecordingStore:
    """A ``LocalStore`` that records every call as ``(method, kwargs)``."""

    def __init__(self, inner: LocalStore):
        self._inner = inner
        self.calls: list[tuple[str, dict[str, Any]]] = []

    def __getattr__(self, name: str) -> Any:
        method = getattr(self._inner, name)

        def record(*args: Any, **kwargs: Any) -> Any:
            self.calls.append((name, kwargs))
            return method(*args, **kwargs)

        return record

    def keys_of(self, method: str) -> list[str]:
        return [kwargs["Key"] for name, kwargs in self.calls if name == method]


def _store(source: FakeS3, tmp_path: Path) -> RecordingStore:
    return RecordingStore(LocalStore(tmp_path / "store", source_client=source))


def _build(
    source: FakeS3,
    tmp_path: Path,
    build: raw.BuildTables | None = None,
    parent_images: pa.Table | None = None,
    store: RecordingStore | None = None,
) -> raw.RawTables:
    return raw.build_raw(
        source,
        store or _store(source, tmp_path),
        build or _fixture_build(),
        source_bucket=SRC,
        prefix=PREFIX,
        dest_bucket=DST,
        workers=1,
        checkpoint=tmp_path / "checkpoint.parquet",
        parent_images=parent_images,
        csv_reader=make_csv_reader_s3(source),
    )


def _copied(source: FakeS3) -> list[str]:
    return sorted(key for key in source.full_reads if "/images/" in key)


def test_images_keep_annotated_images_with_dimensions(source: FakeS3, tmp_path: Path) -> None:
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
    assert [row["source_fingerprint"] for row in images] == [f'"{JPEG_MD5}"', f'"{PNG_MD5}"']
    assert [row["source_name"] for row in images] == ["a.jpg", "b.png"]
    assert images[0]["attrs"] == (
        '{"source_id": 1, "source_key": "coralnet-public-images/s1/images/10.jpg"}'
    )
    assert all(row["mask_key"] is None for row in images)
    assert result.dropped["no_dimensions"] == 1
    assert result.dropped["no_annotations"] == 1


def test_points_keep_original_row_col_of_kept_images(source: FakeS3, tmp_path: Path) -> None:
    result = _build(source, tmp_path)

    assert result.tables["points.parquet"].to_pylist() == [
        {"image_id": "10", "row": 5, "col": 7, "label_id": 100, "annotator": "default"},
        {"image_id": "10", "row": 8, "col": 9, "label_id": 100, "annotator": "default"},
        {"image_id": "10", "row": 1500, "col": 2900, "label_id": 200, "annotator": "default"},
        {"image_id": "20", "row": 1, "col": 2, "label_id": 200, "annotator": "default"},
    ]


def test_label_counts_count_points_per_image_and_label(source: FakeS3, tmp_path: Path) -> None:
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


def test_staged_images_carry_the_copied_objects_fingerprints(
    source: FakeS3, tmp_path: Path
) -> None:
    store = _store(source, tmp_path)
    result = _build(source, tmp_path, store=store)

    images = result.tables["images.parquet"].to_pylist()
    assert [(row["etag"], row["size"], row["crc64nvme"]) for row in images] == [
        (f'"{JPEG_MD5}"', 12, JPEG_CRC),
        (f'"{PNG_MD5}"', 15, PNG_CRC),
    ]
    assert (tmp_path / "store" / DST / "images/coralnet/s2/20.png").read_bytes() == PNG
    storage_classes = [kw["StorageClass"] for name, kw in store.calls if name == "copy_object"]
    assert storage_classes == ["INTELLIGENT_TIERING", "INTELLIGENT_TIERING"]


def _parent(rows: list[tuple[str, str, str]]) -> pa.Table:
    return pa.Table.from_pylist(
        [
            {
                "image_id": image_id,
                "key": key,
                "etag": '"parent-etag"',
                "size": 99,
                "crc64nvme": "parent-crc",
                "media_type": "image/png" if key.endswith(".png") else "image/jpeg",
                "source_fingerprint": fingerprint,
            }
            for image_id, key, fingerprint in rows
        ]
    )


def test_parent_rows_with_the_same_fingerprint_are_carried_not_copied(
    source: FakeS3, tmp_path: Path
) -> None:
    build = _fixture_build()
    build = raw.BuildTables(
        annotations=pd.concat([build.annotations, _annotations([(1, "11", 0, 0, 100)])]),
        images=pd.concat([build.images, _images([(1, "11", "ok", 10, 10)])]),
        id2name=ID2NAME,
    )
    source.put(SRC, _key(1, "11"), b"\xff\xd8\xff\xe0new-jpeg")
    parent = _parent([("10", "s1/10.jpg", f'"{JPEG_MD5}"'), ("20", "s2/20.png", '"stale"')])

    result = _build(source, tmp_path, build, parent_images=parent)

    images = {row["image_id"]: row for row in result.tables["images.parquet"].to_pylist()}
    assert (images["10"]["etag"], images["10"]["crc64nvme"]) == ('"parent-etag"', "parent-crc")
    assert images["20"]["crc64nvme"] == PNG_CRC
    assert images["11"]["crc64nvme"] == "P2mIxoSjo3Q="
    assert _copied(source) == [_key(1, "11"), _key(2, "20")]


def test_carried_rows_keep_the_parents_media_type_unsniffed(source: FakeS3, tmp_path: Path) -> None:
    source.put(SRC, _key(2, "20"), b"GIF89a\x01\x00")
    parent = _parent([("20", "s2/20.png", source.etag(SRC, _key(2, "20")))])

    result = _build(source, tmp_path, parent_images=parent)

    images = {row["image_id"]: row for row in result.tables["images.parquet"].to_pylist()}
    assert (images["20"]["media_type"], images["20"]["key"]) == ("image/png", "s2/20.png")
    assert result.dropped["unsupported_format"] == 0


def test_annotations_without_an_image_row_are_counted_and_dropped(
    source: FakeS3, tmp_path: Path
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
    source: FakeS3, tmp_path: Path
) -> None:
    del source.objects[(SRC, _key(2, "20"))]

    result = _build(source, tmp_path)

    assert result.tables["images.parquet"].column("image_id").to_pylist() == ["10"]
    assert set(result.tables["points.parquet"].column("image_id").to_pylist()) == {"10"}
    assert result.dropped["source_missing"] == 1


def test_a_missing_image_list_leaves_source_names_null_and_is_counted(
    source: FakeS3, tmp_path: Path, caplog: pytest.LogCaptureFixture
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
    source: FakeS3, tmp_path: Path, caplog: pytest.LogCaptureFixture
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
    source: FakeS3, tmp_path: Path, unsupported: tuple[int, str], kept: str
) -> None:
    build = _fixture_build()
    build.images.loc[build.images["image_id"] == unsupported[1], "header_status"] = "pil_fallback"
    source.put(SRC, _key(*unsupported), b"GIF89a\x01\x00")

    result = _build(source, tmp_path, build)

    assert result.tables["images.parquet"].column("image_id").to_pylist() == [kept]
    assert set(result.tables["points.parquet"].column("image_id").to_pylist()) == {kept}
    assert result.dropped["unsupported_format"] == 1


def _checkpoint(tmp_path: Path, fingerprint: str, crc: str, etag: str | None = None) -> None:
    pd.DataFrame(
        [
            {
                "dest_key": DEST_10,
                "source_fingerprint": fingerprint,
                "etag": fingerprint if etag is None else etag,
                "size": 12,
                "crc64nvme": crc,
            }
        ]
    ).to_parquet(tmp_path / "checkpoint.parquet")


def _put_dest(tmp_path: Path, key: str, data: bytes) -> None:
    LocalStore(tmp_path / "store").put_object(Bucket=DST, Key=key, Body=data)


@pytest.mark.parametrize(
    ("checkpoint_crc", "expected_crc"),
    [("checkpoint-crc", "checkpoint-crc"), (None, JPEG_CRC)],
)
def test_a_rerun_skips_keys_the_destination_holds_with_the_source_etag(
    source: FakeS3, tmp_path: Path, checkpoint_crc: str | None, expected_crc: str
) -> None:
    _put_dest(tmp_path, DEST_10, JPEG)
    if checkpoint_crc is not None:
        _checkpoint(tmp_path, f'"{JPEG_MD5}"', checkpoint_crc)
    store = _store(source, tmp_path)

    result = _build(source, tmp_path, store=store)

    assert _copied(source) == [_key(2, "20")]
    assert DEST_10 not in store.keys_of("copy_object") + store.keys_of("get_object")
    images = result.tables["images.parquet"].to_pylist()
    assert [(row["etag"], row["size"], row["crc64nvme"]) for row in images] == [
        (f'"{JPEG_MD5}"', 12, expected_crc),
        (f'"{PNG_MD5}"', 15, PNG_CRC),
    ]


def _row_10(result: raw.RawTables) -> tuple[str, int, str, str]:
    (row,) = [r for r in result.tables["images.parquet"].to_pylist() if r["image_id"] == "10"]
    return row["etag"], row["size"], row["crc64nvme"], row["source_fingerprint"]


def test_a_checkpoint_row_with_a_stale_etag_gives_no_crc(source: FakeS3, tmp_path: Path) -> None:
    _put_dest(tmp_path, DEST_10, JPEG)
    _checkpoint(tmp_path, f'"{JPEG_MD5}"', "stale-crc", etag='"stale"')
    store = _store(source, tmp_path)

    result = _build(source, tmp_path, store=store)

    assert DEST_10 not in store.keys_of("copy_object")
    assert _row_10(result) == (f'"{JPEG_MD5}"', 12, JPEG_CRC, f'"{JPEG_MD5}"')


def test_a_rerun_skips_a_multipart_original_its_checkpoint_recorded(
    source: FakeS3, tmp_path: Path
) -> None:
    source.multipart.add((SRC, _key(1, "10")))
    _put_dest(tmp_path, DEST_10, JPEG)
    _checkpoint(tmp_path, MULTIPART_10, "checkpoint-crc", etag=f'"{JPEG_MD5}"')
    store = _store(source, tmp_path)

    result = _build(source, tmp_path, store=store)

    touched = store.keys_of("copy_object") + store.keys_of("get_object")
    assert DEST_10 not in touched + store.keys_of("head_object")
    assert _row_10(result) == (f'"{JPEG_MD5}"', 12, "checkpoint-crc", MULTIPART_10)


@pytest.mark.parametrize("checkpointed_fingerprint", [None, '"0ther-2"'])
def test_a_multipart_original_the_destination_holds_unrecorded_is_adopted_by_the_copy(
    source: FakeS3, tmp_path: Path, checkpointed_fingerprint: str | None
) -> None:
    source.multipart.add((SRC, _key(1, "10")))
    _put_dest(tmp_path, DEST_10, JPEG)
    if checkpointed_fingerprint is not None:
        _checkpoint(tmp_path, checkpointed_fingerprint, "stale-crc", etag=f'"{JPEG_MD5}"')
    store = _store(source, tmp_path)

    result = _build(source, tmp_path, store=store)

    assert DEST_10 in store.keys_of("copy_object")
    assert _row_10(result) == (f'"{JPEG_MD5}"', 12, JPEG_CRC, MULTIPART_10)


def test_a_checkpointed_key_whose_object_is_gone_is_copied_again(
    source: FakeS3, tmp_path: Path
) -> None:
    _checkpoint(tmp_path, f'"{JPEG_MD5}"', JPEG_CRC)

    result = _build(source, tmp_path)

    assert _copied(source) == [_key(1, "10"), _key(2, "20")]
    assert (tmp_path / "store" / DST / DEST_10).read_bytes() == JPEG
    assert result.tables["images.parquet"].column("crc64nvme").to_pylist() == [JPEG_CRC, PNG_CRC]


def test_a_destination_key_holding_other_bytes_raises_before_any_copy(
    source: FakeS3, tmp_path: Path
) -> None:
    _put_dest(tmp_path, DEST_10, b"stale")
    _checkpoint(tmp_path, '"stale-fingerprint"', "stale-crc")
    store = _store(source, tmp_path)

    with pytest.raises(ObjectExists, match=DEST_10):
        _build(source, tmp_path, store=store)

    assert store.keys_of("copy_object") == []


def _put_build(source: FakeS3, run: str) -> None:
    build = _fixture_build()
    base = f"etl-outputs/coralnet/{run}/coralnet_"
    for kind, frame in (("annotations", build.annotations), ("images", build.images)):
        source.put(SRC, f"{base}{kind}_{run}.parquet", frame.to_parquet(index=False))
    source.put(SRC, f"{base}id2name_{run}.json", b'{"100": "Acropora", "200": "Sand"}')


def test_cli_publishes_a_coralnet_raw_version_into_the_local_store(
    source: FakeS3, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    repo = tmp_path / "repo"
    repo.mkdir()
    (repo / "uv.lock").write_text("lock")
    monkeypatch.setenv("MERMAID_CORALNET_BUCKET", SRC)
    monkeypatch.delenv("MERMAID_CORALNET_PREFIX", raising=False)
    monkeypatch.setattr(raw, "make_s3_client", lambda **_kw: source)
    monkeypatch.setitem(sys.modules, "ibis", None)
    monkeypatch.setitem(sys.modules, "duckdb", None)
    monkeypatch.setattr(raw, "find_repo_root", lambda _start: repo)
    monkeypatch.setattr(
        publishing,
        "probe_git",
        lambda _dir: GitState(sha="abcdef1234567", dirty=False, pushed=True),
    )
    _put_build(source, "run1")
    store = tmp_path / "store"

    argv = ["build", "coralnet", "raw", "--store", str(store), "--build-run", "run1"]
    assert cli.main([*argv, "--source-ids", "1,2", "--workers", "1"]) == 0

    (version_dir,) = (store / DATASET_BUCKET / "datasets" / "coralnet-raw").iterdir()
    manifest = Manifest.model_validate_json((version_dir / "manifest.json").read_bytes())
    assert manifest.parents == ()
    assert manifest.label_mapping is None
    assert manifest.stats.n_images == 2
    assert manifest.source.revision == "run1"


@pytest.mark.parametrize(
    ("flags", "message"),
    [
        (["--store", "{store}"], "--source-ids"),
        (["--store", "s3", "--source-ids", "1"], "--source-ids"),
        (
            ["--store", "{store}", "--source-ids", "1", "--parent-version", "v1"],
            "--parent-manifest-sha256",
        ),
    ],
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
    argv = ["build", "coralnet", "raw", *(flag.format(store=tmp_path) for flag in flags)]

    with pytest.raises(SystemExit) as exit_info:
        cli.main(argv)

    assert exit_info.value.code == 2
    assert message in capsys.readouterr().err


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
    repo = tmp_path / "repo"
    repo.mkdir()
    (repo / "uv.lock").write_text("lock")
    monkeypatch.setenv("MERMAID_CORALNET_BUCKET", SRC)
    monkeypatch.setattr(raw, "find_repo_root", lambda _start: repo)
    monkeypatch.setattr(publishing, "probe_git", lambda _dir: git)

    def no_s3(**_kw: object) -> None:
        raise AssertionError("the guard must fire before an S3 client is made")

    monkeypatch.setattr(raw, "make_s3_client", no_s3)
    argv = ["build", "coralnet", "raw", "--store", str(tmp_path / "store"), "--source-ids", "1"]

    with pytest.raises(PublishRefused, match=message):
        cli.main(argv)


def test_a_tree_without_a_lock_file_is_refused_before_any_s3_call(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    repo = tmp_path / "repo"
    repo.mkdir()
    monkeypatch.setenv("MERMAID_CORALNET_BUCKET", SRC)
    monkeypatch.setattr(raw, "find_repo_root", lambda _start: repo)
    monkeypatch.setattr(
        publishing,
        "probe_git",
        lambda _dir: GitState(sha="abcdef1234567", dirty=False, pushed=True),
    )

    def no_s3(**_kw: object) -> None:
        raise AssertionError("the guard must fire before an S3 client is made")

    monkeypatch.setattr(raw, "make_s3_client", no_s3)
    argv = ["build", "coralnet", "raw", "--store", str(tmp_path / "store"), "--source-ids", "1"]

    with pytest.raises(PublishRefused, match="no uv.lock"):
        cli.main(argv)


def test_image_list_names_parse_as_exact_strings(source: FakeS3) -> None:
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
    source: FakeS3,
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


def _build_with_extra_images(source: FakeS3, extra: int) -> raw.BuildTables:
    """The fixture build plus ``extra`` annotated source-1 JPEGs, each with distinct bytes."""
    build = _fixture_build()
    ids = [str(30 + n) for n in range(extra)]
    for image_id in ids:
        source.put(SRC, _key(1, image_id), b"\xff\xd8\xff\xe0extra-" + image_id.encode())
    return raw.BuildTables(
        annotations=pd.concat([build.annotations, _annotations([(1, i, 0, 0, 100) for i in ids])]),
        images=pd.concat([build.images, _images([(1, i, "ok", 10, 10) for i in ids])]),
        id2name=ID2NAME,
    )


def _read_checkpoint_rows(tmp_path: Path) -> list[dict[str, Any]]:
    frame = pd.read_parquet(tmp_path / "checkpoint.parquet").sort_values("dest_key")
    return frame.to_dict("records")


def test_the_checkpoint_file_holds_one_row_per_copied_key(source: FakeS3, tmp_path: Path) -> None:
    _build(source, tmp_path)

    assert _read_checkpoint_rows(tmp_path) == [
        {
            "dest_key": "images/coralnet/s1/10.jpg",
            "source_fingerprint": f'"{JPEG_MD5}"',
            "etag": f'"{JPEG_MD5}"',
            "size": 12,
            "crc64nvme": JPEG_CRC,
        },
        {
            "dest_key": "images/coralnet/s2/20.png",
            "source_fingerprint": f'"{PNG_MD5}"',
            "etag": f'"{PNG_MD5}"',
            "size": 15,
            "crc64nvme": PNG_CRC,
        },
    ]


class _SecondCopyFails(RecordingStore):
    """A store whose second ``copy_object`` raises, as a crash mid-run does."""

    def __init__(self, inner: LocalStore):
        super().__init__(inner)
        self.copies = 0

    def copy_object(self, **kwargs: Any) -> Any:
        self.copies += 1
        if self.copies == 2:
            raise RuntimeError("interrupted")
        return self._inner.copy_object(**kwargs)


def test_a_crash_between_checkpoint_writes_leaves_the_rerun_to_adopt_the_unrecorded_copy(
    source: FakeS3, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(raw, "_CHECKPOINT_BATCH", 1)
    monkeypatch.setattr(raw, "_CHECKPOINT_EVERY", 2)
    source.multipart.add((SRC, _key(1, "10")))
    failing = _SecondCopyFails(LocalStore(tmp_path / "store", source_client=source))

    with pytest.raises(RuntimeError, match="interrupted"):
        _build(source, tmp_path, store=failing)

    # The first copy finished, but the crash came before the second batch's checkpoint write.
    assert (tmp_path / "store" / DST / DEST_10).read_bytes() == JPEG
    assert not (tmp_path / "checkpoint.parquet").exists()

    rerun = _store(source, tmp_path)
    result = _build(source, tmp_path, store=rerun)

    assert rerun.keys_of("copy_object") == [DEST_10, "images/coralnet/s2/20.png"]
    assert _row_10(result) == (f'"{JPEG_MD5}"', 12, JPEG_CRC, MULTIPART_10)
    assert result.tables["images.parquet"].column("crc64nvme").to_pylist() == [JPEG_CRC, PNG_CRC]
    assert [
        (r["dest_key"], r["etag"], r["size"], r["crc64nvme"])
        for r in _read_checkpoint_rows(tmp_path)
    ] == [
        (DEST_10, f'"{JPEG_MD5}"', 12, JPEG_CRC),
        ("images/coralnet/s2/20.png", f'"{PNG_MD5}"', 15, PNG_CRC),
    ]


@pytest.mark.parametrize(("extra", "writes"), [(1, 2), (2, 2)], ids=["3_images", "4_images"])
def test_the_checkpoint_is_written_every_interval_and_once_after_the_last_batch(
    source: FakeS3,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    extra: int,
    writes: int,
) -> None:
    monkeypatch.setattr(raw, "_CHECKPOINT_BATCH", 1)
    monkeypatch.setattr(raw, "_CHECKPOINT_EVERY", 2)
    written: list[int] = []
    real_write = raw._write_checkpoint

    def record(path: Path, finished: pd.DataFrame) -> None:
        written.append(len(finished))
        real_write(path, finished)

    monkeypatch.setattr(raw, "_write_checkpoint", record)

    _build(source, tmp_path, _build_with_extra_images(source, extra))

    assert len(written) == writes
    assert written[-1] == 2 + extra


def test_listings_follow_continuation_tokens_across_pages() -> None:
    s3 = FakeS3(page_size=2)
    dest_keys = [f"{raw.IMAGE_PREFIX}s1/{n}.jpg" for n in range(5)]
    source_keys = [_key(1, str(n)) for n in range(5)]
    for key in dest_keys:
        s3.put(DST, key, JPEG)
    for key in source_keys:
        s3.put(SRC, key, PNG)

    held = raw._list_dest_objects(s3, DST)
    etags = raw._list_source_etags(s3, SRC, PREFIX, 1)

    assert held["dest_key"].tolist() == dest_keys
    assert etags == dict.fromkeys(source_keys, quoted_etag(PNG))
    assert held["etag"].tolist() == [quoted_etag(JPEG)] * 5


class _DestinationRaced(FakeS3):
    """A FakeS3 that creates the destination key with other bytes once the image listing ends."""

    def list_objects_v2(self, **kwargs: Any) -> dict[str, Any]:
        response = super().list_objects_v2(**kwargs)
        if kwargs.get("Prefix") == raw.IMAGE_PREFIX:
            self.put(DST, DEST_10, b"other")
        return response


def test_a_key_created_after_the_listing_with_other_bytes_fails_the_copy_as_object_exists(
    source: FakeS3, tmp_path: Path
) -> None:
    racing = _DestinationRaced(objects=source.objects)

    with pytest.raises(ObjectExists, match=DEST_10):
        _build(racing, tmp_path, store=racing)  # type: ignore[arg-type]


def test_a_missing_bucket_env_var_refuses_before_any_s3_client_is_made(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    repo = tmp_path / "repo"
    repo.mkdir()
    (repo / "uv.lock").write_text("lock")
    monkeypatch.delenv("MERMAID_CORALNET_BUCKET", raising=False)
    monkeypatch.setattr(raw, "find_repo_root", lambda _start: repo)
    monkeypatch.setattr(
        publishing,
        "probe_git",
        lambda _dir: GitState(sha="abcdef1234567", dirty=False, pushed=True),
    )

    def no_s3(**_kw: object) -> None:
        raise AssertionError("an unset bucket must be refused before an S3 client is made")

    monkeypatch.setattr(raw, "make_s3_client", no_s3)
    args = argparse.Namespace(
        store=str(tmp_path / "store"),
        build_run="run1",
        source_ids=[1],
        parent_version=None,
        workers=1,
    )

    with pytest.raises(ValueError, match="MERMAID_CORALNET_BUCKET"):
        raw.run_raw(args)
