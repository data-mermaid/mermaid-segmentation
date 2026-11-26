"""Tests for ``segdata build mermaid``: the staged uniform-format MERMAID points dataset."""

from __future__ import annotations

import hashlib
import io
import json
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import pytest
from common.test_featurevector import featurevector_bytes
from s3_fake import FakeS3, crc64nvme, quoted_etag

from segdata_etl import cli
from segdata_etl.common.featurevector import FeatureVectorsInvalid, FeatureVectorsMissing
from segdata_etl.datasets import mermaid
from segdata_spec import DATASET_BUCKET
from segdata_spec.errors import PublishRefused
from segdata_spec.git import GitState
from segdata_spec.schema import LABELS_SCHEMA

from .conftest import jpeg_bytes, png_bytes

SRC_BUCKET = "coral-reef-training"
PARQUET_KEY = "mermaid/mermaid_confirmed_annotations.parquet"
CUTOFF = "2026-08-12"
EXTRACTOR = {
    "class_path": "spacer.EfficientNet",
    "crop_size": 224,
    "feature_dim": 2,
    "weights_uri": "s3://w/effnet.pt",
    "weights_sha256": "0" * 64,
}
BEFORE = datetime(2026, 8, 1, tzinfo=UTC)

_ANNOTATION_SCHEMA = pa.schema(
    [
        ("id", pa.string()),
        ("image_id", pa.string()),
        ("point_id", pa.string()),
        ("row", pa.int64()),
        ("col", pa.int64()),
        ("benthic_attribute_id", pa.string()),
        ("benthic_attribute_name", pa.string()),
        ("growth_form_id", pa.string()),
        ("growth_form_name", pa.string()),
        ("updated_on", pa.timestamp("us", tz="UTC")),
        ("region_id", pa.string()),
        ("region_name", pa.string()),
    ]
)

# Sorted by (ba_id, gf_id or ""): Rubble, Encrusting, Branching; the name order differs.
RUBBLE = ("aaa", "Rubble", None, None)
ENCRUSTING = ("bbb", "Hard coral", "g2", "Encrusting")
BRANCHING = ("bbb", "Hard coral", "g9", "Branching")


def ann(
    image_id: str,
    point_id: str,
    row: int,
    col: int,
    label: tuple[str, str, str | None, str | None] = RUBBLE,
    *,
    updated_on: datetime = BEFORE,
    region: tuple[str, str] = ("r1", "Caribbean"),
) -> dict[str, Any]:
    ba_id, ba_name, gf_id, gf_name = label
    return {
        "id": f"ann-{point_id}",
        "image_id": image_id,
        "point_id": point_id,
        "row": row,
        "col": col,
        "benthic_attribute_id": ba_id,
        "benthic_attribute_name": ba_name,
        "growth_form_id": gf_id,
        "growth_form_name": gf_name,
        "updated_on": updated_on,
        "region_id": region[0],
        "region_name": region[1],
    }


def parquet_bytes(rows: list[dict[str, Any]]) -> bytes:
    sink = io.BytesIO()
    pq.write_table(pa.Table.from_pylist(rows, schema=_ANNOTATION_SCHEMA), sink)
    return sink.getvalue()


def feature_source_key(image_id: str) -> str:
    return f"mermaid/{image_id}_featurevector"


def seed(fake: FakeS3, rows: list[dict[str, Any]], images: dict[str, bytes]) -> None:
    """Seed the annotations, the images, and a vector covering every annotated image's points."""
    fake.put(SRC_BUCKET, PARQUET_KEY, parquet_bytes(rows))
    for image_id, data in images.items():
        fake.put(SRC_BUCKET, f"mermaid/{image_id}.png", data)
    points: dict[str, list[tuple[int, int]]] = {}
    for row in rows:
        points.setdefault(row["image_id"], []).append((row["row"], row["col"]))
    for image_id, image_points in points.items():
        fake.put(SRC_BUCKET, feature_source_key(image_id), featurevector_bytes(image_points))


@pytest.fixture
def repo(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    root = tmp_path / "repo"
    root.mkdir()
    (root / "uv.lock").write_bytes(b"lock")
    monkeypatch.setattr(mermaid, "REPO_ROOT", root)
    git = GitState(sha="abcdef1234567", dirty=False, pushed=True)
    monkeypatch.setattr(mermaid, "probe_git", lambda _: git)
    return root


def extractor_spec(directory: Path) -> Path:
    path = directory / "extractor.json"
    path.write_text(json.dumps(EXTRACTOR))
    return path


def build(fake: FakeS3, stage: Path, *extra: str, cutoff: str = CUTOFF, workers: str = "2") -> int:
    argv = [
        "build",
        "mermaid",
        "--stage-dir",
        str(stage),
        "--annotation-date-cutoff",
        cutoff,
        "--workers",
        workers,
        "--extractor-spec",
        str(extractor_spec(stage.parent)),
        *extra,
    ]
    args = cli.build_parser().parse_args(argv)
    return int(args.run(args, fake))


def version_dirs(stage: Path) -> list[Path]:
    root = stage / DATASET_BUCKET / "datasets" / "mermaid"
    return sorted(root.iterdir()) if root.exists() else []


def only_version(stage: Path) -> Path:
    (path,) = version_dirs(stage)
    return path


def table(version: Path, name: str) -> list[dict[str, Any]]:
    return pq.read_table(version / name).to_pylist()


def copies(version: Path) -> list[dict[str, Any]]:
    lines = (version / "copies.jsonl").read_text().splitlines()
    return [json.loads(line) for line in lines]


def test_end_to_end_stages_images_points_and_copy_plan(repo, tmp_path):
    fake = FakeS3()
    blobs = {"i1": png_bytes(8, 6), "i2": png_bytes(12, 9), "i3": png_bytes(5, 5)}
    rows = [
        ann("i1", "p1", 10, 20, ENCRUSTING),
        ann("i1", "p2", 30, 40, RUBBLE),
        ann("i2", "p3", 1, 2, RUBBLE, region=("r2", "Pacific")),
        ann("i3", "p4", 7, 8, ENCRUSTING),
    ]
    seed(fake, rows, blobs)
    stage = tmp_path / "stage"

    assert build(fake, stage) == 0

    version = only_version(stage)
    images = {r["image_id"]: r for r in table(version, "images.parquet")}
    assert set(images) == {"i1", "i2", "i3"}
    i1 = images["i1"]
    assert i1["key"] == "i1.png"
    assert i1["etag"] == quoted_etag(blobs["i1"]).strip('"')
    assert i1["size"] == len(blobs["i1"])
    assert i1["crc64nvme"] == crc64nvme(blobs["i1"])
    assert i1["source_fingerprint"] == i1["etag"]
    assert (i1["media_type"], i1["width"], i1["height"]) == ("image/png", 8, 6)
    assert i1["group"] == "mermaid"
    assert i1["mask_key"] is None
    assert i1["source_name"] is None
    assert json.loads(i1["attrs"]) == {
        "source_key": "mermaid/i1.png",
        "region_id": "r1",
        "region_name": "Caribbean",
    }
    assert json.loads(images["i2"]["attrs"])["region_name"] == "Pacific"

    points = sorted(table(version, "points.parquet"), key=lambda p: (p["image_id"], p["row"]))
    assert [(p["image_id"], p["row"], p["col"], p["label_id"], p["annotator"]) for p in points] == [
        ("i1", 10, 20, 2, "default"),
        ("i1", 30, 40, 1, "default"),
        ("i2", 1, 2, 1, "default"),
        ("i3", 7, 8, 2, "default"),
    ]
    counts = sorted(
        table(version, "stats/label_counts.parquet"), key=lambda c: (c["image_id"], c["label_id"])
    )
    assert [(c["image_id"], c["label_id"], c["n"]) for c in counts] == [
        ("i1", 1, 1),
        ("i1", 2, 1),
        ("i2", 1, 1),
        ("i3", 2, 1),
    ]

    plan = {c["src_key"]: c for c in copies(version)}
    assert len(plan) == 3
    assert plan["mermaid/i1.png"] == {
        "src_bucket": SRC_BUCKET,
        "src_key": "mermaid/i1.png",
        "src_etag": i1["etag"],
        "etag": i1["etag"],
        "dst_bucket": DATASET_BUCKET,
        "dst_key": "images/mermaid/i1.png",
        "size": len(blobs["i1"]),
        "crc64nvme": crc64nvme(blobs["i1"]),
    }
    manifest = json.loads((version / "manifest.json").read_bytes())
    assert "copies.jsonl" not in manifest["files"]
    assert manifest["image_store"] == {
        "bucket": DATASET_BUCKET,
        "prefix": "images/mermaid/",
        "owner": "segdata",
    }
    assert manifest["source"]["type"] == "mermaid-confirmed-annotations"
    assert manifest["source"]["uri"] == f"s3://{SRC_BUCKET}/{PARQUET_KEY}"
    assert json.loads(manifest["source"]["revision"])["cutoff"] == CUTOFF
    assert manifest["etl"]["command"].startswith("segdata build mermaid --stage-dir ")
    assert fake.ops("put") == [] and fake.ops("copy") == []


def test_jpeg_bytes_under_png_key_get_jpg_key_and_jpeg_media_type(repo, tmp_path):
    fake = FakeS3()
    seed(fake, [ann("j1", "p1", 1, 1)], {"j1": jpeg_bytes(16, 12)})
    stage = tmp_path / "stage"

    build(fake, stage)

    version = only_version(stage)
    (row,) = table(version, "images.parquet")
    assert (row["key"], row["media_type"], row["width"], row["height"]) == (
        "j1.jpg",
        "image/jpeg",
        16,
        12,
    )
    (plan,) = copies(version)
    assert plan["src_key"] == "mermaid/j1.png"
    assert plan["dst_key"] == "images/mermaid/j1.jpg"


def test_multipart_source_etag_records_the_copied_objects_md5_and_keeps_source_fingerprint(
    repo, tmp_path
):
    fake = FakeS3()
    data = png_bytes(9, 7)
    seed(fake, [ann("m1", "p1", 1, 1)], {"m1": data})
    fake.put(SRC_BUCKET, "mermaid/m1.png", data, etag='"b68d19ab-2"')
    stage = tmp_path / "stage"

    build(fake, stage)

    version = only_version(stage)
    (row,) = table(version, "images.parquet")
    assert row["etag"] == hashlib.md5(data).hexdigest()  # noqa: S324
    assert row["crc64nvme"] == crc64nvme(data)
    assert row["size"] == len(data)
    assert row["source_fingerprint"] == "b68d19ab-2"
    assert (row["media_type"], row["width"], row["height"]) == ("image/png", 9, 7)
    (plan,) = copies(version)
    assert plan["src_etag"] == "b68d19ab-2"
    assert plan["etag"] == row["etag"]
    assert (plan["size"], plan["crc64nvme"]) == (len(data), crc64nvme(data))
    full_reads = [c for c in fake.ops("get") if c[2] == "mermaid/m1.png"]
    assert [c[3].get("IfMatch") for c in full_reads] == ['"b68d19ab-2"']
    assert "Range" not in full_reads[0][3]


def test_source_without_crc64nvme_is_read_in_full_and_checksummed(repo, tmp_path):
    fake = FakeS3()
    data = png_bytes(6, 4)
    seed(fake, [ann("n1", "p1", 1, 1)], {"n1": data})
    fake.put(SRC_BUCKET, "mermaid/n1.png", data, crc=False)
    stage = tmp_path / "stage"

    build(fake, stage)

    version = only_version(stage)
    (row,) = table(version, "images.parquet")
    assert row["crc64nvme"] == crc64nvme(data)
    assert row["etag"] == hashlib.md5(data).hexdigest()  # noqa: S324
    assert row["source_fingerprint"] == row["etag"]
    (plan,) = copies(version)
    assert plan["crc64nvme"] == crc64nvme(data)


def test_png_404_falls_back_to_the_jpg_key_without_counting_a_deletion(repo, tmp_path, caplog):
    fake = FakeS3()
    data = jpeg_bytes(10, 8)
    seed(fake, [ann("j1", "p1", 1, 1), ann("gone", "p2", 2, 2)], {})
    fake.put(SRC_BUCKET, "mermaid/j1.jpg", data)
    stage = tmp_path / "stage"

    with caplog.at_level("INFO"):
        build(fake, stage)

    version = only_version(stage)
    (row,) = table(version, "images.parquet")
    assert row["image_id"] == "j1"
    assert json.loads(row["attrs"])["source_key"] == "mermaid/j1.jpg"
    (plan,) = copies(version)
    assert plan["src_key"] == "mermaid/j1.jpg"
    dropped = json.loads((version / "manifest.json").read_bytes())["stats"]["dropped"]
    assert dropped["image_deleted_at_source"] == 1
    assert "1 found only as .jpg" in caplog.text
    assert "j1" in caplog.text


def test_labels_are_ba_gf_pairs_numbered_by_sorted_ids(repo, tmp_path):
    fake = FakeS3()
    rows = [
        ann("i1", "p1", 1, 1, ENCRUSTING),
        ann("i1", "p2", 2, 2, RUBBLE),
        ann("i1", "p3", 3, 3, BRANCHING),
        ann("i1", "p4", 4, 4, RUBBLE),
    ]
    seed(fake, rows, {"i1": png_bytes(4, 4)})
    stage = tmp_path / "stage"

    build(fake, stage)

    version = only_version(stage)
    labels = {r["label_id"]: r for r in table(version, "labels.parquet")}
    assert labels[1] == {
        "label_id": 1,
        "name": "Rubble::",
        "mermaid_benthic_attribute_id": "aaa",
        "mermaid_benthic_attribute": "Rubble",
        "mermaid_growth_form": "",
        "mapping_source": "identity",
    }
    assert labels[2]["name"] == "Hard coral::Encrusting"
    assert labels[2]["mermaid_growth_form"] == "Encrusting"
    assert labels[3]["name"] == "Hard coral::Branching"
    assert labels[3]["mermaid_benthic_attribute_id"] == "bbb"
    assert set(labels) == {1, 2, 3}
    mapping = json.loads((version / "manifest.json").read_bytes())["label_mapping"]
    assert mapping["source"] == "identity"
    assert (mapping["n_labels"], mapping["n_mapped"]) == (3, 3)
    assert len(mapping["sha256"]) == 64


def parent_labels(entries: list[tuple[int, str, str | None]]) -> pa.Table:
    """A parent ``labels.parquet`` from ``(label_id, ba_id, growth_form_name)`` entries."""
    return pa.Table.from_pylist(
        [
            {
                "label_id": label_id,
                "name": f"{ba_id}::{gf or ''}",
                "mermaid_benthic_attribute_id": ba_id,
                "mermaid_benthic_attribute": ba_id,
                "mermaid_growth_form": gf,
                "mapping_source": "identity",
            }
            for label_id, ba_id, gf in entries
        ],
        schema=LABELS_SCHEMA,
    )


def label_ids_by_pair(
    labels: list[tuple[str, str, str | None, str | None]], parent: pa.Table
) -> dict[tuple[str, str | None], int]:
    rows = [ann("i1", f"p{n}", n, n, label) for n, label in enumerate(labels)]
    labelled, _ = mermaid.with_label_ids(pd.DataFrame(rows), parent)
    return {
        (r.benthic_attribute_id, r.growth_form_name): int(r.label_id)
        for r in labelled.itertuples(index=False)
    }


def test_parent_label_ids_are_reused_and_new_pairs_follow_the_parent_maximum():
    parent = parent_labels([(1, "a", None), (2, "c", "")])
    labels = [("a", "Sand", None, None), ("b", "B", "g1", "Branching"), ("c", "Turf", None, None)]

    ids = label_ids_by_pair(labels, parent)

    assert ids == {("a", None): 1, ("c", None): 2, ("b", "Branching"): 3}


def test_parent_label_id_gaps_are_kept_and_new_pairs_number_from_the_maximum():
    parent = parent_labels([(1, "a", None), (5, "c", None)])
    labels = [(ba, ba, None, None) for ba in ("a", "b", "c", "d")]

    ids = label_ids_by_pair(labels, parent)

    assert ids == {("a", None): 1, ("c", None): 5, ("b", None): 6, ("d", None): 7}


def test_parent_label_ids_match_on_the_growth_form_name():
    parent = parent_labels([(4, "b", "Branching")])
    labels = [("b", "B", "g9", "Branching"), ("b", "B", "g3", "Massive")]

    ids = label_ids_by_pair(labels, parent)

    assert ids == {("b", "Branching"): 4, ("b", "Massive"): 5}


def test_cutoff_keeps_the_named_day_and_drops_the_next_midnight(repo, tmp_path):
    fake = FakeS3()
    rows = [
        ann("i1", "p1", 1, 1, updated_on=datetime(2026, 11, 20, 23, 59, 59, 999999, tzinfo=UTC)),
        ann("i1", "p2", 2, 2, updated_on=datetime(2026, 11, 21, tzinfo=UTC)),
    ]
    seed(fake, rows, {"i1": png_bytes(4, 4)})
    stage = tmp_path / "stage"

    build(fake, stage, cutoff="2026-11-20")

    version = only_version(stage)
    assert [(p["row"], p["col"]) for p in table(version, "points.parquet")] == [(1, 1)]
    stats = json.loads((version / "manifest.json").read_bytes())["stats"]
    assert stats["dropped"]["annotation_after_cutoff"] == 1


def test_zero_byte_source_image_is_dropped_without_a_ranged_get(repo, tmp_path):
    fake = FakeS3()
    rows = [ann("i1", "p1", 1, 1), ann("empty", "p2", 2, 2)]
    seed(fake, rows, {"i1": png_bytes(4, 4), "empty": b""})
    stage = tmp_path / "stage"

    assert build(fake, stage) == 0

    version = only_version(stage)
    assert [r["image_id"] for r in table(version, "images.parquet")] == ["i1"]
    dropped = json.loads((version / "manifest.json").read_bytes())["stats"]["dropped"]
    assert dropped["image_unreadable_header"] == 1
    assert [c for c in fake.calls if c[0] == "get" and c[2] == "mermaid/empty.png"] == []


def test_dirty_git_refuses_before_reading_the_annotations(repo, tmp_path, monkeypatch):
    fake = FakeS3()
    seed(fake, [ann("i1", "p1", 1, 1)], {"i1": png_bytes(4, 4)})
    dirty = GitState(sha="abcdef1234567", dirty=True, pushed=True)
    monkeypatch.setattr(mermaid, "probe_git", lambda _: dirty)

    with pytest.raises(PublishRefused, match="uncommitted"):
        build(fake, tmp_path / "stage")

    assert [c for c in fake.calls if c[0] == "get" and c[2] == PARQUET_KEY] == []


def test_image_deleted_at_source_is_dropped_with_its_points(repo, tmp_path):
    fake = FakeS3()
    rows = [
        ann("i1", "p1", 1, 1),
        ann("gone", "p2", 2, 2),
        ann("gone", "p3", 3, 3),
    ]
    seed(fake, rows, {"i1": png_bytes(4, 4)})
    stage = tmp_path / "stage"

    build(fake, stage)

    version = only_version(stage)
    assert [r["image_id"] for r in table(version, "images.parquet")] == ["i1"]
    assert {p["image_id"] for p in table(version, "points.parquet")} == {"i1"}
    dropped = json.loads((version / "manifest.json").read_bytes())["stats"]["dropped"]
    assert dropped["image_deleted_at_source"] == 1
    assert dropped["points_of_deleted_images"] == 2


def test_unreadable_header_drops_the_image_and_logs_its_id(repo, tmp_path, caplog):
    fake = FakeS3()
    rows = [ann("i1", "p1", 1, 1), ann("bad", "p2", 2, 2)]
    seed(fake, rows, {"i1": png_bytes(4, 4), "bad": b"GIF89a-not-a-supported-format"})
    stage = tmp_path / "stage"

    with caplog.at_level("WARNING"):
        build(fake, stage)

    version = only_version(stage)
    assert [r["image_id"] for r in table(version, "images.parquet")] == ["i1"]
    dropped = json.loads((version / "manifest.json").read_bytes())["stats"]["dropped"]
    assert dropped["image_unreadable_header"] == 1
    assert "bad" in caplog.text


def test_annotations_changed_between_head_and_get_fails_and_stages_nothing(repo, tmp_path):
    fake = FakeS3()
    seed(fake, [ann("i1", "p1", 1, 1)], {"i1": png_bytes(4, 4)})
    real_head = fake.head_object

    def stale_head(**kwargs: Any) -> dict[str, Any]:
        response = real_head(**kwargs)
        if kwargs["Key"] == PARQUET_KEY:
            response = {**response, "ETag": '"stale"'}
        return response

    fake.head_object = stale_head  # type: ignore[method-assign]
    stage = tmp_path / "stage"

    with pytest.raises(mermaid.AnnotationsChanged):
        build(fake, stage)

    assert version_dirs(stage) == []


def test_parent_rows_with_unchanged_fingerprint_are_carried_planned_and_not_reread(repo, tmp_path):
    fake = FakeS3()
    a, b = png_bytes(4, 4), png_bytes(6, 6)
    seed(fake, [ann("a", "p1", 1, 1), ann("b", "p2", 2, 2)], {"a": a, "b": b})
    stage = tmp_path / "stage"
    build(fake, stage)
    parent = only_version(stage)
    parent_sha = _sha256(parent / "manifest.json")
    b2 = png_bytes(7, 7)
    fake.put(SRC_BUCKET, "mermaid/b.png", b2)
    fake.calls.clear()

    build(
        fake,
        stage,
        "--parent-version",
        parent.name,
        "--parent-manifest-sha256",
        parent_sha,
    )

    child = version_dirs(stage)[-1]
    assert child != parent
    images = {r["image_id"]: r for r in table(child, "images.parquet")}
    assert images["a"] == {r["image_id"]: r for r in table(parent, "images.parquet")}["a"]
    assert images["b"]["etag"] == quoted_etag(b2).strip('"')
    assert [c["src_key"] for c in copies(child)] == ["mermaid/a.png", "mermaid/b.png"]
    assert not [c for c in fake.ops("get") if c[2] == "mermaid/a.png"]
    manifest = json.loads((child / "manifest.json").read_bytes())
    assert manifest["parents"] == [
        {"dataset": "mermaid", "version": parent.name, "manifest_sha256": parent_sha}
    ]
    assert manifest["diff_from_parent"]["images_changed"] == 1


def test_child_adding_one_pair_keeps_parent_label_ids_so_no_points_are_removed(repo, tmp_path):
    fake = FakeS3()
    parent_rows = [ann("i1", "p1", 1, 1, RUBBLE), ann("i1", "p2", 2, 2, BRANCHING)]
    seed(fake, parent_rows, {"i1": png_bytes(4, 4)})
    stage = tmp_path / "stage"
    build(fake, stage)
    parent = only_version(stage)
    # ENCRUSTING sorts between RUBBLE and BRANCHING, so fresh numbering would shift BRANCHING.
    child_rows = [
        *parent_rows,
        ann("i1", "p3", 3, 3, ENCRUSTING),
        ann("i1", "p4", 4, 4, ENCRUSTING),
    ]
    seed(fake, child_rows, {"i1": png_bytes(4, 4)})

    build(
        fake,
        stage,
        "--parent-version",
        parent.name,
        "--parent-manifest-sha256",
        _sha256(parent / "manifest.json"),
    )

    diff = json.loads((version_dirs(stage)[-1] / "manifest.json").read_bytes())["diff_from_parent"]
    assert (diff["points_removed"], diff["points_added"]) == (0, 2)


def test_parent_flags_must_come_together(repo, tmp_path):
    fake = FakeS3()
    seed(fake, [ann("a", "p1", 1, 1)], {"a": png_bytes(4, 4)})

    with pytest.raises(ValueError, match="parent"):
        build(fake, tmp_path / "stage", "--parent-version", "20260101-abcdef1")

    assert version_dirs(tmp_path / "stage") == []


def test_large_jpeg_header_extends_the_read(repo, tmp_path):
    fake = FakeS3()
    base = jpeg_bytes(20, 10)
    # Two COM segments push the SOF past the first 64 KB range.
    com = b"\xff\xfe" + (65_000).to_bytes(2, "big") + b"\x00" * (65_000 - 2)
    data = base[:2] + com + com + base[2:]
    seed(fake, [ann("big", "p1", 1, 1)], {"big": data})
    stage = tmp_path / "stage"

    build(fake, stage)

    (row,) = table(only_version(stage), "images.parquet")
    assert (row["width"], row["height"]) == (20, 10)
    image_gets = [c for c in fake.ops("get") if c[2] != feature_source_key("big")]
    assert len(image_gets) == 3  # parquet, first range, extended range


def test_stage_dir_is_required_and_dispatches_to_run(monkeypatch, tmp_path):
    parser = cli.build_parser()
    with pytest.raises(SystemExit):
        parser.parse_args(["build", "mermaid", "--annotation-date-cutoff", CUTOFF])
    with pytest.raises(SystemExit):
        parser.parse_args(
            ["build", "mermaid", "--stage-dir", str(tmp_path), "--annotation-date-cutoff", CUTOFF]
        )

    seen: dict[str, Any] = {}

    def fake_run(args: Any, client: Any) -> int:
        seen["args"] = args
        return 7

    monkeypatch.setattr(mermaid, "run", fake_run)
    monkeypatch.setattr(cli, "make_s3_client", lambda **_kw: object())
    code = cli.main(
        [
            "build",
            "mermaid",
            "--stage-dir",
            str(tmp_path),
            "--annotation-date-cutoff",
            CUTOFF,
            "--extractor-spec",
            str(extractor_spec(tmp_path)),
        ]
    )

    assert code == 7
    args = seen["args"]
    assert args.stage_dir == tmp_path
    assert args.annotation_date_cutoff == CUTOFF


def feature_plan(version: Path) -> list[dict[str, Any]]:
    lines = (version / "features" / "efficientnet-v1" / "copies.jsonl").read_text().splitlines()
    return [json.loads(line) for line in lines]


def test_build_stages_a_feature_set_pinning_the_staged_dataset_with_its_copy_plan(repo, tmp_path):
    fake = FakeS3()
    seed(fake, [ann("i1", "p1", 10, 20), ann("i1", "p2", 30, 40)], {"i1": png_bytes(4, 4)})
    vector = featurevector_bytes([(10, 20), (30, 40), (50, 60)])
    fake.put(SRC_BUCKET, "mermaid/i1_featurevector", vector, etag='"src-etag-2"')
    stage = tmp_path / "stage"

    build(fake, stage)

    version = only_version(stage)
    md5 = hashlib.md5(vector).hexdigest()  # noqa: S324
    assert feature_plan(version) == [
        {
            "src_bucket": SRC_BUCKET,
            "src_key": "mermaid/i1_featurevector",
            "src_etag": "src-etag-2",
            "etag": md5,
            "dst_bucket": DATASET_BUCKET,
            "dst_key": "features/mermaid/efficientnet-v1/i1.featurevector",
            "size": len(vector),
            "crc64nvme": crc64nvme(vector),
        }
    ]
    staged = version / "features" / "efficientnet-v1"
    assert table(staged, "index.parquet") == [
        {
            "image_id": "i1",
            "key": "features/mermaid/efficientnet-v1/i1.featurevector",
            "etag": md5,
            "size": len(vector),
            "crc64nvme": crc64nvme(vector),
            "n_points": 2,
        }
    ]
    set_manifest = json.loads((staged / "manifest.json").read_text())
    dataset_sha = hashlib.sha256((version / "manifest.json").read_bytes()).hexdigest()
    assert set_manifest["dataset"] == {
        "name": "mermaid",
        "version": version.name,
        "manifest_sha256": dataset_sha,
    }
    assert set_manifest["extractor"] == EXTRACTOR
    assert sorted(p.name for p in staged.iterdir()) == [
        "copies.jsonl",
        "index.parquet",
        "manifest.json",
    ]
    reads = [c for c in fake.ops("get") if c[2] == "mermaid/i1_featurevector"]
    assert [c[3].get("IfMatch") for c in reads] == ['"src-etag-2"']


def test_missing_feature_vector_fails_the_build_and_stages_nothing(repo, tmp_path):
    fake = FakeS3()
    seed(
        fake,
        [ann("i1", "p1", 1, 1), ann("i2", "p2", 2, 2)],
        {"i1": png_bytes(4, 4), "i2": png_bytes(4, 4)},
    )
    del fake.objects[(SRC_BUCKET, "mermaid/i2_featurevector")]
    stage = tmp_path / "stage"

    with pytest.raises(FeatureVectorsMissing, match="i2"):
        build(fake, stage)

    assert version_dirs(stage) == []


def test_allow_missing_features_keeps_the_image_without_a_feature_row(repo, tmp_path):
    fake = FakeS3()
    seed(
        fake,
        [ann("i1", "p1", 1, 1), ann("i2", "p2", 2, 2)],
        {"i1": png_bytes(4, 4), "i2": png_bytes(4, 4)},
    )
    del fake.objects[(SRC_BUCKET, "mermaid/i2_featurevector")]
    stage = tmp_path / "stage"

    build(fake, stage, "--allow-missing-features")

    version = only_version(stage)
    assert [r["image_id"] for r in table(version, "images.parquet")] == ["i1", "i2"]
    index = table(version / "features" / "efficientnet-v1", "index.parquet")
    assert [r["image_id"] for r in index] == ["i1"]
    assert [line["src_key"] for line in feature_plan(version)] == ["mermaid/i1_featurevector"]


def test_feature_vector_missing_a_point_fails_the_build(repo, tmp_path):
    fake = FakeS3()
    seed(fake, [ann("i1", "p1", 1, 1), ann("i1", "p2", 2, 2)], {"i1": png_bytes(4, 4)})
    fake.put(SRC_BUCKET, "mermaid/i1_featurevector", featurevector_bytes([(1, 1)]))
    stage = tmp_path / "stage"

    with pytest.raises(FeatureVectorsInvalid, match="mermaid/i1_featurevector"):
        build(fake, stage)

    assert version_dirs(stage) == []


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_labels_write_an_empty_growth_form_not_null_for_a_pair_without_one():
    pairs = pd.DataFrame(
        {
            "label_id": [1, 2],
            "benthic_attribute_id": ["aaa", "bbb"],
            "benthic_attribute_name": ["Rubble", "Hard coral"],
            "growth_form_name": [None, "Branching"],
        }
    )

    rows = mermaid.labels_table(pairs).to_pylist()

    assert [r["mermaid_growth_form"] for r in rows] == ["", "Branching"]


def test_load_parent_rows_strips_quotes_from_the_parents_source_fingerprints(tmp_path):
    version = "20261120-abcdef1"
    path = tmp_path / DATASET_BUCKET / "datasets" / "mermaid" / version / "images.parquet"
    path.parent.mkdir(parents=True)
    pq.write_table(pa.table({"image_id": ["a", "b"], "source_fingerprint": ['"e1"', "e2"]}), path)

    rows = mermaid.load_parent_rows(tmp_path, version)

    assert {i: r["source_fingerprint"] for i, r in rows.items()} == {"a": "e1", "b": "e2"}


def test_a_vector_already_in_the_family_is_adopted_and_not_planned_for_copy(repo, tmp_path):
    fake = FakeS3()
    seed(
        fake,
        [ann("i1", "p1", 1, 1), ann("i2", "p2", 2, 2)],
        {"i1": png_bytes(4, 4), "i2": png_bytes(4, 4)},
    )
    family_key = "features/mermaid/efficientnet-v1/i1.featurevector"
    fake.put(DATASET_BUCKET, family_key, fake.objects[(SRC_BUCKET, "mermaid/i1_featurevector")])
    stage = tmp_path / "stage"

    build(fake, stage)

    version = only_version(stage)
    assert [line["dst_key"] for line in feature_plan(version)] == [
        "features/mermaid/efficientnet-v1/i2.featurevector"
    ]
    index = table(version / "features" / "efficientnet-v1", "index.parquet")
    assert [r["key"] for r in index] == [
        family_key,
        "features/mermaid/efficientnet-v1/i2.featurevector",
    ]
    assert not [c for c in fake.ops("get") if c[2] == "mermaid/i1_featurevector"]
