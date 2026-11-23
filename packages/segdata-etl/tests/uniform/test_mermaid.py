"""Tests for ``segdata build mermaid``: the staged uniform-format MERMAID points dataset."""

from __future__ import annotations

import hashlib
import io
import json
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any

import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from segdata_etl import cli
from segdata_etl.datasets import mermaid
from segdata_spec import DATASET_BUCKET
from segdata_spec.publish import GitState

from .conftest import FakeS3, crc64nvme_b64, jpeg_bytes, md5_etag, png_bytes

SRC_BUCKET = "coral-reef-training"
PARQUET_KEY = "mermaid/mermaid_confirmed_annotations.parquet"
CUTOFF = "2026-08-12"
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


def seed(fake: FakeS3, rows: list[dict[str, Any]], images: dict[str, bytes]) -> None:
    fake.put(SRC_BUCKET, PARQUET_KEY, parquet_bytes(rows))
    for image_id, data in images.items():
        fake.put(SRC_BUCKET, f"mermaid/{image_id}.png", data)


@pytest.fixture
def repo(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    root = tmp_path / "repo"
    root.mkdir()
    (root / "uv.lock").write_bytes(b"lock")
    monkeypatch.setattr(mermaid, "REPO_ROOT", root)
    git = GitState(sha="abcdef1234567", dirty=False, pushed=True)
    monkeypatch.setattr(mermaid, "probe_git", lambda _: git)
    return root


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
    assert i1["etag"] == md5_etag(blobs["i1"]).strip('"')
    assert i1["size"] == len(blobs["i1"])
    assert i1["crc64nvme"] == crc64nvme_b64(blobs["i1"])
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
        "dst_bucket": DATASET_BUCKET,
        "dst_key": "images/mermaid/i1.png",
        "size": len(blobs["i1"]),
        "crc64nvme": crc64nvme_b64(blobs["i1"]),
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
        "mermaid_growth_form": None,
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


def test_cutoff_keeps_midnight_and_drops_one_microsecond_later(repo, tmp_path):
    fake = FakeS3()
    midnight = datetime(2026, 8, 12, tzinfo=UTC)
    rows = [
        ann("i1", "p1", 1, 1, updated_on=midnight),
        ann("i1", "p2", 2, 2, updated_on=midnight + timedelta(microseconds=1)),
    ]
    seed(fake, rows, {"i1": png_bytes(4, 4)})
    stage = tmp_path / "stage"

    build(fake, stage)

    version = only_version(stage)
    assert [(p["row"], p["col"]) for p in table(version, "points.parquet")] == [(1, 1)]
    stats = json.loads((version / "manifest.json").read_bytes())["stats"]
    assert stats["dropped"]["annotation_after_cutoff"] == 1


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


def test_parent_rows_with_unchanged_fingerprint_are_carried_and_not_copied(repo, tmp_path):
    fake = FakeS3()
    a, b = png_bytes(4, 4), png_bytes(6, 6)
    seed(fake, [ann("a", "p1", 1, 1), ann("b", "p2", 2, 2)], {"a": a, "b": b})
    stage = tmp_path / "stage"
    build(fake, stage)
    parent = only_version(stage)
    parent_sha = _sha256(parent / "manifest.json")
    b2 = png_bytes(7, 7)
    fake.put(SRC_BUCKET, "mermaid/b.png", b2)

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
    assert images["b"]["etag"] == md5_etag(b2).strip('"')
    assert [c["src_key"] for c in copies(child)] == ["mermaid/b.png"]
    manifest = json.loads((child / "manifest.json").read_bytes())
    assert manifest["parents"] == [
        {"dataset": "mermaid", "version": parent.name, "manifest_sha256": parent_sha}
    ]
    assert manifest["diff_from_parent"]["images_changed"] == 1


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
    assert len(fake.ops("get")) == 3  # parquet, first range, extended range


def test_stage_dir_is_required_and_dispatches_to_run(monkeypatch, tmp_path):
    parser = cli.build_parser()
    with pytest.raises(SystemExit):
        parser.parse_args(["build", "mermaid", "--annotation-date-cutoff", CUTOFF])

    seen: dict[str, Any] = {}

    def fake_run(args: Any, client: Any) -> int:
        seen["args"] = args
        return 7

    monkeypatch.setattr(mermaid, "run", fake_run)
    monkeypatch.setattr(cli, "make_s3_client", lambda: object())
    code = cli.main(
        ["build", "mermaid", "--stage-dir", str(tmp_path), "--annotation-date-cutoff", CUTOFF]
    )

    assert code == 7
    args = seen["args"]
    assert args.stage_dir == tmp_path
    assert args.annotation_date_cutoff == CUTOFF
    assert args.workers == 16


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()
