"""Tests for ``segdata promote``: a staged version uploaded write-once, images then tables then manifest."""

from __future__ import annotations

import hashlib
import json
from datetime import datetime
from pathlib import Path
from typing import Any

import pytest
from s3_fake import FakeS3, client_error, crc64nvme, quoted_etag

from segdata_etl import cli, promote
from segdata_etl.common.s3_objects import ObjectConflict
from segdata_etl.common.stage import next_version
from segdata_spec import DATASET_BUCKET
from segdata_spec.errors import ChecksumMismatch, PublishRefused

from .conftest import jpeg_bytes, png_bytes
from .test_coralscapes_v2 import IMG_A, IMG_B, MASK, MASK_B
from .test_coralscapes_v2 import build as build_coralscapes
from .test_coralscapes_v2 import shard as coralscapes_shard
from .test_mermaid import (  # noqa: F401
    ENCRUSTING,
    RUBBLE,
    ann,
    only_version,
    repo,
    seed,
)
from .test_mermaid import build as build_mermaid

BLOBS = {"i1": png_bytes(8, 6), "i2": png_bytes(12, 9), "i3": jpeg_bytes(5, 5)}
_TABLES = ("images.parquet", "points.parquet", "labels.parquet", "stats/label_counts.parquet")


def stage_mermaid(tmp_path: Path) -> tuple[FakeS3, Path, str]:
    """Stage a mermaid version; the returned fake holds the source images and nothing else."""
    fake = FakeS3()
    rows = [
        ann("i1", "p1", 10, 20, ENCRUSTING),
        ann("i2", "p2", 1, 2, RUBBLE),
        ann("i3", "p3", 7, 8, RUBBLE),
    ]
    seed(fake, rows, BLOBS)
    stage = tmp_path / "stage"
    assert build_mermaid(fake, stage) == 0
    fake.calls.clear()
    return fake, stage, only_version(stage).name


def version_dir(stage: Path, dataset: str, version: str) -> Path:
    return stage / DATASET_BUCKET / "datasets" / dataset / version


def run_promote(client: Any, stage: Path, version: str, dataset: str = "mermaid") -> None:
    promote.promote(client, stage_dir=stage, dataset=dataset, version=version, workers=2)


def writes(fake: FakeS3) -> list[tuple[str, str, str, dict[str, Any]]]:
    return [c for c in fake.calls if c[0] in ("put", "copy")]


def test_images_precede_tables_and_the_manifest_is_the_last_put(repo, tmp_path):  # noqa: F811
    fake, stage, version = stage_mermaid(tmp_path)

    run_promote(fake, stage, version)

    log = writes(fake)
    prefix = f"datasets/mermaid/{version}/"
    kinds = [(op, key.startswith(prefix)) for op, _, key, _ in log]
    assert kinds == [("copy", False)] * 3 + [("put", True)] * 5
    assert log[-1][2] == prefix + "manifest.json"
    assert {k for _, _, k, _ in log[3:-1]} == {prefix + t for t in _TABLES}
    assert all(kwargs["IfNoneMatch"] == "*" for _, _, _, kwargs in log)
    assert all(b == DATASET_BUCKET for _, b, _, _ in log)
    published = fake.objects[(DATASET_BUCKET, prefix + "manifest.json")]
    assert published == (version_dir(stage, "mermaid", version) / "manifest.json").read_bytes()
    assert (DATASET_BUCKET, prefix + "copies.jsonl") not in fake.objects
    for image_id, data in BLOBS.items():
        (key,) = [k for b, k in fake.objects if k.startswith(f"images/mermaid/{image_id}.")]
        assert fake.objects[(DATASET_BUCKET, key)] == data


def test_copy_landing_with_another_crc_raises_before_any_table_is_put(repo, tmp_path):  # noqa: F811
    class WrongCrc(FakeS3):
        def copy_object(self, **kwargs: Any) -> dict[str, Any]:
            response = super().copy_object(**kwargs)
            response["CopyObjectResult"]["ChecksumCRC64NVME"] = crc64nvme(b"other")
            return response

    fake, stage, version = stage_mermaid(tmp_path)
    bad = WrongCrc(**{k: getattr(fake, k) for k in ("objects", "crcs", "etags")})

    with pytest.raises(ObjectConflict):
        run_promote(bad, stage, version)

    prefix = f"datasets/mermaid/{version}/"
    assert not [k for _, k in bad.objects if k.startswith(prefix)]
    assert bad.ops("put") == []


def test_published_manifest_refuses_before_any_write(repo, tmp_path):  # noqa: F811
    fake, stage, version = stage_mermaid(tmp_path)
    key = f"datasets/mermaid/{version}/manifest.json"
    fake.put(DATASET_BUCKET, key, b"{}")
    fake.calls.clear()

    with pytest.raises(PublishRefused, match="already"):
        run_promote(fake, stage, version)

    assert [(op, k) for op, _, k, _ in fake.calls] == [("head", key)]


def test_staged_table_that_no_longer_matches_the_manifest_refuses_with_no_writes(
    repo,  # noqa: F811
    tmp_path,
):
    fake, stage, version = stage_mermaid(tmp_path)
    table = version_dir(stage, "mermaid", version) / "points.parquet"
    table.write_bytes(table.read_bytes() + b"x")

    with pytest.raises(ChecksumMismatch, match="points.parquet"):
        run_promote(fake, stage, version)

    assert fake.calls == []


def rewrite_plan(stage: Path, version: str, edit) -> None:
    path = version_dir(stage, "mermaid", version) / "copies.jsonl"
    lines = [json.loads(line) for line in path.read_text().splitlines()]
    path.write_text("".join(json.dumps(line) + "\n" for line in edit(lines)))


@pytest.mark.parametrize("field", ["crc64nvme", "etag"])
@pytest.mark.parametrize("bad", [None, ""])
def test_plan_line_without_crc_or_etag_refuses_at_preflight(repo, tmp_path, field, bad):  # noqa: F811
    fake, stage, version = stage_mermaid(tmp_path)

    def edit(lines):
        lines[0][field] = bad
        return lines

    rewrite_plan(stage, version, edit)

    with pytest.raises(PublishRefused, match=field):
        run_promote(fake, stage, version)

    assert fake.calls == []


def test_plan_missing_a_line_refuses_at_preflight(repo, tmp_path):  # noqa: F811
    fake, stage, version = stage_mermaid(tmp_path)
    rewrite_plan(stage, version, lambda lines: lines[1:])

    with pytest.raises(PublishRefused, match="keys"):
        run_promote(fake, stage, version)

    assert fake.calls == []


@pytest.mark.parametrize(
    ("field", "edit_value"),
    [
        ("size", lambda v: v + 1),
        ("crc64nvme", lambda v: crc64nvme(b"other")),
        ("etag", lambda v: hashlib.md5(b"other").hexdigest()),  # noqa: S324
    ],
)
def test_plan_line_disagreeing_with_images_parquet_refuses_at_preflight(
    repo,  # noqa: F811
    tmp_path,
    field,
    edit_value,
):
    fake, stage, version = stage_mermaid(tmp_path)

    def edit(lines):
        lines[0][field] = edit_value(lines[0][field])
        return lines

    rewrite_plan(stage, version, edit)

    with pytest.raises(PublishRefused, match="images.parquet"):
        run_promote(fake, stage, version)

    assert fake.calls == []


def test_tables_only_version_without_a_parent_refuses_with_no_writes(repo, tmp_path):  # noqa: F811
    fake, stage, version = stage_mermaid(tmp_path)
    (version_dir(stage, "mermaid", version) / "copies.jsonl").unlink()
    with pytest.raises(PublishRefused, match="parent"):
        run_promote(fake, stage, version)
    assert writes(fake) == []


def test_rerun_after_a_crash_finds_the_images_present_and_writes_the_manifest(repo, tmp_path):  # noqa: F811
    class FailsFirstTable(FakeS3):
        failed = False

        def put_object(self, **kwargs: Any) -> dict[str, Any]:
            if kwargs["Key"].endswith(".parquet") and not self.failed:
                self.failed = True
                raise client_error("InternalError", 500, "PutObject")
            return super().put_object(**kwargs)

    fake, stage, version = stage_mermaid(tmp_path)
    flaky = FailsFirstTable(**{k: getattr(fake, k) for k in ("objects", "crcs", "etags")})

    with pytest.raises(Exception, match="InternalError"):
        run_promote(flaky, stage, version)
    manifest_key = (DATASET_BUCKET, f"datasets/mermaid/{version}/manifest.json")
    assert manifest_key not in flaky.objects
    flaky.calls.clear()

    run_promote(flaky, stage, version)

    assert manifest_key in flaky.objects
    assert len(flaky.ops("copy")) == 3
    assert flaky.ops("put")[-1][2] == manifest_key[1]


def test_rerun_logs_images_as_already_present(repo, tmp_path, caplog):  # noqa: F811
    fake, stage, version = stage_mermaid(tmp_path)
    for plan in (version_dir(stage, "mermaid", version) / "copies.jsonl").read_text().splitlines():
        line = json.loads(plan)
        fake.put(
            DATASET_BUCKET, line["dst_key"], fake.objects[(line["src_bucket"], line["src_key"])]
        )

    with caplog.at_level("INFO"):
        run_promote(fake, stage, version)

    assert "3 already present" in caplog.text
    assert "0 copied" in caplog.text


def test_hf_uploads_land_byte_identical_with_the_planned_crc(monkeypatch, tmp_path):
    version_path = build_coralscapes(monkeypatch, tmp_path)
    stage = tmp_path / "stage"
    fake = FakeS3()

    run_promote(fake, stage, version_path.name, dataset="coralscapes_v2")

    expected = {
        "images/coralscapes_v2/train/a1.jpg": IMG_A,
        "images/coralscapes_v2/train/a1_label.png": MASK,
        "images/coralscapes_v2/validation/b1.png": IMG_B,
        "images/coralscapes_v2/validation/b1_label.png": MASK_B,
    }
    for key, data in expected.items():
        assert fake.objects[(DATASET_BUCKET, key)] == data
        assert fake.crcs[(DATASET_BUCKET, key)] == crc64nvme(data)
    log = writes(fake)
    assert [op for op, *_ in log] == ["put"] * len(log)
    assert log[-1][2].endswith("/manifest.json")
    assert not any(k.endswith(("uploads.jsonl", "copies.jsonl")) for _, k in fake.objects)


def test_hf_upload_whose_local_bytes_differ_from_the_plan_refuses_at_preflight(
    monkeypatch, tmp_path
):
    version_path = build_coralscapes(monkeypatch, tmp_path)
    stage = tmp_path / "stage"
    local = stage / DATASET_BUCKET / "images/coralscapes_v2/train/a1.jpg"
    local.write_bytes(jpeg_bytes(4, 3) + b"\x00")
    fake = FakeS3()

    with pytest.raises(PublishRefused, match="a1.jpg"):
        run_promote(fake, stage, version_path.name, dataset="coralscapes_v2")

    assert fake.calls == []


def test_cli_requires_its_flags_and_dispatches(monkeypatch, tmp_path):
    for argv in (
        ["promote", "--dataset", "d", "--version", "v"],
        ["promote", "--stage-dir", str(tmp_path), "--version", "v"],
        ["promote", "--stage-dir", str(tmp_path), "--dataset", "d"],
    ):
        with pytest.raises(SystemExit):
            cli.build_parser().parse_args(argv)
    seen: dict[str, Any] = {}

    def fake_promote(client: Any, **kwargs: Any) -> None:
        seen.update(kwargs, client=client)

    client = object()
    monkeypatch.setattr(promote, "promote", fake_promote)
    monkeypatch.setattr(cli, "make_s3_client", lambda: client)

    code = cli.main(
        [
            "promote",
            "--stage-dir",
            str(tmp_path),
            "--dataset",
            "d",
            "--version",
            "v",
            "--workers",
            "3",
        ]
    )

    assert code == 0
    assert seen == {
        "client": client,
        "stage_dir": tmp_path,
        "dataset": "d",
        "version": "v",
        "workers": 3,
    }


def copy_destinations(stage: Path, version: str) -> list[dict[str, Any]]:
    path = version_dir(stage, "mermaid", version) / "copies.jsonl"
    return [json.loads(line) for line in path.read_text().splitlines()]


def test_planned_copy_key_holding_other_bytes_refuses_with_only_heads(repo, tmp_path):  # noqa: F811
    fake, stage, version = stage_mermaid(tmp_path)
    taken = copy_destinations(stage, version)[0]["dst_key"]
    fake.put(DATASET_BUCKET, taken, b"other bytes")
    fake.calls.clear()

    with pytest.raises(PublishRefused, match=taken):
        run_promote(fake, stage, version)

    assert {op for op, *_ in fake.calls} == {"head"}
    assert all(c[3].get("ChecksumMode") == "ENABLED" for c in fake.ops("head")[1:])


def test_planned_upload_key_holding_other_bytes_refuses_with_only_heads(monkeypatch, tmp_path):
    version_path = build_coralscapes(monkeypatch, tmp_path)
    fake = FakeS3()
    taken = "images/coralscapes_v2/train/a1.jpg"
    fake.put(DATASET_BUCKET, taken, IMG_A + b"x")

    with pytest.raises(PublishRefused, match=taken):
        run_promote(fake, tmp_path / "stage", version_path.name, dataset="coralscapes_v2")

    assert {op for op, *_ in fake.calls} == {"head"}


def plant_same_length(fake: FakeS3, key: str, data: bytes, differs: str) -> None:
    """Seed ``key`` with ``data`` minus a flipped byte, so the size matches the plan.

    ``differs`` names the fingerprint that catches it: ``etag`` keeps the planned CRC,
    ``crc`` keeps the planned ETag, and ``both`` keeps neither.
    """
    flipped = bytes([data[0] ^ 1]) + data[1:]
    fake.put(DATASET_BUCKET, key, flipped, etag=quoted_etag(data) if differs == "crc" else None)
    if differs == "etag":
        fake.crcs[(DATASET_BUCKET, key)] = crc64nvme(data)


@pytest.mark.parametrize("differs", ["etag", "crc", "both"])
def test_planned_copy_key_holding_same_length_other_bytes_refuses_with_only_heads(
    repo,  # noqa: F811
    tmp_path,
    differs,
):
    fake, stage, version = stage_mermaid(tmp_path)
    line = copy_destinations(stage, version)[0]
    plant_same_length(
        fake, line["dst_key"], fake.objects[(line["src_bucket"], line["src_key"])], differs
    )
    fake.calls.clear()

    with pytest.raises(PublishRefused, match=line["dst_key"]):
        run_promote(fake, stage, version)

    assert {op for op, *_ in fake.calls} == {"head"}


@pytest.mark.parametrize("differs", ["etag", "crc", "both"])
def test_planned_upload_key_holding_same_length_other_bytes_refuses_with_only_heads(
    monkeypatch, tmp_path, differs
):
    version_path = build_coralscapes(monkeypatch, tmp_path)
    fake = FakeS3()
    taken = "images/coralscapes_v2/train/a1.jpg"
    plant_same_length(fake, taken, IMG_A, differs)

    with pytest.raises(PublishRefused, match=taken):
        run_promote(fake, tmp_path / "stage", version_path.name, dataset="coralscapes_v2")

    assert {op for op, *_ in fake.calls} == {"head"}


def test_table_key_holding_other_bytes_refuses_with_only_heads(repo, tmp_path):  # noqa: F811
    fake, stage, version = stage_mermaid(tmp_path)
    taken = f"datasets/mermaid/{version}/images.parquet"
    staged = (version_dir(stage, "mermaid", version) / "images.parquet").read_bytes()
    fake.put(DATASET_BUCKET, taken, staged[:-1] + bytes([staged[-1] ^ 1]))
    fake.calls.clear()

    with pytest.raises(PublishRefused, match=taken):
        run_promote(fake, stage, version)

    assert {op for op, *_ in fake.calls} == {"head"}
    assert all(c[3].get("ChecksumMode") == "ENABLED" for c in fake.ops("head")[1:])


def test_planned_upload_key_holding_identical_bytes_still_promotes(monkeypatch, tmp_path):
    version_path = build_coralscapes(monkeypatch, tmp_path)
    fake = FakeS3()
    fake.put(DATASET_BUCKET, "images/coralscapes_v2/train/a1.jpg", IMG_A)

    run_promote(fake, tmp_path / "stage", version_path.name, dataset="coralscapes_v2")

    manifest_key = f"datasets/coralscapes_v2/{version_path.name}/manifest.json"
    assert (DATASET_BUCKET, manifest_key) in fake.objects


def test_half_uploaded_version_counts_as_taken_yet_promote_resumes_it(repo, tmp_path):  # noqa: F811
    fake, stage, version = stage_mermaid(tmp_path)
    prefix = f"datasets/mermaid/{version}/"
    staged_images = (version_dir(stage, "mermaid", version) / "images.parquet").read_bytes()
    fake.put(DATASET_BUCKET, prefix + "images.parquet", staged_images)

    day, sha = version.split("-", 1)
    free = next_version(
        tmp_path / "empty",
        DATASET_BUCKET,
        "mermaid",
        git_sha=sha,
        today=datetime.strptime(day, "%Y%m%d").date(),  # noqa: DTZ007
        client=fake,
    )
    assert free == f"{version}-2"
    run_promote(fake, stage, version)

    assert (DATASET_BUCKET, prefix + "manifest.json") in fake.objects


def test_zero_image_version_refuses_with_no_writes(monkeypatch, tmp_path):
    bad = shard_with_only_unreadable_images()
    version_path = build_coralscapes(monkeypatch, tmp_path, shards=bad)
    fake = FakeS3()

    with pytest.raises(PublishRefused, match="no images"):
        run_promote(fake, tmp_path / "stage", version_path.name, dataset="coralscapes_v2")

    assert writes(fake) == []


def shard_with_only_unreadable_images() -> dict[str, Any]:
    gif = b"GIF89a-not-supported"
    return {"instances/train-00000-of-00001.parquet": coralscapes_shard(["g1"], [gif], [MASK])}


def test_upload_landing_with_another_crc_raises_and_no_table_is_written(monkeypatch, tmp_path):
    class WrongCrc(FakeS3):
        def put_object(self, **kwargs: Any) -> dict[str, Any]:
            response = super().put_object(**kwargs)
            if kwargs["Key"].startswith("images/"):
                response["ChecksumCRC64NVME"] = crc64nvme(b"other")
            return response

    version_path = build_coralscapes(monkeypatch, tmp_path)
    fake = WrongCrc()

    with pytest.raises(ObjectConflict):
        run_promote(fake, tmp_path / "stage", version_path.name, dataset="coralscapes_v2")

    assert not [k for _, k in fake.objects if k.startswith("datasets/")]


PARENT_VERSION = "20261101-1234567"
PARENT_KEY = f"datasets/parent/{PARENT_VERSION}/manifest.json"
PARENT_MANIFEST = b'{"dataset": "parent"}'


def pin_parent(stage: Path, version: str, pinned: bytes = PARENT_MANIFEST) -> None:
    """Add a parent pin on ``PARENT_MANIFEST`` to the staged mermaid manifest."""
    path = version_dir(stage, "mermaid", version) / "manifest.json"
    manifest = json.loads(path.read_bytes())
    manifest["parents"] = [
        {
            "dataset": "parent",
            "version": PARENT_VERSION,
            "manifest_sha256": hashlib.sha256(pinned).hexdigest(),
        }
    ]
    path.write_text(json.dumps(manifest))


def test_unpublished_parent_refuses_with_no_writes(repo, tmp_path):  # noqa: F811
    fake, stage, version = stage_mermaid(tmp_path)
    pin_parent(stage, version)

    with pytest.raises(PublishRefused, match="parent"):
        run_promote(fake, stage, version)

    assert writes(fake) == []


def test_parent_manifest_with_other_bytes_than_the_pin_refuses_with_no_writes(
    repo,  # noqa: F811
    tmp_path,
):
    fake, stage, version = stage_mermaid(tmp_path)
    pin_parent(stage, version)
    fake.put(DATASET_BUCKET, PARENT_KEY, PARENT_MANIFEST + b" ")

    with pytest.raises(PublishRefused, match="parent"):
        run_promote(fake, stage, version)

    assert writes(fake) == []


def test_parent_published_with_the_pinned_bytes_lets_promote_proceed(repo, tmp_path):  # noqa: F811
    fake, stage, version = stage_mermaid(tmp_path)
    pin_parent(stage, version)
    fake.put(DATASET_BUCKET, PARENT_KEY, PARENT_MANIFEST)
    fake.calls.clear()

    run_promote(fake, stage, version)

    assert (DATASET_BUCKET, f"datasets/mermaid/{version}/manifest.json") in fake.objects


SOURCE_BUCKET = "dense-source"
DENSE = "coralscapes_v2"


def stage_dense(monkeypatch, tmp_path: Path) -> tuple[FakeS3, Path, str]:
    """Stage coralscapes, then move its image uploads into copies.jsonl; masks stay uploads."""
    version_path = build_coralscapes(monkeypatch, tmp_path)
    stage = tmp_path / "stage"
    fake = FakeS3()
    lines = [json.loads(x) for x in (version_path / "uploads.jsonl").read_text().splitlines()]
    copies = []
    for line in lines:
        if line["dst_key"].endswith("_label.png"):
            continue
        data = (stage / line["local_path"]).read_bytes()
        src_key = "src/" + line["dst_key"]
        fake.put(SOURCE_BUCKET, src_key, data)
        copies.append(
            {
                "src_bucket": SOURCE_BUCKET,
                "src_key": src_key,
                "src_etag": quoted_etag(data),
                "dst_bucket": DATASET_BUCKET,
                "dst_key": line["dst_key"],
                "size": len(data),
                "crc64nvme": crc64nvme(data),
                "etag": quoted_etag(data),
            }
        )
    masks = [x for x in lines if x["dst_key"].endswith("_label.png")]
    (version_path / "copies.jsonl").write_text("".join(json.dumps(x) + "\n" for x in copies))
    (version_path / "uploads.jsonl").write_text("".join(json.dumps(x) + "\n" for x in masks))
    fake.calls.clear()
    return fake, stage, version_path.name


def test_dense_version_copies_images_uploads_masks_and_writes_the_manifest_last(
    monkeypatch, tmp_path
):
    fake, stage, version = stage_dense(monkeypatch, tmp_path)

    run_promote(fake, stage, version, dataset=DENSE)

    log = writes(fake)
    image_keys = {f"images/{DENSE}/train/a1.jpg", f"images/{DENSE}/validation/b1.png"}
    mask_keys = {f"images/{DENSE}/train/a1_label.png", f"images/{DENSE}/validation/b1_label.png"}
    assert {k for op, _, k, _ in log if op == "copy"} == image_keys
    assert {k for op, _, k, _ in log[:4] if op == "put"} == mask_keys
    matches = {kw["CopySourceIfMatch"] for op, _, _, kw in log if op == "copy"}
    assert matches == {quoted_etag(IMG_A), quoted_etag(IMG_B)}
    assert fake.objects[(DATASET_BUCKET, f"images/{DENSE}/train/a1.jpg")] == IMG_A
    assert fake.objects[(DATASET_BUCKET, f"images/{DENSE}/train/a1_label.png")] == MASK
    assert log[-1][2] == f"datasets/{DENSE}/{version}/manifest.json"


def test_key_listed_in_both_plan_files_refuses_before_any_write(monkeypatch, tmp_path):
    fake, stage, version = stage_dense(monkeypatch, tmp_path)
    directory = version_dir(stage, DENSE, version)
    copied = json.loads((directory / "copies.jsonl").read_text().splitlines()[0])
    upload = {
        "local_path": f"{DATASET_BUCKET}/{copied['dst_key']}",
        "dst_bucket": DATASET_BUCKET,
        "dst_key": copied["dst_key"],
        "size": copied["size"],
        "crc64nvme": copied["crc64nvme"],
        "content_type": "image/jpeg",
    }
    with (directory / "uploads.jsonl").open("a") as handle:
        handle.write(json.dumps(upload) + "\n")

    with pytest.raises(PublishRefused, match="1 duplicated"):
        run_promote(fake, stage, version, dataset=DENSE)

    assert writes(fake) == []


def test_key_in_neither_plan_file_refuses_before_any_write(monkeypatch, tmp_path):
    fake, stage, version = stage_dense(monkeypatch, tmp_path)
    path = version_dir(stage, DENSE, version) / "uploads.jsonl"
    path.write_text("".join(path.read_text().splitlines(keepends=True)[1:]))

    with pytest.raises(PublishRefused, match="1 missing"):
        run_promote(fake, stage, version, dataset=DENSE)

    assert writes(fake) == []


def stage_tables_only(
    tmp_path: Path, parent_store: dict[str, Any] | None = None
) -> tuple[FakeS3, Path, str, dict[str, Any]]:
    """Stage a mermaid version without plan files, its images already in its image store.

    Its one parent is published with the same image store, or with ``parent_store``.
    """
    fake, stage, version = stage_mermaid(tmp_path)
    directory = version_dir(stage, "mermaid", version)
    parent = json.loads((directory / "manifest.json").read_bytes())
    store = parent["image_store"]
    for line in copy_destinations(stage, version):
        data = fake.objects[(line["src_bucket"], line["src_key"])]
        fake.put(store["bucket"], line["dst_key"], data)
    (directory / "copies.jsonl").unlink()
    parent.update(dataset="parent", version=PARENT_VERSION)
    if parent_store is not None:
        parent["image_store"] = parent_store
    parent_bytes = json.dumps(parent).encode()
    fake.put(DATASET_BUCKET, PARENT_KEY, parent_bytes)
    pin_parent(stage, version, parent_bytes)
    fake.calls.clear()
    return fake, stage, version, store


def test_tables_only_version_on_its_parents_image_store_writes_tables_then_the_manifest(
    repo,  # noqa: F811
    tmp_path,
):
    fake, stage, version, _ = stage_tables_only(tmp_path)

    run_promote(fake, stage, version)

    prefix = f"datasets/mermaid/{version}/"
    log = writes(fake)
    assert [op for op, *_ in log] == ["put"] * 5
    assert {k for _, _, k, _ in log[:-1]} == {prefix + t for t in _TABLES}
    assert log[-1][2] == prefix + "manifest.json"


def test_tables_only_version_whose_parent_has_another_image_store_refuses_with_no_writes(
    repo,  # noqa: F811
    tmp_path,
):
    other = {"bucket": "elsewhere", "prefix": "", "owner": "external"}
    fake, stage, version, _ = stage_tables_only(tmp_path, parent_store=other)

    with pytest.raises(PublishRefused, match="image store"):
        run_promote(fake, stage, version)

    assert writes(fake) == []


@pytest.mark.parametrize("fault", ["missing", "etag"])
def test_tables_only_version_with_an_absent_or_changed_image_refuses_with_no_writes(
    repo,  # noqa: F811
    tmp_path,
    fault,
):
    fake, stage, version, store = stage_tables_only(tmp_path)
    key = next(k for b, k in sorted(fake.objects) if b == store["bucket"] and "/i2." in k)
    if fault == "missing":
        del fake.objects[(store["bucket"], key)]
    else:
        fake.etags[(store["bucket"], key)] = quoted_etag(b"other")

    with pytest.raises(PublishRefused, match=key):
        run_promote(fake, stage, version)

    assert writes(fake) == []


def test_images_and_masks_land_in_intelligent_tiering_and_tables_keep_the_default(
    monkeypatch, tmp_path
):
    fake, stage, version = stage_dense(monkeypatch, tmp_path)

    run_promote(fake, stage, version, dataset=DENSE)

    classes = {key: kw.get("StorageClass") for _, _, key, kw in writes(fake)}
    images = {k for k in classes if k.startswith("images/")}
    assert len(images) == 4
    assert {classes[k] for k in images} == {"INTELLIGENT_TIERING"}
    assert {classes[k] for k in classes.keys() - images} == {None}
