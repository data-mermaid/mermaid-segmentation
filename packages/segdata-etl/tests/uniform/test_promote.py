"""Tests for ``segdata promote``: a staged version uploaded write-once, images then tables then manifest."""

from __future__ import annotations

import hashlib
import json
import shutil
from datetime import datetime
from pathlib import Path
from typing import Any

import pyarrow.compute as pc
import pyarrow.parquet as pq
import pytest
from s3_fake import FakeS3, client_error, crc64nvme, quoted_etag

from segdata_etl import cli, promote
from segdata_etl.common.stage import next_version
from segdata_spec import DATASET_BUCKET
from segdata_spec.errors import ChecksumMismatch, ExtractorMismatch, PublishRefused
from segdata_spec.manifest import Manifest, Pin, dump_manifest

from .conftest import jpeg_bytes, png_bytes
from .test_coralscapes_v2 import IMG_A, IMG_B, MASK, MASK_B
from .test_coralscapes_v2 import build as build_coralscapes
from .test_coralscapes_v2 import shard as coralscapes_shard
from .test_mermaid import (  # noqa: F401
    ENCRUSTING,
    RUBBLE,
    SRC_BUCKET,
    ann,
    only_version,
    repo,
    seed,
)
from .test_mermaid import build as build_mermaid

BLOBS = {"i1": png_bytes(8, 6), "i2": png_bytes(12, 9), "i3": jpeg_bytes(5, 5)}
_TABLES = ("images.parquet", "points.parquet", "labels.parquet", "stats/label_counts.parquet")


def stage_mermaid(
    tmp_path: Path, *extra: str, features: bool = False, drop_vector: str | None = None
) -> tuple[FakeS3, Path, str]:
    """Stage a mermaid version; the returned fake holds the sources and nothing else.

    Without ``features`` the staged feature plan, index and spec are removed. ``drop_vector``
    names an image whose source vector is deleted before the build; ``extra`` goes to the build.
    """
    fake = FakeS3()
    rows = [
        ann("i1", "p1", 10, 20, ENCRUSTING),
        ann("i2", "p2", 1, 2, RUBBLE),
        ann("i3", "p3", 7, 8, RUBBLE),
    ]
    seed(fake, rows, BLOBS)
    if drop_vector is not None:
        del fake.objects[(SRC_BUCKET, f"mermaid/{drop_vector}_featurevector")]
    stage = tmp_path / "stage"
    assert build_mermaid(fake, stage, *extra) == 0
    fake.calls.clear()
    version = only_version(stage)
    if not features:
        shutil.rmtree(version / "features")
    return fake, stage, version.name


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


def test_staged_features_publish_after_the_dataset_manifest(repo, tmp_path):  # noqa: F811
    fake, stage, version = stage_mermaid(tmp_path, features=True)

    run_promote(fake, stage, version)

    prefix = f"datasets/mermaid/{version}/"
    set_prefix = f"{prefix}features/efficientnet-v1/"
    log = [(op, key) for op, _, key, _ in writes(fake)]
    assert [op for op, _ in log[:3]] == ["copy"] * 3
    assert all(key.startswith("images/mermaid/") for _, key in log[:3])
    assert {key for _, key in log[3:7]} == {prefix + t for t in _TABLES}
    assert log[7:9] == [
        ("put", prefix + "manifest.json"),
        ("put", "features/mermaid/efficientnet-v1/_extractor_spec.json"),
    ]
    assert sorted(key for _, key in log[9:12]) == [
        f"features/mermaid/efficientnet-v1/{i}.featurevector" for i in ("i1", "i2", "i3")
    ]
    assert log[12:] == [
        ("put", set_prefix + "index.parquet"),
        ("put", set_prefix + "manifest.json"),
    ]
    feature_copies = [c for c in fake.copy_calls if c["Key"].startswith("features/")]
    assert {c["StorageClass"] for c in feature_copies} == {"STANDARD"}
    set_manifest = json.loads(fake.objects[(DATASET_BUCKET, set_prefix + "manifest.json")])
    dataset_manifest = fake.objects[(DATASET_BUCKET, prefix + "manifest.json")]
    assert (
        set_manifest["dataset"]["manifest_sha256"] == hashlib.sha256(dataset_manifest).hexdigest()
    )
    assert set_manifest["stats"] == {"n_images": 3, "n_missing": 0}
    assert set_manifest == json.loads(
        (version_dir(stage, "mermaid", version) / FAMILY_DIR / "manifest.json").read_bytes()
    )
    spec = fake.objects[(DATASET_BUCKET, SPEC_KEY)]
    assert json.loads(spec) == set_manifest["extractor"]


SPEC_KEY = "features/mermaid/efficientnet-v1/_extractor_spec.json"
FAMILY_DIR = "features/efficientnet-v1"
FEATURE_PLAN = f"{FAMILY_DIR}/copies.jsonl"


def test_stored_extractor_spec_of_another_identity_refuses_before_any_write(repo, tmp_path):  # noqa: F811
    fake, stage, version = stage_mermaid(tmp_path, features=True)
    staged = version_dir(stage, "mermaid", version) / FAMILY_DIR / "manifest.json"
    stored = {**json.loads(staged.read_text())["extractor"], "weights_sha256": "1" * 64}
    fake.put(DATASET_BUCKET, SPEC_KEY, json.dumps(stored).encode())

    with pytest.raises(ExtractorMismatch, match="weights_sha256"):
        run_promote(fake, stage, version)

    assert writes(fake) == []


def test_rerun_after_a_failed_feature_publish_completes_only_the_feature_set(repo, tmp_path):  # noqa: F811
    class FailsSetIndexPut(FakeS3):
        failed = False

        def put_object(self, **kwargs: Any) -> dict[str, Any]:
            set_index = kwargs["Key"].endswith("/features/efficientnet-v1/index.parquet")
            if set_index and not self.failed:
                self.failed = True
                raise client_error("InternalError", 500, "PutObject")
            return super().put_object(**kwargs)

    fake, stage, version = stage_mermaid(tmp_path, features=True)
    flaky = FailsSetIndexPut(**{k: getattr(fake, k) for k in ("objects", "crcs", "etags")})
    with pytest.raises(Exception, match="InternalError"):
        run_promote(flaky, stage, version)
    set_manifest = (
        DATASET_BUCKET,
        f"datasets/mermaid/{version}/features/efficientnet-v1/manifest.json",
    )
    assert set_manifest not in flaky.objects
    flaky.calls.clear()

    run_promote(flaky, stage, version)

    assert set_manifest in flaky.objects
    written = [key for _, _, key, _ in writes(flaky)]
    assert not [
        k
        for k in written
        if k.startswith(("images/", f"datasets/mermaid/{version}/")) and "/features/" not in k
    ]
    assert written[-1] == set_manifest[1]


def test_rerun_after_a_completed_feature_promote_refuses_as_already_published(repo, tmp_path):  # noqa: F811
    fake, stage, version = stage_mermaid(tmp_path, features=True)
    run_promote(fake, stage, version)
    fake.calls.clear()

    with pytest.raises(PublishRefused, match="already published"):
        run_promote(fake, stage, version)

    assert writes(fake) == []


def test_published_set_manifest_with_other_bytes_refuses_before_any_write(repo, tmp_path):  # noqa: F811
    fake, stage, version = stage_mermaid(tmp_path, features=True)
    fake.put(DATASET_BUCKET, f"datasets/mermaid/{version}/{FAMILY_DIR}/manifest.json", b"{}")

    with pytest.raises(PublishRefused, match="already published.*other bytes"):
        run_promote(fake, stage, version)

    assert writes(fake) == []


def test_family_folder_without_its_manifest_refuses_before_any_s3_call(repo, tmp_path):  # noqa: F811
    fake, stage, version = stage_mermaid(tmp_path, features=True)
    (version_dir(stage, "mermaid", version) / FAMILY_DIR / "manifest.json").unlink()

    with pytest.raises(PublishRefused, match="not fully staged"):
        run_promote(fake, stage, version)

    assert fake.calls == []


def test_allow_missing_build_promotes_a_set_recording_the_missing_image(repo, tmp_path):  # noqa: F811
    fake, stage, version = stage_mermaid(
        tmp_path, "--allow-missing-features", features=True, drop_vector="i2"
    )

    run_promote(fake, stage, version)

    key = f"datasets/mermaid/{version}/{FAMILY_DIR}/manifest.json"
    assert json.loads(fake.objects[(DATASET_BUCKET, key)])["stats"] == {
        "n_images": 2,
        "n_missing": 1,
    }


def test_staged_index_that_differs_from_its_set_manifest_refuses_before_any_write(
    repo,  # noqa: F811
    tmp_path,
):
    fake, stage, version = stage_mermaid(tmp_path, features=True)
    index_path = version_dir(stage, "mermaid", version) / FAMILY_DIR / "index.parquet"
    index = pq.read_table(index_path)
    pq.write_table(index.filter(pc.not_equal(index["image_id"], "i2")), index_path)

    with pytest.raises(ChecksumMismatch, match="index.parquet"):
        run_promote(fake, stage, version)

    assert writes(fake) == []


def _drop_first_line(lines: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return lines[1:]


def test_published_manifest_refuses_before_any_write(repo, tmp_path):  # noqa: F811
    fake, stage, version = stage_mermaid(tmp_path)
    key = f"datasets/mermaid/{version}/manifest.json"
    fake.put(DATASET_BUCKET, key, b"{}")
    fake.calls.clear()

    with pytest.raises(PublishRefused, match="already published"):
        run_promote(fake, stage, version)

    assert [(op, k) for op, _, k, _ in fake.calls] == [("head", key), ("get", key)]


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


def rewrite_plan(stage: Path, version: str, edit, name: str = "copies.jsonl") -> None:
    path = version_dir(stage, "mermaid", version) / name
    lines = [json.loads(line) for line in path.read_text().splitlines()]
    path.write_text("".join(json.dumps(line) + "\n" for line in edit(lines)))


def test_plan_missing_a_line_without_a_parent_refuses_at_preflight(repo, tmp_path):  # noqa: F811
    fake, stage, version = stage_mermaid(tmp_path)
    rewrite_plan(stage, version, lambda lines: lines[1:])

    with pytest.raises(PublishRefused, match="needs a published parent"):
        run_promote(fake, stage, version)

    assert writes(fake) == []


@pytest.mark.parametrize(
    ("field", "edit_value"),
    [("size", lambda v: v + 1)],
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


def test_cli_dispatches_to_promote_with_its_flags(monkeypatch, tmp_path):
    seen: dict[str, Any] = {}

    def fake_promote(client: Any, **kwargs: Any) -> None:
        seen.update(kwargs, client=client)

    client = object()
    monkeypatch.setattr(promote, "promote", fake_promote)
    pool_workers: list[Any] = []

    def make_client(**kw: Any) -> object:
        pool_workers.append(kw.get("workers"))
        return client

    monkeypatch.setattr(cli, "make_s3_client", make_client)

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
    assert pool_workers == [3]


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

    ``differs`` names the fingerprint that catches it: ``etag`` keeps the planned CRC and
    ``crc`` keeps the planned ETag.
    """
    flipped = bytes([data[0] ^ 1]) + data[1:]
    fake.put(DATASET_BUCKET, key, flipped, etag=quoted_etag(data) if differs == "crc" else None)
    if differs == "etag":
        fake.crcs[(DATASET_BUCKET, key)] = crc64nvme(data)


@pytest.mark.parametrize("differs", ["etag", "crc"])
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


def publish_parent(
    fake: FakeS3, stage: Path, version: str, parent_store: dict[str, Any] | None = None
) -> dict[str, Any]:
    """Publish a parent of the staged mermaid version, on its image store or ``parent_store``, and pin it.

    Returns the version's image store.
    """
    parent = json.loads((version_dir(stage, "mermaid", version) / "manifest.json").read_bytes())
    store = parent["image_store"]
    parent.update(dataset="parent", version=PARENT_VERSION)
    if parent_store is not None:
        parent["image_store"] = parent_store
    parent_bytes = json.dumps(parent).encode()
    fake.put(DATASET_BUCKET, PARENT_KEY, parent_bytes)
    pin_parent(stage, version, parent_bytes)
    return store


def stage_tables_only(
    tmp_path: Path, parent_store: dict[str, Any] | None = None
) -> tuple[FakeS3, Path, str, dict[str, Any]]:
    """Stage a mermaid version without plan files, its images already in its image store.

    Its one parent is published with the same image store, or with ``parent_store``.
    """
    fake, stage, version = stage_mermaid(tmp_path)
    store = json.loads((version_dir(stage, "mermaid", version) / "manifest.json").read_bytes())[
        "image_store"
    ]
    for line in copy_destinations(stage, version):
        data = fake.objects[(line["src_bucket"], line["src_key"])]
        fake.put(store["bucket"], line["dst_key"], data)
    (version_dir(stage, "mermaid", version) / "copies.jsonl").unlink()
    publish_parent(fake, stage, version, parent_store)
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


TAXONOMY_KEY = "taxonomy/mermaid-benthic/20261124-e6abfad/manifest.json"
TAXONOMY_BYTES = b'{"published": "taxonomy"}'


def pin_taxonomy(stage: Path, version: str, sha: str) -> None:
    """Rewrite the staged manifest so it pins the taxonomy ``sha``."""
    path = version_dir(stage, "mermaid", version) / "manifest.json"
    pin = Pin(name="mermaid-benthic", version="20261124-e6abfad", manifest_sha256=sha)
    manifest = Manifest.model_validate_json(path.read_bytes())
    path.write_bytes(dump_manifest(manifest.model_copy(update={"taxonomy": pin})))


def test_a_pinned_taxonomy_absent_from_the_bucket_refuses_with_no_writes(repo, tmp_path):  # noqa: F811
    fake, stage, version = stage_mermaid(tmp_path)
    pin_taxonomy(stage, version, hashlib.sha256(TAXONOMY_BYTES).hexdigest())

    with pytest.raises(PublishRefused, match="not published"):
        run_promote(fake, stage, version)

    assert writes(fake) == []


def test_a_pinned_taxonomy_published_with_its_sha_lets_the_promote_proceed(repo, tmp_path):  # noqa: F811
    fake, stage, version = stage_mermaid(tmp_path)
    fake.put(DATASET_BUCKET, TAXONOMY_KEY, TAXONOMY_BYTES)
    pin_taxonomy(stage, version, hashlib.sha256(TAXONOMY_BYTES).hexdigest())
    fake.calls.clear()

    run_promote(fake, stage, version)

    assert (DATASET_BUCKET, f"datasets/mermaid/{version}/manifest.json") in fake.objects


def stage_carried(
    tmp_path: Path, parent_store: dict[str, Any] | None = None
) -> tuple[FakeS3, Path, str, str]:
    """Stage a mermaid version whose plan omits image i2, which is already in its image store.

    Returns the fake, the stage dir, the version and the carried key.
    """
    fake, stage, version = stage_mermaid(tmp_path)
    store = json.loads((version_dir(stage, "mermaid", version) / "manifest.json").read_bytes())[
        "image_store"
    ]
    line = next(entry for entry in copy_destinations(stage, version) if "/i2." in entry["dst_key"])
    fake.put(store["bucket"], line["dst_key"], fake.objects[(line["src_bucket"], line["src_key"])])
    rewrite_plan(
        stage, version, lambda lines: [x for x in lines if x["dst_key"] != line["dst_key"]]
    )
    publish_parent(fake, stage, version, parent_store)
    fake.calls.clear()
    return fake, stage, version, line["dst_key"]


def test_a_carried_key_is_headed_in_the_image_store_and_never_copied(repo, tmp_path):  # noqa: F811
    fake, stage, version, carried = stage_carried(tmp_path)

    run_promote(fake, stage, version)

    assert sorted(call["Key"] for call in fake.copy_calls) == sorted(
        line["dst_key"] for line in copy_destinations(stage, version)
    )
    assert carried not in {call["Key"] for call in fake.copy_calls}
    assert ("head", DATASET_BUCKET, carried) in [(op, b, k) for op, b, k, _ in fake.calls]
    assert fake.puts[-1] == f"datasets/mermaid/{version}/manifest.json"


@pytest.mark.parametrize("fault", ["missing", "etag"])
def test_a_carried_key_absent_or_changed_refuses_before_any_write(repo, tmp_path, fault):  # noqa: F811
    fake, stage, version, carried = stage_carried(tmp_path)
    location = (DATASET_BUCKET, carried)
    planned_etag = fake.etag(*location)
    if fault == "missing":
        del fake.objects[location]
    elif fault == "etag":
        fake.etags[location] = quoted_etag(b"other")
    elif fault == "size":
        planned_crc = fake.crc(*location)
        fake.objects[location] += b"x"
        fake.etags[location] = planned_etag
        fake.crcs[location] = planned_crc
    else:
        fake.crcs[location] = crc64nvme(b"other")

    with pytest.raises(PublishRefused, match=carried):
        run_promote(fake, stage, version)

    assert writes(fake) == []


def test_a_stage_holding_only_a_feature_set_publishes_it_against_the_published_dataset(
    repo,  # noqa: F811
    tmp_path,
):
    fake, stage, version = stage_mermaid(tmp_path, features=True)
    run_promote(fake, stage, version)
    set_key = f"datasets/mermaid/{version}/{FAMILY_DIR}/manifest.json"
    del fake.objects[(DATASET_BUCKET, set_key)]
    folder = version_dir(stage, "mermaid", version)
    for name in ("manifest.json", "copies.jsonl", *_TABLES):
        (folder / name).unlink()
    fake.calls.clear()

    run_promote(fake, stage, version)

    assert [key for _, _, key, _ in writes(fake)][-1] == set_key
    assert not [k for _, _, k, _ in writes(fake) if k.startswith("images/")]


def test_a_stage_holding_only_a_feature_set_of_an_unpublished_dataset_refuses(repo, tmp_path):  # noqa: F811
    fake, stage, version = stage_mermaid(tmp_path, features=True)
    (version_dir(stage, "mermaid", version) / "manifest.json").unlink()

    with pytest.raises(PublishRefused, match="is not published"):
        run_promote(fake, stage, version)

    assert writes(fake) == []


def test_an_index_key_no_copy_names_is_carried_when_the_family_holds_it(repo, tmp_path):  # noqa: F811
    fake, stage, version = stage_mermaid(tmp_path, features=True)
    lines = [
        json.loads(line)
        for line in (version_dir(stage, "mermaid", version) / FEATURE_PLAN).read_text().splitlines()
    ]
    first = lines[0]
    fake.put(
        DATASET_BUCKET, first["dst_key"], fake.objects[(first["src_bucket"], first["src_key"])]
    )
    rewrite_plan(stage, version, _drop_first_line, name=FEATURE_PLAN)
    fake.calls.clear()

    run_promote(fake, stage, version)

    assert first["dst_key"] not in {c["Key"] for c in fake.copy_calls}
    set_key = f"datasets/mermaid/{version}/{FAMILY_DIR}/manifest.json"
    assert (DATASET_BUCKET, set_key) in fake.objects
