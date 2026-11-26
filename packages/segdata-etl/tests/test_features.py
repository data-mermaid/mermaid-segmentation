"""Tests for ``segdata features attach``: existing vectors copied into a published version."""

from __future__ import annotations

import io
import json
from pathlib import Path
from typing import Any

import pyarrow as pa
import pyarrow.parquet as pq
import pytest
from common.test_featurevector import featurevector_bytes
from s3_fake import FakeS3, client_error, crc64nvme, md5_hex

from segdata_etl import cli, features
from segdata_etl.common.featurevector import FeatureVectorsInvalid, FeatureVectorsMissing
from segdata_spec import DATASET_BUCKET
from segdata_spec.errors import ExtractorMismatch, PinMismatch
from segdata_spec.manifest import ImageStore, SourceInfo
from segdata_spec.publish import GitState, publish_dataset

VERSION = "20261120-abcdef1"
SRC_BUCKET = "vectors"
TEMPLATE = "s3://vectors/{source_id}/{image_id}.featurevector"
SET_PREFIX = f"datasets/ds/{VERSION}/features/effnet/"
EXTRACTOR = {
    "class_path": "spacer.EfficientNet",
    "crop_size": 224,
    "feature_dim": 2,
    "weights_uri": "s3://w/effnet.pt",
    "weights_sha256": "0" * 64,
}
CLEAN_GIT = GitState(sha="abcdef1234567", dirty=False, pushed=True)
# image_id -> (attrs source_id, points)
IMAGES: dict[str, tuple[str, list[tuple[int, int]]]] = {
    "a": ("s1", [(1, 3), (2, 4)]),
    "b": ("s2", [(5, 6)]),
}


@pytest.fixture
def repo(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    root = tmp_path / "repo"
    root.mkdir()
    (root / "uv.lock").write_bytes(b"lock")
    monkeypatch.setattr(features, "REPO_ROOT", root)
    monkeypatch.setattr(features, "probe_git", lambda _: CLEAN_GIT)
    return root


def publish_version(fake: FakeS3, repo: Path, groups: dict[str, str] | None = None) -> str:
    """Publish dataset ``ds`` with IMAGES and their points; return its manifest sha; clear calls.

    ``groups`` maps image ids to their group (default ``g``).
    """
    groups = groups or {}
    ids = list(IMAGES)
    points = [(i, r, c) for i, (_, pts) in IMAGES.items() for r, c in pts]
    images = pa.table(
        {
            "image_id": ids,
            "key": [f"{i}.jpg" for i in ids],
            "etag": ["e"] * len(ids),
            "size": [1] * len(ids),
            "crc64nvme": pa.nulls(len(ids), pa.string()),
            "media_type": ["image/jpeg"] * len(ids),
            "width": [8] * len(ids),
            "height": [8] * len(ids),
            "group": [groups.get(i, "g") for i in ids],
            "mask_key": pa.nulls(len(ids), pa.string()),
            "mask_etag": pa.nulls(len(ids), pa.string()),
            "mask_size": pa.nulls(len(ids), pa.int64()),
            "mask_crc64nvme": pa.nulls(len(ids), pa.string()),
            "source_fingerprint": ["fp"] * len(ids),
            "source_name": pa.nulls(len(ids), pa.string()),
            "attrs": [json.dumps({"source_id": IMAGES[i][0]}) for i in ids],
        }
    )
    tables = {
        "images.parquet": images,
        "points.parquet": pa.table(
            {
                "image_id": [p[0] for p in points],
                "row": [p[1] for p in points],
                "col": [p[2] for p in points],
                "label_id": [1] * len(points),
                "annotator": ["default"] * len(points),
            }
        ),
        "labels.parquet": pa.table(
            {
                "label_id": [1],
                "name": ["x"],
                "mermaid_benthic_attribute_id": pa.nulls(1, pa.string()),
                "mermaid_benthic_attribute": pa.nulls(1, pa.string()),
                "mermaid_growth_form": pa.nulls(1, pa.string()),
                "mapping_source": ["identity"],
            }
        ),
        "stats/label_counts.parquet": pa.table({"image_id": ["a"], "label_id": [1], "n": [2]}),
    }
    _, sha = publish_dataset(
        fake,
        bucket=DATASET_BUCKET,
        dataset="ds",
        version=VERSION,
        kind="points",
        tables=tables,
        image_store=ImageStore(bucket=DATASET_BUCKET, prefix="images/ds/", owner="segdata"),
        etl_package="segdata-etl",
        etl_command="segdata build ds",
        source=SourceInfo(type="test", uri="s3://src/", revision="r1"),
        repo_dir=repo,
        git=CLEAN_GIT,
    )
    fake.calls.clear()
    return sha


def seed_vector(fake: FakeS3, image_id: str, points: list[tuple[int, int]]) -> bytes:
    data = featurevector_bytes(points)
    fake.put(SRC_BUCKET, f"{IMAGES[image_id][0]}/{image_id}.featurevector", data)
    return data


def attach(fake: FakeS3, tmp_path: Path, sha: str, *extra: str, template: str = TEMPLATE) -> int:
    spec = tmp_path / "extractor.json"
    spec.write_text(json.dumps(EXTRACTOR))
    argv = [
        "features",
        "attach",
        "ds",
        "--version",
        VERSION,
        "--manifest-sha256",
        sha,
        "--family",
        "effnet",
        "--source-template",
        template,
        "--extractor-spec",
        str(spec),
        "--cache-dir",
        str(tmp_path / "cache"),
        "--workers",
        "2",
        *extra,
    ]
    args = cli.build_parser().parse_args(argv)
    return int(args.run(args, fake))


def published_index(fake: FakeS3) -> list[dict[str, Any]]:
    data = fake.objects[(DATASET_BUCKET, SET_PREFIX + "index.parquet")]
    return pq.read_table(io.BytesIO(data)).to_pylist()


def test_template_fills_attrs_and_row_fields_per_image(repo, tmp_path):
    fake = FakeS3()
    sha = publish_version(fake, repo)
    for image_id, (_, pts) in IMAGES.items():
        seed_vector(fake, image_id, pts)

    attach(fake, tmp_path, sha)

    sources = sorted(call["CopySource"]["Key"] for call in fake.copy_calls)
    assert sources == ["s1/a.featurevector", "s2/b.featurevector"]
    assert {call["CopySource"]["Bucket"] for call in fake.copy_calls} == {SRC_BUCKET}


def test_unknown_template_field_fails_before_any_source_call(repo, tmp_path):
    fake = FakeS3()
    sha = publish_version(fake, repo)

    with pytest.raises(ValueError, match="nope"):
        attach(fake, tmp_path, sha, template="s3://vectors/{nope}/{image_id}.featurevector")

    assert {bucket for _, bucket, _, _ in fake.calls} == {DATASET_BUCKET}
    assert fake.copy_calls == []


def test_covering_vectors_are_copied_conditionally_and_published_as_a_set(repo, tmp_path):
    fake = FakeS3()
    sha = publish_version(fake, repo)
    blobs = {i: seed_vector(fake, i, pts) for i, (_, pts) in IMAGES.items()}

    assert attach(fake, tmp_path, sha) == 0

    destinations = sorted(call["Key"] for call in fake.copy_calls)
    assert destinations == [
        "features/ds/effnet/a.featurevector",
        "features/ds/effnet/b.featurevector",
    ]
    assert all(c["IfNoneMatch"] == "*" and c["CopySourceIfMatch"] for c in fake.copy_calls)
    assert sorted(published_index(fake), key=lambda r: r["image_id"]) == [
        {
            "image_id": "a",
            "key": "features/ds/effnet/a.featurevector",
            "etag": md5_hex(blobs["a"]),
            "size": len(blobs["a"]),
            "crc64nvme": crc64nvme(blobs["a"]),
            "n_points": 2,
        },
        {
            "image_id": "b",
            "key": "features/ds/effnet/b.featurevector",
            "etag": md5_hex(blobs["b"]),
            "size": len(blobs["b"]),
            "crc64nvme": crc64nvme(blobs["b"]),
            "n_points": 1,
        },
    ]
    manifest = json.loads(fake.objects[(DATASET_BUCKET, SET_PREFIX + "manifest.json")])
    assert manifest["stats"] == {"n_images": 2, "n_missing": 0}


def test_absent_source_fails_with_no_copy_and_no_set(repo, tmp_path):
    fake = FakeS3()
    sha = publish_version(fake, repo)
    seed_vector(fake, "a", IMAGES["a"][1])

    with pytest.raises(FeatureVectorsMissing, match=r"1 image.*\['b'\]"):
        attach(fake, tmp_path, sha)

    assert fake.copy_calls == []
    assert (DATASET_BUCKET, SET_PREFIX + "manifest.json") not in fake.objects


def test_absent_source_with_allow_missing_publishes_the_rest(repo, tmp_path):
    fake = FakeS3()
    sha = publish_version(fake, repo)
    seed_vector(fake, "a", IMAGES["a"][1])

    attach(fake, tmp_path, sha, "--allow-missing")

    assert [call["Key"] for call in fake.copy_calls] == ["features/ds/effnet/a.featurevector"]
    manifest = json.loads(fake.objects[(DATASET_BUCKET, SET_PREFIX + "manifest.json")])
    assert manifest["stats"] == {"n_images": 1, "n_missing": 1}


def test_vector_missing_one_point_fails_with_no_copy_and_no_set(repo, tmp_path):
    fake = FakeS3()
    sha = publish_version(fake, repo)
    seed_vector(fake, "a", [(1, 3)])
    seed_vector(fake, "b", IMAGES["b"][1])

    with pytest.raises(FeatureVectorsInvalid, match="s3://vectors/s1/a.featurevector"):
        attach(fake, tmp_path, sha)

    assert fake.copy_calls == []
    assert (DATASET_BUCKET, SET_PREFIX + "manifest.json") not in fake.objects


def test_wrong_manifest_sha_fails_before_any_source_head(repo, tmp_path):
    fake = FakeS3()
    publish_version(fake, repo)
    for image_id, (_, pts) in IMAGES.items():
        seed_vector(fake, image_id, pts)

    with pytest.raises(PinMismatch):
        attach(fake, tmp_path, "f" * 64)

    assert fake.ops("head") == [] and fake.copy_calls == []


def test_images_of_every_group_are_copied_and_indexed(repo, tmp_path):
    fake = FakeS3()
    sha = publish_version(fake, repo, groups={"a": "g1", "b": "g2"})
    for image_id, (_, pts) in IMAGES.items():
        seed_vector(fake, image_id, pts)

    attach(fake, tmp_path, sha)

    assert sorted(call["Key"] for call in fake.copy_calls) == [
        "features/ds/effnet/a.featurevector",
        "features/ds/effnet/b.featurevector",
    ]
    index = sorted(published_index(fake), key=lambda r: r["image_id"])
    assert [(r["image_id"], r["n_points"]) for r in index] == [("a", 2), ("b", 1)]


def test_unreadable_vector_is_refused_by_key_with_no_copy(repo, tmp_path):
    fake = FakeS3()
    sha = publish_version(fake, repo)
    fake.put(SRC_BUCKET, "s1/a.featurevector", b'{"rows": [1, 2], "cols": [3, 4]}')
    seed_vector(fake, "b", IMAGES["b"][1])

    with pytest.raises(
        FeatureVectorsInvalid, match=r"^1 .*s3://vectors/s1/a.featurevector: unreadable"
    ):
        attach(fake, tmp_path, sha)

    assert fake.copy_calls == []


def test_template_naming_one_source_for_two_images_fails_before_any_source_call(repo, tmp_path):
    fake = FakeS3()
    sha = publish_version(fake, repo, groups={"a": "g1", "b": "g1"})

    with pytest.raises(ValueError, match=r"same source.*\['a', 'b'\]"):
        attach(fake, tmp_path, sha, template="s3://v/{group}/{group}.featurevector")

    assert {bucket for _, bucket, _, _ in fake.calls} == {DATASET_BUCKET}


def put_family_file(fake: FakeS3, image_id: str, points: list[tuple[int, int]]) -> bytes:
    data = featurevector_bytes(points)
    fake.put(DATASET_BUCKET, f"features/ds/effnet/{image_id}.featurevector", data)
    return data


def source_calls(fake: FakeS3) -> list[tuple[str, str]]:
    return [(op, key) for op, bucket, key, _ in fake.calls if bucket != DATASET_BUCKET]


def test_existing_family_file_is_adopted_and_only_the_template_source_copied(repo, tmp_path):
    fake = FakeS3()
    sha = publish_version(fake, repo)
    adopted = put_family_file(fake, "a", IMAGES["a"][1])
    seed_vector(fake, "a", IMAGES["a"][1])
    copied = seed_vector(fake, "b", IMAGES["b"][1])

    attach(fake, tmp_path, sha)

    assert [c["CopySource"]["Key"] for c in fake.copy_calls] == ["s2/b.featurevector"]
    assert [key for _, key in source_calls(fake)] == ["s2/b.featurevector"] * 2
    index = sorted(published_index(fake), key=lambda r: r["image_id"])
    assert [(r["image_id"], r["key"], r["etag"], r["n_points"]) for r in index] == [
        ("a", "features/ds/effnet/a.featurevector", md5_hex(adopted), 2),
        ("b", "features/ds/effnet/b.featurevector", md5_hex(copied), 1),
    ]


def test_adopted_file_missing_a_point_is_refused_by_its_key(repo, tmp_path):
    fake = FakeS3()
    sha = publish_version(fake, repo)
    put_family_file(fake, "a", [(1, 3)])
    seed_vector(fake, "b", IMAGES["b"][1])

    with pytest.raises(
        FeatureVectorsInvalid, match=f"s3://{DATASET_BUCKET}/features/ds/effnet/a.featurevector"
    ):
        attach(fake, tmp_path, sha)

    assert fake.copy_calls == []
    assert (DATASET_BUCKET, SET_PREFIX + "manifest.json") not in fake.objects


def test_family_claimed_by_another_extractor_refuses_before_any_source_call(repo, tmp_path):
    fake = FakeS3()
    sha = publish_version(fake, repo)
    stored = {**EXTRACTOR, "weights_sha256": "1" * 64}
    fake.put(DATASET_BUCKET, "features/ds/effnet/_extractor_spec.json", json.dumps(stored).encode())
    for image_id, (_, pts) in IMAGES.items():
        seed_vector(fake, image_id, pts)

    with pytest.raises(ExtractorMismatch, match="weights_sha256"):
        attach(fake, tmp_path, sha)

    assert fake.copy_calls == []
    assert source_calls(fake) == []
    assert (DATASET_BUCKET, SET_PREFIX + "manifest.json") not in fake.objects


@pytest.mark.parametrize(
    ("seed_kwargs", "reason"),
    [({"crc": False}, "lacks a CRC64NVME"), ({"etag": '"0123456789abcdef"'}, "ETag")],
)
def test_adopted_file_whose_metadata_disagrees_with_its_bytes_is_refused(
    repo, tmp_path, seed_kwargs, reason
):
    fake = FakeS3()
    sha = publish_version(fake, repo)
    key = "features/ds/effnet/a.featurevector"
    fake.put(DATASET_BUCKET, key, featurevector_bytes(IMAGES["a"][1]), **seed_kwargs)
    seed_vector(fake, "b", IMAGES["b"][1])

    with pytest.raises(FeatureVectorsInvalid, match=f"s3://{DATASET_BUCKET}/{key}: .*{reason}"):
        attach(fake, tmp_path, sha)

    assert fake.copy_calls == []
    assert (DATASET_BUCKET, SET_PREFIX + "manifest.json") not in fake.objects


def test_rerun_after_a_copy_stop_adopts_the_files_already_copied(repo, tmp_path):
    class FailsCopyOfB(FakeS3):
        failed = False

        def copy_object(self, **kwargs: Any) -> dict[str, Any]:
            if kwargs["Key"].endswith("/b.featurevector") and not self.failed:
                self.failed = True
                raise client_error("InternalError", 500, "CopyObject")
            return super().copy_object(**kwargs)

    fake = FailsCopyOfB()
    sha = publish_version(fake, repo)
    for image_id, (_, pts) in IMAGES.items():
        seed_vector(fake, image_id, pts)
    with pytest.raises(Exception, match="InternalError"):
        attach(fake, tmp_path, sha)
    assert (DATASET_BUCKET, "features/ds/effnet/a.featurevector") in fake.objects
    fake.copy_calls.clear()

    attach(fake, tmp_path, sha)

    assert [c["Key"] for c in fake.copy_calls] == ["features/ds/effnet/b.featurevector"]
    assert [r["image_id"] for r in published_index(fake)] == ["a", "b"]
