"""``segdata build catlin_seaview`` from a fake legacy prefix into a LocalStore or a fake S3."""

from __future__ import annotations

import argparse
import hashlib
import io
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pandas as pd
import pyarrow as pa
import pytest
from common.conftest import FakeS3
from PIL import Image

from segdata_etl.common.local_store import LocalStore
from segdata_etl.datasets import catlin_seaview, external
from segdata_spec import DATASET_BUCKET
from segdata_spec.errors import PublishRefused
from segdata_spec.manifest import Pin, SourceInfo
from segdata_spec.publish import GitState, publish_taxonomy
from segdata_spec.read import DatasetVersion, open_version
from segdata_spec.sync import _crc64nvme_b64, sync

LEGACY_BUCKET = "dev-datamermaid-sm-sources"
PREFIX = "external_validation_datasets/catlin_seaview/"
CLEAN_GIT = GitState(sha="abcdef1234567", dirty=False, pushed=True)
TAXONOMY_VERSION = "20261101-1234567"
# (region, image_id) -> (width, height)
IMAGE_SIZES = {("ATL", "1001"): (7, 5), ("ATL", "1002"): (6, 4), ("PAC_USA", "2001"): (5, 3)}
CLASSES = {"unlabeled": 0, "Acropora digitate": 5, "Loose Substrate: Sand": 9, "Water": 12}
ANNOTATIONS = [
    ("ATL", "1001", 1, 2, "Acropora digitate"),
    ("ATL", "1001", 3, 4, "Loose Substrate: Sand"),
    ("ATL", "1002", 0, 0, "Acropora digitate"),
    ("PAC_USA", "2001", 2, 1, "Water"),
]


def _jpeg(width: int, height: int) -> bytes:
    buf = io.BytesIO()
    Image.new("RGB", (width, height), (10, 120, 200)).save(buf, format="JPEG")
    return buf.getvalue()


def _parquet(rows: list[tuple[str, str, int, int, str]]) -> bytes:
    columns = ["region", "image_id", "row", "col", "source_label_name"]
    buf = io.BytesIO()
    pd.DataFrame(rows, columns=columns).to_parquet(buf, index=False)
    return buf.getvalue()


def _legacy(
    s3: FakeS3, annotations: list[tuple[str, str, int, int, str]] = ANNOTATIONS
) -> dict[tuple[str, str], bytes]:
    """Write a Catlin-shaped legacy prefix into ``s3``; return the image bytes by (region, id)."""
    manifest = {
        "dataset_name": "catlin_seaview",
        "source": {"paper": "Seaview Survey (2019)", "raw_data_url": "http://example/Q1281"},
        "ingestion": {"skipped_damaged_or_missing_image_ids": {"ATL": [], "PAC_USA": ["x", "y"]}},
    }
    files = {
        "manifest.json": json.dumps(manifest).encode(),
        "classes.json": json.dumps(CLASSES).encode(),
        "catlin_seaview_annotations.parquet": _parquet(annotations),
    }
    images = {pair: _jpeg(*size) for pair, size in IMAGE_SIZES.items()}
    for (region, image_id), data in images.items():
        files[f"images/{region}/{image_id}.jpg"] = data
    for name, data in files.items():
        s3.objects[(LEGACY_BUCKET, PREFIX + name)] = data
    return images


def _publish_taxonomy(client: Any, repo: Path) -> str:
    tables = {
        "benthic_attributes.parquet": pa.table(
            {
                "id": ["ba-acro", "ba-sand"],
                "name": ["Acropora", "Sand"],
                "parent_id": [None, None],
            }
        ),
        "growth_forms.parquet": pa.table({"id": ["g1"], "name": ["Branching"]}),
    }
    _, sha = publish_taxonomy(
        client,
        bucket=DATASET_BUCKET,
        name="mermaid-benthic",
        version=TAXONOMY_VERSION,
        tables=tables,
        etl_package="segdata-etl",
        etl_command="segdata taxonomy",
        source=SourceInfo(type="mermaid-api", uri="https://api.test", revision="r1"),
        repo_dir=repo,
        git=CLEAN_GIT,
    )
    return sha


@pytest.fixture
def repo(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """A repo root holding ``uv.lock``, with a clean pushed git state."""
    root = tmp_path / "repo"
    root.mkdir()
    (root / "uv.lock").write_text("lock")
    monkeypatch.setattr(catlin_seaview, "find_repo_root", lambda _start: root)
    monkeypatch.setattr(external, "probe_git", lambda _dir: CLEAN_GIT)
    return root


def _args(store: str | Path, taxonomy_sha: str, *extra: str) -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    catlin_seaview.add_arguments(parser, workdir=Path("unused"))
    argv = ["--store", str(store), "--taxonomy-version", TAXONOMY_VERSION]
    return parser.parse_args([*argv, "--taxonomy-sha256", taxonomy_sha, "--workers", "2", *extra])


def _run(args: argparse.Namespace, s3: FakeS3, capsys: pytest.CaptureFixture[str]) -> Pin:
    assert catlin_seaview.run(args, s3) == 0
    lines = dict(line.split(": ", 1) for line in capsys.readouterr().out.splitlines())
    return Pin(
        name="catlin_seaview", version=lines["version"], manifest_sha256=lines["manifest_sha256"]
    )


def _expected_image_rows(images: dict[tuple[str, str], bytes]) -> list[dict[str, Any]]:
    rows = []
    for (region, image_id), data in sorted(images.items(), key=lambda kv: kv[0][1]):
        md5 = hashlib.md5(data).hexdigest()  # noqa: S324
        width, height = IMAGE_SIZES[(region, image_id)]
        rows.append(
            {
                "image_id": image_id,
                "key": f"{region}/{image_id}.jpg",
                "etag": f'"{md5}"',
                "size": len(data),
                "crc64nvme": _crc64nvme_b64(data),
                "media_type": "image/jpeg",
                "width": width,
                "height": height,
                "group": region,
                "mask_key": None,
                "mask_etag": None,
                "mask_size": None,
                "mask_crc64nvme": None,
                "source_fingerprint": md5,
                "source_name": f"{image_id}.jpg",
                "attrs": None,
            }
        )
    return rows


@dataclass(frozen=True)
class Published:
    """A ``catlin_seaview`` version built into a LocalStore from the default legacy prefix."""

    store: Path
    version: DatasetVersion
    images: dict[tuple[str, str], bytes]
    taxonomy_sha: str


@pytest.fixture
def published(repo: Path, tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> Published:
    s3 = FakeS3()
    images = _legacy(s3)
    store = tmp_path / "store"
    taxonomy_sha = _publish_taxonomy(LocalStore(store), repo)
    pin = _run(_args(store, taxonomy_sha), s3, capsys)
    return Published(store, open_version(store, DATASET_BUCKET, pin), images, taxonomy_sha)


def test_build_writes_image_rows_and_copies_the_image_bytes(published: Published) -> None:
    version = published.version
    assert sorted(version.images.to_pylist(), key=lambda r: r["image_id"]) == (
        _expected_image_rows(published.images)
    )
    assert version.manifest.image_store.prefix == "images/catlin_seaview/"
    for (region, image_id), data in published.images.items():
        copied = (
            published.store / DATASET_BUCKET / "images/catlin_seaview" / region / f"{image_id}.jpg"
        )
        assert copied.read_bytes() == data


def test_build_writes_points_labels_and_counts(published: Published) -> None:
    version = published.version
    assert version.points is not None
    assert version.points.to_pylist() == [
        {"image_id": "1001", "row": 1, "col": 2, "label_id": 5, "annotator": "default"},
        {"image_id": "1001", "row": 3, "col": 4, "label_id": 9, "annotator": "default"},
        {"image_id": "1002", "row": 0, "col": 0, "label_id": 5, "annotator": "default"},
        {"image_id": "2001", "row": 2, "col": 1, "label_id": 12, "annotator": "default"},
    ]
    assert [
        (r["label_id"], r["name"], r["mermaid_benthic_attribute_id"])
        for r in version.labels.to_pylist()
    ] == [
        (5, "Acropora digitate", "ba-acro"),
        (9, "Loose Substrate: Sand", "ba-sand"),
        (12, "Water", None),
    ]
    assert version.label_counts.num_rows == 4


def test_build_records_licence_source_dropped_and_command_in_the_manifest(
    published: Published,
) -> None:
    manifest = published.version.manifest
    assert manifest.version.endswith("-abcdef1")
    assert manifest.source.license == "CC BY-NC-SA 4.0"
    assert manifest.source.uri == f"s3://{LEGACY_BUCKET}/{PREFIX}"
    assert manifest.stats.dropped == {"damaged_images": 2}
    assert manifest.etl.command == (
        f"segdata build catlin_seaview --store {published.store}"
        f" --taxonomy-version {TAXONOMY_VERSION} --taxonomy-sha256 {published.taxonomy_sha}"
    )


def test_build_to_s3_copies_into_and_publishes_to_the_dataset_bucket(
    repo: Path,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    s3 = FakeS3()
    images = _legacy(s3)
    taxonomy_sha = _publish_taxonomy(s3, repo)
    monkeypatch.chdir(tmp_path)

    pin = _run(_args("s3", taxonomy_sha), s3, capsys)

    assert {(c["Bucket"], c["Key"]) for c in s3.copy_calls} == {
        (DATASET_BUCKET, f"images/catlin_seaview/{region}/{image_id}.jpg")
        for region, image_id in images
    }
    check = tmp_path / "check"
    sync(s3, {"catlin_seaview": pin}, check, bucket=DATASET_BUCKET)
    assert open_version(check, DATASET_BUCKET, pin).manifest.version == pin.version


def test_build_refuses_a_dirty_tree_before_copying_any_image(
    repo: Path,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    s3 = FakeS3()
    _legacy(s3)
    store = tmp_path / "store"
    taxonomy_sha = _publish_taxonomy(LocalStore(store), repo)
    dirty = GitState(sha=CLEAN_GIT.sha, dirty=True, pushed=True)
    monkeypatch.setattr(external, "probe_git", lambda _dir: dirty)

    with pytest.raises(PublishRefused):
        catlin_seaview.run(_args(store, taxonomy_sha), s3)

    assert not (store / DATASET_BUCKET / "images").exists()
    assert not (store / DATASET_BUCKET / "datasets").exists()


def test_build_refuses_a_point_label_missing_from_classes_json(repo: Path, tmp_path: Path) -> None:
    s3 = FakeS3()
    _legacy(s3, [*ANNOTATIONS, ("ATL", "1002", 1, 1, "Not a class")])
    store = tmp_path / "store"
    taxonomy_sha = _publish_taxonomy(LocalStore(store), repo)

    with pytest.raises(ValueError, match="Not a class"):
        catlin_seaview.run(_args(store, taxonomy_sha), s3)

    assert not (store / DATASET_BUCKET / "images").exists()


def test_build_refuses_an_annotated_image_without_an_image_object(
    repo: Path, tmp_path: Path
) -> None:
    s3 = FakeS3()
    _legacy(s3, [*ANNOTATIONS, ("ATL", "1003", 1, 1, "Water")])
    store = tmp_path / "store"
    taxonomy_sha = _publish_taxonomy(LocalStore(store), repo)

    with pytest.raises(ValueError, match="1003"):
        catlin_seaview.run(_args(store, taxonomy_sha), s3)

    assert not (store / DATASET_BUCKET / "images").exists()


def test_build_refuses_an_image_annotated_under_another_region(repo: Path, tmp_path: Path) -> None:
    s3 = FakeS3()
    moved = [a if a[1] != "1002" else ("PAC_USA", *a[1:]) for a in ANNOTATIONS]
    _legacy(s3, moved)
    store = tmp_path / "store"
    taxonomy_sha = _publish_taxonomy(LocalStore(store), repo)

    with pytest.raises(ValueError, match="PAC_USA"):
        catlin_seaview.run(_args(store, taxonomy_sha), s3)

    assert not (store / DATASET_BUCKET / "images").exists()


@pytest.mark.parametrize(
    ("annotation", "message"),
    [
        (("ATL", "1002", 4, 0, "Water"), "points outside"),
        (("ATL", "1002", 1, 1, "unlabeled"), "not in labels.parquet"),
    ],
    ids=["row-past-image-height", "unlabeled-point"],
)
def test_build_refuses_a_bad_point_before_copying_any_image(
    repo: Path, tmp_path: Path, annotation: tuple[str, str, int, int, str], message: str
) -> None:
    s3 = FakeS3()
    _legacy(s3, [*ANNOTATIONS, annotation])
    store = tmp_path / "store"
    taxonomy_sha = _publish_taxonomy(LocalStore(store), repo)

    with pytest.raises(ValueError, match=message):
        catlin_seaview.run(_args(store, taxonomy_sha), s3)

    assert not (store / DATASET_BUCKET / "images").exists()
