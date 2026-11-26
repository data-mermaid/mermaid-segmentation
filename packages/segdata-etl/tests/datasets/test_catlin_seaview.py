"""``segdata build catlin_seaview`` from a fake ingest prefix into a local stage."""

from __future__ import annotations

import io
import json
from datetime import UTC, datetime
from pathlib import Path

import pandas as pd
import pytest
from s3_fake import FakeS3

from segdata_etl.common.s3_objects import SourceChanged
from segdata_etl.datasets import catlin_seaview
from segdata_spec import DATASET_BUCKET
from segdata_spec.errors import PublishRefused
from segdata_spec.git import GitState

from .conftest import (
    CLEAN_GIT,
    INGEST_BUCKET,
    TAXONOMY_VERSION,
    Staged,
    assert_copies_match_sources,
    assert_curated_label_mapping,
    build_args,
    ingest_manifest_bytes,
    ingested_etag,
    jpeg,
    make_repo,
    no_mask,
    probed_columns,
    publish_fixture_taxonomy,
    stage_version,
)

PREFIX = "external_validation_datasets/catlin_seaview/"
# (region, image_id) -> (width, height)
IMAGE_SIZES = {("ATL", "1001"): (7, 5), ("ATL", "1002"): (6, 4), ("PAC_USA", "2001"): (5, 3)}
CLASSES = {"unlabeled": 0, "Acropora digitate": 5, "Loose Substrate: Sand": 9, "Water": 12}
ANNOTATIONS = [
    ("ATL", "1001", 1, 2, "Acropora digitate"),
    ("ATL", "1001", 3, 4, "Loose Substrate: Sand"),
    ("ATL", "1002", 0, 0, "Acropora digitate"),
    ("PAC_USA", "2001", 2, 1, "Water"),
]


def _parquet(rows: list[tuple[str, str, int, int, str]]) -> bytes:
    columns = ["region", "image_id", "row", "col", "source_label_name"]
    buf = io.BytesIO()
    pd.DataFrame(rows, columns=columns).to_parquet(buf, index=False)
    return buf.getvalue()


def _images() -> dict[tuple[str, str], bytes]:
    return {pair: jpeg(*size) for pair, size in IMAGE_SIZES.items()}


def _ingested(
    s3: FakeS3,
    annotations: list[tuple[str, str, int, int, str]] = ANNOTATIONS,
    images: dict[tuple[str, str], bytes] | None = None,
) -> None:
    """Write a Catlin-shaped ingest prefix into ``s3``; ``images`` replaces the image bytes."""
    files = {
        "manifest.json": ingest_manifest_bytes("catlin_seaview", {"damaged_images": 2}),
        "classes.json": json.dumps(CLASSES).encode(),
        "catlin_seaview_annotations.parquet": _parquet(annotations),
    }
    for (region, image_id), data in (images or _images()).items():
        files[f"images/{region}/{image_id}.jpg"] = data
    for name, data in files.items():
        s3.put(INGEST_BUCKET, PREFIX + name, data)


@pytest.fixture
def repo(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    return make_repo(tmp_path, monkeypatch)


@pytest.fixture
def staged(repo: Path, tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> Staged:
    s3 = FakeS3()
    _ingested(s3)
    return stage_version(catlin_seaview, s3, repo, tmp_path, capsys)


def test_build_writes_image_rows_from_the_ingested_objects(staged: Staged) -> None:
    def row(region: str, image_id: str) -> dict[str, object]:
        width, height = IMAGE_SIZES[(region, image_id)]
        data = jpeg(width, height)
        return {
            "image_id": image_id,
            "key": f"{region}/{image_id}.jpg",
            **probed_columns(data),
            "media_type": "image/jpeg",
            "width": width,
            "height": height,
            "group": region,
            **no_mask(),
            "source_fingerprint": ingested_etag(data),
            "source_name": f"{image_id}.jpg",
            "attrs": None,
        }

    assert sorted(staged.version.images.to_pylist(), key=lambda r: r["image_id"]) == [
        row("ATL", "1001"),
        row("ATL", "1002"),
        row("PAC_USA", "2001"),
    ]
    assert staged.version.manifest.image_store.prefix == "images/catlin_seaview/"


def test_build_plans_a_copy_of_each_ingested_image_and_stages_no_image(staged: Staged) -> None:
    assert_copies_match_sources(
        staged,
        {
            image_id: (f"{PREFIX}images/{region}/{image_id}.jpg", data)
            for (region, image_id), data in _images().items()
        },
    )
    assert not (staged.stage / DATASET_BUCKET / "images").exists()


def test_build_writes_points_labels_and_counts(staged: Staged) -> None:
    version = staged.version
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
    assert version.label_counts.to_pylist() == [
        {"image_id": "1001", "label_id": 5, "n": 1},
        {"image_id": "1001", "label_id": 9, "n": 1},
        {"image_id": "1002", "label_id": 5, "n": 1},
        {"image_id": "2001", "label_id": 12, "n": 1},
    ]
    assert_curated_label_mapping(version.manifest, catlin_seaview.DATASET, version.labels)


def test_build_records_source_and_command_in_the_manifest(staged: Staged) -> None:
    manifest = staged.version.manifest
    assert manifest.version.endswith("-abcdef1")
    assert manifest.source.uri == f"s3://{INGEST_BUCKET}/{PREFIX}"
    assert manifest.etl.command == (
        f"segdata build catlin_seaview --stage-dir {staged.stage}"
        f" --taxonomy-version {TAXONOMY_VERSION} --taxonomy-manifest-sha256 {staged.taxonomy_sha}"
    )


def test_build_records_the_taxonomy_pin_in_the_manifest(staged: Staged) -> None:
    taxonomy = staged.version.manifest.taxonomy
    assert taxonomy is not None
    assert (taxonomy.name, taxonomy.version, taxonomy.manifest_sha256) == (
        "mermaid-benthic",
        TAXONOMY_VERSION,
        staged.taxonomy_sha,
    )


def test_build_takes_the_next_suffix_when_the_bucket_already_holds_the_version(
    repo: Path, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    s3 = FakeS3()
    _ingested(s3)
    base = f"{datetime.now(UTC).date():%Y%m%d}-{CLEAN_GIT.sha[:7]}"
    s3.put(DATASET_BUCKET, f"datasets/catlin_seaview/{base}/images.parquet", b"half-promoted")

    staged = stage_version(catlin_seaview, s3, repo, tmp_path, capsys)

    assert staged.pin.version == f"{base}-2"


@pytest.mark.parametrize(
    "git",
    [
        GitState(sha=CLEAN_GIT.sha, dirty=True, pushed=True),
        GitState(sha=CLEAN_GIT.sha, dirty=False, pushed=False),
    ],
    ids=["dirty", "unpushed"],
)
def test_build_refuses_an_unpublishable_tree_before_reading_s3(
    repo: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch, git: GitState
) -> None:
    s3 = FakeS3()
    _ingested(s3)
    monkeypatch.setattr("segdata_etl.common.stage.probe_git", lambda _dir: git)
    stage = tmp_path / "stage"

    with pytest.raises(PublishRefused):
        catlin_seaview.run(build_args(catlin_seaview, stage, "0" * 64), s3)

    assert s3.calls == []
    assert not stage.exists()


def test_build_drops_an_image_with_an_unreadable_header_and_its_points(
    repo: Path, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    s3 = FakeS3()
    images = _images()
    _ingested(s3, images={**images, ("ATL", "1002"): b"not an image"})

    staged = stage_version(catlin_seaview, s3, repo, tmp_path, capsys)

    assert staged.version.manifest.stats.dropped == {
        "damaged_images": 2,
        "image_unreadable_header": 1,
    }
    assert staged.version.points is not None
    assert [r["image_id"] for r in staged.version.points.to_pylist()] == ["1001", "1001", "2001"]
    assert_copies_match_sources(
        staged,
        {
            "1001": (f"{PREFIX}images/ATL/1001.jpg", images[("ATL", "1001")]),
            "2001": (f"{PREFIX}images/PAC_USA/2001.jpg", images[("PAC_USA", "2001")]),
        },
    )


@pytest.mark.parametrize(
    ("annotations", "message"),
    [
        ([*ANNOTATIONS, ("ATL", "1002", 1, 1, "Not a class")], "Not a class"),
        ([*ANNOTATIONS, ("ATL", "1003", 1, 1, "Water")], "1003"),
        ([a if a[1] != "1002" else ("PAC_USA", *a[1:]) for a in ANNOTATIONS], "PAC_USA"),
        ([*ANNOTATIONS, ("ATL", "1002", 1, 1, "unlabeled")], "not in labels.parquet"),
    ],
    ids=["label-not-in-classes", "image-object-missing", "other-region", "unlabeled-point"],
)
def test_build_refuses_inconsistent_annotations_before_staging(
    repo: Path,
    tmp_path: Path,
    annotations: list[tuple[str, str, int, int, str]],
    message: str,
) -> None:
    s3 = FakeS3()
    _ingested(s3, annotations)
    stage = tmp_path / "stage"
    taxonomy_sha = publish_fixture_taxonomy(s3, repo)

    with pytest.raises(ValueError, match=message):
        catlin_seaview.run(build_args(catlin_seaview, stage, taxonomy_sha), s3)

    assert not stage.exists()


def test_build_drops_a_point_past_the_right_edge_and_records_it(
    repo: Path, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    s3 = FakeS3()
    # ATL/1002 is 6 wide, so col 6 is outside; label 12 would add a count row if it were kept.
    _ingested(s3, [*ANNOTATIONS, ("ATL", "1002", 1, 6, "Water")])

    version = stage_version(catlin_seaview, s3, repo, tmp_path, capsys).version

    assert version.manifest.stats.dropped == {
        "damaged_images": 2,
        "image_unreadable_header": 0,
        "points_out_of_bounds": 1,
    }
    assert version.points is not None
    assert version.points.num_rows == len(ANNOTATIONS)
    assert (1, 6) not in {(r["row"], r["col"]) for r in version.points.to_pylist()}
    assert sum(r["n"] for r in version.label_counts.to_pylist()) == len(ANNOTATIONS)


def test_build_raises_source_changed_and_stages_nothing_when_an_image_changes_while_probed(
    repo: Path, tmp_path: Path
) -> None:
    s3 = FakeS3()
    _ingested(s3)
    key = PREFIX + "images/ATL/1001.jpg"
    real_get = s3.get_object

    def get_after_replacement(*, Bucket: str, Key: str, **kwargs: object) -> dict[str, object]:
        if Key == key:
            s3.put(Bucket, Key, jpeg(9, 9))
        return real_get(Bucket=Bucket, Key=Key, **kwargs)

    s3.get_object = get_after_replacement  # type: ignore[method-assign]
    stage = tmp_path / "stage"
    taxonomy_sha = publish_fixture_taxonomy(s3, repo)

    with pytest.raises(SourceChanged, match="images/ATL/1001.jpg"):
        catlin_seaview.run(build_args(catlin_seaview, stage, taxonomy_sha), s3)

    assert not stage.exists()
