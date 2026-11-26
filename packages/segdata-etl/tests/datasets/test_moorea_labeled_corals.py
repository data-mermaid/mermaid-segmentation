"""``segdata build moorea_labeled_corals`` from a fake ingest prefix into a local stage."""

from __future__ import annotations

import io
import json
from pathlib import Path

import pandas as pd
import pytest
from s3_fake import FakeS3

from segdata_etl import promote
from segdata_etl.datasets import moorea_labeled_corals
from segdata_spec import DATASET_BUCKET

from .conftest import (
    INGEST_BUCKET,
    Staged,
    assert_copies_match_sources,
    assert_curated_label_mapping,
    build_args,
    ingest_manifest_bytes,
    ingested_etag,
    jpeg,
    make_repo,
    no_mask,
    png,
    probed_columns,
    publish_fixture_taxonomy,
    stage_version,
)

PREFIX = "external_validation_datasets/moorea_labeled_corals/"
CLASSES = {"unlabeled": 0, "Acrop": 3, "Sand": 8, "Porit": 11}
# (year, image_id, ext) -> bytes
IMAGES = {
    ("2008", "mcr_a", ".jpg"): jpeg(7, 5),
    ("2008", "mcr_b", ".png"): png(6, 4),
    ("2009", "mcr_c", ".jpg"): jpeg(5, 3),
}
Annotation = tuple[str, str, str, int, int, str, str]
ANNOTATIONS: list[Annotation] = [
    ("2008", "mcr_a", ".jpg", 1, 2, "Acrop", "ACROP"),
    ("2008", "mcr_b", ".png", 3, 5, "Sand", "sand"),
    ("2009", "mcr_c", ".jpg", 2, 4, "Porit", "Porit"),
    ("2009", "mcr_c", ".jpg", 0, 0, "Acrop", "acrop"),
]


def _parquet(rows: list[Annotation]) -> bytes:
    columns = ["year", "image_id", "image_ext", "row", "col"]
    buf = io.BytesIO()
    frame = pd.DataFrame(rows, columns=[*columns, "source_label_name", "raw_label_name"])
    frame.to_parquet(buf, index=False)
    return buf.getvalue()


def _ingested(s3: FakeS3, annotations: list[Annotation] = ANNOTATIONS) -> None:
    dropped = {"damaged_images": 2, "off_annotations": 13}
    files = {
        "manifest.json": ingest_manifest_bytes("moorea_labeled_corals", dropped),
        "classes.json": json.dumps(CLASSES).encode(),
        "moorea_labeled_corals_annotations.parquet": _parquet(annotations),
    }
    for (year, image_id, ext), data in IMAGES.items():
        files[f"images/{year}/{image_id}{ext}"] = data
    for name, data in files.items():
        s3.put(INGEST_BUCKET, PREFIX + name, data)


@pytest.fixture
def repo(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    return make_repo(tmp_path, monkeypatch, moorea_labeled_corals)


@pytest.fixture
def s3() -> FakeS3:
    fake = FakeS3()
    _ingested(fake)
    return fake


@pytest.fixture
def staged(s3: FakeS3, repo: Path, tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> Staged:
    return stage_version(moorea_labeled_corals, s3, repo, tmp_path, capsys)


def test_build_groups_images_by_year_and_keeps_the_png_extension(staged: Staged) -> None:
    def row(year: str, image_id: str, ext: str, media: str, w: int, h: int) -> dict[str, object]:
        data = IMAGES[(year, image_id, f".{ext}")]
        return {
            "image_id": image_id,
            "key": f"{year}/{image_id}.{ext}",
            **probed_columns(data),
            "media_type": media,
            "width": w,
            "height": h,
            "group": year,
            **no_mask(),
            "source_fingerprint": ingested_etag(data),
            "source_name": f"{image_id}.{ext}",
            "attrs": None,
        }

    assert sorted(staged.version.images.to_pylist(), key=lambda r: r["image_id"]) == [
        row("2008", "mcr_a", "jpg", "image/jpeg", 7, 5),
        row("2008", "mcr_b", "png", "image/png", 6, 4),
        row("2009", "mcr_c", "jpg", "image/jpeg", 5, 3),
    ]


def test_build_plans_a_copy_of_each_ingested_image(staged: Staged) -> None:
    assert_copies_match_sources(
        staged,
        {
            image_id: (f"{PREFIX}images/{year}/{image_id}{ext}", data)
            for (year, image_id, ext), data in IMAGES.items()
        },
    )


def test_staged_version_promotes_with_the_images_copied_from_the_ingest_prefix(
    s3: FakeS3, staged: Staged
) -> None:
    promote.promote(
        s3, stage_dir=staged.stage, dataset=staged.pin.name, version=staged.pin.version, workers=2
    )

    prefix = f"datasets/moorea_labeled_corals/{staged.pin.version}/"
    assert (DATASET_BUCKET, prefix + "manifest.json") in s3.objects
    assert {
        key: data
        for (bucket, key), data in s3.objects.items()
        if bucket == DATASET_BUCKET and key.startswith("images/")
    } == {
        f"images/moorea_labeled_corals/{year}/{image_id}{ext}": data
        for (year, image_id, ext), data in IMAGES.items()
    }


def test_build_takes_point_labels_from_the_case_merged_source_label_name(staged: Staged) -> None:
    version = staged.version
    assert version.points is not None
    assert version.points.to_pylist() == [
        {"image_id": "mcr_a", "row": 1, "col": 2, "label_id": 3, "annotator": "default"},
        {"image_id": "mcr_b", "row": 3, "col": 5, "label_id": 8, "annotator": "default"},
        {"image_id": "mcr_c", "row": 2, "col": 4, "label_id": 11, "annotator": "default"},
        {"image_id": "mcr_c", "row": 0, "col": 0, "label_id": 3, "annotator": "default"},
    ]
    assert [
        (r["label_id"], r["mermaid_benthic_attribute_id"]) for r in version.labels.to_pylist()
    ] == [
        (3, "ba-acro"),
        (8, "ba-sand"),
        (11, None),
    ]


def test_build_records_the_curated_label_mapping_in_the_manifest(staged: Staged) -> None:
    assert_curated_label_mapping(
        staged.version.manifest, moorea_labeled_corals.DATASET, staged.version.labels
    )


def test_build_refuses_an_inconsistent_ingest_prefix_before_staging(
    repo: Path, tmp_path: Path
) -> None:
    s3 = FakeS3()
    _ingested(s3, [a if a[1] != "mcr_b" else ("2009", *a[1:]) for a in ANNOTATIONS])
    stage = tmp_path / "stage"
    taxonomy_sha = publish_fixture_taxonomy(s3, repo)

    with pytest.raises(ValueError, match="2009"):
        moorea_labeled_corals.run(build_args(moorea_labeled_corals, stage, taxonomy_sha), s3)

    assert not stage.exists()


def test_build_drops_a_point_past_the_bottom_edge_and_records_it(
    repo: Path, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    s3 = FakeS3()
    # mcr_a is 5 high, so row 5 is outside; label 8 would add a count row if it were kept.
    _ingested(s3, [*ANNOTATIONS, ("2008", "mcr_a", ".jpg", 5, 0, "Sand", "sand")])

    version = stage_version(moorea_labeled_corals, s3, repo, tmp_path, capsys).version

    assert version.manifest.stats.dropped["points_out_of_bounds"] == 1
    assert version.points is not None
    assert version.points.num_rows == len(ANNOTATIONS)
    assert (5, 0) not in {(r["row"], r["col"]) for r in version.points.to_pylist()}
    assert sum(r["n"] for r in version.label_counts.to_pylist()) == len(ANNOTATIONS)
