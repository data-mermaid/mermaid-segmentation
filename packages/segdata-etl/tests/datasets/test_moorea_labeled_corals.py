"""``segdata build moorea_labeled_corals`` from a fake legacy prefix into a LocalStore."""

from __future__ import annotations

import io
import json
from dataclasses import dataclass
from pathlib import Path

import pandas as pd
import pytest
from common.conftest import FakeS3

from segdata_etl.common.local_store import LocalStore
from segdata_etl.datasets import moorea_labeled_corals
from segdata_spec import DATASET_BUCKET
from segdata_spec.read import DatasetVersion, open_version

from .conftest import (
    LEGACY_BUCKET,
    build_args,
    copied_columns,
    jpeg,
    legacy_etag,
    make_repo,
    no_mask,
    png,
    publish_fixture_taxonomy,
    run_builder,
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


def _legacy(
    s3: FakeS3, annotations: list[Annotation] = ANNOTATIONS, *, licence: str | None = "CC BY 4.0"
) -> None:
    manifest = {
        "dataset_name": "moorea_labeled_corals",
        "source": {
            "citation": "Beijbom et al. (2012)",
            **({"license": licence} if licence else {}),
        },
        "damaged_images": ["x", "y"],
        "off_annotations_dropped": 13,
        "case_merged_labels": {"ACROP": "Acrop"},
    }
    files = {
        "manifest.json": json.dumps(manifest).encode(),
        "classes.json": json.dumps(CLASSES).encode(),
        "moorea_labeled_corals_annotations.parquet": _parquet(annotations),
    }
    for (year, image_id, ext), data in IMAGES.items():
        files[f"images/{year}/{image_id}{ext}"] = data
    for name, data in files.items():
        s3.objects[(LEGACY_BUCKET, PREFIX + name)] = data


@dataclass(frozen=True)
class Published:
    store: Path
    version: DatasetVersion
    taxonomy_sha: str


@pytest.fixture
def repo(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    return make_repo(tmp_path, monkeypatch, moorea_labeled_corals)


@pytest.fixture
def published(repo: Path, tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> Published:
    s3 = FakeS3()
    _legacy(s3)
    store = tmp_path / "store"
    taxonomy_sha = publish_fixture_taxonomy(LocalStore(store), repo)
    pin = run_builder(
        moorea_labeled_corals, build_args(moorea_labeled_corals, store, taxonomy_sha), s3, capsys
    )
    return Published(store, open_version(store, DATASET_BUCKET, pin), taxonomy_sha)


def test_build_groups_images_by_year_and_keeps_the_png_extension(published: Published) -> None:
    def row(year: str, image_id: str, ext: str, media: str, w: int, h: int) -> dict[str, object]:
        data = IMAGES[(year, image_id, f".{ext}")]
        return {
            "image_id": image_id,
            "key": f"{year}/{image_id}.{ext}",
            **copied_columns(data),
            "media_type": media,
            "width": w,
            "height": h,
            "group": year,
            **no_mask(),
            "source_fingerprint": legacy_etag(data),
            "source_name": f"{image_id}.{ext}",
            "attrs": None,
        }

    assert sorted(published.version.images.to_pylist(), key=lambda r: r["image_id"]) == [
        row("2008", "mcr_a", "jpg", "image/jpeg", 7, 5),
        row("2008", "mcr_b", "png", "image/png", 6, 4),
        row("2009", "mcr_c", "jpg", "image/jpeg", 5, 3),
    ]
    copied = published.store / DATASET_BUCKET / "images/moorea_labeled_corals/2008/mcr_b.png"
    assert copied.read_bytes() == IMAGES[("2008", "mcr_b", ".png")]


def test_build_takes_point_labels_from_the_case_merged_source_label_name(
    published: Published,
) -> None:
    version = published.version
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


def test_build_records_the_legacy_licence_and_drop_counts_in_the_manifest(
    published: Published,
) -> None:
    manifest = published.version.manifest
    assert manifest.source.license == "CC BY 4.0"
    assert manifest.stats.dropped == {
        "damaged_images": 2,
        "off_annotations": 13,
        "case_merged_labels": 1,
    }
    assert manifest.etl.command.startswith(
        f"segdata build moorea_labeled_corals --store {published.store}"
    )


def test_build_refuses_an_image_annotated_under_another_year(repo: Path, tmp_path: Path) -> None:
    s3 = FakeS3()
    _legacy(s3, [a if a[1] != "mcr_b" else ("2009", *a[1:]) for a in ANNOTATIONS])
    store = tmp_path / "store"
    taxonomy_sha = publish_fixture_taxonomy(LocalStore(store), repo)

    with pytest.raises(ValueError, match="2009"):
        moorea_labeled_corals.run(build_args(moorea_labeled_corals, store, taxonomy_sha), s3)

    assert not (store / DATASET_BUCKET / "images").exists()


def test_build_refuses_a_legacy_manifest_without_a_licence_before_copying(
    repo: Path, tmp_path: Path
) -> None:
    s3 = FakeS3()
    _legacy(s3, licence=None)
    store = tmp_path / "store"
    taxonomy_sha = publish_fixture_taxonomy(LocalStore(store), repo)

    with pytest.raises(ValueError, match="no licence"):
        moorea_labeled_corals.run(build_args(moorea_labeled_corals, store, taxonomy_sha), s3)

    assert not (store / DATASET_BUCKET / "images").exists()
