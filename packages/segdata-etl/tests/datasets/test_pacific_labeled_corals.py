"""``segdata build pacific_labeled_corals`` from a fake legacy prefix into a LocalStore."""

from __future__ import annotations

import io
import json
from dataclasses import dataclass
from pathlib import Path

import pandas as pd
import pytest
from common.conftest import FakeS3

from segdata_etl.common.local_store import LocalStore
from segdata_etl.datasets import pacific_labeled_corals
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
    publish_fixture_taxonomy,
    run_builder,
)

PREFIX = "external_validation_datasets/pacific_labeled_corals/"
CLASSES = {"unlabeled": 0, "Acropora": 2, "Bare substrate": 5, "Pocillopora": 7}
ANNOTATORS = ["archived", "host", *(f"visitor{i}" for i in range(1, 6))]
# (site, subset, image_id, ext) -> bytes
IMAGES = {
    ("heron_reef", "reference", "ref_1", ".JPG"): jpeg(7, 5),
    ("heron_reef", "evaluation", "ev_1", ".jpg"): jpeg(6, 4),
    ("nanwan_bay", "reference", "nb_1", ".jpg"): jpeg(5, 3),
}
NA = None
# site, subset, raw_imageid, image_id, image_ext, row, col, archived, host, visitor1..visitor5
Annotation = tuple[object, ...]
ANNOTATIONS: list[Annotation] = [
    ("heron_reef", "reference", 11, "ref_1", ".JPG", 1, 2, "Acropora", NA, NA, NA, NA, NA, NA),
    (
        "heron_reef",
        "evaluation",
        4,
        "ev_1",
        ".jpg",
        0,
        3,
        "Acropora",
        "Pocillopora",
        "Acropora",
        NA,
        NA,
        NA,
        NA,
    ),
    ("nanwan_bay", "reference", 9, "nb_1", ".jpg", 2, 4, "Bare substrate", NA, NA, NA, NA, NA, NA),
]


def _parquet(rows: list[Annotation]) -> bytes:
    columns = ["site", "subset", "raw_imageid", "image_id", "image_ext", "row", "col", *ANNOTATORS]
    frame = pd.DataFrame(rows, columns=columns).astype(
        {"raw_imageid": "int64", **dict.fromkeys(ANNOTATORS, "string")}
    )
    buf = io.BytesIO()
    frame.to_parquet(buf, index=False)
    return buf.getvalue()


def _legacy(s3: FakeS3, annotations: list[Annotation] = ANNOTATIONS) -> None:
    manifest = {"dataset_name": "pacific_labeled_corals", "source": {"citation": "Beijbom 2015"}}
    files = {
        "manifest.json": json.dumps(manifest).encode(),
        "classes.json": json.dumps(CLASSES).encode(),
        "pacific_labeled_corals_annotations.parquet": _parquet(annotations),
    }
    for (site, subset, image_id, ext), data in IMAGES.items():
        files[f"images/{site}/{subset}/{image_id}{ext}"] = data
    for name, data in files.items():
        s3.objects[(LEGACY_BUCKET, PREFIX + name)] = data


@dataclass(frozen=True)
class Published:
    store: Path
    version: DatasetVersion


@pytest.fixture
def repo(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    return make_repo(tmp_path, monkeypatch, pacific_labeled_corals)


@pytest.fixture
def published(repo: Path, tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> Published:
    s3 = FakeS3()
    _legacy(s3)
    store = tmp_path / "store"
    taxonomy_sha = publish_fixture_taxonomy(LocalStore(store), repo)
    args = build_args(pacific_labeled_corals, store, taxonomy_sha)
    pin = run_builder(pacific_labeled_corals, args, s3, capsys)
    return Published(store, open_version(store, DATASET_BUCKET, pin))


def test_build_groups_images_by_site_with_subset_and_raw_imageid_in_attrs(
    published: Published,
) -> None:
    def row(site: str, subset: str, image_id: str, ext: str, w: int, h: int, raw: int) -> dict:
        data = IMAGES[(site, subset, image_id, ext)]
        return {
            "image_id": image_id,
            "key": f"{site}/{image_id}.jpg",
            **copied_columns(data),
            "media_type": "image/jpeg",
            "width": w,
            "height": h,
            "group": site,
            **no_mask(),
            "source_fingerprint": legacy_etag(data),
            "source_name": f"{image_id}{ext}",
            "attrs": f'{{"raw_imageid": {raw}, "subset": "{subset}"}}',
        }

    assert sorted(published.version.images.to_pylist(), key=lambda r: r["image_id"]) == [
        row("heron_reef", "evaluation", "ev_1", ".jpg", 6, 4, 4),
        row("nanwan_bay", "reference", "nb_1", ".jpg", 5, 3, 9),
        row("heron_reef", "reference", "ref_1", ".JPG", 7, 5, 11),
    ]


def test_build_writes_one_point_row_per_present_annotator_and_counts_each(
    published: Published,
) -> None:
    version = published.version
    assert version.points is not None
    assert version.points.to_pylist() == [
        {"image_id": "ref_1", "row": 1, "col": 2, "label_id": 2, "annotator": "archived"},
        {"image_id": "ev_1", "row": 0, "col": 3, "label_id": 2, "annotator": "archived"},
        {"image_id": "ev_1", "row": 0, "col": 3, "label_id": 7, "annotator": "host"},
        {"image_id": "ev_1", "row": 0, "col": 3, "label_id": 2, "annotator": "visitor1"},
        {"image_id": "nb_1", "row": 2, "col": 4, "label_id": 5, "annotator": "archived"},
    ]
    assert version.label_counts.to_pylist() == [
        {"image_id": "ev_1", "label_id": 2, "n": 2},
        {"image_id": "ev_1", "label_id": 7, "n": 1},
        {"image_id": "nb_1", "label_id": 5, "n": 1},
        {"image_id": "ref_1", "label_id": 2, "n": 1},
    ]


def test_build_records_the_dryad_licence_and_no_drops(published: Published) -> None:
    manifest = published.version.manifest
    assert manifest.source.license == "CC0 1.0 (Dryad)"
    assert manifest.stats.dropped == {}
    assert manifest.etl.command.startswith(
        f"segdata build pacific_labeled_corals --store {published.store}"
    )


@pytest.mark.parametrize(
    ("changed", "position", "message"),
    [
        (("heron_reef", "evaluation", 11, "ref_1"), (3, 3), "evaluation"),
        (("nanwan_bay", "reference", 11, "ref_1"), (3, 3), "nanwan_bay"),
        (("heron_reef", "reference", 12, "ref_1"), (3, 3), "raw_imageid"),
        (("heron_reef", "reference", 11, "ref_1"), (1, 2), "duplicated positions"),
    ],
    ids=["other-subset", "other-site", "two-raw-imageids", "duplicated-position"],
)
def test_build_refuses_inconsistent_annotations_before_copying(
    repo: Path,
    tmp_path: Path,
    changed: tuple[str, str, int, str],
    position: tuple[int, int],
    message: str,
) -> None:
    s3 = FakeS3()
    extra = (*changed, ".JPG", *position, "Acropora", NA, NA, NA, NA, NA, NA)
    _legacy(s3, [*ANNOTATIONS, extra])
    store = tmp_path / "store"
    taxonomy_sha = publish_fixture_taxonomy(LocalStore(store), repo)

    with pytest.raises(ValueError, match=message):
        pacific_labeled_corals.run(build_args(pacific_labeled_corals, store, taxonomy_sha), s3)

    assert not (store / DATASET_BUCKET / "images").exists()
