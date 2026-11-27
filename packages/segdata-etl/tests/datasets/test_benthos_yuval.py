"""``segdata build benthos_yuval`` from a fake ingest prefix of tiles and masks into a local stage."""

from __future__ import annotations

import io
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from s3_fake import FakeS3

from segdata_etl.datasets import benthos_yuval
from segdata_spec import DATASET_BUCKET

from .conftest import (
    INGEST_BUCKET,
    Staged,
    assert_copies_match_sources,
    build_args,
    ingest_manifest_bytes,
    ingested_etag,
    make_repo,
    mask_png,
    plan_lines,
    png,
    probed_columns,
    publish_fixture_taxonomy,
    stage_version,
)

PREFIX = "external_validation_datasets/benthos_yuval/"
CLASSES = {"unlabeled": 0, "algae": 1, "coral": 3, "sand": 4}
# RS24/0_0 holds {0: 5, 1: 7, 3: 4} pixels.
MASK_RS24_0_0 = np.array([[0, 0, 0, 0], [0, 1, 1, 1], [1, 1, 1, 1], [3, 3, 3, 3]])
# An RGB mask: channel 0 is the label, the other channels hold colours.
MASK_CR_0_0 = np.stack([np.full((3, 3), 3), np.full((3, 3), 200), np.full((3, 3), 90)], axis=-1)
# (site, stem) -> (image width, image height, mask array)
TILES = {
    ("RS24", "0_0"): (4, 4, MASK_RS24_0_0),
    ("RS24", "0_4"): (4, 2, np.ones((2, 4))),
    ("CR_DoubleWreck", "0_0"): (3, 3, MASK_CR_0_0),
}
Annotation = tuple[str, str, str, int, int]
ANNOTATIONS: list[Annotation] = [
    ("RS24", "0_0", "unlabeled", 4, 4),
    ("RS24", "0_0", "algae", 4, 4),
    ("RS24", "0_0", "coral", 4, 4),
    ("RS24", "0_4", "algae", 2, 4),
    ("CR_DoubleWreck", "0_0", "coral", 3, 3),
]


def _parquet(rows: list[Annotation]) -> bytes:
    columns = ["site", "image_id", "source_label_name", "tile_height", "tile_width"]
    buf = io.BytesIO()
    pd.DataFrame(rows, columns=columns).to_parquet(buf, index=False)
    return buf.getvalue()


def _ingested(
    s3: FakeS3,
    annotations: list[Annotation] = ANNOTATIONS,
    masks: dict[tuple[str, str], np.ndarray | None] | None = None,
) -> dict[str, bytes]:
    """Write a Benthos-shaped prefix; ``masks`` overrides a tile's mask, None drops it."""
    files = {
        "manifest.json": ingest_manifest_bytes("benthos_yuval"),
        "classes.json": json.dumps(CLASSES).encode(),
        "benthos_yuval_annotations.parquet": _parquet(annotations),
    }
    overrides = masks or {}
    for (site, stem), (width, height, mask) in TILES.items():
        files[f"images/{site}/{stem}.png"] = png(width, height)
        mask = overrides.get((site, stem), mask)
        if mask is not None:
            files[f"labels/{site}/{stem}.png"] = mask_png(mask)
    for name, data in files.items():
        s3.put(INGEST_BUCKET, PREFIX + name, data)
    return files


@pytest.fixture
def repo(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    return make_repo(tmp_path, monkeypatch)


@pytest.fixture
def s3() -> FakeS3:
    fake = FakeS3()
    _ingested(fake)
    return fake


@pytest.fixture
def staged(s3: FakeS3, repo: Path, tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> Staged:
    return stage_version(benthos_yuval, s3, repo, tmp_path, capsys)


def _file(s3: FakeS3, name: str) -> bytes:
    return s3.objects[(INGEST_BUCKET, PREFIX + name)]


def test_build_names_tiles_by_site_with_their_mask_columns(s3: FakeS3, staged: Staged) -> None:
    def row(site: str, stem: str, w: int, h: int) -> dict[str, object]:
        image = _file(s3, f"images/{site}/{stem}.png")
        mask = _file(s3, f"labels/{site}/{stem}.png")
        return {
            "image_id": f"{site}_{stem}",
            "key": f"{site}/{site}_{stem}.png",
            **probed_columns(image),
            "media_type": "image/png",
            "width": w,
            "height": h,
            "group": site,
            "mask_key": f"{site}/{site}_{stem}_mask.png",
            **probed_columns(mask, "mask_"),
            "source_fingerprint": ingested_etag(image),
            "source_name": f"{stem}.png",
            "attrs": None,
        }

    assert sorted(staged.version.images.to_pylist(), key=lambda r: r["image_id"]) == [
        row("CR_DoubleWreck", "0_0", 3, 3),
        row("RS24", "0_0", 4, 4),
        row("RS24", "0_4", 4, 2),
    ]


def test_build_plans_image_copies_and_stages_each_mask_for_upload(
    s3: FakeS3, staged: Staged
) -> None:
    assert_copies_match_sources(
        staged,
        {
            f"{site}_{stem}": (
                f"{PREFIX}images/{site}/{stem}.png",
                _file(s3, f"images/{site}/{stem}.png"),
            )
            for site, stem in TILES
        },
    )
    expected = []
    for site, stem in sorted(TILES):
        mask = _file(s3, f"labels/{site}/{stem}.png")
        key = f"images/benthos_yuval/{site}/{site}_{stem}_mask.png"
        assert (staged.stage / DATASET_BUCKET / key).read_bytes() == mask
        expected.append(
            {
                "local_path": f"{DATASET_BUCKET}/{key}",
                "dst_bucket": DATASET_BUCKET,
                "dst_key": key,
                "size": len(mask),
                "crc64nvme": probed_columns(mask)["crc64nvme"],
                "content_type": "image/png",
            }
        )
    assert plan_lines(staged.stage, staged.pin, "uploads.jsonl") == expected


def test_build_stages_a_dense_version_counting_mask_pixels_without_unlabeled(
    staged: Staged,
) -> None:
    version = staged.version
    assert version.manifest.kind == "dense"
    assert version.points is None
    assert version.label_counts.to_pylist() == [
        {"image_id": "CR_DoubleWreck_0_0", "label_id": 3, "n": 9},
        {"image_id": "RS24_0_0", "label_id": 1, "n": 7},
        {"image_id": "RS24_0_0", "label_id": 3, "n": 4},
        {"image_id": "RS24_0_4", "label_id": 1, "n": 8},
    ]


@pytest.mark.parametrize(
    ("masks", "annotations", "message"),
    [
        ({("RS24", "0_4"): np.ones((3, 4))}, ANNOTATIONS, "RS24_0_4.*size"),
        ({("RS24", "0_4"): np.full((2, 4), 2)}, ANNOTATIONS, r"RS24_0_4.*\[2\]"),
        ({}, [a for a in ANNOTATIONS if a[:3] != ("RS24", "0_0", "coral")], "RS24_0_0.*labels"),
        ({("RS24", "0_4"): None}, ANNOTATIONS, "labels/RS24/0_4.png"),
    ],
    ids=["size-differs", "value-not-a-label", "labels-disagree-with-parquet", "mask-missing"],
)
def test_build_refuses_a_bad_mask_before_staging(
    repo: Path,
    tmp_path: Path,
    masks: dict[tuple[str, str], np.ndarray | None],
    annotations: list[Annotation],
    message: str,
) -> None:
    s3 = FakeS3()
    _ingested(s3, annotations, masks)
    stage = tmp_path / "stage"
    taxonomy_sha = publish_fixture_taxonomy(s3, repo)

    with pytest.raises(ValueError, match=message):
        benthos_yuval.run(build_args(benthos_yuval, stage, taxonomy_sha), s3)

    assert not stage.exists()


def test_build_drops_a_tile_with_an_unreadable_header_with_its_mask_and_counts(
    repo: Path, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    s3 = FakeS3()
    _ingested(s3)
    s3.put(INGEST_BUCKET, PREFIX + "images/RS24/0_4.png", b"not an image")

    staged = stage_version(benthos_yuval, s3, repo, tmp_path, capsys)

    assert staged.version.manifest.stats.dropped == {"image_unreadable_header": 1}
    assert sorted(r["image_id"] for r in staged.version.images.to_pylist()) == [
        "CR_DoubleWreck_0_0",
        "RS24_0_0",
    ]
    assert not (
        staged.stage / DATASET_BUCKET / "images/benthos_yuval/RS24/RS24_0_4_mask.png"
    ).exists()
    assert [r["dst_key"] for r in plan_lines(staged.stage, staged.pin, "uploads.jsonl")] == [
        "images/benthos_yuval/CR_DoubleWreck/CR_DoubleWreck_0_0_mask.png",
        "images/benthos_yuval/RS24/RS24_0_0_mask.png",
    ]
    assert staged.version.label_counts.to_pylist() == [
        {"image_id": "CR_DoubleWreck_0_0", "label_id": 3, "n": 9},
        {"image_id": "RS24_0_0", "label_id": 1, "n": 7},
        {"image_id": "RS24_0_0", "label_id": 3, "n": 4},
    ]


def test_build_refuses_a_mask_with_no_tile_before_staging(repo: Path, tmp_path: Path) -> None:
    s3 = FakeS3()
    _ingested(s3)
    s3.put(INGEST_BUCKET, PREFIX + "labels/RS24/9_9.png", mask_png(np.ones((4, 4))))
    stage = tmp_path / "stage"
    taxonomy_sha = publish_fixture_taxonomy(s3, repo)

    with pytest.raises(ValueError, match=r"benthos masks.*unexpected \['labels/RS24/9_9.png'\]"):
        benthos_yuval.run(build_args(benthos_yuval, stage, taxonomy_sha), s3)

    assert not stage.exists()
