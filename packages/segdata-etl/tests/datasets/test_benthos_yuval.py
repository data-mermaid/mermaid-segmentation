"""``segdata build benthos_yuval`` from a fake legacy prefix of tiles and masks into a LocalStore."""

from __future__ import annotations

import io
import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from common.conftest import FakeS3

from segdata_etl.common.local_store import LocalStore
from segdata_etl.datasets import benthos_yuval
from segdata_spec import DATASET_BUCKET
from segdata_spec.read import DatasetVersion, open_version

from .conftest import (
    LEGACY_BUCKET,
    assert_curated_label_mapping,
    build_args,
    copied_columns,
    legacy_etag,
    make_repo,
    mask_png,
    png,
    publish_fixture_taxonomy,
    run_builder,
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


def _legacy(
    s3: FakeS3,
    annotations: list[Annotation] = ANNOTATIONS,
    masks: dict[tuple[str, str], np.ndarray | None] | None = None,
) -> dict[str, bytes]:
    """Write a Benthos-shaped prefix; ``masks`` overrides a tile's mask, None drops it."""
    files = {
        "manifest.json": json.dumps({"dataset_name": "benthos_yuval", "source": {}}).encode(),
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
        s3.objects[(LEGACY_BUCKET, PREFIX + name)] = data
    return files


@dataclass(frozen=True)
class Published:
    store: Path
    version: DatasetVersion
    files: dict[str, bytes]


@pytest.fixture
def repo(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    return make_repo(tmp_path, monkeypatch, benthos_yuval)


@pytest.fixture
def published(repo: Path, tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> Published:
    s3 = FakeS3()
    files = _legacy(s3)
    store = tmp_path / "store"
    taxonomy_sha = publish_fixture_taxonomy(LocalStore(store), repo)
    pin = run_builder(benthos_yuval, build_args(benthos_yuval, store, taxonomy_sha), s3, capsys)
    return Published(store, open_version(store, DATASET_BUCKET, pin), files)


def test_build_names_tiles_by_site_and_copies_each_mask(published: Published) -> None:
    def row(site: str, stem: str, w: int, h: int) -> dict[str, object]:
        image = published.files[f"images/{site}/{stem}.png"]
        mask = published.files[f"labels/{site}/{stem}.png"]
        return {
            "image_id": f"{site}_{stem}",
            "key": f"{site}/{site}_{stem}.png",
            **copied_columns(image),
            "media_type": "image/png",
            "width": w,
            "height": h,
            "group": site,
            "mask_key": f"{site}/{site}_{stem}_mask.png",
            **copied_columns(mask, "mask_"),
            "source_fingerprint": legacy_etag(image),
            "source_name": f"{stem}.png",
            "attrs": None,
        }

    assert sorted(published.version.images.to_pylist(), key=lambda r: r["image_id"]) == [
        row("CR_DoubleWreck", "0_0", 3, 3),
        row("RS24", "0_0", 4, 4),
        row("RS24", "0_4", 4, 2),
    ]
    copied = published.store / DATASET_BUCKET / "images/benthos_yuval/RS24/RS24_0_0_mask.png"
    assert copied.read_bytes() == published.files["labels/RS24/0_0.png"]


def test_build_publishes_a_dense_version_counting_mask_pixels_without_unlabeled(
    published: Published,
) -> None:
    version = published.version
    assert version.manifest.kind == "dense"
    assert version.points is None
    assert version.label_counts.to_pylist() == [
        {"image_id": "CR_DoubleWreck_0_0", "label_id": 3, "n": 9},
        {"image_id": "RS24_0_0", "label_id": 1, "n": 7},
        {"image_id": "RS24_0_0", "label_id": 3, "n": 4},
        {"image_id": "RS24_0_4", "label_id": 1, "n": 8},
    ]


def test_build_records_the_dryad_licence_and_no_drops(published: Published) -> None:
    manifest = published.version.manifest
    assert manifest.source.license == "CC0 1.0 (Dryad)"
    assert manifest.stats.dropped == {}
    assert manifest.etl.command.startswith(f"segdata build benthos_yuval --store {published.store}")
    assert_curated_label_mapping(
        published.version.manifest, benthos_yuval.DATASET, published.version.labels
    )


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
def test_build_refuses_a_bad_mask_before_copying_any_image(
    repo: Path,
    tmp_path: Path,
    masks: dict[tuple[str, str], np.ndarray | None],
    annotations: list[Annotation],
    message: str,
) -> None:
    s3 = FakeS3()
    _legacy(s3, annotations, masks)
    store = tmp_path / "store"
    taxonomy_sha = publish_fixture_taxonomy(LocalStore(store), repo)

    with pytest.raises(ValueError, match=message):
        benthos_yuval.run(build_args(benthos_yuval, store, taxonomy_sha), s3)

    assert not (store / DATASET_BUCKET / "images").exists()
