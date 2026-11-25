"""run_sync and format_summary over a fixture version served by an in-test fake S3 client."""

from __future__ import annotations

from pathlib import Path

import pytest

from mermaidseg.data import PointDataset
from mermaidseg.data.config import SyncPlan, sync_plan
from mermaidseg.data.sync import format_summary, run_sync
from segdata_spec import DATASET_BUCKET
from segdata_spec.errors import ImageMissing, SplitCoverageError
from segdata_spec.manifest import Pin
from segdata_spec.read import open_version
from segdata_spec.resize import ResizeSpec
from segdata_spec.sync import DatasetSyncResult, SyncResult
from tests.mermaidseg.data.conftest import (
    FakeS3,
    ImageSpec,
    publish_points,
    publish_taxonomy_version,
)

LABELS = {5: "five"}
_RESIZE_1024 = ResizeSpec(max_edge=1024)
IMAGES = [ImageSpec("i1", group="g1"), ImageSpec("i2", group="g2")]
POINTS = [("i1", 2, 3, 5), ("i2", 4, 4, 5)]


def _publish(tmp_path: Path) -> tuple[FakeS3, Pin]:
    return publish_points(tmp_path, images=IMAGES, labels=LABELS, points=POINTS)


def _plan(
    pin: Pin,
    groups: set[str] | None,
    split_file: Path | None = None,
    resize: ResizeSpec | None = None,
) -> SyncPlan:
    return SyncPlan(
        pins={"ds": pin},
        taxonomy=None,
        selection={"ds": groups},
        split_files={"ds": split_file} if split_file is not None else {},
        resize={"ds": resize},
    )


def _image_files(cache_dir: Path) -> set[str]:
    return {p.name for p in (cache_dir / "img-bucket").rglob("*.jpg")}


def test_run_sync_fills_the_cache_so_a_reader_opens_it(tmp_path: Path) -> None:
    fake, pin = _publish(tmp_path / "pub")
    cache_dir = tmp_path / "synced"
    run_sync(_plan(pin, None), cache_dir, fake)
    ds = PointDataset(open_version(cache_dir, DATASET_BUCKET, pin), split="val", global_offset=0)
    assert len(ds) == 2
    _, mask = ds[0]
    assert mask[2, 3] == 5


def test_run_sync_syncs_when_the_split_file_lists_every_group(tmp_path: Path) -> None:
    fake, pin = _publish(tmp_path / "pub")
    split_file = tmp_path / "splits.csv"
    split_file.write_text("group,split\ng1,train\ng2,val\n")
    cache_dir = tmp_path / "synced"
    run_sync(_plan(pin, None, split_file=split_file), cache_dir, fake)
    assert _image_files(cache_dir) == {"i1.jpg", "i2.jpg"}


def test_run_sync_fails_on_an_unlisted_group_before_any_image(tmp_path: Path) -> None:
    fake, pin = _publish(tmp_path / "pub")
    split_file = tmp_path / "splits.csv"
    split_file.write_text("group,split\ng1,train\n")
    cache_dir = tmp_path / "synced"
    with pytest.raises(SplitCoverageError, match=r"\['g2'\]"):
        run_sync(_plan(pin, None, split_file=split_file), cache_dir, fake)
    assert _image_files(cache_dir) == set()


def test_run_sync_checks_coverage_of_unselected_groups_too(tmp_path: Path) -> None:
    fake, pin = _publish(tmp_path / "pub")
    split_file = tmp_path / "splits.csv"
    split_file.write_text("group,split\ng1,train\n")
    cache_dir = tmp_path / "synced"
    with pytest.raises(SplitCoverageError, match=r"\['g2'\]"):
        run_sync(_plan(pin, {"g1"}, split_file=split_file), cache_dir, fake)
    assert _image_files(cache_dir) == set()


def test_sync_plan_split_file_reaches_the_coverage_guard(tmp_path: Path) -> None:
    fake, pin = _publish(tmp_path / "pub")
    (tmp_path / "splits.csv").write_text("group,split\ng1,train\n")
    data_cfg = {
        "datasets": {
            "ds": {
                "version": pin.version,
                "manifest_sha256": pin.manifest_sha256,
                "split_file": "splits.csv",
                "train": {"use": ["train"]},
            }
        }
    }
    plan = sync_plan(data_cfg, repo_root=tmp_path)
    cache_dir = tmp_path / "synced"
    with pytest.raises(SplitCoverageError, match=r"\['g2'\]"):
        run_sync(plan, cache_dir, fake)
    assert _image_files(cache_dir) == set()


def test_run_sync_forwards_the_selection(tmp_path: Path) -> None:
    fake, pin = _publish(tmp_path / "pub")
    cache_dir = tmp_path / "synced"
    run_sync(_plan(pin, {"g1"}), cache_dir, fake)
    assert _image_files(cache_dir) == {"i1.jpg"}


def test_run_sync_fails_on_a_missing_image(tmp_path: Path) -> None:
    fake, pin = _publish(tmp_path / "pub")
    del fake.objects[("img-bucket", "imgs/i2.jpg")]
    with pytest.raises(ImageMissing):
        run_sync(_plan(pin, None), tmp_path / "synced", fake)


def test_run_sync_forwards_the_taxonomy_pin(tmp_path: Path) -> None:
    fake, pin = _publish(tmp_path / "pub")
    taxonomy = publish_taxonomy_version(fake, tmp_path / "pub", "20261120-1234567")
    plan = SyncPlan(
        pins={"ds": pin},
        taxonomy=taxonomy,
        selection={"ds": set()},
        split_files={},
    )
    cache_dir = tmp_path / "synced"

    result = run_sync(plan, cache_dir, fake)

    assert result.taxonomy is not None
    assert result.taxonomy.version == "20261120-1234567"
    folder = cache_dir / DATASET_BUCKET / "taxonomy" / "mermaid-benthic" / "20261120-1234567"
    assert {p.name for p in folder.iterdir()} == {
        "manifest.json",
        "benthic_attributes.parquet",
        "growth_forms.parquet",
    }


def test_format_summary_lists_each_dataset_and_the_taxonomy() -> None:
    def dataset(version: str) -> DatasetSyncResult:
        resize = None if version.startswith("20260101") else _RESIZE_1024
        return DatasetSyncResult(version, "0" * 64, resize)

    result = SyncResult(
        datasets={"a": dataset("20260101-abc1234"), "b": dataset("20260102-abc1234")},
        taxonomy=Pin(name="mermaid-benthic", version="20260103-abc1234", manifest_sha256="1" * 64),
    )
    assert format_summary(result).splitlines() == [
        "a: version 20260101-abc1234, resize original",
        "b: version 20260102-abc1234, resize 5a451a1295a8",
        "taxonomy: version 20260103-abc1234",
    ]


def test_run_sync_forwards_each_datasets_resize(tmp_path: Path) -> None:
    fake, pin = _publish(tmp_path / "pub")
    spec = ResizeSpec(max_edge=4)
    cache_dir = tmp_path / "synced"
    result = run_sync(_plan(pin, None, resize=spec), cache_dir, fake)
    assert result.datasets["ds"].resize == spec
    ds = PointDataset(
        open_version(cache_dir, DATASET_BUCKET, pin, resize=spec), split="val", global_offset=0
    )
    image, _ = ds[0]
    assert image.shape == (4, 4, 3)


def test_run_sync_forwards_max_workers(tmp_path: Path) -> None:
    fake, pin = _publish(tmp_path / "pub")
    with pytest.raises(ValueError, match="max_workers"):
        run_sync(_plan(pin, None), tmp_path / "synced", fake, max_workers=0)


def test_summary_shows_the_resize_id_of_a_resized_dataset(tmp_path: Path) -> None:
    fake, pin = _publish(tmp_path / "pub")
    spec = ResizeSpec(max_edge=4)
    result = run_sync(_plan(pin, None, resize=spec), tmp_path / "synced", fake)
    assert f"resize {spec.id}" in format_summary(result)


def test_sync_plan_carries_each_entrys_resize(tmp_path: Path) -> None:
    data_cfg = {
        "datasets": {
            "a": {
                "version": "20261120-ab12cd3",
                "manifest_sha256": "0" * 64,
                "resize": {"max_edge": 100},
                "train": {"groups": "all"},
            },
            "b": {
                "version": "20261120-ab12cd3",
                "manifest_sha256": "0" * 64,
                "train": {"groups": "all"},
            },
        }
    }
    assert sync_plan(data_cfg, tmp_path).resize == {"a": ResizeSpec(max_edge=100), "b": None}
