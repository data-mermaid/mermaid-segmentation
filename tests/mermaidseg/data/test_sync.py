"""run_sync and format_summary over a fixture version served by an in-test fake S3 client."""

from __future__ import annotations

import hashlib
import io
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import pyarrow as pa
import pyarrow.parquet as pq
import pytest
from botocore.exceptions import ClientError

from mermaidseg.data import PointDataset
from mermaidseg.data.config import DATASET_BUCKET, SyncPlan
from mermaidseg.data.sync import format_summary, run_sync
from segdata_spec.manifest import (
    FileEntry,
    Manifest,
    Pin,
    dump_manifest,
    manifest_sha256,
)
from segdata_spec.read import open_version
from segdata_spec.sync import DatasetSyncResult, SyncResult
from tests.mermaidseg.data.conftest import BUCKET, ImageSpec, write_version

LABELS = {5: "five"}
IMAGES = [ImageSpec("i1", group="g1"), ImageSpec("i2", group="g2")]
POINTS = [("i1", 2, 3, 5), ("i2", 4, 4, 5)]


def _client_error(code: str, status: int) -> ClientError:
    return ClientError(
        {"Error": {"Code": code, "Message": code}, "ResponseMetadata": {"HTTPStatusCode": status}},
        "GetObject",
    )


@dataclass
class FakeS3:
    """The ``get_object`` surface of a boto3 S3 client over ``(bucket, key) -> bytes``."""

    objects: dict[tuple[str, str], bytes] = field(default_factory=dict)

    def get_object(self, Bucket: str, Key: str, IfMatch: str | None = None) -> dict[str, Any]:
        data = self.objects.get((Bucket, Key))
        if data is None:
            raise _client_error("NoSuchKey", 404)
        if IfMatch is not None and hashlib.md5(data).hexdigest() != IfMatch.strip('"'):
            raise _client_error("PreconditionFailed", 412)
        return {"Body": io.BytesIO(data)}


def _md5(path: Path) -> str:
    return hashlib.md5(path.read_bytes()).hexdigest()


def _publish(tmp_path: Path) -> tuple[FakeS3, Pin]:
    """Serve a points fixture version, with the images' ETags rewritten to the MD5 of the bytes."""
    cache_dir, _, pin = write_version(
        tmp_path, kind="points", images=IMAGES, labels=LABELS, points=POINTS
    )
    prefix = cache_dir / BUCKET / "datasets" / pin.name / pin.version
    manifest = Manifest.model_validate_json((prefix / "manifest.json").read_bytes())
    store = manifest.image_store
    images = pq.read_table(prefix / "images.parquet")
    etags = [_md5(cache_dir / store.bucket / (store.prefix + k)) for k in images["key"].to_pylist()]
    images = images.set_column(
        images.schema.get_field_index("etag"), "etag", pa.array(etags, pa.string())
    )
    pq.write_table(images, prefix / "images.parquet")
    data = (prefix / "images.parquet").read_bytes()
    files = {
        **manifest.files,
        "images.parquet": FileEntry(
            sha256=hashlib.sha256(data).hexdigest(), bytes=len(data), rows=images.num_rows
        ),
    }
    manifest_bytes = dump_manifest(manifest.model_copy(update={"files": files}))
    (prefix / "manifest.json").write_bytes(manifest_bytes)

    fake = FakeS3()
    key_prefix = f"datasets/{pin.name}/{pin.version}/"
    for file_name in [*files, "manifest.json"]:
        fake.objects[(DATASET_BUCKET, key_prefix + file_name)] = (prefix / file_name).read_bytes()
    for key in images["key"].to_pylist():
        path = cache_dir / store.bucket / (store.prefix + key)
        fake.objects[(store.bucket, store.prefix + key)] = path.read_bytes()
    pinned = Pin(
        name=pin.name, version=pin.version, manifest_sha256=manifest_sha256(manifest_bytes)
    )
    return fake, pinned


def _plan(pin: Pin, groups: set[str] | None, mode: str = "fail") -> SyncPlan:
    return SyncPlan(
        pins={"ds": pin},
        taxonomy=None,
        selection={"ds": groups},
        on_missing_images={"ds": mode},  # type: ignore[dict-item]
    )


def _image_files(cache_dir: Path) -> set[str]:
    return {p.name for p in (cache_dir / "img-bucket").rglob("*.jpg")}


def test_run_sync_fills_the_cache_so_a_reader_opens_it(tmp_path: Path) -> None:
    fake, pin = _publish(tmp_path / "pub")
    cache_dir = tmp_path / "synced"
    result = run_sync(_plan(pin, None), cache_dir, fake)
    assert result.datasets["ds"].images_dropped == 0
    ds = PointDataset(open_version(cache_dir, DATASET_BUCKET, pin), split="val")
    assert len(ds) == 2
    _, mask = ds[0]
    assert mask[2, 3] == 1


def test_run_sync_forwards_the_selection(tmp_path: Path) -> None:
    fake, pin = _publish(tmp_path / "pub")
    cache_dir = tmp_path / "synced"
    run_sync(_plan(pin, {"g1"}), cache_dir, fake)
    assert _image_files(cache_dir) == {"i1.jpg"}


def test_run_sync_forwards_the_missing_mode_and_reports_the_count(tmp_path: Path) -> None:
    fake, pin = _publish(tmp_path / "pub")
    del fake.objects[("img-bucket", "imgs/i2.jpg")]
    result = run_sync(_plan(pin, None, "drop"), tmp_path / "synced", fake)
    assert result.datasets["ds"].images_dropped == 1
    assert format_summary(result) == "ds: version 20260101-abc1234, images_dropped 1"


def test_run_sync_fails_on_a_missing_image_by_default(tmp_path: Path) -> None:
    fake, pin = _publish(tmp_path / "pub")
    del fake.objects[("img-bucket", "imgs/i2.jpg")]
    with pytest.raises(Exception, match="does not exist"):
        run_sync(_plan(pin, None), tmp_path / "synced", fake)


def test_format_summary_lists_each_dataset_and_the_taxonomy() -> None:
    def dataset(version: str, dropped: int) -> DatasetSyncResult:
        return DatasetSyncResult(version, "0" * 64, dropped, ("x",) * dropped)

    result = SyncResult(
        datasets={"a": dataset("20260101-abc1234", 0), "b": dataset("20260102-abc1234", 3)},
        taxonomy=Pin(name="mermaid-benthic", version="20260103-abc1234", manifest_sha256="1" * 64),
    )
    assert format_summary(result).splitlines() == [
        "a: version 20260101-abc1234, images_dropped 0",
        "b: version 20260102-abc1234, images_dropped 3",
        "taxonomy: version 20260103-abc1234",
    ]
