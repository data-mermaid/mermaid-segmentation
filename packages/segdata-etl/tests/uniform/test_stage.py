from __future__ import annotations

import hashlib
import json
from datetime import UTC, date, datetime
from pathlib import Path
from typing import Any

import pyarrow as pa
import pytest
from s3_fake import FakeS3

from segdata_etl.common import stage
from segdata_etl.common.copy_plan import StagedFile
from segdata_etl.common.stage import (
    next_version,
    stage_version,
)
from segdata_spec import DATASET_BUCKET
from segdata_spec.errors import ManifestNotFound
from segdata_spec.git import GitState, Publishable
from segdata_spec.manifest import ImageStore, ParentRef, SourceInfo

GIT = GitState(sha="abcdef1234567", dirty=False, pushed=True)
SOURCE = SourceInfo(type="test", uri="s3://src/", revision="r1")
STORE = ImageStore(bucket=DATASET_BUCKET, prefix="images/ds/", owner="segdata")


class _Clock(datetime):
    @classmethod
    def now(cls, tz=None):
        # A naive now() is host-local 23:30, the previous day in UTC; a UTC now() is the next day.
        return (
            datetime(2026, 11, 24, 4, 30, tzinfo=UTC)
            if tz is not None
            else datetime(2026, 11, 23, 23, 30)
        )


def _tables() -> dict[str, pa.Table]:
    images = pa.table(
        {
            "image_id": ["a"],
            "key": ["a.jpg"],
            "etag": ["e"],
            "size": [1],
            "crc64nvme": pa.nulls(1, pa.string()),
            "media_type": ["image/jpeg"],
            "width": [8],
            "height": [8],
            "group": ["g"],
            "mask_key": pa.nulls(1, pa.string()),
            "mask_etag": pa.nulls(1, pa.string()),
            "mask_size": pa.nulls(1, pa.int64()),
            "mask_crc64nvme": pa.nulls(1, pa.string()),
            "source_fingerprint": ["fp"],
            "source_name": pa.nulls(1, pa.string()),
            "attrs": [json.dumps({})],
        }
    )
    points = pa.table(
        {
            "image_id": ["a"],
            "row": [1],
            "col": [1],
            "label_id": [1],
            "annotator": ["default"],
        }
    )
    labels = pa.table(
        {
            "label_id": [1],
            "name": ["x"],
            "mermaid_benthic_attribute_id": pa.nulls(1, pa.string()),
            "mermaid_benthic_attribute": pa.nulls(1, pa.string()),
            "mermaid_growth_form": pa.nulls(1, pa.string()),
            "mapping_source": ["identity"],
        }
    )
    counts = pa.table({"image_id": ["a"], "label_id": [1], "n": [1]})
    return {
        "images.parquet": images,
        "points.parquet": points,
        "labels.parquet": labels,
        "stats/label_counts.parquet": counts,
    }


def _stage(
    stage_dir: Path,
    repo_root: Path,
    *,
    client: Any = None,
    tables: dict[str, pa.Table] | None = None,
    **kwargs,
):
    return stage_version(
        stage_dir,
        repo=Publishable(root=repo_root, git=GIT, uv_lock_sha256="0" * 64),
        client=client,
        dataset="ds",
        kind="points",
        tables=_tables() if tables is None else tables,
        image_store=STORE,
        etl_command="segdata build ds",
        source=SOURCE,
        **kwargs,
    )


@pytest.fixture
def repo_root(tmp_path: Path) -> Path:
    root = tmp_path / "repo"
    root.mkdir()
    (root / "uv.lock").write_text("lock")
    return root


def test_stage_version_stages_names_plans_and_prints(
    tmp_path: Path,
    repo_root: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
):
    monkeypatch.setattr(stage, "datetime", _Clock)
    stage_dir = tmp_path / "stage"
    copies = [{"src_bucket": "src", "src_key": "a.jpg", "dst_key": "images/ds/a.jpg"}]
    uploads = [StagedFile("images/ds/m.png", 3, "crc", "image/png")]

    staged = _stage(stage_dir, repo_root, copies=copies, uploads=uploads)

    assert staged.version == "20261124-abcdef1"
    version_dir = stage_dir / DATASET_BUCKET / "datasets" / "ds" / staged.version
    manifest_bytes = (version_dir / "manifest.json").read_bytes()
    assert staged.sha256 == hashlib.sha256(manifest_bytes).hexdigest()
    plan = [json.loads(line) for line in (version_dir / "copies.jsonl").read_text().splitlines()]
    assert [(p["src_key"], p["src_bucket"]) for p in plan] == [("a.jpg", "src")]
    (upload,) = (json.loads(x) for x in (version_dir / "uploads.jsonl").read_text().splitlines())
    assert upload["dst_key"] == "images/ds/m.png"
    assert capsys.readouterr().out == (
        f"version: {staged.version}\nmanifest_sha256: {staged.sha256}\n"
    )


def test_stage_version_reads_a_parent_from_the_extra_parent_dirs(tmp_path: Path, repo_root: Path):
    extra = tmp_path / "cache"
    parent = _stage(extra, repo_root)
    ref = ParentRef(dataset="ds", version=parent.version, manifest_sha256=parent.sha256)

    child = _stage(tmp_path / "stage", repo_root, parents=(ref,), extra_parent_dirs=(extra,))

    assert child.manifest.diff_from_parent is not None
    with pytest.raises(ManifestNotFound):
        _stage(tmp_path / "other", repo_root, parents=(ref,))


def test_next_version_appends_first_free_counter(tmp_path: Path):
    kwargs = {"today": date(2026, 11, 23), "git_sha": "abc1234def5678"}
    assert next_version(tmp_path, "bkt", "ds", **kwargs) == "20261123-abc1234"
    base = tmp_path / "bkt" / "datasets" / "ds"
    (base / "20261123-abc1234").mkdir(parents=True)
    assert next_version(tmp_path, "bkt", "ds", **kwargs) == "20261123-abc1234-2"
    (base / "20261123-abc1234-2").mkdir()
    assert next_version(tmp_path, "bkt", "ds", **kwargs) == "20261123-abc1234-3"


def test_next_version_counts_a_version_published_in_the_bucket_as_taken(tmp_path: Path):
    client = FakeS3()
    kwargs = {"today": date(2026, 11, 23), "git_sha": "abc1234def5678", "client": client}
    client.put("bkt", "datasets/ds/20261123-abc1234/manifest.json", b"{}")
    assert next_version(tmp_path, "bkt", "ds", **kwargs) == "20261123-abc1234-2"
    client.put("bkt", "datasets/ds/20261123-abc1234-2/manifest.json", b"{}")
    assert next_version(tmp_path, "bkt", "ds", **kwargs) == "20261123-abc1234-3"


def test_next_version_with_a_client_still_honours_the_staged_dir(tmp_path: Path):
    kwargs = {"today": date(2026, 11, 23), "git_sha": "abc1234def5678", "client": FakeS3()}
    (tmp_path / "bkt" / "datasets" / "ds" / "20261123-abc1234").mkdir(parents=True)
    assert next_version(tmp_path, "bkt", "ds", **kwargs) == "20261123-abc1234-2"
