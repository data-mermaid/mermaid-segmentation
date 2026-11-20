import hashlib
import subprocess
from pathlib import Path

import pyarrow as pa
import pytest
from conftest import FakeS3

from segdata_spec.errors import (
    ObjectExists,
    PinMismatch,
    PublishRefused,
    SchemaValidationError,
)
from segdata_spec.manifest import ImageStore, ParentRef, SourceInfo
from segdata_spec.publish import (
    GitState,
    compute_diff,
    probe_git,
    publish_dataset,
    publish_taxonomy,
)

BUCKET = "seg-bucket"
VERSION = "20261120-abcdef1"
PREFIX = f"datasets/ds/{VERSION}/"
CLEAN = GitState(sha="abcdef1234567", dirty=False, pushed=True)
STORE = ImageStore(bucket="img-bucket", prefix="images/ds/", owner="segdata")
SOURCE = SourceInfo(type="test", uri="s3://src/", revision="r1")


def _images(rows: list[tuple[str, str, int, str]]) -> pa.Table:
    """Rows are (image_id, etag, size, group)."""
    return pa.table(
        {
            "image_id": [r[0] for r in rows],
            "key": [f"images/{r[0]}.jpg" for r in rows],
            "etag": [r[1] for r in rows],
            "size": [r[2] for r in rows],
            "media_type": ["image/jpeg"] * len(rows),
            "width": [4] * len(rows),
            "height": [4] * len(rows),
            "group": [r[3] for r in rows],
            "source_fingerprint": ["fp"] * len(rows),
            "crc64nvme": pa.nulls(len(rows), pa.string()),
            "mask_key": pa.nulls(len(rows), pa.string()),
            "mask_etag": pa.nulls(len(rows), pa.string()),
            "mask_size": pa.nulls(len(rows), pa.int64()),
            "mask_crc64nvme": pa.nulls(len(rows), pa.string()),
            "attrs": pa.nulls(len(rows), pa.string()),
        }
    )


def _points(rows: list[tuple[str, int, int, int]]) -> pa.Table:
    """Rows are (image_id, row, col, label_id); annotator is fixed."""
    return pa.table(
        {
            "image_id": [r[0] for r in rows],
            "row": [r[1] for r in rows],
            "col": [r[2] for r in rows],
            "label_id": [r[3] for r in rows],
            "annotator": ["default"] * len(rows),
        }
    )


def _dataset_tables(
    images: pa.Table | None = None, points: pa.Table | None = None
) -> dict[str, pa.Table]:
    return {
        "images.parquet": images
        if images is not None
        else _images([("a", "e1", 10, "g1"), ("b", "e2", 20, "g1"), ("c", "e3", 30, "g2")]),
        "points.parquet": points
        if points is not None
        else _points([("a", 1, 1, 5), ("b", 2, 2, 6)]),
        "labels.parquet": pa.table(
            {
                "label_id": [5, 6],
                "name": ["x", "y"],
                "mapping_source": ["curated", "curated"],
                "mermaid_benthic_attribute_id": pa.nulls(2, pa.string()),
                "mermaid_benthic_attribute": pa.nulls(2, pa.string()),
                "mermaid_growth_form": pa.nulls(2, pa.string()),
            }
        ),
        "stats/label_counts.parquet": pa.table(
            {"image_id": ["a", "b"], "label_id": [5, 6], "n": [1, 1]}
        ),
    }


@pytest.fixture
def repo_dir(tmp_path: Path) -> Path:
    (tmp_path / "uv.lock").write_bytes(b"lock-bytes")
    return tmp_path


def _publish(client: FakeS3, repo_dir: Path, **overrides):
    kwargs = {
        "bucket": BUCKET,
        "dataset": "ds",
        "version": VERSION,
        "kind": "points",
        "tables": _dataset_tables(),
        "image_store": STORE,
        "etl_package": "segdata-etl",
        "etl_command": "segdata build ds",
        "source": SOURCE,
        "repo_dir": repo_dir,
        "git": CLEAN,
    }
    kwargs.update(overrides)
    return publish_dataset(client, **kwargs)


def _put_keys(client: FakeS3) -> list[str]:
    return [key for op, _, key, _ in client.calls if op == "put"]


def test_tables_precede_manifest_with_write_once_puts(fake_s3, repo_dir):
    manifest, sha = _publish(fake_s3, repo_dir)

    keys = _put_keys(fake_s3)
    assert keys[-1] == PREFIX + "manifest.json"
    assert sorted(keys[:-1]) == sorted(
        PREFIX + n
        for n in (
            "images.parquet",
            "points.parquet",
            "labels.parquet",
            "stats/label_counts.parquet",
        )
    )
    assert all(kw == {"IfNoneMatch": "*"} for op, _, _, kw in fake_s3.calls if op == "put")
    stored = fake_s3.objects[(BUCKET, PREFIX + "manifest.json")]
    assert sha == hashlib.sha256(stored).hexdigest()
    images_bytes = fake_s3.objects[(BUCKET, PREFIX + "images.parquet")]
    entry = manifest.files["images.parquet"]
    assert entry.sha256 == hashlib.sha256(images_bytes).hexdigest()
    assert (entry.bytes, entry.rows) == (len(images_bytes), 3)
    assert manifest.etl.git_sha == "abcdef1234567"
    assert manifest.etl.git_dirty is False
    assert manifest.etl.uv_lock_sha256 == hashlib.sha256(b"lock-bytes").hexdigest()


def test_stats_are_derived_from_tables(fake_s3, repo_dir):
    manifest, _ = _publish(fake_s3, repo_dir, dropped={"missing_image": 4})

    assert manifest.stats.model_dump() == {
        "n_images": 3,
        "n_points": 2,
        "n_groups": 2,
        "image_bytes": 60,
        "dropped": {"missing_image": 4},
    }


def test_existing_manifest_refuses_and_writes_nothing(fake_s3, repo_dir):
    fake_s3.put(BUCKET, PREFIX + "manifest.json", b"{}")

    with pytest.raises(PublishRefused):
        _publish(fake_s3, repo_dir)

    assert _put_keys(fake_s3) == []


def test_existing_table_raises_and_manifest_is_not_written(fake_s3, repo_dir):
    fake_s3.put(BUCKET, PREFIX + "images.parquet", b"old")

    with pytest.raises(ObjectExists):
        _publish(fake_s3, repo_dir)

    assert fake_s3.objects[(BUCKET, PREFIX + "images.parquet")] == b"old"
    assert (BUCKET, PREFIX + "manifest.json") not in fake_s3.objects


@pytest.mark.parametrize(
    "git",
    [
        GitState(sha="abcdef1", dirty=True, pushed=True),
        GitState(sha="abcdef1", dirty=False, pushed=False),
    ],
)
def test_dirty_or_unpushed_tree_refuses_before_any_write(fake_s3, repo_dir, git):
    with pytest.raises(PublishRefused):
        _publish(fake_s3, repo_dir, git=git)

    assert fake_s3.objects == {}


def _run(cwd: Path, *args: str) -> None:
    subprocess.run(["git", *args], cwd=cwd, check=True, capture_output=True)  # noqa: S603, S607


def test_probe_git_reports_dirty_and_pushed(tmp_path):
    origin = tmp_path / "origin.git"
    work = tmp_path / "work"
    work.mkdir()
    _run(tmp_path, "init", "--bare", "-b", "main", str(origin))
    _run(work, "init", "-b", "main")
    _run(work, "config", "user.name", "t")
    _run(work, "config", "user.email", "t@example.com")
    _run(work, "remote", "add", "origin", str(origin))
    tracked = work / "f.txt"
    tracked.write_text("one")
    _run(work, "add", "f.txt")
    _run(work, "commit", "-m", "one")
    _run(work, "push", "-u", "origin", "main")

    state = probe_git(work)
    assert (state.dirty, state.pushed) == (False, True)
    assert len(state.sha) == 40

    (work / "untracked.txt").write_text("x")
    assert probe_git(work).dirty is False

    tracked.write_text("two")
    assert probe_git(work).dirty is True

    _run(work, "commit", "-am", "two")
    local = probe_git(work)
    assert (local.dirty, local.pushed) == (False, False)


def test_diff_counts_image_and_point_changes():
    parent = {
        "images.parquet": _images([("a", "e1", 1, "g"), ("b", "e2", 2, "g"), ("c", "e3", 3, "g")]),
        "points.parquet": _points([("a", 1, 1, 5), ("b", 2, 2, 5)]),
    }
    child = {
        "images.parquet": _images([("b", "e2", 2, "g"), ("c", "e9", 3, "g"), ("d", "e4", 4, "g")]),
        "points.parquet": _points([("b", 2, 2, 5), ("b", 3, 3, 6), ("d", 1, 1, 5)]),
    }

    diff = compute_diff(parent, child)

    assert diff.model_dump() == {
        "images_added": 1,
        "images_removed": 1,
        "images_changed": 1,
        "points_added": 2,
        "points_removed": 1,
    }


def _parent_version(fake_s3, repo_dir):
    tables = _dataset_tables(
        images=_images([("a", "e1", 10, "g1"), ("b", "e2", 20, "g1")]),
        points=_points([("a", 1, 1, 5), ("b", 2, 2, 6)]),
    )
    _, sha = _publish(
        fake_s3, repo_dir, dataset="parent", version="20261101-1234567", tables=tables
    )
    return ParentRef(dataset="parent", version="20261101-1234567", manifest_sha256=sha)


def test_parent_pin_mismatch_writes_nothing(fake_s3, repo_dir):
    parent = _parent_version(fake_s3, repo_dir)
    bad = parent.model_copy(update={"manifest_sha256": "0" * 64})
    before = len(_put_keys(fake_s3))

    with pytest.raises(PinMismatch):
        _publish(fake_s3, repo_dir, parents=(bad,))

    assert len(_put_keys(fake_s3)) == before


def test_no_parents_gives_no_diff(fake_s3, repo_dir):
    manifest, _ = _publish(fake_s3, repo_dir)

    assert manifest.diff_from_parent is None


def test_diff_is_taken_against_the_first_parent_only(fake_s3, repo_dir):
    first = _parent_version(fake_s3, repo_dir)
    ghost = ParentRef(dataset="ghost", version="20261001-7654321", manifest_sha256="1" * 64)
    child = _dataset_tables(
        images=_images([("b", "e2", 20, "g1"), ("c", "e3", 30, "g2")]),
        points=_points([("b", 2, 2, 6)]),
    )

    manifest, _ = _publish(fake_s3, repo_dir, parents=(first, ghost), tables=child)

    assert manifest.parents == (first, ghost)
    assert manifest.diff_from_parent.model_dump() == {
        "images_added": 1,
        "images_removed": 1,
        "images_changed": 0,
        "points_added": 0,
        "points_removed": 1,
    }


def test_taxonomy_tables_precede_manifest(fake_s3, repo_dir):
    tables = {
        "benthic_attributes.parquet": pa.table(
            {"id": ["1"], "name": ["Coral"], "parent_id": [None]}
        ),
        "growth_forms.parquet": pa.table({"id": ["2"], "name": ["Branching"]}),
    }

    manifest, sha = publish_taxonomy(
        fake_s3,
        bucket=BUCKET,
        name="mermaid-benthic",
        version=VERSION,
        tables=tables,
        etl_package="segdata-etl",
        etl_command="segdata taxonomy",
        source=SOURCE,
        repo_dir=repo_dir,
        git=CLEAN,
    )

    prefix = f"taxonomy/mermaid-benthic/{VERSION}/"
    assert _put_keys(fake_s3) == [
        prefix + "benthic_attributes.parquet",
        prefix + "growth_forms.parquet",
        prefix + "manifest.json",
    ]
    assert sha == hashlib.sha256(fake_s3.objects[(BUCKET, prefix + "manifest.json")]).hexdigest()
    assert manifest.files["growth_forms.parquet"].rows == 1


def test_invalid_table_raises_before_any_write(fake_s3, repo_dir):
    tables = _dataset_tables()
    tables["labels.parquet"] = tables["labels.parquet"].drop(["name"])

    with pytest.raises(SchemaValidationError):
        _publish(fake_s3, repo_dir, tables=tables)

    assert fake_s3.calls == []


def test_missing_uv_lock_refuses_before_any_write(fake_s3, tmp_path):
    with pytest.raises(PublishRefused, match="uv.lock"):
        _publish(fake_s3, tmp_path)

    assert fake_s3.calls == []


def test_retry_after_a_partial_publish_adopts_identical_tables(fake_s3, repo_dir):
    _publish(fake_s3, repo_dir)
    del fake_s3.objects[(BUCKET, PREFIX + "manifest.json")]
    fake_s3.calls.clear()

    _, sha = _publish(fake_s3, repo_dir)

    stored = fake_s3.objects[(BUCKET, PREFIX + "manifest.json")]
    assert sha == hashlib.sha256(stored).hexdigest()
    assert len(_put_keys(fake_s3)) == 5
