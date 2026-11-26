import hashlib
import subprocess
from pathlib import Path

import pyarrow as pa
import pytest

from segdata_spec import stage
from segdata_spec.errors import (
    ChecksumMismatch,
    ManifestNotFound,
    ObjectExists,
    PinMismatch,
    PublishRefused,
    SchemaValidationError,
)
from segdata_spec.git import GitState
from segdata_spec.layout import cache_path
from segdata_spec.manifest import ImageStore, ParentRef, SourceInfo
from segdata_spec.stage import (
    compute_diff,
    read_parent_tables,
    stage_dataset,
    stage_taxonomy,
)

BUCKET = "seg-bucket"
VERSION = "20261120-abcdef1"
PREFIX = f"datasets/ds/{VERSION}/"
TABLES = (
    "images.parquet",
    "points.parquet",
    "labels.parquet",
    "stats/label_counts.parquet",
)
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
            "source_name": pa.nulls(len(rows), pa.string()),
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
    repo = tmp_path / "repo"
    repo.mkdir()
    (repo / "uv.lock").write_bytes(b"lock-bytes")
    return repo


@pytest.fixture
def stage_dir(tmp_path: Path) -> Path:
    return tmp_path / "stage"


def _stage(stage_dir: Path, repo_dir: Path, **overrides):
    kwargs = {
        "parent_dirs": (stage_dir,),
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
    return stage_dataset(stage_dir, **kwargs)


def _version_dir(stage_dir: Path, prefix: str = PREFIX) -> Path:
    return cache_path(stage_dir, BUCKET, prefix)


def _files(stage_dir: Path) -> list[Path]:
    return sorted(p for p in stage_dir.rglob("*") if p.is_file()) if stage_dir.exists() else []


@pytest.fixture
def write_log(monkeypatch) -> list[Path]:
    """Paths written, in order."""
    written: list[Path] = []
    real = stage._atomic_write

    def spy(path: Path, data: bytes) -> None:
        written.append(path)
        real(path, data)

    monkeypatch.setattr(stage, "_atomic_write", spy)
    return written


def test_tables_precede_manifest(stage_dir, repo_dir, write_log):
    manifest, sha = _stage(stage_dir, repo_dir)

    version_dir = _version_dir(stage_dir)
    assert write_log[-1] == version_dir / "manifest.json"
    assert sorted(p.relative_to(version_dir).as_posix() for p in write_log[:-1]) == sorted(TABLES)
    stored = (version_dir / "manifest.json").read_bytes()
    assert sha == hashlib.sha256(stored).hexdigest()
    images_bytes = (version_dir / "images.parquet").read_bytes()
    entry = manifest.files["images.parquet"]
    assert entry.sha256 == hashlib.sha256(images_bytes).hexdigest()
    assert (entry.bytes, entry.rows) == (len(images_bytes), 3)
    assert manifest.etl.git_sha == "abcdef1234567"
    assert manifest.etl.git_dirty is False
    assert manifest.etl.uv_lock_sha256 == hashlib.sha256(b"lock-bytes").hexdigest()


def test_stats_are_derived_from_tables(stage_dir, repo_dir):
    manifest, _ = _stage(stage_dir, repo_dir, dropped={"missing_image": 4})

    assert manifest.stats.model_dump() == {
        "n_images": 3,
        "n_points": 2,
        "n_groups": 2,
        "image_bytes": 60,
        "dropped": {"missing_image": 4},
    }


def test_existing_staged_manifest_refuses_and_writes_nothing(stage_dir, repo_dir):
    version_dir = _version_dir(stage_dir)
    version_dir.mkdir(parents=True)
    (version_dir / "manifest.json").write_bytes(b"{}")

    with pytest.raises(PublishRefused):
        _stage(stage_dir, repo_dir)

    assert _files(stage_dir) == [version_dir / "manifest.json"]
    assert (version_dir / "manifest.json").read_bytes() == b"{}"


def test_existing_different_table_raises_and_manifest_is_not_written(stage_dir, repo_dir):
    version_dir = _version_dir(stage_dir)
    version_dir.mkdir(parents=True)
    (version_dir / "images.parquet").write_bytes(b"old")

    with pytest.raises(ObjectExists):
        _stage(stage_dir, repo_dir)

    assert (version_dir / "images.parquet").read_bytes() == b"old"
    assert not (version_dir / "manifest.json").exists()


@pytest.mark.parametrize(
    "git",
    [
        GitState(sha="abcdef1", dirty=True, pushed=True),
        GitState(sha="abcdef1", dirty=False, pushed=False),
    ],
)
def test_dirty_or_unpushed_tree_refuses_before_any_write(stage_dir, repo_dir, git):
    with pytest.raises(PublishRefused):
        _stage(stage_dir, repo_dir, git=git)

    assert _files(stage_dir) == []


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


def _parent_version(stage_dir, repo_dir):
    tables = _dataset_tables(
        images=_images([("a", "e1", 10, "g1"), ("b", "e2", 20, "g1")]),
        points=_points([("a", 1, 1, 5), ("b", 2, 2, 6)]),
    )
    _, sha = _stage(
        stage_dir, repo_dir, dataset="parent", version="20261101-1234567", tables=tables
    )
    return ParentRef(dataset="parent", version="20261101-1234567", manifest_sha256=sha)


def test_parent_pin_mismatch_writes_nothing(stage_dir, repo_dir):
    parent = _parent_version(stage_dir, repo_dir)
    bad = parent.model_copy(update={"manifest_sha256": "0" * 64})
    before = _files(stage_dir)

    with pytest.raises(PinMismatch):
        _stage(stage_dir, repo_dir, parents=(bad,))

    assert _files(stage_dir) == before


def test_no_parents_gives_no_diff(stage_dir, repo_dir):
    manifest, _ = _stage(stage_dir, repo_dir)

    assert manifest.diff_from_parent is None


def test_diff_is_taken_against_the_first_parent_only(stage_dir, repo_dir):
    first = _parent_version(stage_dir, repo_dir)
    ghost = ParentRef(dataset="ghost", version="20261001-7654321", manifest_sha256="1" * 64)
    child = _dataset_tables(
        images=_images([("b", "e2", 20, "g1"), ("c", "e3", 30, "g2")]),
        points=_points([("b", 2, 2, 6)]),
    )

    manifest, _ = _stage(stage_dir, repo_dir, parents=(first, ghost), tables=child)

    assert manifest.parents == (first, ghost)
    assert manifest.diff_from_parent.model_dump() == {
        "images_added": 1,
        "images_removed": 1,
        "images_changed": 0,
        "points_added": 0,
        "points_removed": 1,
    }


def test_stage_reads_the_parent_from_a_later_parent_dir(tmp_path, repo_dir):
    cache, stage_dir = tmp_path / "cache", tmp_path / "stage"
    parent = _parent_version(cache, repo_dir)

    manifest, _ = _stage(stage_dir, repo_dir, parent_dirs=(stage_dir, cache), parents=(parent,))

    assert manifest.diff_from_parent is not None


def _taxonomy_kwargs(repo_dir: Path, tables: dict) -> dict:
    return {
        "bucket": BUCKET,
        "name": "mermaid-benthic",
        "version": VERSION,
        "tables": tables,
        "etl_package": "segdata-etl",
        "etl_command": "segdata taxonomy",
        "source": SOURCE,
        "repo_dir": repo_dir,
        "git": CLEAN,
    }


def _taxonomy_tables() -> dict[str, pa.Table]:
    return {
        "benthic_attributes.parquet": pa.table(
            {"id": ["1"], "name": ["Coral"], "parent_id": [None]}
        ),
        "growth_forms.parquet": pa.table({"id": ["2"], "name": ["Branching"]}),
    }


def test_taxonomy_tables_precede_manifest(stage_dir, repo_dir, write_log):
    tables = _taxonomy_tables()

    manifest, sha = stage_taxonomy(stage_dir, **_taxonomy_kwargs(repo_dir, tables))

    version_dir = cache_path(stage_dir, BUCKET, f"taxonomy/mermaid-benthic/{VERSION}/")
    assert write_log == [
        version_dir / "benthic_attributes.parquet",
        version_dir / "growth_forms.parquet",
        version_dir / "manifest.json",
    ]
    assert sha == hashlib.sha256((version_dir / "manifest.json").read_bytes()).hexdigest()
    assert manifest.files["growth_forms.parquet"].rows == 1


def test_taxonomy_existing_staged_manifest_refuses(stage_dir, repo_dir):
    version_dir = cache_path(stage_dir, BUCKET, f"taxonomy/mermaid-benthic/{VERSION}/")
    version_dir.mkdir(parents=True)
    (version_dir / "manifest.json").write_bytes(b"{}")

    with pytest.raises(PublishRefused):
        stage_taxonomy(stage_dir, **_taxonomy_kwargs(repo_dir, _taxonomy_tables()))

    assert _files(stage_dir) == [version_dir / "manifest.json"]


def test_invalid_table_raises_before_any_write(stage_dir, repo_dir):
    tables = _dataset_tables()
    tables["labels.parquet"] = tables["labels.parquet"].drop(["name"])

    with pytest.raises(SchemaValidationError):
        _stage(stage_dir, repo_dir, tables=tables)

    assert _files(stage_dir) == []


def test_missing_uv_lock_refuses_before_any_write(stage_dir, tmp_path):
    with pytest.raises(PublishRefused, match="uv.lock"):
        _stage(stage_dir, tmp_path)

    assert _files(stage_dir) == []


def test_rerun_after_a_partial_stage_adopts_identical_tables(stage_dir, repo_dir):
    _stage(stage_dir, repo_dir)
    version_dir = _version_dir(stage_dir)
    (version_dir / "manifest.json").unlink()

    _, sha = _stage(stage_dir, repo_dir)

    assert sha == hashlib.sha256((version_dir / "manifest.json").read_bytes()).hexdigest()
    assert len(_files(stage_dir)) == 5


def test_read_parent_tables_uses_the_first_dir_holding_the_manifest(tmp_path, repo_dir):
    first, second = tmp_path / "first", tmp_path / "second"
    parent = _parent_version(second, repo_dir)

    tables = read_parent_tables([first, second], BUCKET, parent)

    assert sorted(tables) == ["images.parquet", "points.parquet"]
    parent_dir = _version_dir(second, "datasets/parent/20261101-1234567/")
    assert tables["images.parquet"] == (parent_dir / "images.parquet").read_bytes()


def test_read_parent_tables_prefers_the_earlier_dir(tmp_path, repo_dir):
    first, second = tmp_path / "first", tmp_path / "second"
    parent = _parent_version(first, repo_dir)
    _parent_version(second, repo_dir)
    parent_dir = _version_dir(second, "datasets/parent/20261101-1234567/")
    (parent_dir / "images.parquet").write_bytes(b"differs")

    assert read_parent_tables([first, second], BUCKET, parent)["images.parquet"] != b"differs"


def test_read_parent_tables_names_a_parent_found_in_no_dir(tmp_path):
    ghost = ParentRef(dataset="ghost", version="20261001-7654321", manifest_sha256="1" * 64)

    with pytest.raises(ManifestNotFound, match="datasets/ghost/20261001-7654321/manifest.json"):
        read_parent_tables([tmp_path / "a", tmp_path / "b"], BUCKET, ghost)


def test_read_parent_tables_refuses_a_manifest_that_misses_its_pin(tmp_path, repo_dir):
    parent = _parent_version(tmp_path, repo_dir)
    bad = parent.model_copy(update={"manifest_sha256": "0" * 64})

    with pytest.raises(PinMismatch):
        read_parent_tables([tmp_path], BUCKET, bad)


def test_read_parent_tables_refuses_a_table_that_misses_its_manifest(tmp_path, repo_dir):
    parent = _parent_version(tmp_path, repo_dir)
    parent_dir = _version_dir(tmp_path, "datasets/parent/20261101-1234567/")
    (parent_dir / "points.parquet").write_bytes(b"tampered")

    with pytest.raises(ChecksumMismatch, match="points.parquet"):
        read_parent_tables([tmp_path], BUCKET, parent)


def test_read_parent_tables_refuses_a_listed_table_missing_from_disk(tmp_path, repo_dir):
    parent = _parent_version(tmp_path, repo_dir)
    parent_dir = _version_dir(tmp_path, "datasets/parent/20261101-1234567/")
    (parent_dir / "images.parquet").unlink()

    with pytest.raises(ChecksumMismatch, match="missing"):
        read_parent_tables([tmp_path], BUCKET, parent)


def _dirty_repo(root: Path) -> Path:
    """A repo with a committed uv.lock and tracked file, the tracked file then modified."""
    root.mkdir()

    def run(*args: str) -> None:
        subprocess.run(["git", *args], cwd=root, check=True, capture_output=True)  # noqa: S603, S607

    run("init", "-b", "main")
    run("config", "user.name", "t")
    run("config", "user.email", "t@example.com")
    (root / "uv.lock").write_text("lock")
    (root / "f.txt").write_text("one")
    run("add", ".")
    run("-c", "commit.gpgsign=false", "commit", "-m", "one")
    (root / "f.txt").write_text("two")
    return root


@pytest.mark.parametrize("which", ["dataset", "taxonomy"])
def test_staging_without_a_git_state_probes_the_repo_and_refuses_a_dirty_one(
    which, tmp_path, stage_dir, monkeypatch
):
    repo = _dirty_repo(tmp_path / "dirty")
    not_a_repo = tmp_path / "outside"
    not_a_repo.mkdir()
    monkeypatch.chdir(not_a_repo)

    with pytest.raises(PublishRefused, match="uncommitted"):
        if which == "dataset":
            _stage(stage_dir, repo, git=None)
        else:
            kwargs = {**_taxonomy_kwargs(repo, _taxonomy_tables()), "git": None}
            stage_taxonomy(stage_dir, **kwargs)

    assert _files(stage_dir) == []
