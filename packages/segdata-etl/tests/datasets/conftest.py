"""Fixtures shared by the external builder tests: legacy files, a fixture taxonomy, run helpers.

Every builder reads its legacy prefix and the taxonomy from one ``FakeS3`` and stages the version
under a local ``--stage-dir``; the plan files are read back as JSON lines.
"""

from __future__ import annotations

import argparse
import hashlib
import io
import json
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from types import ModuleType
from typing import Any

import numpy as np
import pyarrow as pa
import pytest
from PIL import Image
from s3_fake import FakeS3, crc64nvme

from segdata_etl import cli
from segdata_etl.label_maps.curated import curated_source
from segdata_spec import DATASET_BUCKET
from segdata_spec._marker import SyncMarker, dump_marker
from segdata_spec.git import GitState
from segdata_spec.layout import resize_id, synced_marker_dir
from segdata_spec.manifest import Manifest, Pin, SourceInfo
from segdata_spec.publish import publish_taxonomy
from segdata_spec.read import DatasetVersion, open_version

LEGACY_BUCKET = "dev-datamermaid-sm-sources"
CLEAN_GIT = GitState(sha="abcdef1234567", dirty=False, pushed=True)
TAXONOMY_VERSION = "20261101-1234567"


def jpeg(width: int, height: int) -> bytes:
    buf = io.BytesIO()
    Image.new("RGB", (width, height), (10, 120, 200)).save(buf, format="JPEG")
    return buf.getvalue()


def png(width: int, height: int) -> bytes:
    buf = io.BytesIO()
    Image.new("RGB", (width, height), (10, 120, 200)).save(buf, format="PNG")
    return buf.getvalue()


def mask_png(values: np.ndarray) -> bytes:
    """Encode a ``(height, width)`` uint8 array as an L PNG, or ``(h, w, 3)`` as an RGB PNG."""
    buf = io.BytesIO()
    Image.fromarray(values.astype(np.uint8)).save(buf, format="PNG")
    return buf.getvalue()


def publish_fixture_taxonomy(client: Any, repo: Path) -> str:
    """Publish a three-attribute taxonomy into ``client``'s dataset bucket; return its sha256."""
    tables = {
        "benthic_attributes.parquet": pa.table(
            {
                "id": ["ba-acro", "ba-sand", "ba-hc"],
                "name": ["Acropora", "Sand", "Hard coral"],
                "parent_id": [None, None, None],
            }
        ),
        "growth_forms.parquet": pa.table({"id": ["g1"], "name": ["Branching"]}),
    }
    _, sha = publish_taxonomy(
        client,
        bucket=DATASET_BUCKET,
        name="mermaid-benthic",
        version=TAXONOMY_VERSION,
        tables=tables,
        etl_package="segdata-etl",
        etl_command="segdata taxonomy",
        source=SourceInfo(type="mermaid-api", uri="https://api.test", revision="r1"),
        repo_dir=repo,
        git=CLEAN_GIT,
    )
    return sha


def make_repo(tmp_path: Path, monkeypatch: pytest.MonkeyPatch, builder: ModuleType) -> Path:
    """Create a repo root holding ``uv.lock`` for ``builder``, with a clean pushed git state."""
    root = tmp_path / "repo"
    root.mkdir()
    (root / "uv.lock").write_text("lock")
    monkeypatch.setattr(builder, "find_repo_root", lambda _start: root)
    monkeypatch.setattr(builder, "probe_git", lambda _dir: CLEAN_GIT)
    return root


def build_args(
    builder: ModuleType, stage: Path, taxonomy_sha: str, *extra: str
) -> argparse.Namespace:
    """Parse the builder's command line, with the taxonomy cached next to the stage."""
    parser = argparse.ArgumentParser(parents=[cli._uniform_parser()])
    builder.add_arguments(parser, workdir=Path("unused"))
    argv = ["--stage-dir", str(stage), "--cache-dir", str(stage.parent / "cache")]
    argv += ["--taxonomy-version", TAXONOMY_VERSION, "--taxonomy-manifest-sha256", taxonomy_sha]
    return parser.parse_args([*argv, "--workers", "2", *extra])


def run_builder(
    builder: ModuleType, args: argparse.Namespace, s3: Any, capsys: pytest.CaptureFixture[str]
) -> Pin:
    """Run ``builder`` and return the pin it prints."""
    assert builder.run(args, s3) == 0
    lines = dict(line.split(": ", 1) for line in capsys.readouterr().out.splitlines())
    return Pin(
        name=builder.DATASET, version=lines["version"], manifest_sha256=lines["manifest_sha256"]
    )


@dataclass(frozen=True)
class Staged:
    """A version a builder staged, and the stage directory holding it."""

    stage: Path
    pin: Pin
    version: DatasetVersion
    taxonomy_sha: str


def stage_version(
    builder: ModuleType,
    s3: FakeS3,
    repo: Path,
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> Staged:
    """Publish the fixture taxonomy into ``s3``, run ``builder`` and open the staged version."""
    stage = tmp_path / "stage"
    taxonomy_sha = publish_fixture_taxonomy(s3, repo)
    pin = run_builder(builder, build_args(builder, stage, taxonomy_sha), s3, capsys)
    write_sync_marker(stage, pin)
    return Staged(stage, pin, open_version(stage, DATASET_BUCKET, pin), taxonomy_sha)


def write_sync_marker(stage: Path, pin: Pin) -> None:
    """Mark a staged version as synced with every group, which ``open_version`` requires."""
    directory = synced_marker_dir(stage, resize_id(None), DATASET_BUCKET, pin.name, pin.version)
    directory.mkdir(parents=True, exist_ok=True)
    (directory / "marker.json").write_bytes(dump_marker(SyncMarker(None, None, ())))


def plan_lines(stage: Path, pin: Pin, name: str) -> list[dict[str, Any]]:
    """Read one staged plan file of ``pin`` as a list of JSON objects."""
    path = stage / DATASET_BUCKET / "datasets" / pin.name / pin.version / name
    return [json.loads(line) for line in path.read_text().splitlines()]


def legacy_etag(data: bytes) -> str:
    return hashlib.md5(data).hexdigest()  # noqa: S324


def probed_columns(data: bytes, prefix: str = "") -> dict[str, Any]:
    """Return the unquoted etag, size and crc64nvme a single-part object of ``data`` carries."""
    return {
        f"{prefix}etag": legacy_etag(data),
        f"{prefix}size": len(data),
        f"{prefix}crc64nvme": crc64nvme(data),
    }


def no_mask() -> dict[str, Any]:
    return {"mask_key": None, "mask_etag": None, "mask_size": None, "mask_crc64nvme": None}


def assert_copies_match_sources(staged: Staged, sources: Mapping[str, tuple[str, bytes]]) -> None:
    """Assert ``copies.jsonl`` plans one copy per image row, from its legacy object.

    ``sources`` maps each image id to the full legacy key and the bytes it holds.
    """
    prefix = f"images/{staged.pin.name}/"
    rows = {row["image_id"]: row for row in staged.version.images.to_pylist()}
    assert set(rows) == set(sources)
    expected = [
        {
            "src_bucket": LEGACY_BUCKET,
            "src_key": sources[image_id][0],
            "src_etag": legacy_etag(sources[image_id][1]),
            "dst_bucket": DATASET_BUCKET,
            "dst_key": prefix + rows[image_id]["key"],
            **probed_columns(sources[image_id][1]),
        }
        for image_id in sorted(sources)
    ]
    assert plan_lines(staged.stage, staged.pin, "copies.jsonl") == expected
    for image_id, (_, data) in sources.items():
        row = rows[image_id]
        assert {k: row[k] for k in ("etag", "size", "crc64nvme")} == probed_columns(data)


def assert_curated_label_mapping(manifest: Manifest, dataset: str, labels: pa.Table) -> None:
    """Assert the manifest's label mapping is the curated map of ``dataset`` over ``labels``."""
    mapping = manifest.label_mapping
    assert mapping is not None
    assert mapping.source == "curated"
    assert mapping.n_labels == labels.num_rows
    assert mapping.git_ref == CLEAN_GIT.sha
    assert mapping.sha256 == curated_source(dataset).sha256
