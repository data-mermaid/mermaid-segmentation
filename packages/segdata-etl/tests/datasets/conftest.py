"""Fixtures shared by the external builder tests: legacy files, a fixture taxonomy, run helpers."""

from __future__ import annotations

import argparse
import hashlib
import io
from pathlib import Path
from types import ModuleType
from typing import Any

import numpy as np
import pyarrow as pa
import pytest
from PIL import Image

from segdata_etl.common import publishing
from segdata_etl.label_maps.curated import load_curated
from segdata_spec import DATASET_BUCKET
from segdata_spec.manifest import Manifest, Pin, SourceInfo
from segdata_spec.publish import GitState, publish_taxonomy
from segdata_spec.sync import _crc64nvme_b64

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
    """Publish a three-attribute taxonomy; return its manifest sha256."""
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
    monkeypatch.setattr(publishing, "probe_git", lambda _dir: CLEAN_GIT)
    return root


def build_args(builder: ModuleType, store: str | Path, taxonomy_sha: str) -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    builder.add_arguments(parser, workdir=Path("unused"))
    argv = ["--store", str(store), "--taxonomy-version", TAXONOMY_VERSION]
    return parser.parse_args([*argv, "--taxonomy-manifest-sha256", taxonomy_sha, "--workers", "2"])


def run_builder(
    builder: ModuleType, args: argparse.Namespace, s3: Any, capsys: pytest.CaptureFixture[str]
) -> Pin:
    """Run ``builder`` and return the pin it prints."""
    assert builder.run(args, s3) == 0
    lines = dict(line.split(": ", 1) for line in capsys.readouterr().out.splitlines())
    return Pin(
        name=builder.DATASET, version=lines["version"], manifest_sha256=lines["manifest_sha256"]
    )


def copied_columns(data: bytes, prefix: str = "") -> dict[str, Any]:
    """Return the etag, size and crc64nvme columns S3 reports for a copy of ``data``."""
    md5 = hashlib.md5(data).hexdigest()  # noqa: S324
    return {
        f"{prefix}etag": f'"{md5}"',
        f"{prefix}size": len(data),
        f"{prefix}crc64nvme": _crc64nvme_b64(data),
    }


def no_mask() -> dict[str, Any]:
    return {"mask_key": None, "mask_etag": None, "mask_size": None, "mask_crc64nvme": None}


def legacy_etag(data: bytes) -> str:
    return hashlib.md5(data).hexdigest()  # noqa: S324


def assert_curated_label_mapping(manifest: Manifest, dataset: str, labels: pa.Table) -> None:
    """Assert the manifest's label mapping is the curated map of ``dataset``, counted from ``labels``."""
    targets = labels.column("mermaid_benthic_attribute").to_pylist()
    ids = labels.column("mermaid_benthic_attribute_id").to_pylist()
    mapping = manifest.label_mapping
    assert mapping is not None
    assert mapping.source == "curated"
    assert mapping.n_labels == labels.num_rows
    assert mapping.n_mapped == sum(t is not None for t in targets)
    assert mapping.n_unresolved == sum(
        t is not None and i is None for t, i in zip(targets, ids, strict=True)
    )
    assert mapping.git_ref == CLEAN_GIT.sha
    assert mapping.sha256 == load_curated(dataset)[1].sha256
