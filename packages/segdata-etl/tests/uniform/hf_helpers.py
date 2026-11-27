"""Fixtures for the Hugging Face dense builds: tiny parquet shards in place of the hub."""

from __future__ import annotations

import io
import json
from pathlib import Path
from typing import Any

import pyarrow as pa
import pyarrow.parquet as pq
import pytest
from PIL import Image

from segdata_etl.common import hf_dense
from segdata_spec.git import GitState

IMAGE_STRUCT = pa.struct([("bytes", pa.binary()), ("path", pa.string())])


def image_col(blobs: list[bytes]) -> pa.Array:
    """An HF ``Image`` column: one ``{bytes, path}`` struct per blob."""
    return pa.array([{"bytes": b, "path": None} for b in blobs], type=IMAGE_STRUCT)


def mask_png(pixels: list[int], width: int, height: int) -> bytes:
    """A single-channel PNG whose row-major pixel values are ``pixels``."""
    image = Image.new("L", (width, height))
    image.putdata(pixels)
    buf = io.BytesIO()
    image.save(buf, format="PNG")
    return buf.getvalue()


def write_shard(path: Path, columns: dict[str, pa.Array]) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    pq.write_table(pa.table(columns), path)
    return path


class FakeHub:
    """Maps repo file names to local files and records what the build asked for."""

    def __init__(self, files: dict[str, Path]):
        self.files = files
        self.listed: list[tuple[str, str]] = []
        self.downloaded: list[tuple[str, str, str]] = []

    def list_repo_files(self, repo: str, revision: str) -> list[str]:
        self.listed.append((repo, revision))
        return sorted(self.files)

    def download(self, repo: str, filename: str, repo_type: str, revision: str) -> str:
        assert repo_type == "dataset"
        self.downloaded.append((repo, filename, revision))
        return str(self.files[filename])


def install_hub(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    *,
    vocabulary: tuple[str, Any],
    curated_csv: tuple[str, str],
    shards: dict[str, dict[str, pa.Array]],
) -> FakeHub:
    """Point the build at fixture shards, a vocabulary file, a curated CSV and a clean git state."""
    hub_dir = tmp_path / "hub"
    files = {name: write_shard(hub_dir / name, cols) for name, cols in shards.items()}
    vocab_name, vocab = vocabulary
    (hub_dir / vocab_name).write_text(json.dumps(vocab))
    files[vocab_name] = hub_dir / vocab_name
    hub = FakeHub(files)
    monkeypatch.setattr(hf_dense, "_list_repo_files", hub.list_repo_files)
    monkeypatch.setattr(hf_dense, "hf_hub_download", hub.download)
    curated_dir = tmp_path / "curated"
    curated_dir.mkdir()
    (curated_dir / f"{curated_csv[0]}.csv").write_text(curated_csv[1])
    monkeypatch.setattr(hf_dense, "CURATED_ROOT", curated_dir)
    root = tmp_path / "repo"
    root.mkdir()
    (root / "uv.lock").write_bytes(b"lock")
    monkeypatch.setattr("segdata_spec.git.find_repo_root", lambda _start: root)
    git = GitState(sha="abcdef1234567", dirty=False, pushed=True)
    monkeypatch.setattr("segdata_spec.git.probe_git", lambda _: git)
    return hub
