"""Tests for ``segdata build ucsd_mosaics``: site groups, id policy and the research-only license."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from segdata_etl import cli
from segdata_etl.common import hf_dense
from segdata_spec import DATASET_BUCKET
from segdata_spec.publish import GitState, PublishRefused

from .conftest import FakeS3, jpeg_bytes
from .hf_helpers import image_col, install_hub, mask_png

CSV = "label_id,name,mermaid_benthic_attribute\n1,Acropora (branching),acropora\n"
CLASSES = [
    {
        "id": 1,
        "raw_legend_id": 0,
        "name": "Acropora (branching)",
        "description": "",
        "color_rgb": [],
    },
    {"id": 2, "raw_legend_id": 1, "name": "Fungia", "description": "", "color_rgb": []},
]
IMG = jpeg_bytes(2, 2)
MASK = mask_png([0, 1, 2, 2], 2, 2)


def shard(site: str, filenames: list[str], original_split: str = "train") -> dict[str, Any]:
    n = len(filenames)
    return {
        "image": image_col([IMG] * n),
        "label": image_col([MASK] * n),
        "filename": pa.array(filenames),
        "site": pa.array([site] * n),
        "original_split": pa.array([original_split] * n),
    }


def build(monkeypatch, tmp_path, shards) -> Path:
    install_hub(
        monkeypatch,
        tmp_path,
        vocabulary=("classes.json", CLASSES),
        curated_csv=("ucsd_mosaics", CSV),
        shards=shards,
    )
    stage = tmp_path / "stage"
    args = cli.build_parser().parse_args(["build", "ucsd_mosaics", "--stage-dir", str(stage)])
    assert args.run(args, FakeS3()) == 0
    (version,) = (stage / DATASET_BUCKET / "datasets" / "ucsd_mosaics").iterdir()
    return version


def images(version: Path) -> dict[str, dict[str, Any]]:
    rows = pq.read_table(version / "images.parquet").to_pylist()
    return {r["image_id"]: r for r in rows}


def test_site_becomes_a_lowercase_group_with_original_case_in_attrs(monkeypatch, tmp_path):
    shards = {"data/palwave13-00000-of-00001.parquet": shard("PALWave13", ["tile_1.jpg"], "test")}

    version = build(monkeypatch, tmp_path, shards)

    row = images(version)["tile_1"]
    assert row["group"] == "palwave13"
    assert row["key"] == "palwave13/tile_1.jpg"
    assert row["mask_key"] == "palwave13/tile_1_label.png"
    assert row["source_name"] == "tile_1.jpg"
    assert json.loads(row["attrs"]) == {"original_split": "test", "site": "PALWave13"}
    manifest = json.loads((version / "manifest.json").read_bytes())
    assert manifest["source"]["uri"] == "hf://datasets/josauder/UCSD-mosaics-mirror"
    assert manifest["source"]["revision"] == "6a6483841f8ee20bd7d8ee2aa760f10d1e9db704"
    assert (
        manifest["source"]["license"] == "research-only (LICENSE file absent at revision 6a64838)"
    )
    assert (version.parents[2] / "images/ucsd_mosaics/palwave13/tile_1.jpg").read_bytes() == IMG


def test_stems_shared_across_sites_make_every_image_id_group_qualified(
    monkeypatch, tmp_path, caplog
):
    shards = {
        "data/fr3-00000-of-00001.parquet": shard("FR3", ["tile_1.jpg", "only_fr3.jpg"]),
        "data/fr5-00000-of-00001.parquet": shard("FR5", ["tile_1.jpg"]),
    }

    with caplog.at_level("INFO"):
        version = build(monkeypatch, tmp_path, shards)

    rows = images(version)
    assert set(rows) == {"fr3/tile_1", "fr3/only_fr3", "fr5/tile_1"}
    assert rows["fr5/tile_1"]["key"] == "fr5/tile_1.jpg"
    assert "group-qualified" in caplog.text


def test_unique_stems_are_the_image_ids(monkeypatch, tmp_path):
    shards = {
        "data/fr3-00000-of-00001.parquet": shard("FR3", ["a.jpg"]),
        "data/fr5-00000-of-00001.parquet": shard("FR5", ["b.jpg"]),
    }

    version = build(monkeypatch, tmp_path, shards)

    assert set(images(version)) == {"a", "b"}


def test_stem_repeated_within_one_site_raises_naming_the_pair_before_any_download(
    monkeypatch, tmp_path
):
    shards = {"data/palwave13-00000-of-00001.parquet": shard("PALWave13", ["a.jpg", "sub/a.jpg"])}

    with pytest.raises(ValueError, match=r"palwave13.*a"):
        build(monkeypatch, tmp_path, shards)


def test_dirty_git_refuses_before_any_shard_download(monkeypatch, tmp_path):
    hub = install_hub(
        monkeypatch,
        tmp_path,
        vocabulary=("classes.json", CLASSES),
        curated_csv=("ucsd_mosaics", CSV),
        shards={"data/palwave13-00000-of-00001.parquet": shard("PALWave13", ["a.jpg"])},
    )
    dirty = GitState(sha="abcdef1234567", dirty=True, pushed=True)
    monkeypatch.setattr(hf_dense, "probe_git", lambda _: dirty)
    args = cli.build_parser().parse_args(
        ["build", "ucsd_mosaics", "--stage-dir", str(tmp_path / "stage")]
    )

    with pytest.raises(PublishRefused, match="uncommitted"):
        args.run(args, FakeS3())

    assert hub.downloaded == []
