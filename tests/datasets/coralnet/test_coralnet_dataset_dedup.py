"""CoralNetDataset ``image_blacklist`` / ``annotation_overrides`` on a tiny local-cache manifest."""

from __future__ import annotations

import json

import pandas as pd
import pytest

from mermaidseg.datasets.coralnet import CoralNetDataset
from mermaidseg.datasets.coralnet import coralnet_dataset as cnd
from mermaidseg.datasets.local_cache import LocalS3Cache

BUCKET = "bucket"
MANIFEST = "manifest.parquet"


@pytest.fixture
def cache(tmp_path):
    root = tmp_path / "cache"
    (root / BUCKET / cnd._CORALNET_ID2NAME_KEY).parent.mkdir(parents=True)
    (root / BUCKET / cnd._CORALNET_ID2NAME_KEY).write_text(
        json.dumps({"1": "Acropora", "2": "Hard coral", "3": "Sand"})
    )
    # Image 10 (source 1) and image 20 (source 2) are copies of one photo; image 11 is unique.
    manifest = pd.DataFrame(
        {
            "source_id": pd.array([1, 1, 1, 2, 2], dtype="int32"),
            "image_id": ["10", "10", "11", "20", "20"],
            "row": pd.array([5, 6, 7, 5, 9], dtype="int16"),
            "col": pd.array([5, 6, 7, 5, 9], dtype="int16"),
            "coralnet_id": pd.array([2, 3, 3, 1, 3], dtype="int32"),
            "source_label_name": ["2", "3", "3", "1", "3"],
            "image_s3_key": ["k10", "k10", "k11", "k20", "k20"],
        }
    )
    manifest.to_parquet(root / BUCKET / MANIFEST, index=False)
    LocalS3Cache.reset_singleton()
    LocalS3Cache.configure(root, offline=True)
    yield tmp_path
    LocalS3Cache.reset_singleton()


def _make(**kwargs) -> CoralNetDataset:
    return CoralNetDataset(manifest_path=MANIFEST, source_bucket=BUCKET, **kwargs)


def test_image_blacklist_list_and_file(cache):
    assert len(_make()) == 3
    ds = _make(image_blacklist=[20])
    assert sorted(ds.df_images["image_id"]) == ["10", "11"]
    assert "20" not in set(ds.df_annotations["image_id"])

    path = cache / "blacklist.parquet"
    pd.DataFrame({"source_id": [1], "image_id": ["11"]}).to_parquet(path)
    ds = _make(image_blacklist=str(path))
    assert sorted(ds.df_images["image_id"]) == ["10", "20"]


def test_annotation_overrides_replace_points(cache):
    path = cache / "overrides.parquet"
    # Merged points for the kept copy (image 10): finer label at (5, 5), plus (9, 9) from image 20.
    pd.DataFrame(
        {
            "source_id": [1, 1, 1],
            "image_id": ["10", "10", "10"],
            "row": [5, 6, 9],
            "col": [5, 6, 9],
            "coralnet_id": [1, 3, 3],
        }
    ).to_parquet(path)
    ds = _make(image_blacklist=["20"], annotation_overrides=str(path))

    ann = ds.df_annotations
    img10 = ann[ann["image_id"] == "10"].sort_values("row")
    assert img10[["row", "col", "coralnet_id"]].to_numpy().tolist() == [
        [5, 5, 1],
        [6, 6, 3],
        [9, 9, 3],
    ]
    assert img10["source_label_name"].tolist() == ["acropora", "sand", "sand"]
    assert set(img10["image_key"]) == {"k10"}
    assert ann.loc[ann["image_id"] == "11", "coralnet_id"].tolist() == [3]  # untouched
    assert sorted(ds.df_images["image_id"]) == ["10", "11"]
    assert ann["row"].dtype == "int16" and ann["coralnet_id"].dtype == "int32"


def test_annotation_overrides_respect_source_whitelist(cache):
    path = cache / "overrides.parquet"
    pd.DataFrame(
        {"source_id": [1], "image_id": ["10"], "row": [1], "col": [1], "coralnet_id": [1]}
    ).to_parquet(path)
    ds = _make(whitelist_sources=[2], annotation_overrides=str(path))
    assert sorted(ds.df_images["image_id"]) == ["20"]


def test_parquet_refs_resolve_against_repo_root(cache, monkeypatch):
    # A relative reference is a repo file when one exists (configs/coralnet/...), else an S3 key.
    repo = cache / "repo"
    (repo / "configs/coralnet").mkdir(parents=True)
    pd.DataFrame({"image_id": ["20"]}).to_parquet(repo / "configs/coralnet/blacklist.parquet")
    pd.DataFrame({"image_id": ["11"]}).to_parquet(cache / "cache" / BUCKET / "bucket_blacklist.parquet")
    monkeypatch.setattr(cnd, "_REPO_ROOT", repo)

    ds = _make(image_blacklist="configs/coralnet/blacklist.parquet")
    assert sorted(ds.df_images["image_id"]) == ["10", "11"]
    ds = _make(image_blacklist="bucket_blacklist.parquet")
    assert sorted(ds.df_images["image_id"]) == ["10", "20"]
