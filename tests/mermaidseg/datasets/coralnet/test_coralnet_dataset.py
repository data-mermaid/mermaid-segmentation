"""CoralNetDataset ``image_blacklist`` / ``annotation_overrides`` on a tiny local-cache manifest."""

from __future__ import annotations

import hashlib
import json
import shutil

import numpy as np
import pandas as pd
import pytest
from PIL import Image

from mermaidseg.datasets.coralnet import CoralNetDataset
from mermaidseg.datasets.coralnet import coralnet_dataset as cnd
from mermaidseg.datasets.local_cache import LocalS3Cache

BUCKET = "bucket"
RUN = "20260623_nogit"
MANIFEST = f"etl-outputs/coralnet/{RUN}/coralnet_training_manifest_{RUN}.parquet"
ID2NAME_KEY = f"etl-outputs/coralnet/{RUN}/coralnet_id2name_{RUN}.json"


@pytest.fixture
def cache(tmp_path):
    root = tmp_path / "cache"
    (root / BUCKET / MANIFEST).parent.mkdir(parents=True)
    (root / BUCKET / ID2NAME_KEY).write_text(
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


def test_custom_manifest_reads_id2name_named_after_its_directory(cache):
    run_dir = cache / "cache" / BUCKET / "x" / "runA"
    run_dir.mkdir(parents=True)
    shutil.copy(cache / "cache" / BUCKET / MANIFEST, run_dir / "m.parquet")
    (run_dir / "coralnet_id2name_runA.json").write_text(
        json.dumps({"1": "Porites", "2": "Algae", "3": "Rubble"})
    )
    ds = CoralNetDataset(manifest_path="x/runA/m.parquet", source_bucket=BUCKET)
    assert sorted(ds.df_annotations["coralnet_name"].unique()) == ["Algae", "Porites", "Rubble"]


def test_manifest_sha256_is_the_hash_of_the_manifest_bytes_parsed(cache):
    ds = CoralNetDataset(manifest_path=MANIFEST, source_bucket=BUCKET)
    expected = hashlib.sha256((cache / "cache" / BUCKET / MANIFEST).read_bytes()).hexdigest()
    assert ds.manifest_sha256 == expected


def test_manifest_without_directory_is_rejected(cache):
    with pytest.raises(ValueError, match="run directory"):
        CoralNetDataset(manifest_path="m.parquet", source_bucket=BUCKET)


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
    pd.DataFrame({"image_id": ["11"]}).to_parquet(
        cache / "cache" / BUCKET / "bucket_blacklist.parquet"
    )
    monkeypatch.setattr(cnd, "_REPO_ROOT", repo)

    ds = _make(image_blacklist="configs/coralnet/blacklist.parquet")
    assert sorted(ds.df_images["image_id"]) == ["10", "11"]
    ds = _make(image_blacklist="bucket_blacklist.parquet")
    assert sorted(ds.df_images["image_id"]) == ["10", "20"]


def test_manifest_path_is_required(cache):
    with pytest.raises(TypeError, match="manifest_path"):
        CoralNetDataset(source_bucket=BUCKET)


def test_whitelist_and_blacklist_sources_together_raise(cache):
    with pytest.raises(ValueError, match="both whitelist and blacklist"):
        _make(whitelist_sources=[1], blacklist_sources=[2])


def test_blacklist_sources_drops_those_sources_and_their_images(cache):
    ds = _make(blacklist_sources=[1])

    assert sorted(ds.df_images["image_id"]) == ["20"]
    assert set(ds.df_annotations["source_id"]) == {2}


def test_annotation_overrides_for_images_missing_from_the_manifest_are_dropped(cache, caplog):
    path = cache / "overrides.parquet"
    pd.DataFrame(
        {
            "source_id": [1, 1],
            "image_id": ["10", "99"],
            "row": [1, 2],
            "col": [1, 2],
            "coralnet_id": [1, 1],
        }
    ).to_parquet(path)

    with caplog.at_level("WARNING", logger=cnd.logger.name):
        ds = _make(annotation_overrides=str(path))

    assert "99" not in set(ds.df_annotations["image_id"])
    assert sorted(ds.df_images["image_id"]) == ["10", "11", "20"]
    img10 = ds.df_annotations[ds.df_annotations["image_id"] == "10"]
    assert img10[["row", "col"]].to_numpy().tolist() == [[1, 1]]
    assert "1 images are not in the manifest" in caplog.text


def test_read_image_returns_the_rgb_array_for_the_manifest_key(cache):
    path = cache / "cache" / BUCKET / "k10"
    Image.new("L", (4, 3), 7).save(path, format="PNG")

    image = _make().read_image(image_id="10", source_id=1, image_key="k10")

    assert image.shape == (3, 4, 3)
    assert np.all(image == 7)


@pytest.mark.parametrize("image_key", [None, "", float("nan")])
def test_read_image_without_a_manifest_key_raises_key_error(cache, image_key):
    with pytest.raises(KeyError, match="No manifest entry for source_id=1 image_id=10"):
        _make().read_image(image_id="10", source_id=1, image_key=image_key)
