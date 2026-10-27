"""Static sparse copies of the dense training datasets, loaded back through the real loaders.

Source datasets are tiny local HF-style directories, so nothing is downloaded.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from datasets import Dataset, Features, Image, Value
from datasets import config as hf_config
from PIL import Image as PILImage

from mermaidseg.datasets.coralscapes_v2.coralscapes_v2_dataset import CoralscapesV2Dataset
from mermaidseg.datasets.ucsd_mosaics.ucsd_mosaics_dataset import UCSDMosaicsDataset
from scripts.build_sparse_dense_datasets import _parse_args, build, point_rng, sparsify_mask

SIZE = 24
N_POINTS = 5
PADDING = 1


@pytest.fixture(autouse=True)
def _isolated_hf_cache(tmp_path, monkeypatch):
    monkeypatch.setattr(hf_config, "HF_DATASETS_CACHE", tmp_path / "hf-datasets-cache")
    monkeypatch.setenv("HF_HUB_OFFLINE", "1")


def _dense_mask(seed: int, n_labels: int) -> np.ndarray:
    """Blocky mask with a 0 border and labels 1..n_labels inside."""
    rng = np.random.default_rng(seed)
    blocks = rng.integers(1, n_labels + 1, size=(SIZE // 4, SIZE // 4), dtype=np.uint8)
    mask = np.kron(blocks, np.ones((4, 4), dtype=np.uint8))
    mask[:2] = 0
    mask[:, :2] = 0
    return mask


def _write_source(root: Path, splits: dict[str, int], n_labels: int, extra: dict) -> Path:
    """HF-style local dataset: data/<split>-*.parquet + README configs, like the mirrors."""
    features = Features({"image": Image(), "label": Image(), **{k: Value("string") for k in extra}})
    data_files = []
    for s, (split, n) in enumerate(splits.items()):
        rng = np.random.default_rng(s)
        rows = {
            "image": [
                PILImage.fromarray(rng.integers(0, 255, (SIZE, SIZE, 3), dtype=np.uint8))
                for _ in range(n)
            ],
            "label": [PILImage.fromarray(_dense_mask(100 * s + i, n_labels)) for i in range(n)],
            **{k: [v(split, i) for i in range(n)] for k, v in extra.items()},
        }
        (root / "data").mkdir(parents=True, exist_ok=True)
        Dataset.from_dict(rows, features=features).to_parquet(
            root / "data" / f"{split}-00000-of-00001.parquet"
        )
        data_files.append(f"  - split: {split}\n    path: data/{split}-*")
    (root / "README.md").write_text(
        "---\nconfigs:\n- config_name: default\n  data_files:\n" + "\n".join(data_files) + "\n---\n"
    )
    return root


def _build(dataset: str, source: Path, out: Path, *extra: str) -> Path:
    return build(
        _parse_args(
            [
                "--dataset",
                dataset,
                "--output-dir",
                str(out),
                "--hf-repo-id",
                str(source),
                "--n-points",
                str(N_POINTS),
                "--padding",
                str(PADDING),
                *extra,
            ]
        )
    )


def _check_sparse(dense: np.ndarray, sparse: np.ndarray, points: pd.DataFrame) -> None:
    assert len(points) == min(N_POINTS, int((dense > 0).sum()))
    near_point = np.zeros(dense.shape, dtype=bool)
    for row, col, label in points[["row", "col", "label"]].to_numpy():
        assert dense[row, col] == label != 0
        assert sparse[row, col] != 0
        near_point[
            max(row - PADDING, 0) : row + PADDING + 1, max(col - PADDING, 0) : col + PADDING + 1
        ] = True
    assert not sparse[~near_point].any()
    assert set(np.unique(sparse[near_point])) <= set(points["label"]) | {0}


def test_sparsify_mask_points_footprint_and_clipping():
    mask = np.zeros((10, 12), dtype=np.uint8)
    mask[0, 0] = 7
    sparse, points = sparsify_mask(mask, n_points=25, padding=3, rng=point_rng(0, "train", 0))
    np.testing.assert_array_equal(points, [[0, 0, 7]])
    expected = np.zeros_like(mask)
    expected[:4, :4] = 7
    np.testing.assert_array_equal(sparse, expected)
    assert sparse.dtype == mask.dtype


def test_sparsify_mask_samples_distinct_labelled_pixels():
    mask = _dense_mask(0, n_labels=6)
    sparse, points = sparsify_mask(mask, n_points=25, padding=0, rng=point_rng(0, "train", 3))
    assert points.shape == (25, 3)
    assert len({(r, c) for r, c, _ in points}) == 25
    assert (mask[points[:, 0], points[:, 1]] == points[:, 2]).all()
    assert (points[:, 2] > 0).all()
    assert int((sparse > 0).sum()) == 25
    np.testing.assert_array_equal(sparse[points[:, 0], points[:, 1]], points[:, 2])


def test_sparsify_mask_without_labels_is_empty():
    mask = np.zeros((8, 8), dtype=np.uint8)
    sparse, points = sparsify_mask(mask, n_points=25, padding=3, rng=point_rng(0, "train", 0))
    assert points.shape == (0, 3)
    assert not sparse.any()


def test_points_depend_only_on_seed_split_and_index():
    mask = _dense_mask(1, n_labels=4)

    def pick(seed: int, split: str, index: int) -> np.ndarray:
        return sparsify_mask(mask, N_POINTS, PADDING, point_rng(seed, split, index))[1]

    np.testing.assert_array_equal(pick(0, "train", 5), pick(0, "train", 5))
    assert not np.array_equal(pick(0, "train", 5), pick(1, "train", 5))
    assert not np.array_equal(pick(0, "train", 5), pick(0, "fr3", 5))
    assert not np.array_equal(pick(0, "train", 5), pick(0, "train", 6))


def test_rejects_existing_output_without_overwrite(tmp_path):
    out = tmp_path / "out"
    out.mkdir()
    with pytest.raises(SystemExit, match="--overwrite"):
        build(_parse_args(["--dataset", "coralscapes_v2", "--output-dir", str(out)]))


def test_coralscapes_copy_has_only_train_and_loads(tmp_path):
    source = _write_source(
        tmp_path / "src", {"train": 4, "validation": 2, "test": 2}, n_labels=95, extra={}
    )
    out = _build("coralscapes_v2", source, tmp_path / "coralscapes_v2_sparse")

    assert not (tmp_path / ".coralscapes_v2_sparse.partial").exists()
    run = json.loads((out / "run.json").read_text())
    assert list(run["splits"]) == ["train"]
    assert run["splits"]["train"]["n_images"] == 4
    assert run["splits"]["train"]["n_points"] == 4 * N_POINTS
    assert run["source_revision_sha"] is None

    dense = CoralscapesV2Dataset(split="train", hf_repo_id=str(source))
    sparse = CoralscapesV2Dataset(split="train", hf_repo_id=str(out))
    with pytest.raises(KeyError):
        CoralscapesV2Dataset(split="validation", hf_repo_id=str(out))
    points = pd.read_parquet(out / "points.parquet")
    assert len(sparse) == len(dense) == 4
    for i in range(len(sparse)):
        dense_image, dense_mask = dense[i]
        sparse_image, sparse_mask = sparse[i]
        np.testing.assert_array_equal(sparse_image, dense_image)
        native_dense = np.asarray(dense.dataset[i]["label"])
        native_sparse = np.asarray(sparse.dataset[i]["label"])
        _check_sparse(native_dense, native_sparse, points[points["index"] == i])
        assert int((sparse_mask > 0).sum()) <= N_POINTS * (2 * PADDING + 1) ** 2
        assert set(np.unique(sparse_mask)) <= set(np.unique(dense_mask))


def test_ucsd_copy_keeps_every_site_and_classes(tmp_path):
    classes = [{"id": i, "name": f"class {i}"} for i in range(1, 35)]
    source = _write_source(
        tmp_path / "src",
        {"fr3": 3, "palwave13": 2},
        n_labels=34,
        extra={
            "filename": lambda split, i: f"{split}_{i}.png",
            "site": lambda split, i: split.upper(),
            "original_split": lambda split, i: "train" if i % 2 == 0 else "test",
        },
    )
    (source / "classes.json").write_text(json.dumps(classes))
    out = _build("ucsd_mosaics", source, tmp_path / "ucsd_sparse")

    assert json.loads((out / "classes.json").read_text()) == classes
    run = json.loads((out / "run.json").read_text())
    assert list(run["splits"]) == ["fr3", "palwave13"]

    dense = UCSDMosaicsDataset(hf_repo_id=str(source))
    sparse = UCSDMosaicsDataset(hf_repo_id=str(out))
    assert sparse.sites == dense.sites == ["fr3", "palwave13"]
    assert sparse.source_id2name == dense.source_id2name
    assert len(sparse) == len(dense) == 5
    assert sparse.dataset["filename"] == dense.dataset["filename"]
    assert sparse.dataset["original_split"] == dense.dataset["original_split"]

    points = pd.read_parquet(out / "points.parquet")
    offsets = {"fr3": 0, "palwave13": 3}
    for split, offset in offsets.items():
        split_points = points[points["split"] == split]
        for index in split_points["index"].unique():
            i = offset + int(index)
            native_dense = np.asarray(dense.dataset[i]["label"])
            native_sparse = np.asarray(sparse.dataset[i]["label"])
            _check_sparse(native_dense, native_sparse, split_points[split_points["index"] == index])
            np.testing.assert_array_equal(sparse[i][0], dense[i][0])


def test_rebuild_is_identical(tmp_path):
    source = _write_source(tmp_path / "src", {"train": 3}, n_labels=95, extra={})
    out = _build("coralscapes_v2", source, tmp_path / "out")
    first = pd.read_parquet(out / "points.parquet")
    first_masks = [
        np.asarray(r["label"])
        for r in CoralscapesV2Dataset(split="train", hf_repo_id=str(out)).dataset
    ]

    _build("coralscapes_v2", source, out, "--overwrite")
    pd.testing.assert_frame_equal(pd.read_parquet(out / "points.parquet"), first)
    again = CoralscapesV2Dataset(split="train", hf_repo_id=str(out)).dataset
    for mask, row in zip(first_masks, again, strict=True):
        np.testing.assert_array_equal(np.asarray(row["label"]), mask)
