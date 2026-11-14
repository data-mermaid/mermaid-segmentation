"""Per-label counts and sample export for the dataset appendix. No cache or cluster access."""

from __future__ import annotations

import multiprocessing as mp
from concurrent.futures import ProcessPoolExecutor

import numpy as np
import pandas as pd
import pytest
from PIL import Image

from scripts.dataset_table_stats import count_dense_mask, count_sparse
from scripts.export_appendix_data import (
    COUNT_COLUMNS,
    dataset_rng,
    dense_mask_label_counts,
    export_coralnet,
    export_dense,
    export_sparse,
    finalize_counts,
    reduce_label_masks,
    sparse_label_counts,
    tally_frame,
)
from tests._appendix_stubs import DenseStub, SparseStub, build_bundle, points_frame, read_json


def _annotations() -> pd.DataFrame:
    return points_frame(
        [
            ("a", 0, 0, "Coral"),
            ("a", 1, 1, "Coral"),
            ("a", 2, 2, "Sand"),
            ("b", 0, 0, "Coral"),
            ("b", 1, 1, "unlabeled"),
            ("c", 0, 0, "unlabeled"),
        ]
    )


def test_sparse_label_counts_per_label_and_drop_ignore():
    counts = sparse_label_counts(_annotations()).set_index("source_label")
    assert counts.loc["Coral"].tolist() == [2, 3, 3]
    assert counts.loc["Sand"].tolist() == [1, 1, 1]
    assert "unlabeled" not in counts.index


def test_sparse_label_counts_sum_to_table_totals():
    annotations = _annotations()
    images = annotations.drop_duplicates("image_id")[["image_id"]]
    totals = count_sparse(annotations, images, {"coral", "sand"})
    counts = sparse_label_counts(annotations)
    assert counts["annotations"].sum() == totals.source_annotations
    assert counts["pixels"].sum() == totals.source_pixels


def test_dense_mask_label_counts_components_pixels_and_unknown_ids():
    mask = np.zeros((5, 5), dtype=np.uint8)
    mask[0, 0] = 1
    mask[4, 4] = 1
    mask[2, 1:4] = 2
    mask[0, 4] = 7
    mask[4, 0] = 3
    names = {1: "coral", 2: "sand", 3: "unlabeled"}
    tally = dense_mask_label_counts(mask, names)
    assert tally == {"coral": [1, 2, 2], "sand": [1, 1, 3], "id_7": [1, 1, 1]}

    totals = count_dense_mask(mask, names, set())
    assert sum(v[1] for v in tally.values()) == totals.source_components
    assert sum(v[2] for v in tally.values()) == totals.source_pixels


def test_reduce_label_masks_counts_images_per_label():
    masks = [np.zeros((3, 3), dtype=np.uint8) for _ in range(5)]
    for i, mask in enumerate(masks):
        mask[0, 0] = 1
        if i % 2:
            mask[2, 2] = 2
    tally, n = reduce_label_masks(masks, {1: "coral", 2: "sand"}, workers=1, pool=None)
    assert n == 5
    assert tally == {"coral": [5, 5, 5], "sand": [2, 2, 2]}


def test_reduce_label_masks_matches_in_spawn_pool():
    masks = [np.eye(4, dtype=np.uint8) * (i % 3) for i in range(9)]
    names = {1: "coral", 2: "sand"}
    serial, _ = reduce_label_masks(masks, names, workers=1, pool=None)
    with ProcessPoolExecutor(max_workers=2, mp_context=mp.get_context("spawn")) as pool:
        pooled, n = reduce_label_masks(masks, names, workers=2, pool=pool)
    assert n == 9
    assert pooled == serial


def test_finalize_counts_sums_duplicates_and_flags_mapped():
    first = tally_frame({"Coral": [1, 2, 3], "Junk": [1, 1, 1]})
    second = tally_frame({"Coral": [4, 5, 6]})
    for frame in (first, second):
        frame.insert(0, "split", "train")
        frame.insert(0, "dataset_key", "ucsd_mosaics")
    table = finalize_counts([first, second], {"ucsd_mosaics": {"coral"}})
    assert list(table.columns) == list(COUNT_COLUMNS)
    coral = table.set_index("source_label").loc["Coral"]
    assert (coral["images"], coral["annotations"], coral["pixels"]) == (5, 7, 9)
    assert bool(coral["mapped"]) is True
    assert bool(table.set_index("source_label").loc["Junk", "mapped"]) is False


def test_dataset_rng_is_deterministic_per_key():
    a = dataset_rng(0, "coralnet").permutation(50)
    assert (a == dataset_rng(0, "coralnet").permutation(50)).all()
    assert not (a == dataset_rng(0, "mermaid").permutation(50)).all()


def test_export_sparse_scales_points_with_image(tmp_path):
    ds = SparseStub(points_frame([("a", 10, 20, "Coral"), ("a", 30, 50, "Sand")]), size=(40, 60))
    records = export_sparse(
        "moorea_labeled_corals", [("train", ds)], tmp_path, n_images=3, seed=0, max_edge=30
    )
    assert [r["file"] for r in records] == ["00"]
    assert Image.open(tmp_path / "00_image.jpg").size == (30, 20)
    points = pd.read_csv(tmp_path / "00_points.csv")
    assert points["row"].tolist() == pytest.approx([5.0, 15.0])
    assert points["col"].tolist() == pytest.approx([10.0, 25.0])


class BenthosStub:
    """BenthosYuvalCoralsDataset-shaped stub: square and partial orthomosaic tiles."""

    def __init__(self, tiles: list[tuple[str, str, np.ndarray, np.ndarray]]):
        self._classes_global = {"unlabeled": 0, "coral": 1}
        self.df_images = pd.DataFrame(
            {
                "site": [site for site, _, _, _ in tiles],
                "image_id": [image_id for _, image_id, _, _ in tiles],
            }
        )
        self._tiles = {(site, image_id): (image, label) for site, image_id, image, label in tiles}

    def __len__(self) -> int:
        return len(self.df_images)

    def read_image(self, image_id: str, site: str, **_: object) -> np.ndarray:
        image, _ = self._tiles[(site, image_id)]
        return image

    def read_label(self, image_id: str, site: str, **_: object) -> np.ndarray:
        _, label = self._tiles[(site, image_id)]
        return label


def test_export_dense_benthos_keeps_only_square_tiles(tmp_path):
    labeled = np.zeros((8, 8), dtype=np.uint8)
    labeled[:4] = 1
    edge_label = np.zeros((8, 6), dtype=np.uint8)
    edge_label[:4] = 1
    mismatch_label = np.zeros((8, 5), dtype=np.uint8)
    mismatch_label[:4] = 1
    ds = BenthosStub(
        [
            ("RS24", "edge", np.zeros((8, 6, 3), dtype=np.uint8), edge_label),
            (
                "RS24",
                "corner",
                np.zeros((6, 4, 3), dtype=np.uint8),
                np.full((6, 4), 1, dtype=np.uint8),
            ),
            ("RS24", "mismatch", np.zeros((8, 8, 3), dtype=np.uint8), mismatch_label),
            (
                "RS24",
                "empty",
                np.zeros((8, 8, 3), dtype=np.uint8),
                np.zeros((8, 8), dtype=np.uint8),
            ),
            ("RS24", "square", np.full((8, 8, 3), 10, dtype=np.uint8), labeled),
        ]
    )
    records = export_dense("benthos_yuval", [("val", ds)], tmp_path, n_images=5, seed=0, max_edge=0)
    assert [record["image"] for record in records] == ["RS24/square"]
    assert Image.open(tmp_path / "00_image.jpg").size == (8, 8)


def test_export_dense_keeps_native_ids_and_skips_unlabeled(tmp_path):
    empty = np.zeros((8, 8), dtype=np.uint8)
    ignore_only = np.full((8, 8), 3, dtype=np.uint8)
    labeled = empty.copy()
    labeled[:4] = 2
    ds = DenseStub([empty, ignore_only, labeled], ["coral", "sand", "unlabeled / unidentified"])
    records = export_dense(
        "ucsd_mosaics", [("train", ds)], tmp_path, n_images=5, seed=0, max_edge=0
    )
    assert len(records) == 1 and records[0]["image"] == "tile_2.png"
    saved = np.asarray(Image.open(tmp_path / "00_label.png"))
    assert set(np.unique(saved).tolist()) == {0, 2}
    assert read_json(tmp_path / "classes.json")["2"] == "sand"


def test_export_coralnet_picks_sources_with_enough_images(tmp_path):
    build_bundle(tmp_path, n_images=2)
    sources = read_json(tmp_path / "samples" / "coralnet" / "sources.json")
    assert len(sources) == 2
    for entry in sources:
        src_dir = tmp_path / "samples" / "coralnet" / f"s{entry['source_id']}"
        assert len(entry["image_ids"]) == 2
        assert sorted(p.name for p in src_dir.glob("*_points.csv")) == [
            "00_points.csv",
            "01_points.csv",
        ]
        assert all(iid.startswith(f"{entry['source_id']}_") for iid in entry["image_ids"])


def test_export_coralnet_requires_an_eligible_source(tmp_path):
    frame = points_frame([("1_0", 0, 0, "cca")])
    frame["source_id"] = 1
    ds = SparseStub(frame, image_key={"1_0": "k.jpg"}, source_id={"1_0": 1})
    with pytest.raises(RuntimeError, match="at least 3 images"):
        export_coralnet([("train", ds)], tmp_path, n_sources=1, n_images=3, seed=0, max_edge=0)
