"""In-memory dataset stubs and a synthetic bundle for the appendix figure scripts."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from PIL import Image

from scripts.dataset_table_stats import CSV_COLUMNS, DATASET_SPECS
from scripts.export_appendix_data import export_coralnet, export_dense, export_sparse


class SparseStub:
    """BaseCoralDataset-shaped stub: df_images, df_annotations, _ann_positions, read_image."""

    def __init__(self, points: pd.DataFrame, size: tuple[int, int] = (40, 60), **image_cols):
        self.df_annotations = points.reset_index(drop=True)
        images = self.df_annotations.drop_duplicates("image_id")[["image_id"]].copy()
        for column, values in image_cols.items():
            images[column] = images["image_id"].map(values)
        self.df_images = images.reset_index(drop=True)
        self._ann_positions = self.df_annotations.groupby("image_id", sort=False).indices
        self.size = size

    def __len__(self) -> int:
        return len(self.df_images)

    def read_image(self, image_id: str, **_: Any) -> np.ndarray:
        h, w = self.size
        seed = sum(map(ord, str(image_id)))
        return np.random.default_rng(seed).integers(0, 255, (h, w, 3), dtype=np.uint8)


class DenseStub:
    """UCSDMosaicsDataset-shaped stub: ``dataset`` rows plus ``class_table``."""

    def __init__(self, labels: list[np.ndarray], class_names: list[str]):
        self.class_table = [{"id": i, "name": n} for i, n in enumerate(class_names, start=1)]
        self.dataset = [
            {
                "image": Image.fromarray(np.full((*lab.shape, 3), 40 * i % 255, dtype=np.uint8)),
                "label": Image.fromarray(lab.astype(np.uint8)),
                "filename": f"tile_{i}.png",
            }
            for i, lab in enumerate(labels)
        ]

    def __len__(self) -> int:
        return len(self.dataset)


def points_frame(rows: list[tuple[str, float, float, str]]) -> pd.DataFrame:
    return pd.DataFrame(rows, columns=["image_id", "row", "col", "source_label_name"])


def build_bundle(root: Path, n_images: int = 2) -> Path:
    """Write a bundle shaped like ``export_appendix_data.py`` output into ``root``."""
    samples = root / "samples"
    sparse = SparseStub(
        points_frame(
            [
                ("a", 5, 5, "Coral"),
                ("a", 10, 30, "Sand"),
                ("b", 20, 20, "Turf <algae>"),
                ("b", 30, 50, "Coral"),
            ]
        )
    )
    export_sparse(
        "moorea_labeled_corals",
        [("train", sparse)],
        samples / "moorea_labeled_corals",
        n_images=n_images,
        seed=0,
        max_edge=32,
    )

    labels = [np.zeros((16, 24), dtype=np.uint8) for _ in range(3)]
    labels[0][:8] = 1
    labels[1][:, :12] = 2
    labels[2][4:10, 4:10] = 3
    dense = DenseStub(labels, ["coral", "sand", "unlabeled / unidentified"])
    export_dense(
        "ucsd_mosaics",
        [("train", dense)],
        samples / "ucsd_mosaics",
        n_images=n_images,
        seed=0,
        max_edge=0,
    )

    coralnet_points = []
    for sid in (1, 2, 3):
        for i in range(3):
            image_id = f"{sid}_{i}"
            coralnet_points += [
                (image_id, 4, 4, "cca"),
                (image_id, 12, 30, f"label_{sid}"),
                (image_id, 30, 10, "turf"),
            ]
    frame = points_frame(coralnet_points)
    frame["source_id"] = frame["image_id"].str.split("_").str[0].astype(int)
    keys = {iid: f"key/{iid}.jpg" for iid in frame["image_id"]}
    sources = {iid: int(iid.split("_")[0]) for iid in frame["image_id"]}
    coralnet = SparseStub(frame, image_key=keys, source_id=sources)
    export_coralnet(
        [("train", coralnet)],
        samples / "coralnet",
        n_sources=2,
        n_images=n_images,
        seed=0,
        max_edge=0,
    )

    rows = []
    for k, spec in enumerate(DATASET_SPECS):
        counts = dict.fromkeys(CSV_COLUMNS[5:], 0)
        counts.update(
            n_classes=3,
            images_train=10 * (k + 1),
            images_test=k,
            annotations_train=100 * (k + 1),
            annotations_val=50 if spec.key == "coralnet" else 0,
            pixels_train=1000 * (k + 1),
            pixels_val=70 if spec.key == "coralnet" else 0,
            images_val=5 if spec.key == "coralnet" else 0,
            annotations_test=k,
            pixels_test=k,
        )
        rows.append(
            {
                "dataset": spec.display,
                "dataset_key": spec.key,
                "annotation_type": spec.annotation_type,
                "image_type": spec.image_type,
                "level": "source",
                **counts,
            }
        )
    rows.append({**rows[0], "dataset": "Total", "dataset_key": "", "level": "source"})
    pd.DataFrame(rows, columns=list(CSV_COLUMNS)).to_csv(root / "dataset_table.csv", index=False)

    pd.DataFrame(
        {
            "dataset_key": ["coralnet", "coralnet", "mermaid", "ucsd_mosaics", "coralnet"],
            "split": ["train", "val", "train", "train", "train"],
            "source_label": ["cca", "cca", "Sand", "coral", "unmapped thing"],
            "mapped": [True, True, True, True, False],
            "images": [5, 1, 3, 2, 9],
            "annotations": [50, 4, 30, 7, 90],
            "pixels": [50, 4, 30, 700, 90],
        }
    ).to_csv(root / "label_counts.csv", index=False)
    return root


def read_json(path: Path) -> Any:
    return json.loads(Path(path).read_text())
