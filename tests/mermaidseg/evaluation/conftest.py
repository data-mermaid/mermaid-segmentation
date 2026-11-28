"""A tiny synced cache holding the versions the evaluation suite reads, and its data config."""

from __future__ import annotations

import csv
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from mermaidseg.data.config import DataSection
from segdata_spec.manifest import Pin
from tests.mermaidseg.data.conftest import ImageSpec, write_version

# CoralNet group -> split in the split file; every split has a group the others lack.
CORALNET_SPLITS = {"s1": "train", "s2": "train", "s3": "val", "s4": "test", "s5": "excluded"}
CORALNET_IMAGES = {
    "s1": ["c1a", "c1b", "c1c"],
    "s2": ["c2a"],
    "s3": ["c3a", "c3b"],
    "s4": ["c4a"],
    "s5": ["c5a"],
}
CORALNET_LABELS = [
    {"label_id": 1, "name": "Acro", "mermaid_benthic_attribute": "Acropora"},
    {"label_id": 2, "name": "Sand", "mermaid_benthic_attribute": "Sand"},
]
PACIFIC_LABELS = [
    {"label_id": 1, "name": "Porites", "mermaid_benthic_attribute": "Porites"},
    {"label_id": 2, "name": "Sand", "mermaid_benthic_attribute": "Sand"},
]


@dataclass(frozen=True)
class EvalCache:
    """The cache directory, the ``data`` section that pins it, and the CoralNet split file."""

    cache_dir: Path
    data: dict[str, Any]
    split_file: Path

    @property
    def section(self) -> DataSection:
        return DataSection.model_validate(self.data)


def _pin_entry(pin: Pin) -> dict[str, Any]:
    return {"version": pin.version, "manifest_sha256": pin.manifest_sha256}


def _write_split_file(path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["group", "split"])
        writer.writerows(CORALNET_SPLITS.items())
    return path


def write_coralnet(tmp_path: Path) -> Pin:
    """Write the CoralNet points version: per image an Acro point at (0, 0), Sand at (1, 1)."""
    images = [ImageSpec(i, group=g) for g, ids in CORALNET_IMAGES.items() for i in ids]
    points = [p for s in images for p in ((s.image_id, 0, 0, 1), (s.image_id, 1, 1, 2))]
    _, _, pin = write_version(
        tmp_path,
        kind="points",
        images=images,
        labels=CORALNET_LABELS,
        points=points,
        name="coralnet",
    )
    return pin


def write_eval_cache(tmp_path: Path) -> EvalCache:
    """Write CoralNet, Pacific (two annotators) and a dense ``coralscapes_v2`` into one cache."""
    coralnet = write_coralnet(tmp_path)
    _, _, pacific = write_version(
        tmp_path,
        kind="points",
        images=[ImageSpec("p1", group="heron_reef"), ImageSpec("p2", group="line_islands")],
        labels=PACIFIC_LABELS,
        points=[
            ("p1", 2, 2, 1, "host"),
            ("p1", 2, 2, 2, "archived"),
            ("p2", 3, 3, 2, "archived"),
        ],
        name="pacific_labeled_corals",
    )
    mask = np.zeros((8, 8), np.uint8)
    mask[:4] = 1
    cache_dir, _, coralscapes = write_version(
        tmp_path,
        kind="dense",
        images=[ImageSpec("cs1", group="test", mask=mask)],
        labels={1: "acanthaster planci"},
        name="coralscapes_v2",
    )
    split_file = _write_split_file(tmp_path / "splits" / "coralnet.csv")
    data = {
        "taxonomy": {"version": "20261121-ab12cd3", "manifest_sha256": "c" * 64},
        "datasets": {
            "coralnet": {
                **_pin_entry(coralnet),
                "split_file": str(split_file),
                "train": {"use": ["train", "val"]},
                "test": {"use": ["test"]},
            },
            "pacific_labeled_corals": {
                **_pin_entry(pacific),
                "val": {"groups": "all", "annotators": ["host", "archived"]},
            },
            "coralscapes_v2": {**_pin_entry(coralscapes), "test": {"groups": ["test"]}},
        },
    }
    return EvalCache(cache_dir=cache_dir, data=data, split_file=split_file)
