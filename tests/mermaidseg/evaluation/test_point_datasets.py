"""run_point_evaluation: per-unit blocks, macro means, and per-unit hierarchical metrics."""

from __future__ import annotations

import json
import math
from collections.abc import Callable, Collection
from pathlib import Path
from types import SimpleNamespace
from typing import Any, cast

import numpy as np
import pandas as pd
import pyarrow as pa
import pytest
import torch
import yaml
from numpy.typing import NDArray
from PIL import Image

from mermaidseg.data.point_dataset import PointDataset
from mermaidseg.dataset_reconciliation.concept_schema import ConceptSchema
from mermaidseg.evaluation.gt_mapping import ClassLookup, ConceptLookup, build_concept_lookup
from mermaidseg.evaluation.hierarchical import HIER_HEADLINE_KEYS
from mermaidseg.evaluation.metrics import REPORTED_BINARY_CONCEPTS
from mermaidseg.evaluation.point_datasets import (
    PointEvalDataset,
    run_point_evaluation,
    unit_scalar_keys,
    unit_scalars,
)
from mermaidseg.inference.predictor import CBMPredictor

_REPO = Path(__file__).resolve().parents[3]
_CSV = _REPO / "configs" / "class_to_concepts.csv"
LABELS = [
    "sand",
    "montastraea",
    "porites",
    "macroalgae",
    "acropora cervicornis - bleached",
    "trash",
    "non-acropora encrusting",
    "dead pocillopora",
    "chondrilla spp.",
    "millepora",
    "palythoa",
]
DEAD = {"dead pocillopora"}
CORAL = {
    "montastraea",
    "porites",
    "acropora cervicornis - bleached",
    "non-acropora encrusting",
    "millepora",
}
SIZE = 8
NUM_CLASSES = 5


def _concept_names() -> list[str]:
    data = yaml.safe_load((_REPO / "configs" / "data_config.yaml").read_text())["data"]
    schema = ConceptSchema.from_csv(_CSV, sources=set(data["datasets"]))
    return [name for _, name in sorted(schema.channel_id2name().items())]


def fake_reader(
    root: Path,
    images: list[dict[str, Any]],
    annotations: list[dict[str, Any]],
    pixels: Callable[[str], NDArray[np.uint8]],
    fail: Collection[str] = (),
) -> PointDataset:
    """The surface of a PointDataset that PointEvalDataset reads; ``fail`` images have no file."""
    names = sorted({a["source_label_name"] for a in annotations})
    label_ids = {name: i + 1 for i, name in enumerate(names)}
    root.mkdir(parents=True, exist_ok=True)
    for image in images:
        if image["image_id"] not in fail:
            Image.fromarray(pixels(image["image_id"])).save(root / f"{image['image_id']}.png")
    points = pd.DataFrame(
        [
            {
                "image_id": a["image_id"],
                "row": a["row"],
                "col": a["col"],
                "label_id": label_ids[a["source_label_name"]],
            }
            for a in annotations
        ],
        columns=["image_id", "row", "col", "label_id"],
    )
    version = SimpleNamespace(
        labels=pa.table({"label_id": list(label_ids.values()), "name": list(label_ids)}),
        image_path=lambda row: root / f"{row['image_id']}.png",
    )
    reader = SimpleNamespace(images=pd.DataFrame(images), points=points, version=version)
    return cast(PointDataset, reader)


def fake_annotations(units: list[str], seed: int = 0) -> list[dict[str, Any]]:
    """Twelve random 8x8 points on each of three images of sources ``10`` and ``20``."""
    rng = np.random.default_rng(seed)
    anns: list[dict[str, Any]] = []
    for unit in ["10", "20"]:
        for k in range(3):
            for _ in range(12):
                anns.append(
                    {
                        "image_id": f"{unit}_{k}",
                        "row": int(rng.integers(0, SIZE)),
                        "col": int(rng.integers(0, SIZE)),
                        "source_label_name": LABELS[int(rng.integers(0, len(LABELS)))],
                    }
                )
    return [a for a in anns if a["image_id"].split("_")[0] in units]


def fake_coralnet(root: Path, units: list[str]) -> PointDataset:
    """Sources ``units`` of two CoralNet-like sources; each image is one uniform value."""
    images = [{"source_id": u, "image_id": f"{u}_{k}"} for u in units for k in range(3)]

    def pixels(image_id: str) -> NDArray[np.uint8]:
        unit, k = image_id.split("_")
        return np.full((SIZE, SIZE, 3), 10 * int(unit) + int(k), dtype=np.uint8)

    return fake_reader(root, images, fake_annotations(units), pixels)


class StubPredictor:
    """Predictions are a deterministic function of the image, independent of batching."""

    input_size = (SIZE, SIZE)
    mean = (0.5,) * 3
    std = (0.25,) * 3
    device = torch.device("cpu")
    sample_at_points = staticmethod(CBMPredictor.sample_at_points)

    def __init__(self, num_concepts: int):
        self.num_concepts = num_concepts
        self.seen_images: list[torch.Tensor] = []

    def forward_probs(self, images: torch.Tensor):
        self.seen_images.append(images)
        class_maps, concept_maps = [], []
        for image in images:
            seed = int(round(float(image.sum()) * 1000)) % (2**31)
            g = torch.Generator().manual_seed(seed)
            class_maps.append(torch.rand(NUM_CLASSES, SIZE, SIZE, generator=g))
            concept_maps.append(torch.rand(self.num_concepts, SIZE, SIZE, generator=g))
        return torch.stack(class_maps), torch.stack(concept_maps)


def _run(tmp_path: Path, units: list[str], batch_size: int = 2) -> dict:
    names = _concept_names()
    id2label = {i: f"c{i}" for i in range(NUM_CLASSES)}
    class_lookup = ClassLookup(
        {label: 1 + i % (NUM_CLASSES - 1) for i, label in enumerate(LABELS)}, id2label
    )
    concept_lookup = build_concept_lookup("coralnet", LABELS, names, _CSV)
    return run_point_evaluation(
        eval_name="coralnet_val",
        reader=fake_coralnet(tmp_path / "images", units),
        unit_col="source_id",
        predictor=cast(CBMPredictor, StubPredictor(len(names))),
        class_lookup=class_lookup,
        concept_lookup=concept_lookup,
        concept_names=names,
        id2label=id2label,
        output_dir=tmp_path,
        binary_concepts=REPORTED_BINARY_CONCEPTS,
        taxonomy_csv=_CSV,
        batch_size=batch_size,
        num_workers=0,
    )


def _same(a: float, b: float) -> bool:
    return (math.isnan(a) and math.isnan(b)) or a == pytest.approx(b, abs=1e-12)


@pytest.fixture(scope="module")
def both(tmp_path_factory):
    return _run(tmp_path_factory.mktemp("both"), ["10", "20"])


def test_macro_is_the_mean_of_unit_values(both):
    keys = unit_scalar_keys(
        list(both["pooled"]["taxonomic"]), REPORTED_BINARY_CONCEPTS, with_hier=True
    )
    assert set(both["macro"]) == set(keys)
    per_unit = [unit_scalars(d, REPORTED_BINARY_CONCEPTS) for d in both["units"].values()]
    for key in keys:
        values = [u[key] for u in per_unit if not math.isnan(u[key])]
        expected = float(np.mean(values)) if values else float("nan")
        assert _same(both["macro"][key], expected), key


def test_pooled_counts_are_the_sum_of_units(both):
    for rank, rd in both["pooled"]["taxonomic"].items():
        assert rd["n_all"] == sum(u["taxonomic"][rank]["n_all"] for u in both["units"].values())
    for concept in REPORTED_BINARY_CONCEPTS:
        for key in ("tp", "fp", "fn", "tn"):
            assert both["pooled"][concept][key] == sum(
                u[concept][key] for u in both["units"].values()
            )
    assert "per_taxon" in both["pooled"]["taxonomic"]["genus"]


@pytest.mark.parametrize("unit", ["10"])
def test_unit_metrics_equal_a_run_on_that_unit_alone(tmp_path, both, unit):
    alone = _run(tmp_path, [unit], batch_size=3)
    assert set(alone["units"]) == {unit}
    unit_d = both["units"][unit]
    for key, value in unit_scalars(unit_d, REPORTED_BINARY_CONCEPTS).items():
        assert _same(value, unit_scalars(alone["pooled"], REPORTED_BINARY_CONCEPTS)[key]), key
    for key in HIER_HEADLINE_KEYS:
        a = unit_d["hierarchical"].get(key, float("nan"))
        b = alone["pooled"]["hierarchical"].get(key, float("nan"))
        assert _same(a, b), key
    assert unit_d["hierarchical"] == pytest.approx(alone["pooled"]["hierarchical"], nan_ok=True)


def test_living_and_coral_counts_follow_the_live_column(both):
    table = pd.read_csv(_CSV, dtype=str).fillna("not_given")
    table = table[table["source_dataset_source"] == "coralnet"]
    table = table.assign(label=table["source_label_class_name"].str.lower()).drop_duplicates(
        "label"
    )
    table = table.set_index("label").loc[LABELS]
    labels = pd.DataFrame(fake_annotations(["10", "20"]))["source_label_name"]
    assert labels.isin(DEAD).any() and labels.isin(CORAL).any()
    for rank, rd in both["pooled"]["taxonomic"].items():
        values = table[rank].str.lower()
        taxon = set(values.index[~values.isin(["none", "not_given"])])
        assert rd["n_living"] == int((labels.isin(taxon) & ~labels.isin(DEAD)).sum()), rank
        assert rd["n_coral"] == int((labels.isin(taxon) & labels.isin(CORAL)).sum()), rank
        assert rd["n_all"] - rd["n_none"] - rd["n_living"] == int(
            (labels.isin(taxon) & labels.isin(DEAD)).sum()
        ), rank


def test_images_are_normalised_with_the_predictors_mean_and_std(tmp_path):
    names = _concept_names()
    predictor = StubPredictor(len(names))
    id2label = {i: f"c{i}" for i in range(NUM_CLASSES)}
    run_point_evaluation(
        eval_name="coralnet_val",
        reader=fake_coralnet(tmp_path / "images", ["10"]),
        unit_col="source_id",
        predictor=cast(CBMPredictor, predictor),
        class_lookup=ClassLookup(dict.fromkeys(LABELS, 1), id2label),
        concept_lookup=build_concept_lookup("coralnet", LABELS, names, _CSV),
        concept_names=names,
        id2label=id2label,
        output_dir=tmp_path,
        binary_concepts=REPORTED_BINARY_CONCEPTS,
        taxonomy_csv=_CSV,
        batch_size=1,
        num_workers=0,
    )
    batch = torch.cat(predictor.seen_images)
    raw = torch.tensor([100, 101, 102], dtype=torch.float32) / 255
    expected = (raw - 0.5) / 0.25
    for image, value in zip(batch, expected, strict=True):
        torch.testing.assert_close(image, torch.full_like(image, float(value)))


# -- tiny hand-derived runs ------------------------------------------------------------
TINY_CONCEPTS = ["kingdom__animalia", "kingdom__none", "bleached"]
TINY_ID2LABEL = {0: "ignore", 1: "coral", 2: "sand"}
TINY_MEAN = (0.5, 0.4, 0.3)
TINY_STD = (0.25, 0.5, 0.1)


def tiny_reader(root, images, annotations, *, fail=(), gray=False):
    """4x8 images of value 100; ``fail`` images are unreadable, ``gray`` saves one channel."""
    shape = (4, 8) if gray else (4, 8, 3)
    return fake_reader(
        root, images, annotations, lambda _: np.full(shape, 100, dtype=np.uint8), fail=fail
    )


class RowBandPredictor:
    """Class 1 and ``bleached`` own the top half of the input rows, class 2 the bottom half."""

    input_size = (4, 4)
    mean = TINY_MEAN
    std = TINY_STD
    device = torch.device("cpu")
    sample_at_points = staticmethod(CBMPredictor.sample_at_points)

    def forward_probs(self, images):
        n = images.shape[0]
        class_maps = torch.full((n, 3, 4, 4), 0.1)
        class_maps[:, 1, :2, :] = 0.9
        class_maps[:, 2, 2:, :] = 0.9
        concept_maps = torch.full((n, 3, 4, 4), 0.5)
        concept_maps[:, 2, :2, :] = 0.9
        concept_maps[:, 2, 2:, :] = 0.1
        return class_maps, concept_maps


def _tiny_lookups():
    class_lookup = ClassLookup({"coral": 1, "sand": 2}, TINY_ID2LABEL)
    rows = {
        "coral": np.array([2, 1, 2], dtype=np.int8),
        "sand": np.array([1, 2, 1], dtype=np.int8),
    }
    return class_lookup, ConceptLookup(rows, len(TINY_CONCEPTS))


def _tiny_images(*ids, unit="u"):
    return [{"source_id": unit, "image_id": i} for i in ids]


def _tiny_annotations(image_id, points):
    return [
        {"image_id": image_id, "row": r, "col": c, "source_label_name": label}
        for r, c, label in points
    ]


def _tiny_run(tmp_path, reader, *, concept_names=TINY_CONCEPTS, **kwargs):
    class_lookup, concept_lookup = _tiny_lookups()
    return run_point_evaluation(
        eval_name="tiny",
        reader=reader,
        unit_col="source_id",
        predictor=cast(CBMPredictor, RowBandPredictor()),
        class_lookup=class_lookup,
        concept_lookup=concept_lookup,
        concept_names=concept_names,
        id2label=TINY_ID2LABEL,
        output_dir=tmp_path,
        batch_size=kwargs.pop("batch_size", 2),
        num_workers=0,
        **kwargs,
    )


def _tiny_dataset_item(reader, idx=0):
    class_lookup, concept_lookup = _tiny_lookups()
    dataset = PointEvalDataset(
        reader,
        "source_id",
        (4, 4),
        TINY_MEAN,
        TINY_STD,
        class_lookup,
        concept_lookup,
    )
    return dataset[idx]


def test_points_are_sampled_by_row_and_column_on_a_non_square_image(tmp_path):
    # Source image is 4 rows x 8 columns; predictions vary with the row only.
    # (row, col): (0, 7) top -> 1, (3, 0) bottom -> 2, (1, 3) top -> 1, (2, 5) bottom -> 2.
    points = [(0, 7, "coral"), (3, 0, "sand"), (1, 3, "sand"), (2, 5, "sand")]
    reader = tiny_reader(tmp_path / "images", _tiny_images("a"), _tiny_annotations("a", points))
    result = _tiny_run(tmp_path, reader)
    # Ground truth 1, 2, 2, 2 against predictions 1, 2, 1, 2: three of four correct.
    assert result["pooled"]["num_points"] == 4
    assert result["pooled"]["class_accuracy"] == pytest.approx(0.75)
    # bleached is true, false, false, false; predicted (top half positive) true, false, true,
    # false: one true positive, one false positive, two true negatives.
    bleached = result["pooled"]["binary"]["per_concept"]["bleached"]
    assert (bleached["tp"], bleached["fp"], bleached["fn"], bleached["tn"]) == (1, 1, 0, 2)


def test_an_unreadable_image_is_counted_failed_and_contributes_no_points(tmp_path):
    images = _tiny_images("a", "b")
    annotations = _tiny_annotations("a", [(0, 0, "coral")]) + _tiny_annotations(
        "b", [(0, 0, "coral"), (1, 1, "sand")]
    )
    reader = tiny_reader(tmp_path / "images", images, annotations, fail={"b"})
    item = _tiny_dataset_item(reader, idx=1)
    assert item["ok"] is False
    assert item["rows"].size == 0 and item["concept_rows"].shape == (0, 3)
    assert item["image"].shape == (3, 4, 4)

    result = _tiny_run(tmp_path, reader)
    assert (result["images_processed"], result["images_failed"]) == (2, 1)
    assert result["pooled"]["num_points"] == 1
    payload = json.loads((tmp_path / "tiny" / "metrics.json").read_text())
    assert payload["images_failed"] == 1


def test_a_one_channel_image_is_read_as_three_identical_channels(tmp_path):
    reader = tiny_reader(
        tmp_path / "images", _tiny_images("a"), _tiny_annotations("a", [(0, 0, "coral")]), gray=True
    )
    item = _tiny_dataset_item(reader)
    assert item["ok"] is True
    assert (item["H"], item["W"]) == (4, 8)
    expected = torch.tensor(
        [(100 / 255 - m) / sd for m, sd in zip(TINY_MEAN, TINY_STD, strict=True)]
    )
    assert item["image"].shape == (3, 4, 4)
    torch.testing.assert_close(item["image"][:, 0, 0], expected)


def test_max_images_per_unit_keeps_the_first_images_of_each_unit(tmp_path):
    images = _tiny_images("a1", "a2", "a3") + _tiny_images("b1", "b2", unit="v")
    annotations = (
        _tiny_annotations("a1", [(0, 0, "coral")])
        + _tiny_annotations("a2", [(0, 0, "coral")] * 2)
        + _tiny_annotations("a3", [(0, 0, "coral")] * 3)
        + _tiny_annotations("b1", [(0, 0, "coral")] * 4)
        + _tiny_annotations("b2", [(0, 0, "coral")] * 5)
    )
    result = _tiny_run(
        tmp_path, tiny_reader(tmp_path / "images", images, annotations), max_images_per_unit=1
    )
    assert result["images_processed"] == 2
    assert {u: d["num_points"] for u, d in result["units"].items()} == {"u": 1, "v": 4}
    payload = json.loads((tmp_path / "tiny" / "metrics.json").read_text())
    assert (payload["num_units_done"], payload["num_units_total"]) == (2, 2)


def test_binary_concepts_missing_from_the_concept_names_raise_before_any_output(tmp_path):
    reader = tiny_reader(
        tmp_path / "images", _tiny_images("a"), _tiny_annotations("a", [(0, 0, "coral")])
    )
    with pytest.raises(RuntimeError, match=r"\['anthropogenic'\]"):
        _tiny_run(tmp_path, reader, binary_concepts=("bleached", "anthropogenic"))
    assert not (tmp_path / "tiny").exists()


def test_unmapped_labels_are_reported_per_lookup_and_written_to_metrics_json(tmp_path):
    class_lookup = ClassLookup({"coral": 1, "sand": 0}, TINY_ID2LABEL)
    concept_lookup = ConceptLookup({"coral": np.array([2, 1, 0], np.int8)}, 3)
    # 'sand' maps to class 0 but has no concept row; 'mystery' has neither; names are lowercased.
    annotations = _tiny_annotations("a", [(0, 0, "CORAL"), (1, 1, "sand"), (2, 2, "Mystery")])
    result = run_point_evaluation(
        eval_name="tiny",
        reader=tiny_reader(tmp_path / "images", _tiny_images("a"), annotations),
        unit_col="source_id",
        predictor=cast(CBMPredictor, RowBandPredictor()),
        class_lookup=class_lookup,
        concept_lookup=concept_lookup,
        concept_names=TINY_CONCEPTS,
        id2label=TINY_ID2LABEL,
        output_dir=tmp_path,
        num_workers=0,
    )
    assert result["unmapped_class_labels"] == ["mystery", "sand"]
    assert result["unmapped_concept_labels"] == ["mystery", "sand"]
    payload = json.loads((tmp_path / "tiny" / "metrics.json").read_text())
    assert payload["unmapped_class_labels"] == ["mystery", "sand"]
    assert payload["unmapped_concept_labels"] == ["mystery", "sand"]


def test_progress_callback_reports_images_processed_after_each_batch(tmp_path):
    ids = ["a", "b", "c", "d", "e"]
    annotations = [row for i in ids for row in _tiny_annotations(i, [(0, 0, "coral")])]
    calls: list[tuple[int, int]] = []
    _tiny_run(
        tmp_path,
        tiny_reader(tmp_path / "images", _tiny_images(*ids), annotations),
        batch_size=2,
        progress_cb=lambda done, total: calls.append((done, total)),
    )
    assert calls == [(2, 5), (4, 5), (5, 5)]
