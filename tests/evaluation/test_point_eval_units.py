"""run_point_evaluation: per-unit blocks, macro means, and per-unit hierarchical metrics."""

from __future__ import annotations

import csv
import json
import math
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import torch

from mermaidseg.evaluation.gt_mapping import ClassLookup, build_concept_lookup
from mermaidseg.evaluation.hierarchical import HIER_HEADLINE_KEYS
from mermaidseg.evaluation.metrics import REPORTED_BINARY_CONCEPTS
from mermaidseg.evaluation.point_datasets import (
    run_point_evaluation,
    unit_scalar_keys,
    unit_scalars,
)
from mermaidseg.evaluation.predictor import CBMPredictor

_REPO = Path(__file__).resolve().parents[2]
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
CORAL = {"montastraea", "porites", "acropora cervicornis - bleached", "non-acropora encrusting", "millepora"}
SIZE = 8
NUM_CLASSES = 5


def _concept_names() -> list[str]:
    raw = json.loads((_REPO / "demo" / "concept_id2name.json").read_text())
    return [name for _, name in sorted(raw.items(), key=lambda kv: int(kv[0]))]


class FakeDataset:
    """Two CoralNet-like sources with a few 8x8 images and random point labels."""

    def __init__(self, units: list[str], seed: int = 0):
        rng = np.random.default_rng(seed)
        images, anns = [], []
        for unit in ["10", "20"]:
            for k in range(3):
                image_id = f"{unit}_{k}"
                images.append({"source_id": unit, "image_id": image_id})
                for _ in range(12):
                    anns.append(
                        {
                            "image_id": image_id,
                            "row": int(rng.integers(0, SIZE)),
                            "col": int(rng.integers(0, SIZE)),
                            "source_label_name": LABELS[int(rng.integers(0, len(LABELS)))],
                        }
                    )
        df_images = pd.DataFrame(images)
        self.df_images = df_images[df_images["source_id"].isin(units)].reset_index(drop=True)
        df_ann = pd.DataFrame(anns)
        self.df_annotations = df_ann[df_ann["image_id"].isin(self.df_images["image_id"])]

    def read_image(self, image_id: str, **_):
        value = 10 * int(image_id.split("_")[0]) + int(image_id.split("_")[1])
        return np.full((SIZE, SIZE, 3), value, dtype=np.uint8)


class StubPredictor:
    """Predictions are a deterministic function of the image, independent of batching."""

    input_size = (SIZE, SIZE)
    device = torch.device("cpu")
    sample_at_points = staticmethod(CBMPredictor.sample_at_points)

    def __init__(self, num_concepts: int):
        self.num_concepts = num_concepts

    def forward_probs(self, images: torch.Tensor):
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
        base_ds=FakeDataset(units),
        unit_col="source_id",
        key_cols=["image_id"],
        predictor=StubPredictor(len(names)),
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


def test_units_carry_concept_and_hierarchical_blocks(both):
    assert set(both["units"]) == {"10", "20"}
    for d in both["units"].values():
        for concept in REPORTED_BINARY_CONCEPTS:
            assert {"accuracy", "precision", "recall", "f1", "n_true"} <= set(d[concept])
        assert "consistency/overall" in d["hierarchical"]
        for rd in d["taxonomic"].values():
            assert {"macro_f1_all", "macro_f1_living", "f1_none"} <= set(rd)
    assert "per_taxon" in both["pooled"]["taxonomic"]["genus"]


def test_macro_is_the_mean_of_unit_values(both):
    keys = unit_scalar_keys(list(both["pooled"]["taxonomic"]), REPORTED_BINARY_CONCEPTS, with_hier=True)
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


@pytest.mark.parametrize("unit", ["10", "20"])
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


def test_pooled_hierarchical_unchanged_by_per_unit_accumulators(tmp_path, both):
    rerun = _run(tmp_path, ["10", "20"], batch_size=5)
    assert both["pooled"]["hierarchical"] == pytest.approx(
        rerun["pooled"]["hierarchical"], nan_ok=True
    )


@pytest.mark.parametrize(
    ("label", "live", "coral"),
    [
        ("porites", True, True),
        ("dead pocillopora", False, False),
        ("dead coral", False, False),
        ("a. galapagensis (dead)", False, False),
        ("a. galapagensis (alive)", True, True),
        ("millepora", True, True),
        ("cred-octocoral", True, True),
        ("palythoa", True, False),
        ("chondrilla spp.", True, False),  # sponge: live column ignored outside corals / echinoderms
        ("macroalgae", True, False),
        ("sand", True, False),
    ],
)
def test_point_scopes_from_the_concept_table(label, live, coral):
    lookup = build_concept_lookup("coralnet", [label], _concept_names(), _CSV)
    assert lookup.scopes(label) == (live, coral)


def test_living_and_coral_counts_follow_the_live_column(both):
    table = pd.read_csv(_CSV, dtype=str).fillna("not_given")
    table = table[table["source_dataset_source"] == "coralnet"]
    table = table.assign(label=table["source_label_class_name"].str.lower()).drop_duplicates("label")
    table = table.set_index("label").loc[LABELS]
    labels = FakeDataset(["10", "20"]).df_annotations["source_label_name"]
    assert labels.isin(DEAD).any() and labels.isin(CORAL).any()
    for rank, rd in both["pooled"]["taxonomic"].items():
        values = table[rank].str.lower()
        taxon = set(values.index[~values.isin(["none", "not_given"])])
        assert rd["n_living"] == int((labels.isin(taxon) & ~labels.isin(DEAD)).sum()), rank
        assert rd["n_coral"] == int((labels.isin(taxon) & labels.isin(CORAL)).sum()), rank
        assert rd["n_all"] - rd["n_none"] - rd["n_living"] == int(
            (labels.isin(taxon) & labels.isin(DEAD)).sum()
        ), rank


def test_per_unit_csv_has_new_columns(both, tmp_path_factory):
    out = tmp_path_factory.mktemp("csv")
    _run(out, ["10", "20"])
    with (out / "coralnet_val" / "per_unit.csv").open() as handle:
        header = next(csv.reader(handle))
    for column in (
        "tax_genus_macro_f1_all",
        "tax_genus_macro_f1_living",
        "tax_genus_acc_coral",
        "tax_genus_macro_f1_coral",
        "tax_kingdom_f1_none",
        "bleached_f1",
        "anthropogenic_precision",
        "hier/consistency/overall",
    ):
        assert column in header
