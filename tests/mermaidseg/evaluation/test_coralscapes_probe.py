"""Coralscapes probe: label spaces, probe loading, strided windows, concept code tables, SVM folding."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import torch
import yaml

from mermaidseg.dataset_reconciliation.concept_schema import ConceptSchema
from mermaidseg.evaluation import coralscapes_probe
from mermaidseg.evaluation.coralscapes_probe import (
    CORALSCAPES39,
    CORALSCAPES_V2,
    CORALSCAPES_V2_ID2NAME,
    LabelSpace,
    ProbeConcepts,
    _class_taxonomy_targets,
    _dense_images,
    _load_probe_entry,
    _predict_classes_from_activated,
    _require_label,
    _require_rgb,
    _sample_activated,
    _score_classes,
    _split_rows,
    _union_probe_images,
    build_coralscapes_concept_codes,
    coralscapes_v2_to_legacy_lut,
    evaluate_coralscapes_probe,
    label_md5,
    label_space_from_name,
    load_probes,
    reassemble_concept_logits,
)
from mermaidseg.evaluation.hierarchical import HierarchicalScore
from mermaidseg.evaluation.metrics import ClassConfusion
from tests.mermaidseg.data.conftest import ImageSpec, open_written

_REPO = Path(__file__).resolve().parents[3]
_CSV = _REPO / "configs" / "class_to_concepts.csv"
HEIGHT, WIDTH = 1024, 2048


def test_unknown_split_raises(tmp_path: Path):
    probe = _probe_json(tmp_path / "probe.json", label_space="coralscapes_v2", num_classes=95)
    with pytest.raises(ValueError, match="unknown Coralscapes splits"):
        evaluate_coralscapes_probe(
            CORALSCAPES_V2,
            [probe],
            None,  # type: ignore[arg-type]
            predictor=None,  # type: ignore[arg-type]
            output_dir=tmp_path / "out",
            concepts=ProbeConcepts(["bleached", "anthropogenic"], _CSV),
            splits=["train"],
        )


def _csv_without_sand(tmp_path: Path) -> Path:
    frame = pd.read_csv(_CSV)
    name = frame["source_label_class_name"].astype(str).str.lower()
    source = frame["source_dataset_source"].astype(str).str.lower()
    frame = frame.loc[~((source == "coralscapes_v2") & (name == "sand"))]
    path = tmp_path / "class_to_concepts.csv"
    frame.to_csv(path, index=False)
    return path


def test_missing_coralscapes_class_is_not_treated_as_not_given(tmp_path: Path):
    path = _csv_without_sand(tmp_path)
    with pytest.raises(RuntimeError, match="sand"):
        build_coralscapes_concept_codes(path, "bleached")


def _head(class_id: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """A one-class linear head that predicts ``class_id`` for every pixel."""
    weff = torch.zeros(1, 1)
    beff = torch.zeros(1)
    classes = torch.tensor([class_id], dtype=torch.long)
    return weff, beff, classes


def _space_with_lut(lut: np.ndarray) -> LabelSpace:
    return LabelSpace("test", "test", 39, {}, lut)


def test_one_activated_map_updates_every_budget():
    activated = torch.zeros(1, 2, 2)
    label = np.array([[1, 0], [2, 1]], dtype=np.uint8)
    lut = np.zeros(96, dtype=np.uint8)
    lut[1] = 2
    lut[2] = 3
    heads = {5: _head(2), 10: _head(3), 20: _head(2), 50: _head(3)}
    confusions = {k: ClassConfusion(40, ignore_index=0) for k in heads}

    _score_classes(activated, label, _space_with_lut(lut), heads, confusions)

    assert confusions[5].total == 3
    assert int(confusions[5].cm[2, 2]) == 2
    assert int(confusions[5].cm[3, 2]) == 1
    assert int(confusions[10].cm[3, 3]) == 1
    assert int(confusions[10].cm[2, 3]) == 2
    assert np.array_equal(confusions[20].cm, confusions[5].cm)
    assert np.array_equal(confusions[50].cm, confusions[10].cm)
    # The ignored pixel (39-class id 0) is not a counted ground-truth class.
    assert int(confusions[5].cm[0].sum()) == 0


def test_union_probe_images_groups_budgets_on_one_stem():
    image_a = {
        "site": "s",
        "stem": "a",
        "label_md5": "m",
        "height": 1,
        "width": 1,
    }
    probes = {
        5: {"images": [{**image_a, "points": [{"class_id": 1}]}]},
        10: {
            "images": [
                {**image_a, "points": [{"class_id": 2}]},
                {
                    "site": "s",
                    "stem": "b",
                    "label_md5": "n",
                    "height": 1,
                    "width": 1,
                    "points": [{"class_id": 3}],
                },
            ]
        },
    }
    images = _union_probe_images(probes)
    assert [image["stem"] for image in images] == ["a", "b"]
    assert set(images[0]["points_by_k"]) == {5, 10}

    probes[20] = {"images": [{**image_a, "label_md5": "other", "points": [{"class_id": 1}]}]}
    with pytest.raises(RuntimeError, match="label md5"):
        _union_probe_images(probes)


def test_svm_folding_matches_sklearn():
    from mermaidseg.evaluation.coralscapes_probe import _fit_linear_svm, _folded_linear

    rng = np.random.default_rng(0)
    d = 12
    class_ids = [1, 4, 9]
    xs, ys = [], []
    for ci in class_ids:
        xs.append(rng.normal(ci * 0.6, 1.0, size=(30, d)))
        ys += [ci] * 30
    x = np.concatenate(xs).astype(np.float32)
    y = np.array(ys)
    scaler, svm, acc = _fit_linear_svm(x, y, 1.0)
    assert 0.0 <= acc <= 1.0
    weff, beff, classes = _folded_linear(scaler, svm, torch.device("cpu"))
    xt = rng.normal(0, 3, size=(100, d)).astype(np.float32)
    ref = svm.predict(scaler.transform(xt))
    scores = torch.from_numpy(xt) @ weff.t() + beff
    got = classes[scores.argmax(1)].numpy()
    assert (got == ref).all()


class _WindowPredictor:
    """Window logits are the crop mean times ``scale`` in channel 0, ``second`` of it in channel 1.

    Channels past the first two, up to ``n_channels``, are zero.
    """

    device = torch.device("cpu")

    def __init__(self, scale: float = 1.0, second=lambda w: w, n_channels: int = 2):
        self.scale = scale
        self.second = second
        self.n_channels = n_channels
        self.crops: list[np.ndarray] = []
        self.forward_calls = 0

    def preprocess(self, crop):
        self.crops.append(crop)
        return torch.full((3, 2, 2), float(crop.mean()) * self.scale)

    def forward_concept_logits(self, images):
        self.forward_calls += 1
        window = images[:, 0]
        zeros = [torch.zeros_like(window)] * (self.n_channels - 2)
        return torch.stack([window, self.second(window), *zeros], dim=1)

    def activate_concept_logits(self, logits):
        return logits


def _banded_image() -> np.ndarray:
    """Pixel value ``col // 512`` in all channels: four 512-column bands valued 0, 1, 2, 3."""
    image = np.zeros((HEIGHT, WIDTH, 3), dtype=np.uint8)
    for band in range(4):
        image[:, band * 512 : (band + 1) * 512] = band
    return image


def test_reassemble_runs_left_center_right_crops_and_averages_their_overlaps():
    predictor = _WindowPredictor()
    image = _banded_image()
    logits = reassemble_concept_logits(predictor, image)  # type: ignore[arg-type]
    assert [crop.shape for crop in predictor.crops] == [(HEIGHT, 1024, 3)] * 3
    assert [float(crop.mean()) for crop in predictor.crops] == [0.5, 1.5, 2.5]
    assert predictor.forward_calls == 1
    assert logits.shape == (2, HEIGHT, WIDTH)
    # Windows are worth 0.5 / 1.5 / 2.5; the bands are left, left+center, center+right, right.
    for band, expected in enumerate([0.5, 1.0, 2.0, 2.5]):
        got = logits[:, :, band * 512 : (band + 1) * 512]
        assert torch.allclose(got, torch.full_like(got, expected)), band


@pytest.mark.parametrize(
    ("shape", "match"),
    [
        ((HEIGHT, 1024, 3), "must be 1024x2048"),
    ],
)
def test_require_rgb_rejects_wrong_channels_and_size(shape, match):
    with pytest.raises(ValueError, match=match):
        _require_rgb(np.zeros(shape, dtype=np.uint8), "img")


@pytest.mark.parametrize("shape", [(HEIGHT, 1024), (WIDTH, HEIGHT)])
def test_require_label_rejects_non_2d_and_wrong_size(shape):
    with pytest.raises(ValueError, match="img label must be 1024x2048"):
        _require_label(np.zeros(shape, dtype=np.uint8), "img")


def test_sample_activated_indexes_row_then_col_on_a_non_square_map():
    # value = 100 * channel + 10 * row + col on a 3-row x 5-col map.
    k, r, c = torch.meshgrid(torch.arange(2), torch.arange(3), torch.arange(5), indexing="ij")
    activated = (100 * k + 10 * r + c).float()
    got = _sample_activated(activated, np.array([2, 0]), np.array([4, 1]))
    assert got.tolist() == [[24.0, 1.0], [124.0, 101.0]]


@pytest.mark.parametrize(
    ("rows", "cols", "match"),
    [
        ([0, 1], [0], "row/col shape mismatch"),
        ([-1], [0], "outside the 3x5"),
    ],
)
def test_sample_activated_rejects_misaligned_and_out_of_bounds_points(rows, cols, match):
    with pytest.raises(ValueError, match=match):
        _sample_activated(torch.zeros(2, 3, 5), np.array(rows), np.array(cols))


def test_class_prediction_in_strips_covers_a_short_last_strip():
    # Rows 0..4 carry feature x0 = row. Class 7 scores x0, class 9 scores a constant 2.5,
    # so rows 0-2 predict 9 and rows 3-4 predict 7. Strips of 2 rows end in a 1-row strip.
    activated = torch.zeros(2, 5, 3)
    activated[0] = torch.arange(5.0)[:, None]
    weff = torch.tensor([[1.0, 0.0], [0.0, 0.0]])
    beff = torch.tensor([0.0, 2.5])
    classes = torch.tensor([7, 9])
    pred = _predict_classes_from_activated(activated, weff, beff, classes, strip_rows=2)
    assert pred.tolist() == [[9] * 3, [9] * 3, [9] * 3, [7] * 3, [7] * 3]


def _probe_json(path: Path, **overrides) -> Path:
    """A probe file; an override of None leaves that field out."""
    probe = {
        "label_space": "coralscapes39",
        "points_per_class_target": 5,
        "num_classes": 39,
        "images": [],
        **overrides,
    }
    path.write_text(json.dumps({k: v for k, v in probe.items() if v is not None}))
    return path


def _image(stem: str, class_id: int) -> dict:
    return {
        "site": "s",
        "stem": stem,
        "label_md5": "m",
        "height": 1,
        "width": 1,
        "points": [{"row": 0, "col": 0, "class_id": class_id}],
    }


@pytest.mark.parametrize(
    ("label_space", "num_classes", "expected"),
    [("coralscapes_v2", 95, CORALSCAPES_V2), ("coralscapes39", 39, CORALSCAPES39)],
    ids=["coralscapes_v2", "coralscapes39"],
)
def test_load_probes_resolves_the_files_label_space(tmp_path, label_space, num_classes, expected):
    path = _probe_json(tmp_path / "a.json", label_space=label_space, num_classes=num_classes)
    assert load_probes([path]).label_space is expected


def test_load_probes_keys_each_file_by_its_budget_and_unions_images_by_stem(tmp_path):
    first = _probe_json(tmp_path / "a.json", points_per_class_target=5, images=[_image("x", 1)])
    second = _probe_json(
        tmp_path / "b.json",
        points_per_class_target=10,
        images=[_image("x", 2), _image("y", 3)],
    )
    loaded = load_probes([first, second])
    assert sorted(loaded.by_k) == [5, 10]
    assert [image["stem"] for image in loaded.images] == ["x", "y"]
    assert loaded.images[0]["points_by_k"] == {
        5: [{"row": 0, "col": 0, "class_id": 1}],
        10: [{"row": 0, "col": 0, "class_id": 2}],
    }
    assert loaded.images[1]["points_by_k"] == {10: [{"row": 0, "col": 0, "class_id": 3}]}


def test_load_probes_rejects_files_of_different_label_spaces(tmp_path):
    v2 = _probe_json(tmp_path / "a.json", label_space="coralscapes_v2", num_classes=95)
    other = _probe_json(tmp_path / "b.json", points_per_class_target=10)
    with pytest.raises(RuntimeError, match="coralscapes_v2.*coralscapes39"):
        load_probes([v2, other])


def test_load_probes_rejects_a_file_with_no_label_space(tmp_path):
    with pytest.raises(ValueError, match=r"a\.json has no label_space"):
        load_probes([_probe_json(tmp_path / "a.json", label_space=None)])


def test_label_space_from_name_rejects_an_unknown_name():
    with pytest.raises(ValueError, match="'v3'"):
        label_space_from_name("v3")


def test_load_probes_rejects_a_probe_whose_width_differs_from_its_label_space(tmp_path):
    with pytest.raises(RuntimeError, match="num_classes is 95, expected 39"):
        load_probes([_probe_json(tmp_path / "a.json", num_classes=95)])


def test_load_probes_rejects_two_probes_with_the_same_budget(tmp_path):
    paths = [_probe_json(tmp_path / "a.json"), _probe_json(tmp_path / "b.json")]
    with pytest.raises(RuntimeError, match="two probe sets target k=5"):
        load_probes(paths)


def test_load_probes_rejects_an_empty_list():
    with pytest.raises(RuntimeError, match="no probe sets"):
        load_probes([])


def _full_concept_names() -> list[str]:
    data = yaml.safe_load((_REPO / "configs" / "data_config.yaml").read_text())["data"]
    schema = ConceptSchema.from_csv(_CSV, sources=set(data["datasets"]))
    return [name for _, name in sorted(schema.channel_id2name().items())]


def test_class_taxonomy_targets_has_a_not_given_column_zero_then_one_column_per_class():
    names = _full_concept_names()
    hier = HierarchicalScore(names, _CSV)
    targets = _class_taxonomy_targets(hier, names, _CSV, torch.device("cpu"))
    assert targets.shape == (len(hier.ranks), 96)
    assert (targets[:, 0] == 0).all()
    acropora_id = next(i for i, n in CORALSCAPES_V2_ID2NAME.items() if n == "acropora alive")
    genus = hier.ranks.index("genus")
    first_genus = next(i for i, n in enumerate(names) if n.startswith("genus__"))
    assert int(targets[genus, acropora_id]) == names.index("genus__acropora") - first_genus + 1


def test_a_missing_reported_binary_concept_raises_before_loading_images(tmp_path):
    probe = _probe_json(tmp_path / "probe.json", label_space="coralscapes_v2", num_classes=95)
    with pytest.raises(RuntimeError, match=r"\['anthropogenic'\]"):
        evaluate_coralscapes_probe(
            CORALSCAPES_V2,
            [probe],
            None,  # type: ignore[arg-type]
            predictor=None,  # type: ignore[arg-type]
            output_dir=tmp_path / "out",
            concepts=ProbeConcepts(["bleached", "sand"], _CSV),
        )


def test_conflicting_cells_for_one_coralscapes_class_raise(tmp_path):
    frame = pd.read_csv(_CSV)
    name = frame["source_label_class_name"].astype(str).str.lower()
    source = frame["source_dataset_source"].astype(str).str.lower()
    sand = frame[(source == "coralscapes_v2") & (name == "sand")].iloc[:1].copy()
    sand["bleached"] = "true" if sand["bleached"].iloc[0] != "true" else "false"
    path = tmp_path / "conflict.csv"
    pd.concat([frame, sand]).to_csv(path, index=False)
    with pytest.raises(RuntimeError, match="sand' has conflicting bleached cells"):
        build_coralscapes_concept_codes(path, "bleached")


def test_score_classes_rejects_a_lut_that_emits_an_id_beyond_the_label_space():
    lut = np.zeros(96, dtype=np.uint8)
    lut[1] = 40
    confusion = ClassConfusion(40, ignore_index=0)
    with pytest.raises(RuntimeError, match="test label id 40 is outside 0..39"):
        _score_classes(
            torch.zeros(1, 1, 1),
            np.ones((1, 1), np.uint8),
            _space_with_lut(lut),
            {5: _head(1)},
            {5: confusion},
        )
    assert confusion.total == 0


def _stub_version_reads(monkeypatch, label: np.ndarray) -> None:
    """Every probe and split image is the banded image over ``label``."""
    monkeypatch.setattr(coralscapes_probe, "_dense_images", lambda _version: None)
    monkeypatch.setattr(coralscapes_probe, "_split_rows", lambda _frame, _split: [{}])
    monkeypatch.setattr(coralscapes_probe, "_load_row", lambda _v, _row: (_banded_image(), label))
    monkeypatch.setattr(
        coralscapes_probe, "_load_probe_entry", lambda _v, _f, _e: (_banded_image(), label)
    )


def _three_class_entry() -> dict:
    def points(class_id, cols):
        return [
            {"row": 100 * (i + 1), "col": col, "class_id": class_id} for i, col in enumerate(cols)
        ]

    return {
        "site": "s",
        "stem": "a",
        "label_md5": "m",
        "height": HEIGHT,
        "width": WIDTH,
        "points": points(1, [10, 100, 300, 450])
        + points(2, [600, 700, 800, 900])
        + points(3, [1600, 1800, 1900, 2000]),
    }


def test_coralscapes39_run_fits_one_svm_per_budget_and_scores_each_in_one_forward(
    tmp_path, monkeypatch
):
    lut = coralscapes_v2_to_legacy_lut()
    v2_for = {target: int(np.flatnonzero(lut == target)[0]) for target in (1, 2, 3)}
    # Native label: 39-class id 1 on columns 0:512 and 1024:1536, 2 on 512:1024, 3 after.
    label = np.full((HEIGHT, WIDTH), v2_for[1], dtype=np.uint8)
    label[:, 512:1024] = v2_for[2]
    label[:, 1536:] = v2_for[3]
    _stub_version_reads(monkeypatch, label)
    probe_paths = [
        _probe_json(
            tmp_path / f"k{k}.json", points_per_class_target=k, images=[_three_class_entry()]
        )
        for k in (5, 10)
    ]
    # Bands 0..3 carry window logits (w, w^2) = (2, 4), (4, 16), (8, 64), (10, 100): points on a
    # convex curve, so each class is linearly separable from the others.
    predictor = _WindowPredictor(scale=4.0, second=lambda w: w**2)

    out = evaluate_coralscapes_probe(
        CORALSCAPES39,
        probe_paths,
        None,  # type: ignore[arg-type]
        predictor=predictor,  # type: ignore[arg-type]
        output_dir=tmp_path / "out",
        splits=["test"],
    )

    # One train image shared by both budgets plus one test image: two forwards.
    assert predictor.forward_calls == 2
    result = out["coralscapes39_test"]
    assert sorted(result["by_k"]) == ["10", "5"]
    # The SVM predicts 1, 2, 3, 3 for the four bands; band 2 is labelled 1, the only error.
    # Class 1: tp 512 of 1024 gt and 512 predicted -> 1/2; class 2: 1; class 3: 512 of 1024 -> 1/2.
    for k in ("5", "10"):
        scores = result["by_k"][k]["test"]
        assert scores["status"] == "done"
        assert scores["num_images"] == 1
        assert scores["accuracy"] == pytest.approx(0.75)
        assert scores["miou"] == pytest.approx(2 / 3)
        assert result["by_k"][k]["probe"]["classes_present"] == [1, 2, 3]
        assert result["by_k"][k]["probe"]["num_points"] == 12
    on_disk = json.loads((tmp_path / "out" / "coralscapes39_test" / "metrics.json").read_text())
    assert on_disk["by_k"]["5"]["test"]["accuracy"] == pytest.approx(0.75)
    assert not {"bleached", "anthropogenic", "hierarchical"} & set(on_disk)


def test_capped_split_loads_only_the_images_it_scores(tmp_path, monkeypatch):
    lut = coralscapes_v2_to_legacy_lut()
    v2_for = {target: int(np.flatnonzero(lut == target)[0]) for target in (1, 2, 3)}
    label = np.full((HEIGHT, WIDTH), v2_for[1], dtype=np.uint8)
    label[:, 512:1024] = v2_for[2]
    label[:, 1536:] = v2_for[3]
    _stub_version_reads(monkeypatch, label)
    monkeypatch.setattr(coralscapes_probe, "_split_rows", lambda _frame, _split: [{}, {}])
    loads: list[object] = []

    def counting_load(_version, row):
        loads.append(row)
        return _banded_image(), label

    monkeypatch.setattr(coralscapes_probe, "_load_row", counting_load)
    probe = _probe_json(
        tmp_path / "k5.json", points_per_class_target=5, images=[_three_class_entry()]
    )

    out = evaluate_coralscapes_probe(
        CORALSCAPES39,
        [probe],
        None,  # type: ignore[arg-type]
        predictor=_WindowPredictor(scale=4.0, second=lambda w: w**2),  # type: ignore[arg-type]
        output_dir=tmp_path / "out",
        splits=["test"],
        max_test_images=1,
    )

    assert out["coralscapes39_test"]["by_k"]["5"]["test"]["num_images"] == 1
    assert len(loads) == 1


def _kingdom_and_binary_concept_names() -> list[str]:
    kingdoms = [name for name in _full_concept_names() if name.startswith("kingdom__")]
    return [*kingdoms, "bleached", "anthropogenic"]


def test_coralscapes_v2_run_writes_the_by_k_shape_with_unannotated_concept_counts(
    tmp_path, monkeypatch
):
    # Native ids 1, 2, 3 on the bands; the top row (2048 pixels) is unannotated.
    label = np.full((HEIGHT, WIDTH), 1, dtype=np.uint8)
    label[:, 512:1024] = 2
    label[:, 1536:] = 3
    label[0, :] = 0
    _stub_version_reads(monkeypatch, label)
    probe = _probe_json(
        tmp_path / "probe.json",
        label_space="coralscapes_v2",
        num_classes=95,
        points_per_class_target=10,
        images=[_three_class_entry()],
    )
    names = _kingdom_and_binary_concept_names()
    predictor = _WindowPredictor(scale=4.0, second=lambda w: w**2, n_channels=len(names))

    out = evaluate_coralscapes_probe(
        CORALSCAPES_V2,
        [probe],
        None,  # type: ignore[arg-type]
        predictor=predictor,  # type: ignore[arg-type]
        output_dir=tmp_path / "out",
        concepts=ProbeConcepts(names, _CSV),
        splits=["val"],
    )

    on_disk = json.loads((tmp_path / "out" / "coralscapes_val" / "metrics.json").read_text())
    assert list(out) == ["coralscapes_val"]
    assert set(on_disk) == {
        "eval",
        "split",
        "label_space",
        "num_classes",
        "by_k",
        "hierarchical",
        "bleached",
        "anthropogenic",
    }
    assert (on_disk["eval"], on_disk["split"]) == ("coralscapes", "val")
    assert (on_disk["label_space"], on_disk["num_classes"]) == ("coralscapes_v2", 95)
    assert list(on_disk["by_k"]) == ["10"]
    assert on_disk["by_k"]["10"]["probe"]["points_per_class_target"] == 10
    assert on_disk["by_k"]["10"]["val"]["status"] == "done"
    assert "hierarchical" in on_disk
    for concept in ("bleached", "anthropogenic"):
        assert on_disk[concept]["n_unannotated"] == 2048


def test_a_two_class_probe_predicts_the_second_class_where_it_scores_higher():
    from mermaidseg.evaluation.coralscapes_probe import _fit_linear_svm, _folded_linear

    x = np.array([[2.0, 2.0]] * 4 + [[10.0, 10.0]] * 4, dtype=np.float32)
    y = np.array([1] * 4 + [2] * 4)
    scaler, svm, _ = _fit_linear_svm(x, y, 1.0)
    weff, beff, classes = _folded_linear(scaler, svm, torch.device("cpu"))
    activated = torch.tensor([[[2.0, 10.0]], [[2.0, 10.0]]])  # (K=2, H=1, W=2)
    pred = _predict_classes_from_activated(activated, weff, beff, classes)
    assert pred.tolist() == [[1, 2]]


def test_a_two_class_probe_predicts_the_first_class_at_a_score_of_exactly_zero():
    weff = torch.tensor([[1.0, 0.0]])  # one score row, as a two-class LinearSVC gives
    beff = torch.tensor([-2.0])
    classes = torch.tensor([7, 9])
    # scores: 1*1-2 = -1, 1*2-2 = 0, 1*3-2 = 1
    activated = torch.tensor([[[1.0, 2.0, 3.0]], [[0.0, 0.0, 0.0]]])  # (K=2, H=1, W=3)
    pred = _predict_classes_from_activated(activated, weff, beff, classes)
    assert pred.tolist() == [[7, 7, 9]]


def _two_band_label(top: int, bottom: int) -> np.ndarray:
    label = np.full((HEIGHT, WIDTH), bottom, dtype=np.uint8)
    label[: HEIGHT // 2] = top
    return label


@pytest.fixture
def version(tmp_path: Path):
    """A train image (labels 2 over 5), a validation image (7 over 9) and a test image (all 3)."""
    site = '{"site": "site1"}'
    return open_written(
        tmp_path,
        kind="dense",
        images=[
            ImageSpec("site1_a", group="train", mask=_two_band_label(2, 5), attrs=site),
            ImageSpec("site1_b", group="validation", mask=_two_band_label(7, 9), attrs=site),
            ImageSpec("site1_c", group="test", mask=np.full((HEIGHT, WIDTH), 3, np.uint8)),
        ],
        labels={i: CORALSCAPES_V2_ID2NAME[i] for i in (2, 3, 5, 7, 9)},
        name="coralscapes_v2",
    )


def test_a_probe_entry_reads_the_train_image_with_its_stem_as_image_id(version):
    entry = {"site": "site1", "stem": "site1_a", "label_md5": label_md5(_two_band_label(2, 5))}

    image, label = _load_probe_entry(version, _dense_images(version), entry)

    assert image.shape == (HEIGHT, WIDTH, 3)
    assert [int(label[r, c]) for r, c in [(0, 0), (511, 2047), (512, 0), (1023, 1000)]] == [
        2,
        2,
        5,
        5,
    ]


def test_a_probe_stem_that_is_not_a_train_image_raises(version):
    with pytest.raises(KeyError, match="site1_b"):
        _load_probe_entry(
            version,
            _dense_images(version),
            {"site": "site1", "stem": "site1_b", "label_md5": "m"},
        )


def test_a_probe_label_md5_that_differs_from_the_version_mask_raises(version):
    actual = label_md5(_two_band_label(2, 5))
    entry = {"site": "site1", "stem": "site1_a", "label_md5": "0" * 32}

    with pytest.raises(ValueError, match=f"site1_a.*{'0' * 32}.*{actual}"):
        _load_probe_entry(version, _dense_images(version), entry)


@pytest.mark.parametrize(
    ("split", "image_id", "top", "bottom"), [("val", "site1_b", 7, 9), ("test", "site1_c", 3, 3)]
)
def test_a_split_selects_its_group_rows_with_the_native_label_ids(
    version, split, image_id, top, bottom
):
    rows = _split_rows(_dense_images(version), split)

    assert [row["image_id"] for row in rows] == [image_id]
    _, label = coralscapes_probe._load_row(version, rows[0])
    assert (int(label[0, 0]), int(label[HEIGHT - 1, WIDTH - 1])) == (top, bottom)
