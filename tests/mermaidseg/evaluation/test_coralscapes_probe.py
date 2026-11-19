"""Coralscapes probe: 39-class scoring on shared features, strided windows, concept code tables, SVM folding."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import torch

from mermaidseg.datasets.coralscapes_v2.coralscapes_v2_dataset import CORALSCAPES_V2_ID2NAME
from mermaidseg.evaluation.coralscapes_probe import (
    LEGACY_NUM_CLASSES,
    average_window_logits,
    build_coralscapes_concept_codes,
    coralscapes_windows,
    evaluate_coralscapes_probe,
    legacy_label,
    score_legacy_activated,
    union_probe_images,
)
from mermaidseg.evaluation.metrics import ClassConfusion

_CSV = Path(__file__).resolve().parents[3] / "configs" / "class_to_concepts.csv"


def test_windows_and_overlap_average():
    windows = coralscapes_windows()
    assert [name for name, *_box in windows] == ["left", "center", "right"]
    assert windows[0][1:] == (0, 0, 1024, 1024)
    assert windows[1][1:] == (0, 512, 1024, 1536)
    assert windows[2][1:] == (0, 1024, 1024, 2048)

    # A constant map stays constant under bilinear upsampling, so the overlap
    # bands are the unweighted means of the windows that cover them.
    values = (1.0, 3.0, 5.0)
    logits = torch.stack([torch.full((1, 2, 2), value) for value in values], dim=0)
    boxes = [window[1:] for window in windows]
    averaged = average_window_logits(logits, boxes, 1024, 2048)
    assert averaged.shape == (1, 1024, 2048)
    assert torch.allclose(averaged[:, :, 0:512], torch.full((1, 1024, 512), 1.0))
    assert torch.allclose(averaged[:, :, 512:1024], torch.full((1, 1024, 512), 2.0))
    assert torch.allclose(averaged[:, :, 1024:1536], torch.full((1, 1024, 512), 4.0))
    assert torch.allclose(averaged[:, :, 1536:2048], torch.full((1, 1024, 512), 5.0))


def test_uncovered_pixel_raises():
    logits = torch.zeros(1, 1, 2, 2)
    with pytest.raises(RuntimeError, match="no window coverage"):
        average_window_logits(logits, [(0, 0, 4, 4)], height=8, width=8)


def test_unknown_split_raises(tmp_path: Path):
    probe = tmp_path / "probe.json"
    probe.write_text(
        '{"images": [], "num_images": 0, "total_points": 0, "points_per_class_target": 10}'
    )
    with pytest.raises(ValueError, match="unknown Coralscapes splits"):
        evaluate_coralscapes_probe(
            predictor=None,  # type: ignore[arg-type]
            probe_json=probe,
            coralscapes_root=None,
            output_dir=tmp_path / "out",
            concept_names=["bleached", "anthropogenic"],
            taxonomy_csv=_CSV,
            splits=["train"],
        )


def test_coralscapes_anthropogenic_codes_match_the_concept_table():
    codes = build_coralscapes_concept_codes(_CSV, "anthropogenic")
    assert int(codes[0]) == 0
    by_name = {name: int(codes[class_id]) for class_id, name in CORALSCAPES_V2_ID2NAME.items()}
    positives = {name for name, code in by_name.items() if code == 2}
    assert positives == {
        "human",
        "human-made structure",
        "transect line",
        "transect tools",
        "trash",
    }
    assert {name for name, code in by_name.items() if code == 0} == {
        "water column",
        "water surface",
    }
    assert by_name["sand"] == 1


def _csv_with_sand_cell(tmp_path: Path, cell: str | None, drop_sand: bool = False) -> Path:
    frame = pd.read_csv(_CSV)
    name = frame["source_label_class_name"].astype(str).str.lower()
    source = frame["source_dataset_source"].astype(str).str.lower()
    sand = (source == "coralscapes_v2") & (name == "sand")
    if drop_sand:
        frame = frame.loc[~sand].copy()
    else:
        frame.loc[sand, "bleached"] = cell
    path = tmp_path / "class_to_concepts.csv"
    frame.to_csv(path, index=False)
    return path


def test_missing_coralscapes_class_is_not_treated_as_not_given(tmp_path: Path):
    path = _csv_with_sand_cell(tmp_path, cell=None, drop_sand=True)
    with pytest.raises(RuntimeError, match="sand"):
        build_coralscapes_concept_codes(path, "bleached")


def test_unexpected_bleached_cell_raises(tmp_path: Path):
    path = _csv_with_sand_cell(tmp_path, cell="maybe")
    with pytest.raises(RuntimeError, match="maybe"):
        build_coralscapes_concept_codes(path, "bleached")


def _head(class_id: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """A one-class linear head that predicts ``class_id`` for every pixel."""
    weff = torch.zeros(1, 1)
    beff = torch.zeros(1)
    classes = torch.tensor([class_id], dtype=torch.long)
    return weff, beff, classes


def test_one_activated_map_updates_every_budget():
    activated = torch.zeros(1, 2, 2)
    label = np.array([[1, 0], [2, 1]], dtype=np.uint8)
    lut = np.zeros(96, dtype=np.uint8)
    lut[1] = 2
    lut[2] = 3
    heads = {5: _head(2), 10: _head(3), 20: _head(2), 50: _head(3)}
    confusions = {k: ClassConfusion(LEGACY_NUM_CLASSES + 1, ignore_index=0) for k in heads}

    score_legacy_activated(activated, label, lut, heads, confusions)

    assert confusions[5].total == 3
    assert int(confusions[5].cm[2, 2]) == 2
    assert int(confusions[5].cm[3, 2]) == 1
    assert int(confusions[10].cm[3, 3]) == 1
    assert int(confusions[10].cm[2, 3]) == 2
    assert np.array_equal(confusions[20].cm, confusions[5].cm)
    assert np.array_equal(confusions[50].cm, confusions[10].cm)
    # The ignored pixel (legacy id 0) is not a counted ground-truth class.
    assert int(confusions[5].cm[0].sum()) == 0


def test_legacy_label_rejects_ids_outside_the_lut():
    lut = np.zeros(96, dtype=np.uint8)
    with pytest.raises(RuntimeError, match="0..95"):
        legacy_label(np.array([[96]], dtype=np.uint8), lut)


def test_score_legacy_requires_a_confusion_per_head():
    activated = torch.zeros(1, 1, 1)
    label = np.zeros((1, 1), dtype=np.uint8)
    lut = np.zeros(96, dtype=np.uint8)
    with pytest.raises(ValueError, match="different budgets"):
        score_legacy_activated(
            activated,
            label,
            lut,
            {5: _head(1)},
            {10: ClassConfusion(LEGACY_NUM_CLASSES + 1, ignore_index=0)},
        )


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
    images = union_probe_images(probes)
    assert [image["stem"] for image in images] == ["a", "b"]
    assert set(images[0]["points_by_k"]) == {5, 10}

    probes[20] = {"images": [{**image_a, "label_md5": "other", "points": [{"class_id": 1}]}]}
    with pytest.raises(RuntimeError, match="label md5"):
        union_probe_images(probes)


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
