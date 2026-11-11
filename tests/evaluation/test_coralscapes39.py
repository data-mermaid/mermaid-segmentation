"""39-class Coralscapes probe: shared features, no model and no HuggingFace."""

from __future__ import annotations

import numpy as np
import pytest
import torch

from mermaidseg.evaluation.coralscapes_probe import (
    LEGACY_NUM_CLASSES,
    legacy_label,
    score_legacy_activated,
    union_probe_images,
)
from mermaidseg.evaluation.metrics import ClassConfusion


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
