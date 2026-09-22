"""Unit tests for the evaluation metric accumulators."""

from __future__ import annotations

import numpy as np

from mermaidseg.evaluation.metrics import (
    BinaryConceptStats,
    ClassConfusion,
    ConceptLayout,
    MetricBundle,
    TaxonomicRankAccuracy,
)


def test_class_confusion_accuracy_and_ignore():
    cm = ClassConfusion(num_classes=3, ignore_index=0)
    gt = np.array([1, 1, 2, 0, 2])
    pred = np.array([1, 2, 2, 1, 2])
    cm.update(gt, pred)
    # GT==0 dropped -> 4 scored points, 3 correct.
    assert cm.total == 4
    assert abs(cm.accuracy() - 3 / 4) < 1e-9


def test_class_confusion_miou():
    cm = ClassConfusion(num_classes=3, ignore_index=0)
    # class 1: gt {1,1}, pred {1,2} -> tp1, fn1; class 2: gt{2}, pred{2} -> tp1
    cm.update(np.array([1, 1, 2]), np.array([1, 2, 2]))
    iou = cm.per_class_iou()
    # class1: tp=1, union = gt(2)+pred(1)-tp(1)=2 -> 0.5
    assert abs(iou[1] - 0.5) < 1e-9
    # class2: tp=1, union = gt(1)+pred(2)-1 = 2 -> 0.5
    assert abs(iou[2] - 0.5) < 1e-9
    assert abs(cm.miou() - 0.5) < 1e-9


def test_class_confusion_pooling():
    a = ClassConfusion(3)
    b = ClassConfusion(3)
    a.update(np.array([1, 2]), np.array([1, 1]))
    b.update(np.array([2, 2]), np.array([2, 2]))
    pooled = a.clone()
    pooled.merge(b)
    assert pooled.total == 4
    # 3 correct out of 4
    assert abs(pooled.accuracy() - 3 / 4) < 1e-9


def test_taxonomic_rank_accuracy_all_vs_living():
    # rank with 3 channels: [animalia, plantae, none]; none is local index 2.
    rank = TaxonomicRankAccuracy("kingdom", channel_indices=[0, 1, 2], none_local_index=2)
    # GT rows use 0/1/2. Two given pixels:
    #  p0: animalia active (correct pred), living
    #  p1: none active (pred none too -> correct), NOT living
    #  p2: not_given (all zeros) -> ignored
    gt = np.array(
        [
            [2, 1, 1],  # animalia
            [1, 1, 2],  # none
            [0, 0, 0],  # not_given
        ]
    )
    pred = np.array(
        [
            [0.9, 0.05, 0.05],  # argmax 0 == animalia -> correct
            [0.1, 0.1, 0.8],  # argmax 2 == none -> correct
            [0.5, 0.3, 0.2],  # ignored
        ],
        dtype=np.float32,
    )
    rank.update(gt, pred)
    assert rank.total_all == 2
    assert abs(rank.acc_all - 1.0) < 1e-9
    assert rank.total_living == 1  # only the animalia pixel
    assert abs(rank.acc_living - 1.0) < 1e-9


def test_taxonomic_living_excludes_none_error():
    rank = TaxonomicRankAccuracy("kingdom", channel_indices=[0, 1, 2], none_local_index=2)
    gt = np.array([[2, 1, 1]])  # living animalia
    pred = np.array([[0.1, 0.1, 0.8]], dtype=np.float32)  # predicts none -> wrong
    rank.update(gt, pred)
    assert rank.acc_all == 0.0
    assert rank.total_living == 1
    assert rank.acc_living == 0.0


def test_binary_concept_stats_matches_sklearn():
    from sklearn.metrics import accuracy_score, f1_score

    rng = np.random.default_rng(0)
    n = 200
    # channel 0 only; GT in {1(False),2(True)}; some not_given(0) ignored.
    gt_vals = rng.choice([0, 1, 2], size=n, p=[0.2, 0.4, 0.4])
    probs = rng.random(n).astype(np.float32)
    gt_rows = gt_vals[:, None]
    pred_rows = probs[:, None]
    stats = BinaryConceptStats(["c"], [0])
    stats.update(gt_rows, pred_rows)

    valid = gt_vals > 0
    y_true = (gt_vals[valid] == 2).astype(int)
    y_pred = (probs[valid] > 0.5).astype(int)
    per = stats.per_concept()["c"]
    assert abs(per["accuracy"] - accuracy_score(y_true, y_pred)) < 1e-9
    assert abs(per["f1"] - f1_score(y_true, y_pred, zero_division=0)) < 1e-9
    assert per["n_valid"] == int(valid.sum())


def test_concept_layout_and_bundle():
    names = [
        "kingdom__animalia",
        "kingdom__none",
        "class__hexacorallia",
        "class__none",
        "live",
        "sand",
    ]
    layout = ConceptLayout.from_concept_names(names)
    assert [r for r, _, _ in layout.taxo] == ["kingdom", "class"]
    assert layout.binary_names == ["live", "sand"]

    bundle = MetricBundle.create(num_classes=4, layout=layout)
    gt_class = np.array([1, 2])
    pred_class = np.array([1, 3])
    gt_rows = np.array([[2, 1, 2, 1, 2, 1], [0, 0, 0, 0, 1, 2]])
    pred_probs = np.array(
        [[0.9, 0.1, 0.8, 0.2, 0.9, 0.1], [0.4, 0.6, 0.5, 0.5, 0.2, 0.8]], dtype=np.float32
    )
    bundle.update(gt_class, pred_class, gt_rows, pred_probs)
    d = bundle.to_dict(class_id2name={0: "ignore", 1: "a", 2: "b", 3: "c"})
    assert d["num_points"] == 2
    assert abs(d["class_accuracy"] - 0.5) < 1e-9
    assert "kingdom" in d["taxonomic"]


def test_bundle_merge_pooling():
    names = ["kingdom__animalia", "kingdom__none", "live"]
    layout = ConceptLayout.from_concept_names(names)
    a = MetricBundle.create(4, layout)
    b = MetricBundle.create(4, layout)
    a.update(np.array([1]), np.array([1]), np.array([[2, 1, 2]]), np.array([[0.9, 0.1, 0.9]], np.float32))
    b.update(np.array([2]), np.array([3]), np.array([[2, 1, 1]]), np.array([[0.9, 0.1, 0.2]], np.float32))
    pooled = a.clone()
    pooled.merge(b)
    assert pooled.num_points == 2
    assert pooled.class_confusion.total == 2
    assert pooled.taxo[0].total_all == 2
