"""Unit tests for the evaluation metric accumulators."""

from __future__ import annotations

import numpy as np
import pytest

from mermaidseg.evaluation.metrics import (
    BinaryConceptScore,
    BinaryConceptStats,
    ClassConfusion,
    ConceptLayout,
    MetricBundle,
    TaxonomicRankAccuracy,
    concept_block_from_binary,
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
    # The source accumulator is untouched: 2 points, 1 correct.
    assert a.total == 2
    assert abs(a.accuracy() - 1 / 2) < 1e-9
    assert b.total == 2


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
    from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score

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
    assert abs(per["precision"] - precision_score(y_true, y_pred, zero_division=0)) < 1e-9
    assert abs(per["recall"] - recall_score(y_true, y_pred, zero_division=0)) < 1e-9
    assert abs(per["f1"] - f1_score(y_true, y_pred, zero_division=0)) < 1e-9
    assert per["n_valid"] == int(valid.sum())
    assert per["n_not_given"] == int((gt_vals == 0).sum())
    assert per["n_true"] == int((gt_vals == 2).sum())
    assert per["n_false"] == int((gt_vals == 1).sum())

    # Extra not_given rows must not move TP/FP/FN/TN.
    before = (per["tp"], per["fp"], per["fn"], per["tn"])
    stats.update(np.zeros((5, 1), dtype=np.int8), np.ones((5, 1), dtype=np.float32))
    after = stats.per_concept()["c"]
    assert (after["tp"], after["fp"], after["fn"], after["tn"]) == before
    assert after["n_not_given"] == per["n_not_given"] + 5


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
    assert layout.binary_indices == [4, 5]
    # (rank, global channel indices, local index of the rank's "none" channel)
    assert layout.taxo == [("kingdom", [0, 1], 1), ("class", [2, 3], 1)]

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
    # live: row0 True/0.9 -> TP, row1 False/0.2 -> TN.
    # sand: row0 False/0.1 -> TN, row1 True/0.8 -> TP.
    per = bundle.binary.per_concept()
    assert (per["live"]["tp"], per["live"]["fp"], per["live"]["fn"], per["live"]["tn"]) == (
        1,
        0,
        0,
        1,
    )
    assert (per["sand"]["tp"], per["sand"]["fp"], per["sand"]["fn"], per["sand"]["tn"]) == (
        1,
        0,
        0,
        1,
    )


def test_concept_block_is_copied_per_concept_and_missing_raises():
    stats = BinaryConceptStats(["bleached", "anthropogenic"], [0, 1])
    stats.update(np.array([[2, 1], [1, 2], [0, 2]]), np.array([[0.9, 0.1], [0.2, 0.8], [0.9, 0.3]]))
    binary = stats.to_dict()
    anthro = concept_block_from_binary(binary, "anthropogenic")
    assert (anthro["tp"], anthro["fp"], anthro["fn"], anthro["tn"]) == (1, 0, 1, 1)
    assert anthro["threshold"] == 0.5
    assert concept_block_from_binary(binary, "bleached")["n_not_given"] == 1
    with pytest.raises(RuntimeError, match="'sand'"):
        concept_block_from_binary(binary, "sand")


def test_binary_concept_score_separates_true_false_not_given_and_unannotated():
    score = BinaryConceptScore(threshold=0.5)
    # code 2 / 0.9 -> TP; code 2 / 0.5 -> FN (exactly 0.5 is negative)
    # code 1 / 0.9 -> FP; code 1 / 0.1 -> TN
    # code 0, annotated -> not_given; code 0, unannotated -> not a negative
    codes = np.array([2, 2, 1, 1, 0, 0])
    probs = np.array([0.9, 0.5, 0.9, 0.1, 0.99, 0.01])
    unannotated = np.array([False, False, False, False, False, True])
    score.update(codes, probs, unannotated)
    got = score.to_dict()
    assert (got["tp"], got["fn"], got["fp"], got["tn"]) == (1, 1, 1, 1)
    assert got["n_true"] == 2
    assert got["n_false"] == 2
    assert got["n_not_given"] == 1
    assert got["n_unannotated"] == 1
    assert got["precision"] == 0.5
    assert got["recall"] == 0.5
    assert abs(got["accuracy"] - 0.5) < 1e-9
    # F1 = 2tp / (2tp + fp + fn) = 2 / 4
    assert got["f1"] == 0.5


def test_binary_concept_score_rejects_unannotated_marked_true():
    score = BinaryConceptScore()
    with pytest.raises(ValueError, match="unannotated"):
        score.update(np.array([2]), np.array([0.9]), np.array([True]))


def test_bundle_merge_pooling():
    names = ["kingdom__animalia", "kingdom__none", "live"]
    layout = ConceptLayout.from_concept_names(names)
    a = MetricBundle.create(4, layout)
    b = MetricBundle.create(4, layout)
    a.update(
        np.array([1]), np.array([1]), np.array([[2, 1, 2]]), np.array([[0.9, 0.1, 0.9]], np.float32)
    )
    b.update(
        np.array([2]), np.array([3]), np.array([[2, 1, 1]]), np.array([[0.9, 0.1, 0.2]], np.float32)
    )
    pooled = a.clone()
    pooled.merge(b)
    assert pooled.num_points == 2
    assert pooled.class_confusion.total == 2
    assert pooled.taxo[0].total_all == 2
    # live: a has one TP, b has one TN; the pool has both.
    pooled_live = pooled.binary.per_concept()["live"]
    assert (pooled_live["tp"], pooled_live["fp"], pooled_live["fn"], pooled_live["tn"]) == (
        1,
        0,
        0,
        1,
    )
    # The source bundle is unchanged and shares no binary state with the clone.
    assert a.num_points == 1
    assert a.class_confusion.total == 1
    assert a.taxo[0].total_all == 1
    a_live = a.binary.per_concept()["live"]
    assert (a_live["tp"], a_live["tn"]) == (1, 0)
    assert pooled.binary is not a.binary


def test_class_confusion_drops_out_of_range_gt_and_clips_out_of_range_pred():
    cm = ClassConfusion(num_classes=3)
    # GT 3 and -1 are dropped; pred 5 clips to 2 and pred -2 clips to 0.
    cm.update(np.array([1, 3, -1, 2, 2]), np.array([1, 1, 1, 5, -2]))
    assert cm.cm.tolist() == [[0, 0, 0], [0, 1, 0], [1, 0, 1]]


def test_class_confusion_miou_skips_nan_classes_and_the_ignore_class():
    cm = ClassConfusion(num_classes=3, ignore_index=0)
    # A prediction of 0 gives class 0 IoU 0.0; class 2 is never seen, so its IoU is NaN.
    # Class 1: tp=1, union=2 -> 0.5. Only class 1 is averaged.
    cm.update(np.array([1, 1]), np.array([1, 0]))
    assert cm.miou() == 0.5


def test_class_confusion_to_dict():
    cm = ClassConfusion(num_classes=4, ignore_index=0)
    # Rows: gt 1 -> [pred 1, pred 0]; gt 2 -> [pred 1]. Class 3 is never seen.
    cm.update(np.array([1, 1, 2]), np.array([1, 0, 1]))
    assert cm.to_dict(id2name={1: "a", 2: "b"}) == {
        "accuracy": 1 / 3,
        "miou": 1 / 6,
        "total": 3,
        "per_class": {
            "a": {"iou": 1 / 3, "gt_count": 2, "pred_count": 2, "tp": 1},
            "b": {"iou": 0.0, "gt_count": 1, "pred_count": 0, "tp": 0},
            "3": {"iou": None, "gt_count": 0, "pred_count": 0, "tp": 0},
        },
    }


@pytest.mark.parametrize(
    ("prob", "tp", "fn"),
    [(0.5, 0, 1), (np.nextafter(0.5, 1.0), 1, 0)],
    ids=["exactly-threshold-is-negative", "just-above-is-positive"],
)
def test_binary_concept_stats_threshold_boundary(prob, tp, fn):
    stats = BinaryConceptStats(["c"], [0])
    stats.update(np.array([[2]]), np.array([[prob]], dtype=np.float64))
    per = stats.per_concept()["c"]
    assert (per["tp"], per["fn"]) == (tp, fn)


@pytest.mark.parametrize("code", [-1, 3])
def test_binary_concept_stats_rejects_codes_outside_0_to_2(code):
    stats = BinaryConceptStats(["c"], [0])
    with pytest.raises(ValueError, match=rf"found \[{code}\]"):
        stats.update(np.array([[1], [code]]), np.array([[0.9], [0.9]]))


@pytest.mark.parametrize(
    ("second_codes", "expected"),
    [
        # Second concept is all true negatives: accuracy 1.0, F1 undefined and skipped.
        ([1, 1], {"macro_accuracy": 0.75, "macro_f1": 2 / 3, "n_concepts_scored": 2}),
        # Second concept is all not_given: skipped from both macros.
        ([0, 0], {"macro_accuracy": 0.5, "macro_f1": 2 / 3, "n_concepts_scored": 1}),
    ],
    ids=["true-negatives-only", "not-given-only"],
)
def test_binary_concept_stats_macro_scores_skip_undefined_concepts(second_codes, expected):
    stats = BinaryConceptStats(["a", "b"], [0, 1])
    # Concept a: code 2 / 0.9 -> TP, code 1 / 0.9 -> FP: accuracy 0.5, F1 2/3.
    gt = np.array([[2, second_codes[0]], [1, second_codes[1]]])
    probs = np.array([[0.9, 0.1], [0.9, 0.1]])
    stats.update(gt, probs)
    got = stats.to_dict()
    assert {key: got[key] for key in expected} == pytest.approx(expected)
