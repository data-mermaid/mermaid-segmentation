"""Unit tests for the evaluation metric accumulators."""

from __future__ import annotations

import copy

import numpy as np
import pytest

from mermaidseg.evaluation.metrics import (
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


def test_concept_layout_groups_ranks_and_binary_channels_by_index():
    names = [
        "kingdom__animalia",
        "kingdom__none",
        "class__hexacorallia",
        "class__none",
        "live",
        "sand",
    ]
    layout = ConceptLayout.from_concept_names(names)
    assert layout.taxo == [("kingdom", [0, 1], 1), ("class", [2, 3], 1)]
    assert layout.binary_names == ["live", "sand"]
    assert layout.binary_indices == [4, 5]


def test_concept_layout_without_a_none_channel_has_no_none_index():
    layout = ConceptLayout.from_concept_names(["kingdom__animalia", "kingdom__plantae", "live"])
    assert layout.taxo == [("kingdom", [0, 1], None)]


def test_metric_bundle_to_dict_reports_class_taxonomic_and_binary_values():
    layout = ConceptLayout.from_concept_names(["kingdom__animalia", "kingdom__none", "live"])
    bundle = MetricBundle.create(num_classes=3, layout=layout)
    bundle.update(
        np.array([1, 2, 1]),
        np.array([1, 1, 0]),
        np.array([[2, 1, 2], [1, 2, 1], [0, 0, 2]]),
        np.array([[0.9, 0.1, 0.9], [0.2, 0.8, 0.9], [0.5, 0.5, 0.1]], dtype=np.float32),
    )
    names = {0: "ignore", 1: "coral", 2: "sand"}

    plain = bundle.to_dict(class_id2name=names)
    assert plain["num_points"] == 3
    assert plain["class_points_scored"] == 3
    assert plain["class_accuracy"] == pytest.approx(1 / 3)
    # coral: tp 1, gt 2, pred 2 -> 1/3; sand: tp 0, gt 1, pred 0 -> 0.
    assert plain["class_miou"] == pytest.approx(1 / 6)
    assert plain["taxonomic"]["kingdom"]["n_all"] == 2
    assert plain["taxonomic"]["kingdom"]["acc_all"] == 1.0
    live = plain["binary"]["per_concept"]["live"]
    assert (live["tp"], live["fp"], live["fn"], live["tn"]) == (1, 1, 1, 0)
    assert "class_detail" not in plain
    assert "per_taxon" not in plain["taxonomic"]["kingdom"]

    detailed = bundle.to_dict(
        class_id2name=names, include_class_detail=True, include_taxon_detail=True
    )
    per_class = detailed["class_detail"]["per_class"]
    assert list(per_class) == ["coral", "sand"]
    assert per_class["coral"] == {
        "iou": pytest.approx(1 / 3),
        "gt_count": 2,
        "pred_count": 2,
        "tp": 1,
    }
    assert per_class["sand"] == {"iou": 0.0, "gt_count": 1, "pred_count": 0, "tp": 0}
    assert set(detailed["taxonomic"]["kingdom"]["per_taxon"]) == {"animalia", "none"}


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


def test_binary_stats_excludes_unannotated_rows_from_every_count():
    stats = BinaryConceptStats(["a"], [0])
    # Annotated: code 2 / 0.9 -> TP, code 1 / 0.1 -> TN, code 0 -> not_given.
    # Unannotated: two code-0 rows with confident probabilities count nowhere else.
    codes = np.array([[2], [1], [0], [0], [0]])
    probs = np.array([[0.9], [0.1], [0.99], [0.99], [0.01]])
    unannotated = np.array([False, False, False, True, True])
    stats.update(codes, probs, unannotated=unannotated)
    got = stats.per_concept()["a"]
    assert (got["tp"], got["fp"], got["fn"], got["tn"]) == (1, 0, 0, 1)
    assert got["n_not_given"] == 1
    assert got["n_unannotated"] == 2


def test_binary_stats_rejects_an_unannotated_row_with_a_given_code():
    stats = BinaryConceptStats(["a", "b"], [0, 1])
    with pytest.raises(ValueError, match="unannotated"):
        stats.update(np.array([[0, 2]]), np.array([[0.9, 0.9]]), unannotated=np.array([True]))


def test_binary_stats_merge_sums_unannotated_counts():
    a = BinaryConceptStats(["c"], [0])
    b = BinaryConceptStats(["c"], [0])
    a.update(np.array([[0]]), np.array([[0.9]]), unannotated=np.array([True]))
    b.update(np.array([[0], [0]]), np.array([[0.9], [0.9]]), unannotated=np.array([True, True]))
    a.merge(b)
    assert a.per_concept()["c"]["n_unannotated"] == 3


def test_bundle_merge_pools_class_taxonomic_and_binary_counts():
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
    assert pooled.binary is not None
    live = pooled.binary.per_concept()["live"]
    # a contributes one true positive, b one true negative.
    assert (live["tp"], live["fp"], live["fn"], live["tn"]) == (1, 0, 0, 1)


def _random_rank_rows(rng, n: int, n_channels: int, none_index: int, p_not_given: float = 0.15):
    """GT 0/1/2 rows (some not_given), predicted probabilities, and the GT/pred indices."""
    gt_idx = rng.integers(0, n_channels, size=n)
    # Skew the GT so some taxa are absent and none is common.
    gt_idx[rng.random(n) < 0.3] = none_index
    gt_idx[gt_idx == 1] = 0
    given = rng.random(n) >= p_not_given
    gt_rows = np.ones((n, n_channels), dtype=np.int8)
    gt_rows[np.arange(n), gt_idx] = 2
    gt_rows[~given] = 0
    probs = rng.random((n, n_channels)).astype(np.float32)
    return gt_rows, probs, gt_idx, probs.argmax(axis=1), given


def test_taxonomic_macro_f1_matches_sklearn():
    from sklearn.metrics import f1_score

    rng = np.random.default_rng(0)
    n_channels, none_index = 6, 4
    gt_rows, probs, gt_idx, pred_idx, given = _random_rank_rows(rng, 500, n_channels, none_index)
    rank = TaxonomicRankAccuracy(
        "genus", channel_indices=list(range(n_channels)), none_local_index=none_index
    )
    rank.update(gt_rows, probs)
    d = rank.to_dict()

    y_true, y_pred = gt_idx[given], pred_idx[given]
    present = sorted(set(y_true.tolist()))
    assert 1 not in present  # a taxon without GT support is excluded from the mean
    expected_all = f1_score(y_true, y_pred, labels=present, average="macro", zero_division=0)
    assert d["macro_f1_all"] == pytest.approx(expected_all, abs=1e-12)
    assert d["n_taxa_all"] == len(present)

    living = y_true != none_index
    present_living = sorted(set(y_true[living].tolist()))
    expected_living = f1_score(
        y_true[living], y_pred[living], labels=present_living, average="macro", zero_division=0
    )
    assert d["macro_f1_living"] == pytest.approx(expected_living, abs=1e-12)
    assert d["n_taxa_living"] == len(present_living)

    expected_none = f1_score(y_true == none_index, y_pred == none_index, zero_division=0)
    assert d["f1_none"] == pytest.approx(expected_none, abs=1e-12)
    assert d["n_none"] == int((y_true == none_index).sum())
    assert d["acc_all"] == pytest.approx(float((y_true == y_pred).mean()))
    assert d["acc_living"] == pytest.approx(float((y_true[living] == y_pred[living]).mean()))


def test_f1_none_is_nan_without_none_channel_or_support():
    rank = TaxonomicRankAccuracy("kingdom", channel_indices=[0, 1])
    rank.update(np.array([[2, 1]]), np.array([[0.9, 0.1]], np.float32))
    assert np.isnan(rank.to_dict()["f1_none"])
    rank = TaxonomicRankAccuracy("kingdom", channel_indices=[0, 1, 2], none_local_index=2)
    rank.update(np.array([[2, 1, 1]]), np.array([[0.9, 0.1, 0.0]], np.float32))
    assert np.isnan(rank.to_dict()["f1_none"])
    assert np.isnan(TaxonomicRankAccuracy("kingdom", [0, 1, 2], 2).to_dict()["macro_f1_all"])


def test_taxonomic_merge_equals_single_accumulator():
    rng = np.random.default_rng(7)
    gt_rows, probs, *_ = _random_rank_rows(rng, 300, 5, 3)
    live = rng.random(300) < 0.8
    coral = live & (rng.random(300) < 0.5)
    whole = TaxonomicRankAccuracy("order", list(range(5)), 3, values=list("abcde"))
    whole.update(gt_rows, probs, live=live, coral=coral)
    parts = [
        TaxonomicRankAccuracy("order", list(range(5)), 3, values=list("abcde")) for _ in range(3)
    ]
    for part, idx in zip(parts, np.array_split(np.arange(300), 3), strict=True):
        part.update(gt_rows[idx], probs[idx], live=live[idx], coral=coral[idx])
    merged = parts[0].clone()
    merged.merge(parts[1])
    merged.merge(parts[2])
    assert merged.to_dict(include_detail=True) == whole.to_dict(include_detail=True)
    assert whole.total_coral > 0


def test_living_and_coral_scopes_match_sklearn():
    from sklearn.metrics import f1_score

    rng = np.random.default_rng(0)
    n_channels, none_index = 6, 4
    gt_rows, probs, gt_idx, pred_idx, given = _random_rank_rows(rng, 600, n_channels, none_index)
    # live=False on non-none points stands for dead coral; coral is a subset of live.
    live = rng.random(600) < 0.7
    coral = live & (rng.random(600) < 0.5)
    rank = TaxonomicRankAccuracy("genus", list(range(n_channels)), none_local_index=none_index)
    rank.update(gt_rows, probs, live=live, coral=coral)
    d = rank.to_dict()

    def expected(mask):
        y_true, y_pred = gt_idx[mask], pred_idx[mask]
        labels = sorted(set(y_true.tolist()))
        f1 = f1_score(y_true, y_pred, labels=labels, average="macro", zero_division=0)
        return float((y_true == y_pred).mean()), f1, int(mask.sum()), len(labels)

    not_none = given & (gt_idx != none_index)
    for scope, mask in (("living", not_none & live), ("coral", not_none & coral)):
        acc, f1, n, n_taxa = expected(mask)
        assert d[f"acc_{scope}"] == pytest.approx(acc, abs=1e-12), scope
        assert d[f"macro_f1_{scope}"] == pytest.approx(f1, abs=1e-12), scope
        assert (d[f"n_{scope}"], d[f"n_taxa_{scope}"]) == (n, n_taxa), scope
    # The all scope ignores the flags.
    acc, f1, n, _ = expected(given)
    assert (d["acc_all"], d["n_all"]) == (pytest.approx(acc), n)
    assert d["macro_f1_all"] == pytest.approx(f1, abs=1e-12)


def test_dead_coral_counts_in_all_but_not_living():
    # [acropora, porites, none]: a dead acropora predicted acropora, a live porites
    # predicted acropora, and a sand point predicted none.
    rank = TaxonomicRankAccuracy(
        "genus", [0, 1, 2], none_local_index=2, values=["acropora", "porites", "none"]
    )
    gt = np.array([[2, 1, 1], [1, 2, 1], [1, 1, 2]])
    pred = np.array([[0.9, 0.1, 0.0], [0.8, 0.2, 0.0], [0.0, 0.1, 0.9]], dtype=np.float32)
    live = np.array([False, True, True])
    rank.update(gt, pred, live=live, coral=live)
    d = rank.to_dict(include_detail=True)
    assert (d["n_all"], d["n_living"], d["n_coral"]) == (3, 1, 1)
    assert d["acc_all"] == pytest.approx(2 / 3)
    assert d["acc_living"] == 0.0 and d["acc_coral"] == 0.0
    # Living: only porites is a GT taxon; the dead acropora's prediction is not a false positive.
    assert d["n_taxa_living"] == 1 and d["macro_f1_living"] == 0.0
    assert d["per_taxon"]["acropora"]["f1_living"] is None
    assert d["per_taxon"]["acropora"]["support"] == 1
    assert d["per_taxon"]["acropora"]["support_living"] == 0


def test_without_flags_living_is_not_none_and_coral_is_empty():
    rank = TaxonomicRankAccuracy("kingdom", [0, 1, 2], none_local_index=2)
    rank.update(np.array([[2, 1, 1], [1, 1, 2]]), np.array([[0.9, 0, 0.1], [0, 0, 1]], np.float32))
    d = rank.to_dict()
    assert (d["n_living"], d["n_coral"]) == (1, 0)
    assert np.isnan(d["acc_coral"]) and np.isnan(d["macro_f1_coral"])
    with pytest.raises(ValueError, match="live must have shape"):
        rank.update(
            np.array([[2, 1, 1]]), np.array([[1, 0, 0]], np.float32), live=np.array([True, True])
        )


def _confusion_with_data() -> ClassConfusion:
    cm = ClassConfusion(3)
    cm.update(np.array([1, 2]), np.array([1, 1]))
    return cm


def test_class_confusion_clone_is_independent():
    source = _confusion_with_data()
    clone = source.clone()
    clone.update(np.array([2, 2]), np.array([2, 2]))
    assert source.total == 2
    assert clone.total == 4


def test_taxonomic_clone_is_independent():
    source = TaxonomicRankAccuracy("kingdom", [0, 1], none_local_index=1, values=["a", "none"])
    source.update(np.array([[2, 1]]), np.array([[0.9, 0.1]], np.float32), live=np.array([True]))
    clone = source.clone()
    clone.update(np.array([[2, 1]]), np.array([[0.9, 0.1]], np.float32), live=np.array([True]))
    clone.values.append("extra")
    clone.channel_indices.append(9)
    assert (source.total_all, source.total_living) == (1, 1)
    assert (clone.total_all, clone.total_living) == (2, 2)
    assert source.values == ["a", "none"]
    assert source.channel_indices == [0, 1]


def test_binary_stats_clone_is_independent():
    source = BinaryConceptStats(["c"], [0])
    source.update(np.array([[2], [0]]), np.array([[0.9], [0.9]]))
    clone = source.clone()
    clone.update(np.array([[2], [0]]), np.array([[0.9], [0.9]]))
    assert source.per_concept()["c"]["tp"] == 1
    assert source.per_concept()["c"]["n_not_given"] == 1
    assert clone.per_concept()["c"]["tp"] == 2
    assert clone.per_concept()["c"]["n_not_given"] == 2


def test_metric_bundle_clone_is_independent():
    layout = ConceptLayout.from_concept_names(["kingdom__animalia", "kingdom__none", "live"])
    args = (
        np.array([1]),
        np.array([1]),
        np.array([[2, 1, 2]]),
        np.array([[0.9, 0.1, 0.9]], np.float32),
    )
    source = MetricBundle.create(3, layout)
    source.update(*args)
    clone = source.clone()
    clone.update(*args)
    assert source.num_points == 1
    assert source.class_confusion.total == 1
    assert source.taxo[0].total_all == 1
    assert source.binary is not None and source.binary.per_concept()["live"]["tp"] == 1
    assert clone.num_points == 2
    assert clone.class_confusion.total == 2
    assert clone.taxo[0].total_all == 2
    assert clone.binary is not None and clone.binary.per_concept()["live"]["tp"] == 2


@pytest.mark.parametrize(
    ("target", "other", "match"),
    [
        (ClassConfusion(3), ClassConfusion(4), "num_classes"),
        (
            TaxonomicRankAccuracy("kingdom", [0, 1]),
            TaxonomicRankAccuracy("kingdom", [0, 2]),
            "different channels",
        ),
        (BinaryConceptStats(["a"], [0]), BinaryConceptStats(["b"], [0]), "different concepts"),
    ],
    ids=["class_confusion", "taxonomic", "binary"],
)
def test_merge_rejects_a_mismatched_layout_and_leaves_the_target_unchanged(target, other, match):
    before = copy.deepcopy(target.__dict__)
    with pytest.raises(ValueError, match=match):
        target.merge(other)
    after = target.__dict__
    assert after.keys() == before.keys()
    for key, value in before.items():
        assert np.array_equal(after[key], value), key


def test_class_confusion_to_dict_excludes_ignore_and_names_classes():
    cm = ClassConfusion(num_classes=5, ignore_index=0)
    # Class 0 is never scored as ground truth but is predicted once, so its IoU is 0
    # and would drag the mean down if mIoU did not exclude it. Class 4 is unused.
    cm.update(np.array([1, 1, 2, 3, 0]), np.array([1, 2, 2, 0, 1]))
    got = cm.to_dict(id2name={1: "a", 2: "b"})
    assert got["total"] == 4
    assert got["accuracy"] == pytest.approx(0.5)
    assert got["miou"] == pytest.approx(1 / 3)  # (0.5 + 0.5 + 0.0) / 3, class 4 is NaN
    assert got["per_class"] == {
        "a": {"iou": 0.5, "gt_count": 2, "pred_count": 1, "tp": 1},
        "b": {"iou": 0.5, "gt_count": 1, "pred_count": 2, "tp": 1},
        "3": {"iou": 0.0, "gt_count": 1, "pred_count": 0, "tp": 0},
        "4": {"iou": None, "gt_count": 0, "pred_count": 0, "tp": 0},
    }
    assert cm.miou() == pytest.approx(1 / 3)


def test_class_confusion_class_ids_select_the_reported_and_averaged_classes():
    cm = ClassConfusion(num_classes=5, ignore_index=0)
    cm.update(np.array([1, 1, 2, 3, 0]), np.array([1, 2, 2, 0, 1]))
    assert cm.miou(class_ids=[1, 3]) == pytest.approx(0.25)
    got = cm.to_dict(class_ids=[1, 3])
    assert list(got["per_class"]) == ["1", "3"]
    assert got["miou"] == pytest.approx(0.25)


def test_class_confusion_drops_out_of_range_gt_and_clips_out_of_range_predictions():
    cm = ClassConfusion(num_classes=3, ignore_index=0)
    # GT -1 and 5 are dropped; predictions 9 and -4 clip to classes 2 and 0.
    cm.update(np.array([1, 1, -1, 5, 2]), np.array([9, -4, 1, 1, 2]))
    assert cm.cm.tolist() == [[0, 0, 0], [1, 0, 1], [0, 0, 1]]


def test_binary_stats_scores_only_the_selected_channels_with_a_strict_threshold():
    # Column 0 is not a selected channel; reading it would change every count.
    stats = BinaryConceptStats(["a", "b", "c"], [1, 2, 3])
    gt = np.array(
        [
            [2, 2, 1, 0],
            [2, 2, 1, 0],
            [2, 1, 1, 0],
            [2, 1, 2, 0],
        ]
    )
    probs = np.array(
        [
            [0.99, 0.9, 0.1, 0.9],
            [0.99, 0.5, 0.1, 0.9],  # exactly 0.5 is negative
            [0.99, 0.9, 0.1, 0.9],
            [0.99, 0.1, 0.9, 0.9],
        ]
    )
    stats.update(gt, probs)
    got = stats.to_dict()
    per = got["per_concept"]
    # a: TP, FN (0.5), FP, TN -> 1/1/1/1.   b: TN, TN, TN, TP.   c: never given.
    assert (per["a"]["tp"], per["a"]["fn"], per["a"]["fp"], per["a"]["tn"]) == (1, 1, 1, 1)
    assert (per["b"]["tp"], per["b"]["fn"], per["b"]["fp"], per["b"]["tn"]) == (1, 0, 0, 3)
    assert per["c"]["n_valid"] == 0
    assert per["c"]["n_not_given"] == 4
    assert got["n_concepts_scored"] == 2
    assert got["macro_accuracy"] == pytest.approx((0.5 + 1.0) / 2)
    assert got["macro_f1"] == pytest.approx((0.5 + 1.0) / 2)


def test_binary_stats_macro_is_nan_when_no_concept_is_scored():
    stats = BinaryConceptStats(["a"], [0])
    stats.update(np.array([[0]]), np.array([[0.9]]))
    got = stats.to_dict()
    assert got["n_concepts_scored"] == 0
    assert np.isnan(got["macro_accuracy"]) and np.isnan(got["macro_f1"])


@pytest.mark.parametrize("bad_code", [3, -1])
def test_binary_stats_rejects_codes_outside_0_1_2(bad_code):
    stats = BinaryConceptStats(["a"], [0])
    with pytest.raises(ValueError, match=rf"found \[{bad_code}\]"):
        stats.update(np.array([[2], [bad_code]]), np.array([[0.9], [0.9]]))


def test_binary_stats_rejects_misaligned_names_and_channels():
    with pytest.raises(ValueError, match="must align"):
        BinaryConceptStats(["a", "b"], [0])


@pytest.mark.parametrize(
    ("kwargs", "match"),
    [
        ({"confusion": np.zeros((3, 3), dtype=np.int64)}, "confusion must be"),
        ({"confusion_coral": np.zeros((2, 3), dtype=np.int64)}, "confusion_coral must be"),
        ({"values": ["a", "b", "c"]}, "2 channels but 3 values"),
    ],
)
def test_taxonomic_rejects_matrices_and_values_that_do_not_match_the_channels(kwargs, match):
    with pytest.raises(ValueError, match=match):
        TaxonomicRankAccuracy("kingdom", [0, 1], **kwargs)


def test_concept_block_from_binary_rejects_an_entry_missing_a_required_field():
    binary = {"per_concept": {"bleached": {"tp": 1, "fp": 0}}}
    with pytest.raises(RuntimeError, match="bleached stats are missing fields"):
        concept_block_from_binary(binary, "bleached")
