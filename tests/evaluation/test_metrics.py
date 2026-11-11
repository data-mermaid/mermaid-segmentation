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


def test_binary_concept_score_rejects_unannotated_marked_true():
    score = BinaryConceptScore()
    with pytest.raises(ValueError, match="unannotated"):
        score.update(np.array([2]), np.array([0.9]), np.array([True]))


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


@pytest.mark.parametrize("seed", [0, 1, 2])
def test_taxonomic_macro_f1_matches_sklearn(seed):
    from sklearn.metrics import f1_score

    rng = np.random.default_rng(seed)
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


def test_taxonomic_living_f1_ignores_nonliving_predicted_as_taxon():
    # [animalia, plantae, none]. A none pixel predicted animalia is a false positive for
    # animalia in macro_f1_all only; a living pixel predicted none is a false negative.
    rank = TaxonomicRankAccuracy("kingdom", channel_indices=[0, 1, 2], none_local_index=2)
    gt = np.array([[2, 1, 1], [2, 1, 1], [1, 1, 2]])
    pred = np.array([[0.9, 0, 0.1], [0.1, 0, 0.9], [0.9, 0, 0.1]], dtype=np.float32)
    rank.update(gt, pred)
    d = rank.to_dict()
    # living: animalia tp=1, fn=1 (predicted none), no fp -> 2/3.
    assert d["macro_f1_living"] == pytest.approx(2 / 3)
    # all: animalia tp=1 fp=1 fn=1 -> 0.5; none tp=0 -> 0.
    assert d["macro_f1_all"] == pytest.approx(0.25)
    assert d["f1_none"] == pytest.approx(0.0)
    assert d["n_taxa_all"] == 2
    assert d["n_taxa_living"] == 1


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
    parts = [TaxonomicRankAccuracy("order", list(range(5)), 3, values=list("abcde")) for _ in range(3)]
    for part, idx in zip(parts, np.array_split(np.arange(300), 3), strict=True):
        part.update(gt_rows[idx], probs[idx], live=live[idx], coral=coral[idx])
    merged = parts[0].clone()
    merged.merge(parts[1])
    merged.merge(parts[2])
    assert merged.to_dict(include_detail=True) == whole.to_dict(include_detail=True)
    assert whole.total_coral > 0


@pytest.mark.parametrize("seed", [0, 1, 2])
def test_living_and_coral_scopes_match_sklearn(seed):
    from sklearn.metrics import f1_score

    rng = np.random.default_rng(seed)
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
    rank = TaxonomicRankAccuracy("genus", [0, 1, 2], none_local_index=2, values=["acropora", "porites", "none"])
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
        rank.update(np.array([[2, 1, 1]]), np.array([[1, 0, 0]], np.float32), live=np.array([True, True]))
