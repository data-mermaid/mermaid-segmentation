"""Calibration and rare-taxon metrics of HierarchicalMetrics, with hand-counted pixels."""

from __future__ import annotations

import math

import pytest
import torch

from mermaidseg.dataset_reconciliation.taxonomy_tree import TaxonomyTree
from mermaidseg.model.hierarchical_metrics import HierarchicalMetrics
from mermaidseg.model.rank_outputs import RankBatch


def _tree(channels: list[str], rank: str = "kingdom") -> TaxonomyTree:
    return TaxonomyTree(ranks=(rank,), channels={rank: channels}, edges={})


def _batch(
    scores: torch.Tensor, target: list[int], pred: list[int], rank: str = "kingdom"
) -> dict[str, RankBatch]:
    """One rank's batch over ``len(target)`` pixels laid out as (1, 1, W); target 0 is unlabelled."""
    target_t = torch.tensor(target).reshape(1, 1, -1)
    return {
        rank: RankBatch(
            rank=rank,
            scores=scores,
            target=target_t,
            pred=torch.tensor(pred).reshape(1, 1, -1),
            valid=target_t != 0,
        )
    }


def _one_hot_scores(pred: list[int], num_channels: int) -> torch.Tensor:
    scores = torch.zeros(1, num_channels, 1, len(pred))
    for pixel, dense in enumerate(pred):
        scores[0, dense - 1, 0, pixel] = 1.0
    return scores


# Channels a and none, four bins. Confidences .9 (bin 3), .6 and .7 (bin 2); pixel 3 is unlabelled.
#   pixel 0: GT a, predicted a (correct)         pixel 1: GT none, predicted a (wrong)
#   pixel 2: GT none, predicted none (correct)   pixel 3: no GT
CALIBRATION_CHANNELS = ["a", "none"]
CALIBRATION_SCORES = torch.tensor([[[[0.9, 0.6, 0.3, 0.2]], [[0.1, 0.4, 0.7, 0.8]]]])
CALIBRATION_TARGET = [1, 2, 2, 0]
CALIBRATION_PRED = [1, 1, 2, 2]


def _calibrated(**kwargs) -> HierarchicalMetrics:
    return HierarchicalMetrics(_tree(CALIBRATION_CHANNELS), n_bins=4, **kwargs)


def _calibration_batch() -> dict[str, RankBatch]:
    return _batch(CALIBRATION_SCORES, CALIBRATION_TARGET, CALIBRATION_PRED)


def test_calibration_bins_confidence_and_reports_ece_per_ground_truth_subset():
    metrics = _calibrated()

    metrics.update(_calibration_batch())
    out = {k: v for k, v in metrics.compute().items() if k.startswith(("ece", "confidence"))}

    # All 3 labelled pixels: bin 3 holds (.9, correct); bin 2 holds (.6, wrong) and (.7, correct),
    # mean confidence .65 against accuracy .5. ECE = 1/3 * .1 + 2/3 * .15 = 2/15.
    expected = {
        "ece/kingdom": 2 / 15,
        "confidence/kingdom": 2.2 / 3,
        "ece/kingdom/accuracy": 2 / 3,
        "ece/kingdom/n": 3.0,
        # Taxon GT is pixel 0 alone: confidence .9, correct.
        "ece/kingdom/taxon_gt": 0.1,
        "ece/kingdom/taxon_gt/confidence": 0.9,
        "ece/kingdom/taxon_gt/accuracy": 1.0,
        "ece/kingdom/taxon_gt/n": 1.0,
        # 'none' GT is pixels 1 and 2, both in bin 2.
        "ece/kingdom/none_gt": 0.15,
        "ece/kingdom/none_gt/confidence": 0.65,
        "ece/kingdom/none_gt/accuracy": 0.5,
        "ece/kingdom/none_gt/n": 2.0,
        "ece/mean": 2 / 15,
        "ece/mean/taxon_gt": 0.1,
        "ece/mean/none_gt": 0.15,
    }
    assert out == {key: pytest.approx(value, rel=1e-5) for key, value in expected.items()}


def test_calibration_puts_a_confidence_of_one_in_the_last_bin():
    metrics = HierarchicalMetrics(_tree(["a", "b"]), n_bins=4)
    scores = torch.tensor([[[[1.0]], [[0.0]]]])

    metrics.update(_batch(scores, target=[1], pred=[1]))
    out = metrics.compute()

    assert out["ece/kingdom"] == pytest.approx(0.0)
    assert out["ece/kingdom/n"] == 1.0


def test_calibration_omits_a_ground_truth_subset_with_no_pixels():
    metrics = HierarchicalMetrics(_tree(["a", "b"]), n_bins=4)
    scores = torch.tensor([[[[0.9]], [[0.1]]]])

    metrics.update(_batch(scores, target=[1], pred=[1]))
    out = metrics.compute()

    assert not [key for key in out if "none_gt" in key]
    assert out["ece/kingdom/taxon_gt/n"] == 1.0


@pytest.mark.parametrize(
    ("transform", "scores"),
    [
        pytest.param("none", [0.75, 0.25], id="none_keeps_probabilities"),
        pytest.param("softmax", [math.log(3.0), 0.0], id="softmax_of_logits"),
        pytest.param("renormalize", [3.0, 1.0], id="renormalize_positive_scores"),
    ],
)
def test_calibration_confidence_follows_the_score_transform(transform, scores):
    metrics = HierarchicalMetrics(_tree(["a", "b"]), n_bins=4, score_transform=transform)

    metrics.update(_batch(torch.tensor(scores).reshape(1, 2, 1, 1), target=[1], pred=[1]))

    assert metrics.compute()["confidence/kingdom"] == pytest.approx(0.75)


# Taxa a, b, c, d and none. Supports and hits: a 4 pixels / 3 right, b 2 / 2, c 1 / 0, d unseen,
# 'none' 1 / 1. One unlabelled pixel is ignored.
RARE_CHANNELS = ["a", "b", "c", "d", "none"]
RARE_TARGET = [1, 1, 1, 1, 2, 2, 3, 5, 0]
RARE_PRED = [1, 1, 1, 2, 2, 2, 1, 5, 1]


def _rare_metrics(**kwargs) -> HierarchicalMetrics:
    metrics = HierarchicalMetrics(_tree(RARE_CHANNELS), **kwargs)
    scores = _one_hot_scores(RARE_PRED, num_channels=len(RARE_CHANNELS))
    metrics.update(_batch(scores, RARE_TARGET, RARE_PRED))
    return metrics


def _rare(metrics: HierarchicalMetrics) -> dict[str, float]:
    return {k: v for k, v in metrics.compute().items() if k.startswith("rare/")}


def test_rare_taxon_metrics_split_seen_taxa_into_head_and_tail_by_support():
    # Seen taxa a (support 4, acc .75), b (2, 1.0), c (1, 0.0); 'none' and unseen d are left out.
    # Support quantiles of [1, 2, 4]: 0.5 -> 2, 0.8 -> 3.2, so tail is {b, c} and head is {a}.
    out = _rare(_rare_metrics())

    assert out == {
        "rare/kingdom/n_taxa_seen": 3.0,
        "rare/kingdom/macro_acc": pytest.approx(1.75 / 3),
        "rare/kingdom/tail_macro_acc": pytest.approx(0.5),
        "rare/kingdom/tail_n_taxa": 2.0,
        "rare/kingdom/head_macro_acc": pytest.approx(0.75),
        "rare/kingdom/head_n_taxa": 1.0,
        "rare/kingdom/head_tail_gap": pytest.approx(0.25),
    }


def test_rare_taxon_metrics_ignore_taxa_below_min_support():
    # min_support 2 keeps a and b. Support quantiles of [2, 4]: 0.5 -> 3, 0.8 -> 3.6.
    out = _rare(_rare_metrics(min_support=2))

    assert out == {
        "rare/kingdom/n_taxa_seen": 2.0,
        "rare/kingdom/macro_acc": pytest.approx(0.875),
        "rare/kingdom/tail_macro_acc": pytest.approx(1.0),
        "rare/kingdom/tail_n_taxa": 1.0,
        "rare/kingdom/head_macro_acc": pytest.approx(0.75),
        "rare/kingdom/head_n_taxa": 1.0,
        "rare/kingdom/head_tail_gap": pytest.approx(-0.25),
    }


def test_rare_taxon_strata_include_taxa_exactly_on_the_cuts():
    # Supports 1..5 with quantiles 0.25 and 0.75 cut at 2 and 4: tail {a, b}, mid {c}, head {d, e}.
    channels = ["a", "b", "c", "d", "e"]
    target = [1] * 1 + [2] * 2 + [3] * 3 + [4] * 4 + [5] * 5
    # Taxon b is right once in two, d three times in four; the rest are all right.
    pred = [1] + [2, 1] + [3] * 3 + [4, 4, 4, 1] + [5] * 5
    metrics = HierarchicalMetrics(_tree(channels), tail_quantile=0.25, head_quantile=0.75)

    metrics.update(_batch(_one_hot_scores(pred, 5), target, pred))
    out = _rare(metrics)

    assert {k: v for k, v in out.items() if "n_taxa" in k or "macro_acc" in k} == {
        "rare/kingdom/n_taxa_seen": 5.0,
        "rare/kingdom/macro_acc": pytest.approx((1.0 + 0.5 + 1.0 + 0.75 + 1.0) / 5),
        "rare/kingdom/tail_macro_acc": pytest.approx(0.75),
        "rare/kingdom/tail_n_taxa": 2.0,
        "rare/kingdom/mid_macro_acc": pytest.approx(1.0),
        "rare/kingdom/mid_n_taxa": 1.0,
        "rare/kingdom/head_macro_acc": pytest.approx(0.875),
        "rare/kingdom/head_n_taxa": 2.0,
    }


def test_reset_clears_accumulated_counts_so_a_reused_metric_matches_a_fresh_one():
    metrics = _calibrated()
    metrics.update(_calibration_batch())
    first = metrics.compute()

    metrics.reset()
    assert metrics.compute() == {}
    metrics.update(_calibration_batch())

    assert metrics.compute() == first
    assert first["ece/kingdom/n"] == 3.0


def test_update_ignores_ranks_the_metric_is_not_scoring():
    metrics = HierarchicalMetrics(_tree(["a", "b"], rank="phylum"), ranks=("phylum",))
    scores = torch.tensor([[[[0.9]], [[0.1]]]])

    metrics.update(_batch(scores, target=[1], pred=[1], rank="kingdom"))

    assert metrics.compute() == {}


def _available_devices() -> list[str]:
    """CUDA and MPS when present; CPU is covered by the calibration tests."""
    devices: list[str] = []
    if torch.cuda.is_available():
        devices.append("cuda")
    if torch.backends.mps.is_available():
        devices.append("mps")
    return devices


def test_cpu_accumulators_are_float64():
    metrics = _calibrated()

    metrics.update(_calibration_batch())

    # float32 stops resolving unit increments past 2**24 pixels in one bin.
    for accumulator in (metrics._bin_count, metrics._bin_conf, metrics._bin_correct):
        assert accumulator["kingdom"].dtype == torch.float64


@pytest.mark.parametrize("device", _available_devices())
def test_accumulation_on_the_batch_device_gives_the_cpu_values(device):
    metrics = _calibrated()
    batch = {
        rank: RankBatch(
            rank=rank,
            scores=b.scores.to(device),
            target=b.target.to(device),
            pred=b.pred.to(device),
            valid=b.valid.to(device),
        )
        for rank, b in _calibration_batch().items()
    }

    metrics.update(batch)
    out = metrics.compute()

    assert out["ece/kingdom"] == pytest.approx(2 / 15, rel=1e-5)
    assert out["ece/kingdom/none_gt/n"] == 2.0
