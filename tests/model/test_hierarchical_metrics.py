"""HierarchicalMetrics on a toy order -> family -> genus taxonomy."""

from __future__ import annotations

import pandas as pd
import pytest
import torch

from mermaidseg.dataset_reconciliation.taxonomy_tree import TaxonomyTree, build_taxonomy_tree
from mermaidseg.model.hierarchical_metrics import HierarchicalMetrics
from mermaidseg.model.rank_outputs import RankBatch

RANKS = ("order", "family", "genus")
# favites has no recorded parent; none is a channel but not a taxon.
PATHS = [
    ("scleractinia", "acroporidae", "acropora"),
    ("scleractinia", "acroporidae", "montipora"),
    ("scleractinia", "poritidae", "porites"),
    ("alcyonacea", "alcyoniidae", "sarcophyton"),
    ("not_given", "not_given", "favites"),
    ("none", "none", "none"),
]
# Channel order is sorted: genus = acropora, favites, montipora, none, porites, sarcophyton.


def _tree() -> TaxonomyTree:
    rows = [{"source_dataset_source": "toy", **dict(zip(RANKS, p, strict=True))} for p in PATHS]
    return build_taxonomy_tree(ranks=RANKS, df=pd.DataFrame(rows))


TREE = _tree()


def _batch(rank: str, scores: list[list[float]], targets: list[str | None]) -> RankBatch:
    """One pixel per score row; the prediction is the argmax, a ``None`` target is not_given."""
    index = TREE.index_of(rank)
    s = torch.tensor(scores, dtype=torch.float64).T.reshape(1, len(index), 1, len(scores))
    target = torch.tensor([index[t] + 1 if t else 0 for t in targets]).reshape(1, 1, -1)
    return RankBatch(rank, s, target, s.argmax(dim=1) + 1, target.ne(0))


def _onehot(rank: str, preds: list[str]) -> list[list[float]]:
    channels = TREE.channels[rank]
    return [[1.0 if c == p else 0.0 for c in channels] for p in preds]


def _predicted(rank: str, preds: list[str], targets: list[str | None]) -> RankBatch:
    return _batch(rank, _onehot(rank, preds), targets)


def _subset(out: dict[str, float], prefix: str) -> dict[str, float]:
    return {k: v for k, v in out.items() if k.startswith(prefix)}


def test_ece_bins_the_gap_between_confidence_and_accuracy():
    metrics = HierarchicalMetrics(TREE)
    scores = [
        [0.9, 0.0, 0.05, 0.0, 0.05, 0.0],  # acropora, correct: bin 13
        [0.05, 0.0, 0.9, 0.0, 0.05, 0.0],  # montipora, wrong: bin 13
        [0.2, 0.1, 0.2, 0.0, 0.3, 0.2],  # porites, correct: bin 4
        [0.7, 0.0, 0.1, 0.1, 0.1, 0.0],  # acropora on a 'none' pixel, wrong: bin 10
    ]
    targets = ["acropora", "acropora", "porites", "none"]
    metrics.update({"genus": _batch("genus", scores, targets)})
    out = metrics.compute()
    # all: bin 13 |0.5 - 0.9| * 2/4 + bin 4 |1 - 0.3| * 1/4 + bin 10 |0 - 0.7| * 1/4
    # taxon_gt: bin 13 0.4 * 2/3 + bin 4 0.7 * 1/3; none_gt: bin 10 0.7
    assert {k: out[k] for k in ("ece/genus", "ece/genus/taxon_gt", "ece/genus/none_gt")} == (
        pytest.approx({"ece/genus": 0.55, "ece/genus/taxon_gt": 0.5, "ece/genus/none_gt": 0.7})
    )
    assert out["ece/mean"] == pytest.approx(0.55)


def test_severity_is_normalized_by_the_rank_depth_plus_one():
    metrics = HierarchicalMetrics(TREE)
    # acropora -> porites meet at order (distance 2); acropora -> montipora are siblings (1).
    batch = _predicted(
        "genus", ["porites", "montipora", "acropora"], ["acropora", "acropora", "acropora"]
    )
    metrics.update({"genus": batch})
    out = metrics.compute()
    # Genus is rank index 2, so the maximum distance is 3.
    keys = ("severity/genus", "severity/genus/normalized", "severity/mean_normalized")
    assert {k: out[k] for k in keys} == pytest.approx(
        {
            "severity/genus": 1.5,
            "severity/genus/normalized": 0.5,
            "severity/mean_normalized": 0.5,
        }
    )


# Baseline pixels: acroporidae/acropora is consistent, poritidae/acropora is not.
@pytest.mark.parametrize(
    ("family", "genus", "expected"),
    [
        (
            "acroporidae",
            "none",
            {"ok": 0.5, "trunc_inc": 1 / 3, "frac": 1 / 3, "trunc": 1, "nn": 0, "np": 0},
        ),
        ("none", "none", {"ok": 0.5, "trunc_inc": 0.5, "frac": 0.0, "trunc": 0, "nn": 1, "np": 0}),
        (
            "acroporidae",
            "favites",
            {"ok": 0.5, "trunc_inc": 0.5, "frac": 0.0, "trunc": 0, "nn": 0, "np": 1},
        ),
    ],
    ids=["truncated", "none-none", "no-parent"],
)
def test_consistency_sets_aside_truncated_none_none_and_parentless_pixels(family, genus, expected):
    metrics = HierarchicalMetrics(TREE)
    families = ["acroporidae", "poritidae", family]
    genera = ["acropora", "acropora", genus]
    metrics.update(
        {
            "family": _predicted("family", families, families),
            "genus": _predicted("genus", genera, genera),
        }
    )
    key = "consistency/family->genus"
    assert _subset(metrics.compute(), key) == pytest.approx(
        {
            key: expected["ok"],
            f"{key}/truncated_as_inconsistent": expected["trunc_inc"],
            f"{key}/truncated_frac": expected["frac"],
            f"{key}/n_scored": 2.0,
            f"{key}/n_truncated": float(expected["trunc"]),
            f"{key}/n_none_none": float(expected["nn"]),
            f"{key}/n_no_parent": float(expected["np"]),
        }
    )


def test_consistency_overall_labeled_counts_only_pixels_with_fine_ground_truth():
    metrics = HierarchicalMetrics(TREE)
    families = ["acroporidae", "poritidae", "acroporidae", "acroporidae"]
    genera = ["acropora", "acropora", "acropora", "none"]
    # The last two pixels (one consistent, one truncated) have no genus ground truth.
    metrics.update(
        {
            "family": _predicted("family", families, families),
            "genus": _predicted("genus", genera, ["acropora", "acropora", None, None]),
        }
    )
    out = metrics.compute()
    assert out["consistency/overall"] == pytest.approx(2 / 3)
    assert _subset(out, "consistency/overall_labeled") == pytest.approx(
        {
            "consistency/overall_labeled": 0.5,
            "consistency/overall_labeled/truncated_as_inconsistent": 0.5,
            "consistency/overall_labeled/truncated_frac": 0.0,
            "consistency/overall_labeled/n_scored": 2.0,
            "consistency/overall_labeled/n_truncated": 0.0,
        }
    )


def test_rare_taxon_strata_split_taxa_by_support_quantiles():
    metrics = HierarchicalMetrics(TREE)
    pairs = [
        *[("acropora", "acropora")] * 3,  # support 3, accuracy 1
        *[("montipora", "montipora")] * 2,  # support 3, accuracy 2/3
        ("montipora", "acropora"),
        ("porites", "porites"),  # support 2, accuracy 1/2
        ("porites", "acropora"),
        ("favites", "acropora"),  # support 1, accuracy 0
        ("sarcophyton", "acropora"),  # support 1, accuracy 0
        ("none", "none"),  # not a taxon: excluded
    ]
    targets, preds = zip(*pairs, strict=True)
    metrics.update({"genus": _predicted("genus", list(preds), list(targets))})
    # Supports [1, 1, 2, 3, 3]: median 2 (tail <= 2), 0.8 quantile 3 (head >= 3), no mid.
    assert _subset(metrics.compute(), "rare/genus") == pytest.approx(
        {
            "rare/genus/n_taxa_seen": 5.0,
            "rare/genus/macro_acc": 13 / 30,
            "rare/genus/tail_macro_acc": 1 / 6,
            "rare/genus/tail_n_taxa": 3.0,
            "rare/genus/head_macro_acc": 5 / 6,
            "rare/genus/head_n_taxa": 2.0,
            "rare/genus/head_tail_gap": 2 / 3,
        }
    )
