"""The evaluation summary: ``summary.json`` and one Markdown block per section."""

from __future__ import annotations

import json

import pytest

from mermaidseg.evaluation.run import REGISTRY
from mermaidseg.evaluation.summary import (
    _concept_lines,
    _point_lines,
    _val_loss_lines,
    write_summary,
)

_POOLED = {
    "class_accuracy": 0.5,
    "taxonomic": {
        "genus": {
            "acc_all": 0.8,
            "acc_living": 0.7,
            "macro_f1_all": 0.6,
            "macro_f1_living": 0.5,
            "f1_none": 0.4,
            "acc_coral": 0.3,
            "macro_f1_coral": 0.2,
            "n_coral": 9,
        }
    },
    "binary": {"macro_accuracy": 0.9, "macro_f1": 0.85},
}
_BLEACHED = {
    "accuracy": 0.1,
    "precision": 0.2,
    "recall": 0.3,
    "f1": 0.4,
    "n_true": 5,
    "n_false": 6,
    "n_not_given": 7,
    "n_unannotated": 0,
}
_HIERARCHICAL = {
    "consistency/overall": 0.91,
    "severity/mean_normalized": 0.12,
    "ece/mean": 0.03,
}


def test_concept_lines_report_each_reported_binary_concept_then_the_hierarchy():
    section = {"bleached": _BLEACHED, "hierarchical": _HIERARCHICAL}

    assert _concept_lines(section) == [
        "- bleached acc/precision/recall/F1: 0.1000/0.2000/0.3000/0.4000 "
        "(n_true=5 n_false=6 n_not_given=7 n_unannotated=0)",
        "- hierarchical: consistency 0.9100, severity (mean normalized) 0.1200, ECE (mean) 0.0300",
    ]


def test_concept_lines_skip_concepts_and_hierarchy_that_are_not_dicts():
    section = {"bleached": None, "anthropogenic": "n/a", "hierarchical": None}

    assert _concept_lines(section) == []


def test_point_lines_format_the_pooled_metrics_under_the_title():
    pooled = {**_POOLED, "bleached": _BLEACHED}

    assert _point_lines("Pooled title", pooled) == [
        "## Pooled title",
        "- class accuracy: 0.5000",
        "- taxonomic acc_all/acc_living: genus=0.8000/0.7000",
        "- taxonomic macro F1 all/living, none-class F1: genus=0.6000/0.5000, 0.4000",
        "- live coral taxonomic acc/macro F1: genus=0.3000/0.2000 (n=9)",
        "- binary macro acc/F1: 0.9000/0.8500",
        "- bleached acc/precision/recall/F1: 0.1000/0.2000/0.3000/0.4000 "
        "(n_true=5 n_false=6 n_not_given=7 n_unannotated=0)",
        "",
    ]


def test_point_lines_show_n_a_for_a_missing_binary_block_and_taxonomic_scores():
    pooled = {"class_accuracy": 0.5, "taxonomic": {"genus": {"acc_all": 0.8, "acc_living": 0.7}}}

    lines = _point_lines("T", pooled)

    assert "- binary macro acc/F1: n/a/n/a" in lines
    assert "- taxonomic macro F1 all/living, none-class F1: genus=n/a/n/a, n/a" in lines


_LOSS_SECTION = {
    "overall": {
        "loss": 1.0,
        "classification": 0.5,
        "multi_hot/bleached": 0.1,
        "genus": 0.2,
        "family": 0.3,
        "genus/accuracy": 0.9,
    },
    "per_dataset": {"coralnet": {"loss": 0.4, "multi_hot/bleached": 0.25, "other": 5.0}},
}


def test_val_loss_lines_list_headline_keys_then_ranks_then_each_dataset():
    assert _val_loss_lines(_LOSS_SECTION) == [
        "## Validation loss (overall)",
        "- loss: 1.0000",
        "- classification: 0.5000",
        "- multi_hot/bleached: 0.1000",
        "- ranks: genus=0.2000, family=0.3000",
        "- coralnet: loss=0.4000, multi_hot/bleached=0.2500",
        "",
    ]


def test_val_loss_lines_use_the_given_title_and_tolerate_an_empty_section():
    assert _val_loss_lines({}, "Custom") == ["## Custom (overall)", ""]


def test_write_summary_writes_the_summary_as_json(tmp_path):
    summary = {"bundle": "b", "split": "val", "coralnet_val": {"error": "X: y"}}

    write_summary(summary, tmp_path, REGISTRY)

    assert json.loads((tmp_path / "summary.json").read_text()) == summary


@pytest.mark.parametrize(
    ("key", "section", "expected_lines"),
    [
        (
            "coralnet_val",
            {"error": "RuntimeError: boom"},
            ["## coralnet_val: ERROR - RuntimeError: boom"],
        ),
        (
            "coralnet_train",
            {"pooled": _POOLED},
            ["## CoralNet (pooled over train sources)", "- class accuracy: 0.5000"],
        ),
        (
            "pacific",
            {"pooled": _POOLED},
            ["## Pacific Labeled Corals (pooled over regions)", "- class accuracy: 0.5000"],
        ),
        ("val_loss", _LOSS_SECTION, ["## Validation loss (overall)", "- loss: 1.0000"]),
        (
            "train_loss",
            _LOSS_SECTION,
            ["## CoralNet train-source loss (overall)", "- loss: 1.0000"],
        ),
        ("test_loss", _LOSS_SECTION, ["## Test loss (overall)", "- loss: 1.0000"]),
        (
            "coralscapes39_test",
            {
                "by_k": {
                    "50": {
                        "test": {"accuracy": 0.7, "miou": 0.4},
                        "probe": {"train_accuracy": 0.95, "num_points": 1950},
                    }
                }
            },
            [
                "## CoralscapesV2 39-class linear probe (test)",
                "- k=50: test accuracy 0.7000, mIoU 0.4000 (probe train acc 0.9500, 1950 pixels)",
            ],
        ),
        (
            "coralscapes_val",
            {
                "by_k": {
                    "10": {
                        "val": {"accuracy": 0.6, "miou": 0.3},
                        "probe": {"train_accuracy": 0.9, "num_points": 945},
                    }
                },
                "bleached": _BLEACHED,
            },
            [
                "## CoralscapesV2 95-class linear probe (val)",
                "- k=10: val accuracy 0.6000, mIoU 0.3000 (probe train acc 0.9000, 945 pixels)",
                "- bleached acc/precision/recall/F1: 0.1000/0.2000/0.3000/0.4000 "
                "(n_true=5 n_false=6 n_not_given=7 n_unannotated=0)",
            ],
        ),
    ],
)
def test_write_summary_renders_each_section_kind_under_its_own_heading(
    tmp_path, key, section, expected_lines
):
    write_summary({key: section}, tmp_path, REGISTRY)

    lines = (tmp_path / "summary.md").read_text().splitlines()
    assert lines[:2] == ["# CBM evaluation summary", ""]
    for expected in expected_lines:
        assert expected in lines


def test_write_summary_leaves_scalar_entries_out_of_the_markdown(tmp_path):
    summary = {"bundle": "outputs/b", "input_size": [64, 64], "num_removed_concepts": 3}

    write_summary(summary, tmp_path, REGISTRY)

    assert (tmp_path / "summary.md").read_text() == "# CBM evaluation summary\n"
