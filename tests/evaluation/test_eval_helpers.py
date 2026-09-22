"""Tests for GT mapping, benthos spec/scoring, and SVM folding."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import torch

_REPO_ROOT = Path(__file__).resolve().parents[2]
_CSV = _REPO_ROOT / "configs" / "class_to_concepts.csv"
_BENTHOS_SPEC = _REPO_ROOT / "configs" / "eval" / "benthos_zero_shot.yaml"


def test_concept_lookup_matches_schema():
    """build_concept_lookup must reproduce ConceptSchema encoding exactly."""
    from mermaidseg.dataset_reconciliation.concept_schema import ConceptSchema
    from mermaidseg.evaluation.gt_mapping import build_concept_lookup

    sources = {"coralscapes_v2", "pacific_labeled_corals", "benthos_yuval"}
    schema = ConceptSchema.from_csv(_CSV, sources=sources)
    names = list(schema.channel_names)
    import pandas as pd

    df = pd.read_csv(_CSV)
    df["source_label_class_name"] = df["source_label_class_name"].astype(str).str.lower()
    df["source_dataset_source"] = df["source_dataset_source"].astype(str).str.lower()
    checked = 0
    for src in sources:
        labels = df[df["source_dataset_source"] == src]["source_label_class_name"].unique().tolist()
        lk = build_concept_lookup(src, labels, names, _CSV)
        for lab in labels:
            ref = schema.row_for(src, lab).astype(np.int8)
            got = lk.row(lab)
            assert np.array_equal(ref, got), (src, lab)
            checked += 1
    assert checked > 100


def test_class_lookup_direct_and_rollup():
    from mermaidseg.evaluation.gt_mapping import build_class_lookup

    id2label = {0: "ignore", 1: "acropora", 2: "hard coral", 3: "background", 4: "sand"}
    # Pacific: 'acropora'->'acropora' (direct); 'other scleractinians'->'hard coral' (direct);
    # 'unclear'->'background'; unknown label -> 0.
    hierarchy = {"acropora": "hard coral", "hard coral": None, "background": None}
    names = ["acropora", "other scleractinians", "unclear", "sand", "totally unknown label"]
    lk = build_class_lookup("pacific_labeled_corals", names, id2label, hierarchy)
    assert lk.class_id("acropora") == 1
    assert lk.class_id("other scleractinians") == 2
    assert lk.class_id("unclear") == 3
    assert lk.class_id("sand") == 4
    assert lk.class_id("totally unknown label") == 0


def test_class_lookup_rollup_to_parent():
    from mermaidseg.evaluation.gt_mapping import build_class_lookup

    # 'acropora' not in id2label -> roll up to parent 'hard coral' which is.
    id2label = {0: "ignore", 1: "hard coral"}
    hierarchy = {"acropora": "hard coral", "hard coral": None}
    lk = build_class_lookup("pacific_labeled_corals", ["acropora"], id2label, hierarchy)
    assert lk.class_id("acropora") == 1


def test_benthos_spec_load_and_validate():
    from mermaidseg.evaluation.benthos_eval import (
        _build_gt_name_to_spec,
        _validate_spec,
        load_spec,
    )
    from mermaidseg.model.concept_expr import ConceptResolver

    classes = load_spec(_BENTHOS_SPEC)
    names = [
        "class__hexacorallia",
        "class__octocorallia",
        "phylum__porifera",
        "live",
        "sand",
        "hard_substrate",
        "anthropogenic",
        "algae",
        "calcifying",
    ]
    _validate_spec(classes, ConceptResolver(names))  # should not raise
    gt = _build_gt_name_to_spec(classes)
    assert gt["coral"] == 0 and gt["sand"] == 2

    with pytest.raises(ValueError, match="not present in this model"):
        _validate_spec(classes, ConceptResolver(["live", "sand"]))


def test_benthos_scoring_and_nodata():
    from mermaidseg.evaluation.benthos_eval import load_spec
    from mermaidseg.evaluation.metrics import ClassConfusion
    from mermaidseg.evaluation.predictor import CBMPredictor
    from mermaidseg.model.concept_expr import ConceptResolver, evaluate as eval_expr

    classes = load_spec(_BENTHOS_SPEC)
    names = [
        "class__hexacorallia",
        "class__octocorallia",
        "phylum__porifera",
        "live",
        "sand",
        "hard_substrate",
        "anthropogenic",
        "algae",
        "calcifying",
    ]
    resolver = ConceptResolver(names)
    k, hh = len(names), 4
    cp = np.zeros((k, hh, hh), np.float32)
    cp[names.index("class__hexacorallia"), :2, :2] = 0.9
    cp[names.index("live"), :2, :2] = 0.9
    cp[names.index("sand"), 2:, 2:] = 0.95
    score = np.stack([eval_expr(e["expr"], cp, resolver) for e in classes], axis=0)
    up = CBMPredictor.upsample_scores(torch.from_numpy(score), 8, 8)
    pred = up.argmax(0).numpy()

    gt = np.full((8, 8), -1, np.int64)
    gt[:4, :4] = 0  # Hard Coral
    gt[4:, 4:] = 2  # Sand
    nodata = np.zeros((8, 8), bool)
    nodata[0, 0] = True
    valid = (gt >= 0) & (~nodata)
    cm = ClassConfusion(len(classes), ignore_index=None)
    cm.update(gt[valid], pred[valid])
    assert int(valid.sum()) == 31
    assert cm.accuracy() == 1.0


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
