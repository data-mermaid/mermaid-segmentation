"""Grouped metrics on a per-point dump must equal the streaming MetricBundle metrics."""

import numpy as np
import pandas as pd
import pytest

from mermaidseg.evaluation.metrics import ConceptLayout, MetricBundle
from mermaidseg.evaluation.point_dump import (
    binary_metrics,
    class_metrics,
    confusions,
    point_records,
    summary_by,
    taxon_metrics,
)

CONCEPTS = [
    "kingdom__animalia", "kingdom__plantae", "kingdom__none",
    "class__anthozoa", "class__hydrozoa", "class__none",
    "genus__acropora", "genus__porites", "genus__millepora", "genus__none",
    "bleached", "live",
]
ID2LABEL = {0: "background", 1: "acropora", 2: "porites", 3: "sand", 4: "turf"}


def _softmax_groups(x, layout):
    out = x.copy()
    for _, idx, _ in layout.taxo:
        e = np.exp(x[:, idx] - x[:, idx].max(axis=1, keepdims=True))
        out[:, idx] = e / e.sum(axis=1, keepdims=True)
    out[:, layout.binary_indices] = 1 / (1 + np.exp(-x[:, layout.binary_indices]))
    return out


def _random_gt_rows(rng, n, layout):
    rows = np.zeros((n, len(CONCEPTS)), np.int8)
    for _, idx, _ in layout.taxo:
        given = rng.random(n) < 0.8
        rows[np.ix_(given, idx)] = 1
        pick = rng.integers(0, len(idx), n)
        rows[given, np.array(idx)[pick[given]]] = 2
    rows[:, layout.binary_indices] = rng.integers(0, 3, (n, len(layout.binary_indices)))
    return rows


@pytest.fixture(scope="module")
def data():
    rng = np.random.default_rng(0)
    layout = ConceptLayout.from_concept_names(CONCEPTS)
    n = 3000
    class_probs = rng.dirichlet(np.ones(len(ID2LABEL)), n).astype(np.float32)
    concept_probs = _softmax_groups(rng.normal(size=(n, len(CONCEPTS))) * 2, layout).astype(np.float32)
    gt_class = rng.integers(0, len(ID2LABEL), n)
    gt_rows = _random_gt_rows(rng, n, layout)
    units = rng.choice(["s1", "s2", "s3"], n)
    rec = point_records(class_probs, concept_probs, gt_class, gt_rows, layout, ID2LABEL)
    df = pd.DataFrame({"source_id": units, **rec})
    for c in ["gt_mermaid", "pred_mermaid"] + [f"{p}_{r}" for r, _, _ in layout.taxo for p in ("gt", "pred")]:
        df[c] = df[c].astype("category")
    bundles = {}
    for u in ["s1", "s2", "s3"]:
        m = units == u
        b = MetricBundle.create(max(ID2LABEL) + 1, layout)
        b.update(gt_class[m], class_probs[m].argmax(1), gt_rows[m], concept_probs[m])
        bundles[u] = b
    pooled = MetricBundle.create(max(ID2LABEL) + 1, layout)
    for b in bundles.values():
        pooled.merge(b)
    return df, bundles, pooled, layout


def _check_row(row, ref):
    assert row["num_points"] == ref["num_points"]
    assert row["class_accuracy"] == pytest.approx(ref["class_accuracy"])
    assert row["class_miou"] == pytest.approx(ref["class_miou"])
    for rank, rd in ref["taxonomic"].items():
        assert row[f"{rank}_acc_all"] == pytest.approx(rd["acc_all"])
        assert row[f"{rank}_acc_living"] == pytest.approx(rd["acc_living"])
        assert row[f"{rank}_n_all"] == rd["n_all"]
    assert row["binary_macro_accuracy"] == pytest.approx(ref["binary"]["macro_accuracy"])
    assert row["binary_macro_f1"] == pytest.approx(ref["binary"]["macro_f1"])


def test_summary_pooled_matches_metric_bundle(data):
    df, _, pooled, _ = data
    s = summary_by(df)
    assert len(s) == 1
    _check_row(s.iloc[0], pooled.to_dict())


def test_summary_per_unit_matches_metric_bundle(data):
    df, bundles, _, _ = data
    s = summary_by(df, "source_id").set_index("source_id")
    for u, b in bundles.items():
        _check_row(s.loc[u], b.to_dict())


def test_class_metrics_match_confusion(data):
    df, _, pooled, _ = data
    cm = class_metrics(df).set_index("mermaid_class")
    ref = pooled.class_confusion.to_dict(id2name=ID2LABEL)["per_class"]
    for name, r in ref.items():
        if r["gt_count"] == 0 and r["pred_count"] == 0:
            assert name not in cm.index
            continue
        assert cm.loc[name, "n_gt"] == r["gt_count"]
        assert cm.loc[name, "n_pred"] == r["pred_count"]
        assert cm.loc[name, "tp"] == r["tp"]
        assert cm.loc[name, "iou"] == pytest.approx(r["iou"])


def test_binary_metrics_match(data):
    df, _, pooled, _ = data
    bm = binary_metrics(df).set_index("concept")
    for name, r in pooled.binary.per_concept().items():
        for k in ("tp", "fp", "fn", "tn", "n_not_given"):
            assert bm.loc[name, k] == r[k]
        assert bm.loc[name, "f1"] == pytest.approx(r["f1"], nan_ok=True)


def test_taxon_metrics_sum_to_rank_accuracy(data):
    df, _, pooled, _ = data
    for t in pooled.taxo:
        tm = taxon_metrics(df, t.rank)
        assert tm["n_gt"].sum() == t.total_all
        assert tm["tp"].sum() == t.correct_all
        assert tm["n_pred"].sum() == t.total_all
        per = taxon_metrics(df, t.rank, by="source_id")
        assert per["tp"].sum() == t.correct_all


def test_confusions_and_finest_rank(data):
    df, _, pooled, _ = data
    c = confusions(df, "genus")
    assert (c["gt"].astype(str) != c["pred"].astype(str)).all()
    assert c["n"].sum() == pooled.taxo[-1].total_all - pooled.taxo[-1].correct_all
    has_genus = df["gt_genus"].notna() & (df["gt_genus"].astype(str) != "none")
    assert (df.loc[has_genus, "gt_finest_rank"] == "genus").all()
