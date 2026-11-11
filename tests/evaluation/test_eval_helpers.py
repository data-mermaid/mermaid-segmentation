"""Tests for GT mapping and SVM folding."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import torch

_REPO_ROOT = Path(__file__).resolve().parents[2]
_CSV = _REPO_ROOT / "configs" / "class_to_concepts.csv"


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


def test_fish_concept_and_taxonomy():
    """Fish is a binary concept; unknown ranks below phylum are masked."""
    import pandas as pd

    from mermaidseg.dataset_reconciliation.concept_schema import ConceptSchema

    df = pd.read_csv(_CSV)
    df["source_label_class_name"] = df["source_label_class_name"].astype(str).str.lower()
    df["source_dataset_source"] = df["source_dataset_source"].astype(str).str.lower()
    df["target_label_class_name"] = df["target_label_class_name"].fillna("").astype(str).str.lower()
    sources = set(df["source_dataset_source"])
    schema = ConceptSchema.from_csv(_CSV, sources=sources)
    names = list(schema.channel_names)
    assert "fish" in names
    fish_i = names.index("fish")
    phylum_i = names.index("phylum__chordata")
    rank_idx = {
        rank: [i for i, name in enumerate(names) if name.startswith(f"{rank}__")]
        for rank in ("class", "order", "family", "genus")
    }
    assert all(rank_idx.values())

    fish_sources = {"coralscapes_v2", "coralscapes", "coralnet", "catlin_seaview"}
    seen_fish = set()
    for source in fish_sources:
        row = schema.row_for(source, "fish")
        assert int(row[fish_i]) == 2
        assert int(row[phylum_i]) == 2
        for idx in rank_idx.values():
            assert np.all(row[idx] == 0)
        seen_fish.add(source)
    assert seen_fish == fish_sources

    for source, label in (
        ("coralnet", "damselfish turf + sediments"),
        ("coralnet", "fishing gear"),
        ("catlin_seaview", "fishing gear"),
        ("mermaid", "fishing net"),
        ("coralnet", "crown of thorns starfish"),
        ("mermaid", "tunicate"),
        ("coralnet", "tunicate"),
    ):
        assert int(schema.row_for(source, label)[fish_i]) == 1, (source, label)

    other = df[
        df["source_label_class_name"].eq("other") | df["target_label_class_name"].eq("other")
    ]
    assert len(other) > 0
    for source, label in zip(
        other["source_dataset_source"], other["source_label_class_name"], strict=True
    ):
        assert int(schema.row_for(source, label)[fish_i]) == 0, (source, label)


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
