"""GT mapping: concept and class lookups built from the concept table."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from mermaidseg.evaluation.gt_mapping import build_concept_lookup

_REPO_ROOT = Path(__file__).resolve().parents[3]
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


@pytest.mark.parametrize(
    ("label", "live", "coral"),
    [
        ("porites", True, True),
        ("dead pocillopora", False, False),
        ("a. galapagensis (alive)", True, True),
        ("millepora", True, True),
        ("cred-octocoral", True, True),
        ("palythoa", True, False),
        ("macroalgae", True, False),  # live column ignored outside corals / echinoderms
    ],
)
def test_point_scopes_from_the_concept_table(label, live, coral):
    lookup = build_concept_lookup("coralnet", [label], [], _CSV)
    assert lookup.scopes(label) == (live, coral)
