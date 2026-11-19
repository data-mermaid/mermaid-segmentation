"""ConceptSchema encoding of the concept table."""

from __future__ import annotations

from pathlib import Path

import numpy as np

_REPO_ROOT = Path(__file__).resolve().parents[3]
_CSV = _REPO_ROOT / "configs" / "class_to_concepts.csv"


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
