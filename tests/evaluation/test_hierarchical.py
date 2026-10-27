"""HierarchicalScore: point/pixel concept outputs -> hierarchical metrics."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest
import torch

from mermaidseg.evaluation.gt_mapping import build_concept_lookup
from mermaidseg.evaluation.hierarchical import HierarchicalScore

_REPO = Path(__file__).resolve().parents[2]
_CSV = _REPO / "configs" / "class_to_concepts.csv"


def _concept_names() -> list[str]:
    raw = json.loads((_REPO / "demo" / "concept_id2name.json").read_text())
    return [name for _, name in sorted(raw.items(), key=lambda kv: int(kv[0]))]


def _rows(names: list[str], labels: list[str]) -> torch.Tensor:
    lookup = build_concept_lookup("coralscapes_v2", labels, names, _CSV)
    return torch.from_numpy(np.stack([lookup.row(label) for label in labels]))


def test_dense_targets_follow_the_concept_rows():
    names = _concept_names()
    score = HierarchicalScore(names, _CSV)
    assert score.ranks == ("kingdom", "phylum", "class", "order", "family", "genus")
    rows = torch.cat(
        [_rows(names, ["acropora alive"]), torch.zeros(1, len(names), dtype=torch.int8)]
    )
    targets = score.dense_targets(rows)
    assert targets.shape == (len(score.ranks), 2)
    assert (targets[:, 1] == 0).all()
    genus = score.ranks.index("genus")
    first_genus = next(i for i, n in enumerate(names) if n.startswith("genus__"))
    assert int(targets[genus, 0]) == names.index("genus__acropora") - first_genus + 1


def test_perfect_predictions_are_consistent_and_calibrated():
    names = _concept_names()
    score = HierarchicalScore(names, _CSV)
    rows = _rows(names, ["acropora alive", "sand"])
    probs = (rows == 2).float().T.contiguous()  # (K, N): all mass on the GT channel per rank
    score.update(probs, score.dense_targets(rows))
    out = score.to_dict()
    assert out["consistency/overall"] == 1.0
    assert out["ece/mean"] == pytest.approx(0.0)
    assert "severity/genus" not in out  # no mistakes to score


def test_sibling_genus_mistake_has_distance_one():
    names = _concept_names()
    score = HierarchicalScore(names, _CSV)
    rows = _rows(names, ["acropora alive"])
    probs = (rows == 2).float().T.contiguous()
    probs[names.index("genus__acropora"), 0] = 0.0
    probs[names.index("genus__montipora"), 0] = 1.0
    score.update(probs, score.dense_targets(rows))
    out = score.to_dict()
    assert out["severity/genus"] == 1.0
    assert out["consistency/family->genus"] == 1.0


def test_channels_that_do_not_match_the_taxonomy_raise():
    names = [n for n in _concept_names() if n != "genus__acropora"]
    with pytest.raises(ValueError, match="genus"):
        HierarchicalScore(names, _CSV)
