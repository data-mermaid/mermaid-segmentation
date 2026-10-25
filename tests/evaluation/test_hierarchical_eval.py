"""``evaluate.py --hierarchical``: point-eval hierarchical and taxonomy-loss metrics on a toy taxonomy."""

from __future__ import annotations

import math
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest
import torch

from mermaidseg.dataset_reconciliation.taxonomy_tree import build_taxonomy_tree
from mermaidseg.evaluation.gt_mapping import ClassLookup, build_concept_lookup
from mermaidseg.evaluation.hierarchical_eval import (
    HierarchicalSpec,
    concept_value2id_from_names,
)
from mermaidseg.evaluation.point_datasets import run_point_evaluation
from mermaidseg.evaluation.predictor import CBMPredictor
from mermaidseg.model.taxonomy_loss import TaxonomicCoherenceLoss

RANKS = ["kingdom", "phylum", "class", "order", "family", "genus"]
LINEAGES = {  # genus -> (class, order, family); kingdom animalia, phylum cnidaria
    "acropora": ("hexacorallia", "scleractinia", "acroporidae"),
    "montipora": ("hexacorallia", "scleractinia", "acroporidae"),
    "isopora": ("hexacorallia", "scleractinia", "acroporidae"),
    "porites": ("hexacorallia", "scleractinia", "poritidae"),
    "goniopora": ("hexacorallia", "scleractinia", "poritidae"),
    "sarcophyton": ("octocorallia", "alcyonacea", "alcyoniidae"),
    "sinularia": ("octocorallia", "alcyonacea", "alcyoniidae"),
}
# Per-channel probabilities of the fake model (each rank sums to 1), in sorted channel order.
PROBS = {
    "kingdom__animalia": 1.0,
    "phylum__cnidaria": 1.0,
    "class__hexacorallia": 0.8,
    "class__octocorallia": 0.2,
    "order__alcyonacea": 0.2,
    "order__scleractinia": 0.8,
    "family__acroporidae": 0.60,
    "family__alcyoniidae": 0.15,
    "family__poritidae": 0.25,
    "genus__acropora": 0.08,
    "genus__goniopora": 0.12,
    "genus__isopora": 0.04,
    "genus__montipora": 0.25,
    "genus__porites": 0.33,
    "genus__sarcophyton": 0.10,
    "genus__sinularia": 0.08,
    "live": 0.9,
}
CONCEPT_NAMES = list(PROBS)
SOURCE = "pacific_labeled_corals"


@pytest.fixture
def taxonomy_csv(tmp_path: Path) -> Path:
    rows = [
        {"kingdom": "animalia", "phylum": "cnidaria", "class": c, "order": o, "family": f, "genus": g,
         "source_label_class_name": g, "live": "true"}
        for g, (c, o, f) in LINEAGES.items()
    ]
    rows.append({"kingdom": "animalia", "phylum": "cnidaria", "class": "hexacorallia",
                 "order": "scleractinia", "family": "acroporidae", "genus": "not_given",
                 "source_label_class_name": "acroporidae", "live": "true"})
    df = pd.DataFrame(rows).assign(source_dataset_source=SOURCE)
    path = tmp_path / "class_to_concepts.csv"
    df.to_csv(path, index=False)
    return path


class _FakeCBM(torch.nn.Module):
    """Spatially constant per-rank probabilities; enough of the CBM interface for CBMPredictor."""

    def __init__(self):
        super().__init__()
        self.concept_classifier = torch.nn.Conv2d(len(PROBS), 2, 1)
        self.register_buffer("probs", torch.tensor(list(PROBS.values())))

    def forward(self, images):
        b, _, h, w = images.shape
        hidden = self.probs[None, :, None, None].expand(b, -1, h, w)
        return SimpleNamespace(hidden_states=hidden, logits=torch.zeros(b, 2, h, w))


class _FakePointDataset:
    """Two images in two units; one fully labelled point and one family-only point each."""

    def __init__(self):
        self.df_images = pd.DataFrame({"site": ["a", "b"], "image_id": ["1", "2"]})
        self.df_annotations = pd.DataFrame(
            {
                "image_id": ["1", "1", "2", "2"],
                "row": [3, 10, 5, 20],
                "col": [4, 12, 6, 25],
                "source_label_name": ["acropora", "acroporidae", "acropora", "acroporidae"],
            }
        )

    def read_image(self, **_):
        return np.full((32, 32, 3), 128, dtype=np.uint8)


def _run(tmp_path, taxonomy_csv, spec):
    names = ["acropora", "acroporidae"]
    return run_point_evaluation(
        eval_name=SOURCE,
        base_ds=_FakePointDataset(),
        unit_col="site",
        key_cols=["image_id"],
        predictor=CBMPredictor(_FakeCBM(), input_size=(16, 16), device="cpu"),
        class_lookup=ClassLookup({n: 1 for n in names}, {0: "ignore", 1: "coral"}),
        concept_lookup=build_concept_lookup(SOURCE, names, CONCEPT_NAMES, taxonomy_csv),
        concept_names=CONCEPT_NAMES,
        id2label={0: "ignore", 1: "coral"},
        output_dir=tmp_path / "out",
        num_workers=0,
        hierarchical_spec=spec,
    )


def test_default_run_has_no_hierarchical_block(tmp_path, taxonomy_csv):
    result = _run(tmp_path, taxonomy_csv, spec=None)
    assert "hierarchical" not in result["pooled"]
    assert all("hierarchical" not in unit for unit in result["units"].values())


def test_evaluate_cli_defaults_to_no_hierarchical():
    sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "scripts"))
    import evaluate

    args = evaluate.build_parser().parse_args(["--checkpoint", "x", "--output-dir", "y"])
    assert args.hierarchical is False and args.ambiguity_policy == "permissive"


def test_point_metrics_values(tmp_path, taxonomy_csv):
    spec = HierarchicalSpec.build(CONCEPT_NAMES, taxonomy_csv)
    result = _run(tmp_path, taxonomy_csv, spec)
    h = result["pooled"]["hierarchical"]
    assert h["num_points"] == 4
    assert set(result["units"]) == {"a", "b"}
    assert result["units"]["a"]["hierarchical"]["num_points"] == 2

    # Partial target: the family-only points' genus mass inside Acroporidae.
    assert h["partial/family->genus/subtree_mass"] == pytest.approx(0.08 + 0.04 + 0.25, abs=1e-6)
    assert h["partial/family->genus/argmax_in_subtree"] == 0.0  # argmax is porites
    assert h["partial/family->genus/n"] == 2
    assert "partial/order->family/subtree_mass" not in h  # no point stops at order

    # Expected distance at genus for the acropora points: siblings 1, same order 2,
    # other class (lineages meet at phylum) 4, out of a maximum of 6.
    expected = ((0.25 + 0.04) * 1 + (0.33 + 0.12) * 2 + (0.10 + 0.08) * 4) / 6
    assert h["expected_distance/genus"] == pytest.approx(expected, abs=1e-6)
    assert h["expected_distance/genus/n"] == 2
    assert h["expected_distance/family/n"] == 4

    # KL matches the training KL loss on the same probabilities (log-probs as logits).
    coherence = TaxonomicCoherenceLoss(
        build_taxonomy_tree(taxonomy_csv), concept_value2id_from_names(CONCEPT_NAMES)
    )
    logits = torch.tensor(list(PROBS.values())).log()[None, :, None, None]
    _, kl_components = coherence(logits, torch.zeros_like(logits), torch.zeros(1, 1, 1, dtype=torch.long))
    for pair in ("class->order", "order->family", "family->genus"):
        assert h[f"kl/{pair}"] == pytest.approx(kl_components[f"kl/{pair}"], abs=1e-6)

    # HierarchicalMetrics: argmax lineage hexacorallia/scleractinia/acroporidae/porites.
    assert h["consistency/order->family"] == 1.0
    assert h["consistency/family->genus"] == 0.0
    assert h["severity/genus"] == pytest.approx(2.0)  # acropora predicted as porites
    assert not math.isnan(h["ece/mean"])

    md = (tmp_path / "out" / SOURCE / "metrics.json").read_text()
    assert '"hierarchical"' in md


def test_spec_rejects_mismatched_channel_order(taxonomy_csv):
    names = list(CONCEPT_NAMES)
    i, j = names.index("genus__acropora"), names.index("genus__goniopora")
    names[i], names[j] = names[j], names[i]
    with pytest.raises(ValueError, match="do not match the taxonomy tree"):
        HierarchicalSpec.build(names, taxonomy_csv)


def test_layout_must_put_taxonomic_ranks_first():
    with pytest.raises(ValueError, match="not contiguous"):
        concept_value2id_from_names(["kingdom__animalia", "live", "phylum__cnidaria"])


def test_chunked_updates_match_per_point_updates(taxonomy_csv):
    """Buffering points into chunks gives exactly the per-point result."""
    from mermaidseg.evaluation.hierarchical_eval import HierarchicalPointEval

    spec = HierarchicalSpec.build(CONCEPT_NAMES, taxonomy_csv)
    lookup = build_concept_lookup(SOURCE, ["acropora", "acroporidae", "porites"], CONCEPT_NAMES, taxonomy_csv)
    rng = np.random.default_rng(0)
    rows = np.stack([lookup.row(n) for n in rng.choice(["acropora", "acroporidae", "porites"], 50)])
    logits = torch.tensor(rng.normal(size=(50, len(CONCEPT_NAMES))), dtype=torch.float32)
    probs = torch.cat(
        [logits[:, o : o + n].softmax(dim=1) for _, o, n in spec.slices] + [logits[:, 16:].sigmoid()], dim=1
    )
    per_point = HierarchicalPointEval(spec, flush_points=1)
    chunked = HierarchicalPointEval(spec, flush_points=1000)
    for i in range(0, 50, 7):
        per_point.update(rows[i : i + 7], probs[i : i + 7])
        chunked.update(rows[i : i + 7], probs[i : i + 7])
    a, b = per_point.compute(), chunked.compute()
    assert a.keys() == b.keys()
    for key in a:
        assert a[key] == pytest.approx(b[key], abs=1e-9, nan_ok=True), key
