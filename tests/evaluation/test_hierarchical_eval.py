"""HierarchicalSpec and HierarchicalPointEval on a toy taxonomy.

These cover the taxonomy-loss helpers. Point-eval wiring lives in the local
hierarchical scorer (``mermaidseg.evaluation.hierarchical``), not here.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import torch

from mermaidseg.evaluation.gt_mapping import build_concept_lookup
from mermaidseg.evaluation.hierarchical_eval import (
    HierarchicalSpec,
    concept_value2id_from_names,
)

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
