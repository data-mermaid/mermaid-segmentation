"""Concept-mode prediction consistency between batch_predict and batch_predict_loss.

`postprocess_predicted_concepts` thresholds per-pixel concept scores at 0.5. That
threshold is a *probability* threshold, so concept logits must be passed through sigmoid
first. `batch_predict_loss` (train/val loop) already does this; `batch_predict`
(inference / Evaluator.evaluate_model / predict) historically fed raw logits, so the
same model produced different concept->label maps in eval vs training. These lock the
fix: batch_predict must postprocess sigmoid(logits), and both paths must agree.

Runs fully offline on CPU with a fixed-logits stub model (no HF download, no S3).
"""

from __future__ import annotations

import types

import numpy as np
import pandas as pd
import torch
from torch import nn

import mermaidseg.model.models as models_mod
from mermaidseg.dataset_reconciliation.concepts import postprocess_predicted_concepts
from mermaidseg.io import ConfigDict
from mermaidseg.model.meta import MetaModel

NUM_CONCEPTS = 3
# Winning concept has logit 0.3: sigmoid(0.3)=0.574 > 0.5 (assigned as a probability),
# but the raw logit 0.3 < 0.5 (would be left unassigned = -1). This is exactly the region
# where feeding logits vs sigmoid to postprocess disagrees, so it discriminates the bug.
_WIN_CONCEPT = 1
_WIN_LOGIT = 0.3


class _FixedConceptLogits(nn.Module):
    """Concept-mode stub: forward(x) -> obj with ``.logits`` = fixed concept logits
    ``(B, NUM_CONCEPTS, H, W)`` (channels-first), independent of the input values."""

    def __init__(self, num_classes: int = NUM_CONCEPTS, **_kwargs):
        super().__init__()
        self._param = nn.Parameter(torch.zeros(1))  # so the optimizer has something to own

    def forward(self, x: torch.Tensor):
        b, _c, h, w = x.shape
        logits = torch.full((b, NUM_CONCEPTS, h, w), -5.0)  # sigmoid(-5) ~= 0 -> unassigned
        logits[:, _WIN_CONCEPT] = _WIN_LOGIT
        return types.SimpleNamespace(logits=logits)


def _concept_matrix() -> pd.DataFrame:
    """3 concepts, all at hierarchy level 1 (satisfies postprocess's 'level' lookup)."""
    cols = pd.MultiIndex.from_arrays(
        [list(range(NUM_CONCEPTS)), [1] * NUM_CONCEPTS], names=["concept_id", "level"]
    )
    return pd.DataFrame(np.zeros((1, NUM_CONCEPTS)), columns=cols)


def _make_concept_meta(monkeypatch) -> MetaModel:
    monkeypatch.setattr(models_mod, "FixedConceptLogits", _FixedConceptLogits, raising=False)
    training_kwargs = ConfigDict(
        {
            "training_mode": "concept",
            "optimizer": {"type": "AdamW", "lr": 1e-3},
        }
    )
    return MetaModel(
        run_name="concept-consistency",
        num_classes=NUM_CONCEPTS,
        num_concepts=NUM_CONCEPTS,
        model_kwargs=ConfigDict({"name": "FixedConceptLogits"}),
        device="cpu",
        training_kwargs=training_kwargs,
        concept_matrix=_concept_matrix(),
        conceptid2labelid={i: i + 1 for i in range(NUM_CONCEPTS)},
    )


def test_batch_predict_postprocesses_sigmoid_not_raw_logits(monkeypatch):
    """batch_predict must feed sigmoid(logits) — not raw logits — to postprocess.

    Differential check (robust to postprocess's internal layout): batch_predict's output
    must equal ``postprocess(sigmoid(logits))`` and differ from ``postprocess(raw
    logits)``. The 0.3 winning logit (prob 0.574) sits in the band where the two
    disagree, so the second assertion guarantees the test actually exercises the fix.
    """
    meta = _make_concept_meta(monkeypatch)
    inputs = torch.randn(1, 3, 8, 8)
    logits = meta.model(inputs).logits  # deterministic, input-independent

    outputs, _concept_outputs = meta.batch_predict(inputs)

    cm, c2l = meta.concept_matrix, meta.conceptid2labelid
    from_sigmoid = postprocess_predicted_concepts(
        torch.sigmoid(logits).detach().cpu().numpy(), cm, c2l
    )
    from_raw = postprocess_predicted_concepts(logits.detach().cpu().numpy(), cm, c2l)

    assert torch.equal(outputs.cpu(), from_sigmoid), (
        "batch_predict must postprocess sigmoid(logits)"
    )
    assert not torch.equal(from_sigmoid, from_raw), (
        "sigmoid vs raw must differ, else test is vacuous"
    )
