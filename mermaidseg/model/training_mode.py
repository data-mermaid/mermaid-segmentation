"""Seam for extracting ``MetaModel``'s ``training_mode`` branching into per-mode
adapters.

Scaffold for the deferred "MetaModel mode-adapter" tech-debt item (#7). This is **not yet wired
into MetaModel** — it defines the target interface so the extraction can proceed incrementally and
safely behind the existing characterization tests (``tests/model/test_meta_characterization.py``).

Today MetaModel selects behavior with ``if self.training_mode == ...`` in several places:

- ``__init__``: loss/concept setup and the ``freeze_encoder`` / ``detach_concepts`` defaults.
- ``batch_predict`` (eval/inference): ``"standard"`` returns logits; ``"concept"`` maps concept
  logits back to class predictions; ``"concept-bottleneck"`` splits logits + concept activations.
- ``batch_predict_loss`` (train step): the loss is called with a *different arity per mode*
  (``self.loss(outputs, target_labels)`` for standard vs. the concept / CBM signatures).

The plan: give each mode a small adapter satisfying :class:`TrainingMode`, and have MetaModel hold
one adapter and delegate — so a new mode, or a change to one, is a local edit instead of another
scattered branch. The loss-call-arity divergence pairs with the "uniform loss interface" item (#5):
once losses share one ``forward(prediction, target) -> (loss, components)`` shape, each adapter's
:meth:`TrainingMode.predict_and_loss` collapses to assembling the prediction/target bundle.

Incremental path (each step stays green against the characterization tests):
1. Land ``StandardMode`` and route only ``training_mode == "standard"`` through it.
2. Add ``ConceptMode`` and ``ConceptBottleneckMode``; route the remaining branches.
3. Delete the ``if self.training_mode == ...`` branches from ``batch_predict`` / ``batch_predict_loss``.
"""

from typing import Any, Protocol, runtime_checkable

import torch


@runtime_checkable
class TrainingMode(Protocol):
    """The per-mode behavior MetaModel currently selects with ``if training_mode ==
    ...``.

    A concrete adapter owns how one training mode turns raw model outputs into
    class/concept predictions and into a training loss, so MetaModel can delegate
    instead of branching.
    """

    def predict(self, seg_outputs: Any) -> tuple[torch.Tensor, torch.Tensor | None]:
        """Map raw model outputs to ``(class_outputs, concept_outputs)`` for
        eval/inference.

        ``concept_outputs`` is ``None`` for standard mode.
        """
        ...

    def predict_and_loss(
        self,
        seg_outputs: Any,
        loss_fn: torch.nn.Module,
        target_labels: torch.Tensor,
        target_concepts: torch.Tensor | None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor | None, dict[str, float]]:
        """Compute ``(loss, class_outputs, concept_outputs, loss_components)`` for one
        train step."""
        ...
