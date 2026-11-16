"""Tests for TaxonomicalLoss / DualTaxonomicalLoss."""

from __future__ import annotations

import torch

from mermaidseg.model.loss import CrossEntropyLoss, DualTaxonomicalLoss, TaxonomicalLoss

TINY_HIERARCHY = {
    "acropora": "acroporidae",
    "montipora": "acroporidae",
    "acroporidae": "hard coral",
    "porites": "poritidae",
    "poritidae": "hard coral",
    "hard coral": None,
    "sand": None,
}

ID2LABEL = {
    0: "ignore",
    1: "Acropora",
    2: "Montipora",
    3: "Porites",
    4: "Sand",
    5: "Hard coral",
}


def test_alpha_beta_zero_matches_cross_entropy():
    logits = torch.randn(2, 6, 4, 4, requires_grad=True)
    targets = torch.randint(1, 6, (2, 4, 4))
    targets[0, 0, 0] = 0

    ce = CrossEntropyLoss(ignore_index=0, damping_denominator=0.0)
    tax = TaxonomicalLoss(
        ID2LABEL,
        TINY_HIERARCHY,
        ignore_index=0,
        damping_denominator=0.0,
        alpha=0.0,
        beta=0.0,
    )
    ce_loss, _ = ce(logits, targets)
    tax_loss, components = tax(logits, targets)
    assert torch.allclose(ce_loss, tax_loss, atol=1e-5)
    assert components["tree_distance"] == 0.0
    assert "taxonomical_total" in components


def test_tree_term_lower_for_sibling_than_distant_confusion():
    """Softmax peaked on a sibling should yield lower tree loss than on sand."""
    tax = TaxonomicalLoss(
        ID2LABEL,
        TINY_HIERARCHY,
        ignore_index=0,
        alpha=1.0,
        beta=0.0,
    )
    # GT = Acropora (1); peak on Montipora (2) vs Sand (4)
    sibling_logits = torch.full((1, 6, 1, 1), -10.0)
    sibling_logits[0, 2, 0, 0] = 10.0
    distant_logits = torch.full((1, 6, 1, 1), -10.0)
    distant_logits[0, 4, 0, 0] = 10.0
    targets = torch.tensor([[[1]]])

    _, sib = tax(sibling_logits, targets)
    _, dist = tax(distant_logits, targets)
    assert sib["tree_distance"] < dist["tree_distance"]


def test_level_loss_runs_with_hard_coral_ancestor():
    tax = TaxonomicalLoss(
        ID2LABEL,
        TINY_HIERARCHY,
        ignore_index=0,
        alpha=0.0,
        beta=1.0,
        level_names=["Hard coral"],
    )
    logits = torch.randn(1, 6, 2, 2)
    targets = torch.ones(1, 2, 2, dtype=torch.long)
    loss, components = tax(logits, targets)
    assert torch.isfinite(loss)
    assert "level/Hard coral" in components
    # Regression guard: the level term must be a real (non-zero) loss. A prior implementation
    # collapsed the coarse softmax to a single class, making this identically 0.
    assert components["level/Hard coral"] > 0.0


def test_level_term_produces_gradient():
    """Regression guard for the degenerate multi-level CE: beta must change the
    gradient.

    The GT mixes in-subtree (Acropora/Montipora under Hard coral) and out-of-subtree
    (Sand) pixels, so the binary in/out cross-entropy has real signal in both
    directions.
    """
    targets = torch.tensor([[[1, 2, 4, 4]]])  # Acropora, Montipora (in) + Sand, Sand (out)

    def grad_with(beta: float) -> torch.Tensor:
        torch.manual_seed(0)
        logits = torch.randn(1, 6, 1, 4, requires_grad=True)
        tax = TaxonomicalLoss(
            ID2LABEL,
            TINY_HIERARCHY,
            ignore_index=0,
            alpha=0.0,
            beta=beta,
            level_names=["Hard coral"],
        )
        loss, _ = tax(logits, targets)
        loss.backward()
        return logits.grad.clone()

    delta = (grad_with(1.0) - grad_with(0.0)).abs().max().item()
    assert delta > 0.0, "level (beta) term contributes no gradient — it is inert"


def test_taxonomical_buffers_move_with_module():
    """Regression guard for the device bug: hierarchy tensors must be registered buffers
    so ``module.to(device)`` co-locates them with on-device targets (else GPU indexing
    raises)."""
    tax = TaxonomicalLoss(
        ID2LABEL, TINY_HIERARCHY, ignore_index=0, alpha=1.0, beta=1.0, level_names=["Hard coral"]
    )
    buffers = dict(tax.named_buffers())
    assert "distance_matrix" in buffers
    assert any(name.startswith("level_remap__") for name in buffers)
    # forward is also self-guarding: it moves buffers to the input device on use.
    moved = tax.to(torch.device("cpu"))
    assert all(b.device.type == "cpu" for b in dict(moved.named_buffers()).values())


def test_dual_morph_masked_unknown_channels():
    loss_fn = DualTaxonomicalLoss(
        ID2LABEL,
        TINY_HIERARCHY,
        ignore_index=0,
        alpha=0.0,
        beta=0.0,
        gamma=1.0,
    )
    logits = torch.randn(1, 6, 2, 2)
    targets = torch.ones(1, 2, 2, dtype=torch.long)
    morph_logits = torch.randn(1, 2, 2, 2)
    # All unknown (0) → morphology contribution 0
    morph_targets = torch.zeros(1, 2, 2, 2)
    total, components = loss_fn(logits, targets, morph_logits, morph_targets)
    assert components["morphology"] == 0.0
    assert torch.isfinite(total)

    morph_targets = torch.ones(1, 2, 2, 2) * 2  # True
    total2, components2 = loss_fn(logits, targets, morph_logits, morph_targets)
    assert components2["morphology"] > 0.0
