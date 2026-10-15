#!/usr/bin/env python
"""Offline controlled comparison: CE baseline vs TaxonomicalLoss vs DualTaxonomicalLoss.

No S3/creds. Builds a *structured* synthetic segmentation task where the hierarchy is
real — sibling classes (same benthic parent) get similar colors, distant classes get
distinct colors, with noise so classes are confusable. Trains the SAME model (fresh,
same seed) under each loss on identical data, then evaluates on a held-out split. The
point is to see whether the taxonomical loss trades a little leaf accuracy for *less
severe* mistakes: lower ``mean_tree_distance`` and higher ``ancestor_accuracy`` (getting
"hard coral" right even when unsure of the genus).

This is a controlled mechanism demo, NOT a coral-quality verdict — real data (the S3
path in docs/local-loss-testing.md) is the real comparison.

python scripts/diagnostics/local_loss_compare.py --steps 40
"""

from __future__ import annotations

import argparse

import torch

from mermaidseg.model.eval import Evaluator
from mermaidseg.model.hierarchy_loss import build_distance_matrix, build_level_remaps
from mermaidseg.model.loss import CrossEntropyLoss, DualTaxonomicalLoss, TaxonomicalLoss
from mermaidseg.model.models import LinearDualLoRADINOv3, LinearLoRADINOv3

HIERARCHY = {
    "acropora": "acroporidae",
    "montipora": "acroporidae",
    "porites": "poritidae",
    "acroporidae": "hard coral",
    "poritidae": "hard coral",
    "hard coral": None,
    "sand": None,
    "algae": None,
}
ID2LABEL = {
    0: "ignore",
    1: "Acropora",
    2: "Montipora",
    3: "Porites",
    4: "Sand",
    5: "Algae",
    6: "Hard coral",
}
NUM_CLASSES = max(ID2LABEL) + 1

# Base colors: the three hard-coral genera are near-identical greens (siblings, easily confused);
# "Hard coral" (coarse) a fourth green; Sand yellow and Algae dark are far. So most confusions the
# model makes are *within* the coral subtree — exactly where taxonomy severity should differ.
CLASS_RGB = {
    1: (0.20, 0.80, 0.30),  # Acropora   ┐
    2: (0.26, 0.82, 0.24),  # Montipora  ├ acroporidae (very close)
    6: (0.24, 0.74, 0.30),  # Hard coral │
    3: (0.30, 0.66, 0.40),  # Porites    ┘ poritidae (a bit apart, still green)
    4: (0.90, 0.85, 0.40),  # Sand   (far)
    5: (0.10, 0.20, 0.16),  # Algae  (far)
}


def make_dataset(n_images: int, size: int, patch: int, noise: float, gen: torch.Generator):
    """Return (images (N,3,H,W), labels (N,H,W)) — a grid of class-colored, noisy
    patches."""
    fg = list(CLASS_RGB)
    grid = size // patch
    imgs = torch.zeros(n_images, 3, size, size)
    lbls = torch.zeros(n_images, size, size, dtype=torch.long)
    for n in range(n_images):
        for gy in range(grid):
            for gx in range(grid):
                cid = fg[int(torch.randint(len(fg), (1,), generator=gen))]
                y0, x0 = gy * patch, gx * patch
                color = torch.tensor(CLASS_RGB[cid]).view(3, 1, 1)
                imgs[n, :, y0 : y0 + patch, x0 : x0 + patch] = color
                lbls[n, y0 : y0 + patch, x0 : x0 + patch] = cid
    imgs = imgs + noise * torch.randn(imgs.shape, generator=gen)
    # normalize-ish to look like model input
    imgs = (imgs - imgs.mean()) / (imgs.std() + 1e-6)
    return imgs, lbls


def train_and_eval(loss_name, steps, batch, lr, tr, va, seed, dist, remaps):
    torch.manual_seed(seed)  # same init + same batch order across losses
    is_dual = loss_name == "DualTaxonomicalLoss"
    if is_dual:
        model = LinearDualLoRADINOv3(
            num_classes=NUM_CLASSES,
            input_size=(tr[0].shape[-1],) * 2,
            lora_r=8,
            num_morphology=4,
            morphology_names=list("abcd"),
        )
    else:
        model = LinearLoRADINOv3(
            num_classes=NUM_CLASSES, input_size=(tr[0].shape[-1],) * 2, lora_r=8
        )
    model.train()
    if loss_name == "CrossEntropyLoss":
        loss_fn = CrossEntropyLoss(ignore_index=0)
    elif loss_name == "TaxonomicalLoss":
        loss_fn = TaxonomicalLoss(ID2LABEL, HIERARCHY, ignore_index=0, alpha=0.5, beta=0.3)
    else:
        loss_fn = DualTaxonomicalLoss(
            ID2LABEL, HIERARCHY, ignore_index=0, alpha=0.5, beta=0.3, gamma=0.3
        )
    trainable = [p for p in model.parameters() if p.requires_grad]
    opt = torch.optim.AdamW(trainable, lr=lr)

    imgs, lbls = tr
    n = imgs.shape[0]
    g = torch.Generator().manual_seed(seed)
    for _ in range(steps):
        idx = torch.randint(n, (batch,), generator=g)
        out = model(imgs[idx])
        if is_dual:
            ml = out.morphology_logits.float()
            mt = torch.randint(0, 3, ml.shape).float()
            loss, _ = loss_fn(out.logits.float(), lbls[idx], ml, mt)
        else:
            loss, _ = loss_fn(out.logits.float(), lbls[idx])
        opt.zero_grad()
        loss.backward()
        opt.step()

    model.eval()
    ev = Evaluator(
        num_classes=NUM_CLASSES,
        device="cpu",
        ignore_index=0,
        hierarchy_metrics=True,
        distance_matrix=dist,
        level_remaps=remaps,
    )
    with torch.no_grad():
        preds = model(va[0]).logits.argmax(dim=1)
        ev.accumulate(preds, va[1])
        return ev.compute_and_reset()


def main() -> None:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--steps", type=int, default=40)
    ap.add_argument("--batch", type=int, default=6)
    ap.add_argument("--size", type=int, default=96)
    ap.add_argument("--patch", type=int, default=24)
    ap.add_argument("--noise", type=float, default=0.5)
    ap.add_argument("--lr", type=float, default=2e-3)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    gen = torch.Generator().manual_seed(args.seed)
    tr = make_dataset(24, args.size, args.patch, args.noise, gen)
    va = make_dataset(12, args.size, args.patch, args.noise, gen)
    dist = build_distance_matrix(ID2LABEL, HIERARCHY, ignore_index=0, num_classes=NUM_CLASSES)
    remaps = build_level_remaps(ID2LABEL, HIERARCHY, None, ignore_index=0, num_classes=NUM_CLASSES)

    print(f"structured synthetic · {args.steps} steps · noise={args.noise} · size={args.size}")
    print(
        f"{'loss':22s}{'accuracy':>10s}{'miou':>8s}{'mean_tree_dist':>16s}{'ancestor_acc/hard coral':>26s}"
    )
    print("-" * 82)
    rows = {}
    for name in ("CrossEntropyLoss", "TaxonomicalLoss", "DualTaxonomicalLoss"):
        r = train_and_eval(name, args.steps, args.batch, args.lr, tr, va, args.seed, dist, remaps)
        rows[name] = r
        acc = r.get("accuracy", 0.0)
        miou = r.get("miou", 0.0)
        mtd = r.get("mean_tree_distance", 0.0)
        anc = r.get("ancestor_accuracy/hard coral", 0.0)
        print(f"{name:22s}{acc:>10.3f}{miou:>8.3f}{mtd:>16.3f}{anc:>26.3f}")

    base = rows["CrossEntropyLoss"]
    tax = rows["TaxonomicalLoss"]
    print("\nTaxonomicalLoss vs CE baseline:")
    print(
        f"  mean_tree_distance      {base.get('mean_tree_distance', 0):.3f} -> {tax.get('mean_tree_distance', 0):.3f}  (lower = less severe errors)"
    )
    print(
        f"  ancestor_accuracy(hc)   {base.get('ancestor_accuracy/hard coral', 0):.3f} -> {tax.get('ancestor_accuracy/hard coral', 0):.3f}  (higher = coarse level more often right)"
    )
    print(
        "\nControlled synthetic demo — real coral comparison needs the S3 path (docs/local-loss-testing.md)."
    )


if __name__ == "__main__":
    main()
