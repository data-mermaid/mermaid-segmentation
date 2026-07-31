#!/usr/bin/env python
"""Compare CE vs TaxonomicalLoss vs DualTaxonomicalLoss on the *real* downloaded
CoralNet subset.

Unlike ``local_loss_compare.py`` (structured *synthetic* patches), this trains the same
frozen-backbone LoRA model under each loss on the actual CoralNet imagery + point
annotations in ``data/coralnet_local_subset`` — real reef photos, a real MERMAID benthic
hierarchy, and the real confusion structure between coral genera. It is still a **small
local smoke-scale comparison**, not a convergence run: a handful of images, a few dozen
steps on CPU. Use it to see the losses behave on real data and sanity-check the
hierarchy metrics; use the SageMaker ablation for a real verdict.

No S3, no AWS creds. Images are read from the local ``local_path`` column. The benthic
hierarchy is fetched once from the public MERMAID API (no creds) and cached to disk for
subsequent offline runs.

python scripts/diagnostics/local_loss_compare_coralnet.py --top-k 20 --n-train 20
--steps 40
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from PIL import Image

from mermaidseg.dataset_reconciliation.concepts import initialize_benthic_hierarchy
from mermaidseg.model.eval import Evaluator
from mermaidseg.model.hierarchy_loss import build_distance_matrix, build_level_remaps
from mermaidseg.model.loss import CrossEntropyLoss, DualTaxonomicalLoss, TaxonomicalLoss
from mermaidseg.model.models import LinearDualLoRADINOv3, LinearLoRADINOv3

DATA_ROOT = Path("data/coralnet_local_subset")
MANIFEST = DATA_ROOT / "subset_manifest.parquet"
HIERARCHY_CACHE = DATA_ROOT / "benthic_hierarchy.json"

# ImageNet normalization (what the DINOv3 backbone expects).
MEAN = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
STD = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)

# Non-taxonomic catch-all labels that carry no hierarchy signal — drop to ignore.
DROP_LABELS = {"Other", "Unknown"}


def scatter_points(
    rows: np.ndarray, cols: np.ndarray, values: np.ndarray, size: int, padding: int
) -> np.ndarray:
    """Scatter integer point values into a ``size x size`` mask, dilating each by
    ``padding``."""
    mask = np.zeros((size, size), dtype=np.int64)
    for r, c, v in zip(rows, cols, values, strict=True):
        r0, r1 = max(0, r - padding), min(size, r + padding + 1)
        c0, c1 = max(0, c - padding), min(size, c + padding + 1)
        mask[r0:r1, c0:c1] = v
    return mask


def load_hierarchy() -> dict[str, str | None]:
    """Return the benthic ``{child: parent}`` hierarchy, fetching + caching once (public
    API)."""
    if HIERARCHY_CACHE.exists():
        with HIERARCHY_CACHE.open() as f:
            return json.load(f)
    try:
        hier = initialize_benthic_hierarchy()
    except Exception as exc:  # noqa: BLE001 — surface a clear, actionable message
        raise SystemExit(
            f"Could not fetch the benthic hierarchy from the MERMAID API ({exc}). It is a public "
            "endpoint (no creds) — run once with network access to populate "
            f"{HIERARCHY_CACHE}, after which this script is fully offline."
        ) from exc
    with HIERARCHY_CACHE.open("w") as f:
        json.dump(hier, f)
    return hier


def build_label_space(df: pd.DataFrame, top_k: int) -> dict[int, str]:
    """Compact ``id2label`` (0=ignore) from the ``top_k`` most frequent MERMAID
    labels."""
    counts = df.loc[~df.mermaid_label.isin(DROP_LABELS), "mermaid_label"].value_counts()
    labels = list(counts.head(top_k).index)
    return {0: "ignore", **{i + 1: name for i, name in enumerate(labels)}}


def build_morph_space(df: pd.DataFrame, keep_labels: set[str], top_m: int) -> list[str]:
    """Distinct growth forms (the ``top_m`` most frequent) among the kept coral
    labels."""
    coral = df[df.is_coral_growth_form & df.mermaid_label.isin(keep_labels)]
    counts = coral.growth_form_name.dropna().value_counts()
    return list(counts.head(top_m).index)


def select_images(
    df: pd.DataFrame, keep_labels: set[str], n_train: int, n_val: int, seed: int
) -> tuple[list, list]:
    """Disjoint train/val image ids, preferring images with the most in-subset
    points."""
    in_subset = df[df.mermaid_label.isin(keep_labels)]
    ranked = in_subset.groupby("image_id").size().sort_values(ascending=False)
    chosen = list(ranked.head(n_train + n_val).index)
    rng = np.random.default_rng(seed)
    rng.shuffle(chosen)
    return chosen[:n_train], chosen[n_train : n_train + n_val]


def build_tensors(
    df: pd.DataFrame,
    image_ids: list,
    label2id: dict[str, int],
    morph2idx: dict[str, int],
    input_size: int,
    padding: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Load local images + scatter sparse points into (images, labels, morphology)
    tensors.

    ``labels`` is ``ignore`` (0) everywhere except at annotated in-subset points (scaled
    to the model input size and dilated by ``padding``). ``morphology`` uses the
    0=invalid / 1=False / 2=True encoding: at a coral point with a kept growth form,
    that channel is True and the others False; everywhere else invalid (masked out of
    the loss).
    """
    by_image = dict(iter(df[df.image_id.isin(image_ids)].groupby("image_id")))
    num_morph = len(morph2idx)
    images, labels, morphs = [], [], []
    for iid in image_ids:
        g = by_image[iid]
        pil = Image.open(DATA_ROOT / g.local_path.iloc[0]).convert("RGB")
        w0, h0 = pil.size
        img = pil.resize((input_size, input_size), Image.BILINEAR)
        arr = torch.from_numpy(np.asarray(img, dtype=np.float32) / 255.0).permute(2, 0, 1)
        images.append((arr - MEAN) / STD)

        keep = g[g.mermaid_label.isin(label2id)]
        sy, sx = input_size / h0, input_size / w0
        rows = np.clip(np.round(keep.row.to_numpy() * sy), 0, input_size - 1).astype(np.intp)
        cols = np.clip(np.round(keep.col.to_numpy() * sx), 0, input_size - 1).astype(np.intp)
        label_ids = keep.mermaid_label.map(label2id).to_numpy().astype(np.int64)
        mask = scatter_points(rows, cols, label_ids, input_size, padding)
        labels.append(torch.from_numpy(mask))

        morph = torch.zeros(num_morph, input_size, input_size)
        if num_morph:
            gf = keep.growth_form_name.map(morph2idx)
            has_gf = gf.notna().to_numpy()
            pr, pc = rows[has_gf], cols[has_gf]
            idx = gf.dropna().to_numpy().astype(int)
            for r, c, m in zip(pr, pc, idx, strict=True):
                r0, r1 = max(0, r - padding), min(input_size, r + padding + 1)
                c0, c1 = max(0, c - padding), min(input_size, c + padding + 1)
                morph[:, r0:r1, c0:c1] = 1.0  # all forms False in this patch...
                morph[m, r0:r1, c0:c1] = 2.0  # ...except the observed one (True)
        morphs.append(morph)

    return torch.stack(images), torch.stack(labels), torch.stack(morphs)


def train_and_eval(
    loss_name: str,
    id2label: dict[int, str],
    hierarchy: dict[str, str | None],
    morph_names: list[str],
    tr: tuple[torch.Tensor, torch.Tensor, torch.Tensor],
    va: tuple[torch.Tensor, torch.Tensor, torch.Tensor],
    dist: torch.Tensor,
    remaps: dict[str, torch.Tensor],
    args: argparse.Namespace,
) -> dict[str, float]:
    num_classes = max(id2label) + 1
    input_size = tr[0].shape[-1]
    torch.manual_seed(args.seed)  # identical init + batch order across losses
    is_dual = loss_name == "DualTaxonomicalLoss"
    damp = args.damping
    if is_dual:
        model = LinearDualLoRADINOv3(
            num_classes=num_classes,
            input_size=(input_size, input_size),
            lora_r=args.lora_r,
            num_morphology=len(morph_names),
            morphology_names=morph_names,
        )
        loss_fn = DualTaxonomicalLoss(
            id2label,
            hierarchy,
            ignore_index=0,
            alpha=args.alpha,
            beta=args.beta,
            gamma=args.gamma,
            damping_denominator=damp,
        )
    else:
        model = LinearLoRADINOv3(
            num_classes=num_classes, input_size=(input_size, input_size), lora_r=args.lora_r
        )
        if loss_name == "CrossEntropyLoss":
            loss_fn = CrossEntropyLoss(ignore_index=0, damping_denominator=damp)
        else:
            loss_fn = TaxonomicalLoss(
                id2label,
                hierarchy,
                ignore_index=0,
                alpha=args.alpha,
                beta=args.beta,
                damping_denominator=damp,
            )
    model.train()
    trainable = [p for p in model.parameters() if p.requires_grad]
    opt = torch.optim.AdamW(trainable, lr=args.lr)

    imgs, lbls, morph = tr
    n = imgs.shape[0]
    g = torch.Generator().manual_seed(args.seed)
    for _ in range(args.steps):
        idx = torch.randint(n, (args.batch,), generator=g)
        out = model(imgs[idx])
        if is_dual:
            loss, _ = loss_fn(
                out.logits.float(), lbls[idx], out.morphology_logits.float(), morph[idx]
            )
        else:
            loss, _ = loss_fn(out.logits.float(), lbls[idx])
        opt.zero_grad()
        loss.backward()
        opt.step()

    model.eval()
    ev = Evaluator(
        num_classes=num_classes,
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
    ap.add_argument("--top-k", type=int, default=20, help="most-frequent MERMAID labels to keep")
    ap.add_argument(
        "--top-m", type=int, default=6, help="most-frequent growth forms for the dual head"
    )
    ap.add_argument("--n-train", type=int, default=20)
    ap.add_argument("--n-val", type=int, default=10)
    ap.add_argument("--input-size", type=int, default=224, help="square model input; must be /16")
    ap.add_argument(
        "--padding", type=int, default=8, help="half-size of the dilation patch per point"
    )
    ap.add_argument("--steps", type=int, default=60)
    ap.add_argument("--batch", type=int, default=4)
    ap.add_argument("--lora-r", type=int, default=8)
    ap.add_argument("--lr", type=float, default=1e-3)
    # Loss weights default to the real training config (configs/training_config_dinov3_lora_taxonomical.yaml).
    ap.add_argument("--alpha", type=float, default=0.1, help="tree-distance weight")
    ap.add_argument("--beta", type=float, default=0.1, help="multi-level CE weight")
    ap.add_argument("--gamma", type=float, default=0.1, help="morphology BCE weight (dual)")
    ap.add_argument("--damping", type=float, default=100.0, help="loss damping_denominator")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    if not MANIFEST.exists():
        raise SystemExit(f"Missing {MANIFEST}. Download the CoralNet local subset first.")

    df = pd.read_parquet(MANIFEST)
    id2label = build_label_space(df, args.top_k)
    label2id = {v: k for k, v in id2label.items() if k != 0}
    keep_labels = set(label2id)
    morph_names = build_morph_space(df, keep_labels, args.top_m)
    morph2idx = {name: i for i, name in enumerate(morph_names)}
    hierarchy = load_hierarchy()

    train_ids, val_ids = select_images(df, keep_labels, args.n_train, args.n_val, args.seed)
    tr = build_tensors(df, train_ids, label2id, morph2idx, args.input_size, args.padding)
    va = build_tensors(df, val_ids, label2id, morph2idx, args.input_size, args.padding)

    num_classes = max(id2label) + 1
    dist = build_distance_matrix(id2label, hierarchy, ignore_index=0, num_classes=num_classes)
    remaps = build_level_remaps(id2label, hierarchy, None, ignore_index=0, num_classes=num_classes)
    levels = list(remaps.keys())
    # Headline the level the most classes roll up to (e.g. "hard coral" over "sand") — that is where
    # taxonomy severity is most visible. Descendant count = non-ignore entries in the remap.
    headline = max(levels, key=lambda lv: int((remaps[lv] != 0).sum()), default=None)
    anc_key = f"ancestor_accuracy/{headline}" if headline else None

    print(
        f"real CoralNet subset · {args.n_train} train / {args.n_val} val imgs · "
        f"{args.steps} steps · {num_classes - 1} classes · input={args.input_size}"
    )
    print(f"classes: {', '.join(id2label[i] for i in range(1, num_classes))}")
    print(f"hierarchy levels discovered in label set: {levels or '(none)'}")
    print(f"growth forms (dual): {morph_names or '(none)'}\n")

    hdr = f"{'loss':22s}{'accuracy':>10s}{'miou':>8s}{'mean_tree_dist':>16s}"
    hdr += f"{'anc_acc/' + headline:>24s}" if headline else ""
    print(hdr)
    print("-" * len(hdr))
    rows = {}
    for name in ("CrossEntropyLoss", "TaxonomicalLoss", "DualTaxonomicalLoss"):
        r = train_and_eval(name, id2label, hierarchy, morph_names, tr, va, dist, remaps, args)
        rows[name] = r
        line = (
            f"{name:22s}{r.get('accuracy', 0.0):>10.3f}{r.get('miou', 0.0):>8.3f}"
            f"{r.get('mean_tree_distance', 0.0):>16.3f}"
        )
        if anc_key:
            line += f"{r.get(anc_key, 0.0):>24.3f}"
        print(line)

    base, tax = rows["CrossEntropyLoss"], rows["TaxonomicalLoss"]
    print("\nTaxonomicalLoss vs CE baseline:")
    print(
        f"  mean_tree_distance   {base.get('mean_tree_distance', 0):.3f} -> "
        f"{tax.get('mean_tree_distance', 0):.3f}  (lower = less severe errors)"
    )
    if anc_key:
        print(
            f"  {anc_key}   {base.get(anc_key, 0):.3f} -> {tax.get(anc_key, 0):.3f}  "
            "(higher = coarse level right more often)"
        )
    if levels:
        print("\nancestor_accuracy by level (TaxonomicalLoss):")
        for lv in levels:
            print(f"  {lv:16s} {tax.get(f'ancestor_accuracy/{lv}', 0.0):.3f}")
    print(
        "\nSmall real-data smoke — a convergence verdict needs the SageMaker ablation "
        "(docs/taxonomical_loss_experiments.md)."
    )


if __name__ == "__main__":
    main()
