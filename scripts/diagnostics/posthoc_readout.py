#!/usr/bin/env python
"""Post-hoc per-level readout for a flat (standard-mode) DINOv3 segmentation checkpoint.

The standard model emits a single softmax over N benthic-attribute classes. This utility shows
what that model can already report per point WITHOUT retraining:

  1. Top-K classes + softmax confidence (no roll-up — the full distribution is kept).
  2. A per-LEVEL confidence by aggregating the class softmax UP the benthic-attribute hierarchy:
     P(node) = sum of the probabilities of every leaf class under that node. This recovers a
     confidence at every taxonomic level (e.g. acropora 51% -> acroporidae 52% -> hard coral 99%),
     which the current "roll up to one label" UI discards.

NOT included: growth form. It is a separate axis the flat model never predicts and cannot be
derived from a class (one genus maps to many growth forms) — it needs a trained growth-form head.
See docs/lora-baseline-readout.md and the multi-task-head plan.

Read-only inference. Uses the locally-cached DINOv3 backbone + a downloaded checkpoint dir
(``model.pt`` + ``config.json`` + ``target_id2label.json``, as saved by the training Logger).

Example:
    python scripts/diagnostics/posthoc_readout.py path/to/ckpt_dir \\
        demo/static/nadir_acropora_table.jpg --out out_readout --grid 5
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
from PIL import Image

from mermaidseg.dataset_reconciliation.concepts import (
    initialize_benthic_hierarchy,
    normalize_benthic_hierarchy,
)
from mermaidseg.model.models import LinearLoRADINOv3

# ImageNet normalization (matches the val transform in configs/data_config_coralnet_mermaid.yaml).
_MEAN = np.array([0.485, 0.456, 0.406])
_STD = np.array([0.229, 0.224, 0.225])


def load_model(ckpt_dir: Path) -> tuple[torch.nn.Module, dict[int, str], tuple[int, int]]:
    """Build the flat LoRA model, reload the frozen backbone from cache, load
    LoRA+head."""
    with open(ckpt_dir / "config.json") as f:
        cfg = json.load(f)
    m = cfg["model"]
    with open(ckpt_dir / "target_id2label.json") as f:
        id2label = {int(k): v for k, v in json.load(f).items()}
    model = LinearLoRADINOv3(
        encoder_name=m["encoder_name"],
        num_classes=cfg["num_classes"],
        input_size=tuple(m["input_size"]),
        lora_r=m["lora_r"],
        lora_alpha=m["lora_alpha"],
        lora_dropout=m["lora_dropout"],
        lora_target_modules=tuple(m["lora_target_modules"]),
        lora_bias=m["lora_bias"],
    )
    ck = torch.load(ckpt_dir / "model.pt", map_location="cpu", weights_only=False)
    # strict=False: the frozen backbone base weights are reloaded via from_pretrained and are not
    # in the checkpoint (which stores only LoRA adapters + head).
    model.load_state_dict(ck["model_state_dict"], strict=False)
    model.eval()
    return model, id2label, tuple(m["input_size"])


def infer_probs(model: torch.nn.Module, img_path: Path, size: tuple[int, int]) -> np.ndarray:
    """Return per-pixel softmax probabilities, shape (num_classes, H, W)."""
    h, w = size
    img = Image.open(img_path).convert("RGB").resize((w, h), Image.BILINEAR)
    arr = (np.asarray(img).astype(np.float32) / 255.0 - _MEAN) / _STD
    x = torch.from_numpy(arr.transpose(2, 0, 1)).float().unsqueeze(0)
    with torch.no_grad():
        logits = model(x).logits
        return torch.softmax(logits, dim=1)[0].numpy()


def build_ancestry(id2label: dict[int, str], hierarchy: dict[str, str]):
    """Return (ancestors_of, node_to_channels).

    ``ancestors_of[name]`` is [self, parent, ..., root] via the benthic-attribute
    hierarchy. ``node_to_channels[node]`` is the list of class channels whose ancestry
    contains ``node``.
    """

    def ancestors(name: str) -> list[str]:
        chain, seen, cur = [], set(), name.lower()
        while cur and cur not in seen:
            chain.append(cur)
            seen.add(cur)
            cur = hierarchy.get(cur)
        return chain

    node_to_channels: dict[str, list[int]] = {}
    ancestors_of: dict[str, list[str]] = {}
    for channel, name in id2label.items():  # channel index == class id; 0 is background
        chain = ancestors(name)
        ancestors_of[name] = chain
        for node in chain:
            node_to_channels.setdefault(node, []).append(channel)
    return ancestors_of, node_to_channels


def readout_at_pixel(probs, py, px, id2label, ancestors_of, node_to_channels, topk):
    p = probs[:, py, px]
    order = np.argsort(p)[::-1]
    top = [(id2label.get(int(c), "background"), float(p[int(c)])) for c in order[:topk]]
    # Aggregate up the hierarchy for the top class.
    top_name = id2label.get(int(order[0]), "background")
    ladder = []
    for node in ancestors_of.get(top_name, [top_name]):
        agg = float(
            sum(probs[c, py, px] for c in node_to_channels.get(node, []) if c < probs.shape[0])
        )
        ladder.append((node, agg))
    return top, ladder


def main() -> None:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument(
        "ckpt_dir", type=Path, help="dir with model.pt + config.json + target_id2label.json"
    )
    ap.add_argument("image", type=Path, help="image to run (e.g. a quadrat photo)")
    ap.add_argument("--out", type=Path, default=Path("out_readout"))
    ap.add_argument("--grid", type=int, default=5, help="NxN point grid, like a CoralNet quadrat")
    ap.add_argument("--topk", type=int, default=3)
    args = ap.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)

    print(f"Loading model from {args.ckpt_dir} (backbone from HF cache)...")
    model, id2label, size = load_model(args.ckpt_dir)
    print(f"Running inference on {args.image.name}...")
    probs = infer_probs(model, args.image, size)

    print("Fetching benthic-attribute hierarchy from the MERMAID API...")
    try:
        hierarchy = normalize_benthic_hierarchy(initialize_benthic_hierarchy())
    except Exception as exc:  # noqa: BLE001 — network is optional; degrade to top-K only
        print(f"  WARNING: could not fetch hierarchy ({exc}); emitting top-K without the ladder.")
        hierarchy = {}
    ancestors_of, node_to_channels = build_ancestry(id2label, hierarchy)

    h, w = size
    ys = np.linspace(h / (2 * args.grid), h - h / (2 * args.grid), args.grid).astype(int)
    xs = np.linspace(w / (2 * args.grid), w - w / (2 * args.grid), args.grid).astype(int)
    report = {
        "image": str(args.image),
        "checkpoint": str(args.ckpt_dir),
        "grid": args.grid,
        "points": [],
    }
    for gy, py in enumerate(ys):
        for gx, px in enumerate(xs):
            top, ladder = readout_at_pixel(
                probs, py, px, id2label, ancestors_of, node_to_channels, args.topk
            )
            report["points"].append(
                {
                    "grid": [gy, gx],
                    "px": [int(px), int(py)],
                    "top_k": [{"label": lbl, "conf": round(c, 4)} for lbl, c in top],
                    "hierarchy_ladder": [{"level": n, "conf": round(c, 4)} for n, c in ladder],
                }
            )
    out_json = args.out / "readout.json"
    with open(out_json, "w") as f:
        json.dump(report, f, indent=2)

    print(f"\nPOST-HOC READOUT · {args.image.name} · {args.grid}x{args.grid} grid")
    for pt in report["points"][:4]:
        tk = " · ".join(f"{d['label']} {d['conf'] * 100:.0f}%" for d in pt["top_k"])
        print(f"\n▶ point {tuple(pt['grid'])}  top-{args.topk}: {tk}")
        if len(pt["hierarchy_ladder"]) > 1:
            for d in pt["hierarchy_ladder"]:
                bar = "█" * int(round(d["conf"] * 20))
                print(f"    {d['conf'] * 100:5.0f}%  {bar:<20} {d['level']}")
    print(
        f"\nWrote {out_json} ({len(report['points'])} points). Render: render_readout.py {args.out}"
    )


if __name__ == "__main__":
    main()
