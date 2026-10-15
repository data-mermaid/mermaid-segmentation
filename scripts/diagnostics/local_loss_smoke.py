#!/usr/bin/env python
"""Offline smoke test for the taxonomical / dual-head loss functions — no S3, no creds.

Builds the REAL model (by name, cached DINOv3 backbone) and the REAL loss (from a training
config's ``loss`` block) over a small synthetic label space + hierarchy, feeds synthetic batches,
and runs a few train steps + one eval step. Catches integration bugs that unit tests miss —
model-output → loss shape wiring, buffer/device placement, the hierarchy eval metrics — without
downloading any data.

    python scripts/diagnostics/local_loss_smoke.py \
        --model-config configs/model_config_dinov3_lora_qv_r8.yaml \
        --training-config configs/training_config_dinov3_lora_taxonomical.yaml
"""

from __future__ import annotations

import argparse

import torch
import yaml

import mermaidseg.model.loss as loss_mod
import mermaidseg.model.models as models_mod
from mermaidseg.model.eval import Evaluator
from mermaidseg.model.hierarchy_loss import build_distance_matrix, build_level_remaps

# A small, fixed benthic-attribute tree so the run is fully offline (no MERMAID API call).
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


def _load(cfg_path: str) -> dict:
    with open(cfg_path) as f:
        return yaml.safe_load(f)


def build_model(model_cfg: dict, num_classes: int, image_size: int, num_morph: int):
    m = dict(model_cfg["model"])
    name = m.pop("name")
    m.pop("input_size", None)
    kwargs = dict(m, num_classes=num_classes, input_size=(image_size, image_size))
    if name in ("LinearDualDINOv3", "LinearDualLoRADINOv3"):
        kwargs.setdefault("num_morphology", num_morph)
        kwargs.setdefault("morphology_names", [f"m{i}" for i in range(num_morph)])
    return getattr(models_mod, name)(**kwargs), name


def build_loss(train_cfg: dict, num_classes: int):
    loss_block = dict(train_cfg["training"]["loss"])
    loss_type = loss_block.pop("type")
    loss_cls = getattr(loss_mod, loss_type)
    # Pass the synthetic hierarchy explicitly so the loss never calls the MERMAID API.
    loss = loss_cls(id2label=ID2LABEL, hierarchy=HIERARCHY, **loss_block)
    return loss, loss_type


def main() -> None:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--model-config", required=True)
    ap.add_argument("--training-config", required=True)
    ap.add_argument("--steps", type=int, default=3)
    ap.add_argument("--batch-size", type=int, default=2)
    ap.add_argument("--image-size", type=int, default=128, help="small for speed; must be /16")
    ap.add_argument("--num-morphology", type=int, default=5)
    ap.add_argument("--device", default="cpu")
    args = ap.parse_args()

    device = torch.device(args.device)
    num_classes = max(ID2LABEL) + 1
    torch.manual_seed(0)

    model_cfg = _load(args.model_config)
    train_cfg = _load(args.training_config)
    model, model_name = build_model(model_cfg, num_classes, args.image_size, args.num_morphology)
    model = model.to(device).train()
    loss_fn, loss_type = build_loss(train_cfg, num_classes)
    loss_fn = loss_fn.to(device)  # co-locate hierarchy buffers with the batch
    is_dual = "Dual" in loss_type
    print(
        f"model={model_name}  loss={loss_type}  classes={num_classes}  dual={is_dual}  device={device}"
    )

    trainable = [p for p in model.parameters() if p.requires_grad]
    opt = torch.optim.AdamW(trainable, lr=1e-3)

    H = W = args.image_size

    def synth_batch():
        images = torch.randn(args.batch_size, 3, H, W, device=device)
        targets = torch.randint(0, num_classes, (args.batch_size, H, W), device=device)
        targets[:, :4, :4] = 0  # a few ignore pixels
        return images, targets

    print(f"\n--- {args.steps} train steps (synthetic batches) ---")
    for step in range(args.steps):
        images, targets = synth_batch()
        out = model(images)
        if is_dual:
            morph_logits = out.morphology_logits.float()
            morph_targets = torch.randint(0, 3, morph_logits.shape, device=device).float()
            loss, comps = loss_fn(out.logits.float(), targets, morph_logits, morph_targets)
        else:
            loss, comps = loss_fn(out.logits.float(), targets)
        opt.zero_grad()
        loss.backward()
        gnorm = torch.nn.utils.clip_grad_norm_(trainable, 1e9)  # measure, don't clip
        opt.step()
        assert torch.isfinite(loss), "loss is not finite"
        assert torch.isfinite(gnorm), "gradient is not finite"
        comp_str = "  ".join(f"{k}={v:.4f}" for k, v in comps.items() if k != "taxonomical_total")
        print(f"  step {step}: total={loss.item():.4f}  |grad|={gnorm.item():.3f}  {comp_str}")

    print("\n--- eval step (hierarchy metrics) ---")
    dist = build_distance_matrix(ID2LABEL, HIERARCHY, ignore_index=0, num_classes=num_classes)
    remaps = build_level_remaps(
        ID2LABEL,
        HIERARCHY,
        train_cfg["training"]["loss"].get("level_names"),
        ignore_index=0,
        num_classes=num_classes,
    )
    ev = Evaluator(
        num_classes=num_classes,
        device=str(device),
        ignore_index=0,
        hierarchy_metrics=True,
        distance_matrix=dist,
        level_remaps=remaps,
    )
    with torch.no_grad():
        images, targets = synth_batch()
        preds = model(images).logits.argmax(dim=1)
        ev.accumulate(preds, targets)
        results = ev.compute_and_reset()
    for k, v in results.items():
        print(f"  {k} = {v:.4f}")

    print("\nPASS — loss forward/backward, optimizer step, and hierarchy eval all ran offline.")


if __name__ == "__main__":
    main()
