#!/usr/bin/env python3
"""Per-image hard coral cover predicted by a model, next to the ground truth at the same points.

For every image of the given CoralNet sources (same preprocessing and point sampling as
``evaluate.py``), with hard coral = ``order__scleractinia``:

- ``n_hard_gt``: annotated points whose concept row is positive for hard coral (ground truth);
- ``n_hard_pred``: annotated points where the order head's argmax is hard coral;
- ``p_hard_sum``: sum over annotated points of the predicted hard coral probability;
- ``dense_hard``: share of all pixels (at the model input resolution) whose order argmax is hard coral;
- ``dense_p``: mean predicted hard coral probability over all pixels.

Live hard coral (``order__scleractinia`` and ``live``): annotated as live hard coral when both concepts are positive
(dead coral, ``live`` negative, is not); predicted when the order argmax is hard coral and ``P(live) > 0.5``:

- ``n_live_gt`` / ``n_live_pred``: annotated points that are / are predicted live hard coral;
- ``dense_live``: share of all pixels predicted live hard coral.

``--points-out`` also writes one row per annotated point (label, ground truth, prediction, probability).

Usage:
    python scripts/coral_cover_predict.py --split test --sources 4957 3606 3697 \
        --checkpoint nbs/model_checkpoints/cbm_scratch100k_base_cd/model_epoch0 \
        --out eval_out/coral_cover/coralnet_test_pred_base_step100000.parquet
For a run trained on another concept CSV, pass its concept map and pinned CSV:
    --concept-id2name eval_out/scratch100k_hier/<RUN>/concept_id2name.json \
    --class-to-concepts nbs/model_checkpoints/<RUN>/class_to_concepts.csv
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import yaml
from torch.utils.data import DataLoader

_REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path[:0] = [str(_REPO_ROOT), str(_REPO_ROOT / "demo")]

from inference import build_model, load_artifacts  # noqa: E402  (demo module)

from mermaidseg.datasets.coralnet.coralnet_dataset import CoralNetDataset  # noqa: E402
from mermaidseg.datasets.local_cache import setup_local_cache  # noqa: E402
from mermaidseg.evaluation.gt_mapping import build_concept_lookup  # noqa: E402
from mermaidseg.evaluation.point_datasets import PointEvalDataset, _collate, _worker_init  # noqa: E402
from mermaidseg.evaluation.predictor import CBMPredictor  # noqa: E402

HARD_CORAL = "order__scleractinia"
LIVE = "live"
LIVE_THRESHOLD = 0.5  # binary concepts are positive when prob > threshold (as in evaluation.metrics)


class _NoClassLookup:
    """Class ids are not needed here; the dataset only requires the interface."""

    @staticmethod
    def class_id(_name: str) -> int:
        return 0


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--split", default="test", choices=["val", "test"])
    p.add_argument("--sources", type=int, nargs="+", default=None,
                   help="CoralNet source ids (default: the split's whitelist_sources)")
    p.add_argument("--data-config", type=Path, default=_REPO_ROOT / "configs/data_config_dedup.yaml")
    p.add_argument("--concept-id2name", type=Path,
                   default=_REPO_ROOT / "eval_out/scratch100k_hier/concept_id2name.json")
    p.add_argument("--class-to-concepts", type=Path, default=_REPO_ROOT / "configs/class_to_concepts.csv",
                   help="concept CSV for the ground truth (the *_taxv2 runs pin theirs in <RUN>/class_to_concepts.csv)")
    p.add_argument("--resolution", type=int, default=256)
    p.add_argument("--batch-size", type=int, default=16)
    p.add_argument("--num-workers", type=int, default=16)
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--points-out", type=Path, default=None, help="optional per-point parquet")
    args = p.parse_args()

    data = yaml.safe_load(args.data_config.read_text())
    data = data.get("data", data)
    setup_local_cache(data)
    split = (data.get("coralnet") or {}).get(args.split) or {}
    base = CoralNetDataset(whitelist_sources=args.sources or split.get("whitelist_sources"),
                           image_blacklist=split.get("image_blacklist"),
                           annotation_overrides=split.get("annotation_overrides"),
                           transform=None, split=args.split)

    id2name = json.loads(args.concept_id2name.read_text())
    concept_names = [id2name[k] for k in sorted(id2name, key=int)]
    order_idx = [i for i, n in enumerate(concept_names) if n.startswith("order__")]
    hard_in_order = [concept_names[i] for i in order_idx].index(HARD_CORAL)
    hard_channel = concept_names.index(HARD_CORAL)
    live_channel = concept_names.index(LIVE)

    names = base.df_annotations["source_label_name"].astype(str).str.lower().unique().tolist()
    concept_lookup = build_concept_lookup("coralnet", names, concept_names, args.class_to_concepts)
    size = (args.resolution, args.resolution)
    ds = PointEvalDataset(base, "source_id", ["image_id"], size, _NoClassLookup(), concept_lookup)

    device = torch.device("cuda")
    artifacts = load_artifacts(
        checkpoint=args.checkpoint,
        model_config=_REPO_ROOT / f"configs/model_config_cbm_dpt_lora_vitl_{args.resolution}.yaml",
        id2label=_REPO_ROOT / "demo/id2label.json",
        concept_id2name=args.concept_id2name,
    )
    predictor = CBMPredictor(build_model(artifacts, device), input_size=size, device=device, amp=True)
    labels_of = {
        key[0] if isinstance(key, tuple) else key: g["source_label_name"].astype(str).str.lower().tolist()
        for key, g in base.df_annotations.groupby(["image_id"])
    }
    loader = DataLoader(ds, batch_size=args.batch_size, num_workers=args.num_workers,
                        collate_fn=_collate, worker_init_fn=_worker_init)

    rows, points = [], []
    img_idx = 0
    with torch.no_grad():
        for batch in loader:
            _, concept_probs = predictor.forward_probs(batch["images"])
            order_probs = concept_probs[:, order_idx]
            dense_is_hard = order_probs.argmax(dim=1) == hard_in_order
            dense_hard = dense_is_hard.float().mean(dim=(1, 2)).cpu().numpy()
            dense_p = concept_probs[:, hard_channel].mean(dim=(1, 2)).cpu().numpy()
            dense_live = (dense_is_hard & (concept_probs[:, live_channel] > LIVE_THRESHOLD)).float().mean(dim=(1, 2))
            dense_live = dense_live.cpu().numpy()
            for b in range(concept_probs.shape[0]):
                meta = ds.df_images.iloc[img_idx]
                img_idx += 1
                if not batch["ok"][b] or batch["rows"][b].shape[0] == 0:
                    continue
                at_points = predictor.sample_at_points(concept_probs[b], batch["rows"][b], batch["cols"][b],
                                                       batch["H"][b], batch["W"][b])  # (C, n_points)
                pred_hard = at_points[order_idx].argmax(dim=0) == hard_in_order
                p_live = at_points[live_channel]
                pred_live = pred_hard & (p_live > LIVE_THRESHOLD)
                gt = np.asarray(batch["concept_rows"][b])[:, hard_channel]
                gt_live = np.asarray(batch["concept_rows"][b])[:, live_channel]
                rows.append({"source_id": int(meta["source_id"]), "image_id": int(meta["image_id"]),
                             "n_points": int(len(gt)), "n_hard_gt": int((gt == 2).sum()),
                             "n_hard_pred": int(pred_hard.sum().item()),
                             "p_hard_sum": float(at_points[hard_channel].sum().item()),
                             "dense_hard": float(dense_hard[b]), "dense_p": float(dense_p[b]),
                             "n_live_gt": int(((gt == 2) & (gt_live == 2)).sum()),
                             "n_live_pred": int(pred_live.sum().item()), "dense_live": float(dense_live[b])})
                if args.points_out is not None:
                    lab = labels_of[meta["image_id"]]
                    # Out-of-bounds points are dropped by the dataset; labels align only if none were.
                    lab = lab if len(lab) == len(gt) else ["<misaligned>"] * len(gt)
                    points.append(pd.DataFrame({
                        "source_id": int(meta["source_id"]), "image_id": int(meta["image_id"]), "label": lab,
                        "gt_code": gt.astype(np.int8), "pred_hard": pred_hard.cpu().numpy(),
                        "p_hard": at_points[hard_channel].cpu().numpy().astype(np.float32),
                        "gt_live_code": gt_live.astype(np.int8), "p_live": p_live.cpu().numpy().astype(np.float32),
                        "pred_live_hard": pred_live.cpu().numpy(),
                        "pred_order": np.array([concept_names[order_idx[i]].removeprefix("order__")
                                                for i in at_points[order_idx].argmax(dim=0).cpu().numpy()])}))

    out = pd.DataFrame(rows)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    out.to_parquet(args.out, index=False)
    print(f"{len(out)} images, {out.n_points.sum()} points -> {args.out}")
    if args.points_out is not None:
        pd.concat(points, ignore_index=True).to_parquet(args.points_out, index=False)
        print(f"points -> {args.points_out}")


if __name__ == "__main__":
    main()
