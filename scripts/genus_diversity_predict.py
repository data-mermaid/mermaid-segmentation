#!/usr/bin/env python3
"""Per-image hard coral genera: ground truth at the annotated points vs a model, for diversity analyses.

For every image of the given CoralNet sources (same preprocessing and point sampling as ``evaluate.py``), and every
scleractinian genus (``order = scleractinia`` in ``class_to_concepts.csv``):

- ``gt_points``: annotated points labelled with that genus (dead coral, ``live`` negative, excluded);
- ``pred_points``: the same annotated points (dead coral excluded) where the genus head's argmax is that genus;
- ``dense_share``: share of all pixels (at the model input resolution) where the genus head's argmax is that genus.

Writes a long table (one row per image and genus with any non-zero value) to ``--out`` and one row per image
(``n_points``, ``n_dead``) to ``<out stem>_images.parquet``. A presence threshold on ``dense_share`` is applied at
analysis time.

Usage:
    python scripts/genus_diversity_predict.py --split test --sources 3697 3606 \
        --checkpoint nbs/model_checkpoints/cbm_scratch100k_base_cd/model_epoch0 \
        --out eval_out/genus_diversity/test_base_step100000.parquet
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

CSV = _REPO_ROOT / "configs/class_to_concepts.csv"


class _NoClassLookup:
    """Class ids are not needed here; the dataset only requires the interface."""

    @staticmethod
    def class_id(_name: str) -> int:
        return 0


def coral_genera() -> set[str]:
    df = pd.read_csv(CSV)
    return set(df.loc[(df["order"] == "scleractinia") & ~df["genus"].isin(["not_given", "none"]), "genus"])


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--split", default="test", choices=["val", "test"])
    p.add_argument("--sources", type=int, nargs="+", required=True)
    p.add_argument("--data-config", type=Path, default=_REPO_ROOT / "configs/data_config_dedup.yaml")
    p.add_argument("--concept-id2name", type=Path,
                   default=_REPO_ROOT / "eval_out/scratch100k_hier/concept_id2name.json")
    p.add_argument("--resolution", type=int, default=256)
    p.add_argument("--batch-size", type=int, default=16)
    p.add_argument("--num-workers", type=int, default=16)
    p.add_argument("--out", type=Path, required=True)
    args = p.parse_args()

    data = yaml.safe_load(args.data_config.read_text())
    data = data.get("data", data)
    setup_local_cache(data)
    split = (data.get("coralnet") or {}).get(args.split) or {}
    base = CoralNetDataset(whitelist_sources=args.sources, image_blacklist=split.get("image_blacklist"),
                           annotation_overrides=split.get("annotation_overrides"), transform=None, split=args.split)

    id2name = json.loads(args.concept_id2name.read_text())
    concept_names = [id2name[k] for k in sorted(id2name, key=int)]
    genus_idx = [i for i, n in enumerate(concept_names) if n.startswith("genus__")]
    genus_names = [concept_names[i].removeprefix("genus__") for i in genus_idx]
    keep = np.array([g in coral_genera() for g in genus_names])  # scleractinian genera among the channels
    live_channel = concept_names.index("live")

    names = base.df_annotations["source_label_name"].astype(str).str.lower().unique().tolist()
    concept_lookup = build_concept_lookup("coralnet", names, concept_names, CSV)
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
    loader = DataLoader(ds, batch_size=args.batch_size, num_workers=args.num_workers,
                        collate_fn=_collate, worker_init_fn=_worker_init)

    long_rows, image_rows = [], []
    img_idx = 0
    n_genus = len(genus_idx)
    with torch.no_grad():
        for batch in loader:
            _, concept_probs = predictor.forward_probs(batch["images"])
            dense_argmax = concept_probs[:, genus_idx].argmax(dim=1)  # (B, H, W)
            dense_share = torch.stack([torch.bincount(a.flatten(), minlength=n_genus) for a in dense_argmax]).float()
            dense_share = (dense_share / dense_argmax[0].numel()).cpu().numpy()
            for b in range(concept_probs.shape[0]):
                meta = ds.df_images.iloc[img_idx]
                img_idx += 1
                if not batch["ok"][b] or batch["rows"][b].shape[0] == 0:
                    continue
                rows = np.asarray(batch["concept_rows"][b])
                dead = rows[:, live_channel] == 1  # live = FALSE
                at_points = predictor.sample_at_points(concept_probs[b], batch["rows"][b], batch["cols"][b],
                                                       batch["H"][b], batch["W"][b])
                pred = at_points[genus_idx].argmax(dim=0).cpu().numpy()
                gt_rows = rows[:, genus_idx] == 2
                gt_points = gt_rows[~dead].sum(axis=0)
                pred_points = np.bincount(pred[~dead], minlength=n_genus)
                source_id, image_id = int(meta["source_id"]), int(meta["image_id"])
                image_rows.append({"source_id": source_id, "image_id": image_id,
                                   "n_points": int(len(rows)), "n_dead": int(dead.sum())})
                nonzero = keep & ((gt_points > 0) | (pred_points > 0) | (dense_share[b] > 0))
                for g in np.flatnonzero(nonzero):
                    long_rows.append((source_id, image_id, genus_names[g], int(gt_points[g]), int(pred_points[g]),
                                      float(dense_share[b, g])))

    args.out.parent.mkdir(parents=True, exist_ok=True)
    out = pd.DataFrame(long_rows, columns=["source_id", "image_id", "genus", "gt_points", "pred_points", "dense_share"])
    out.to_parquet(args.out, index=False)
    images_out = args.out.with_name(args.out.stem + "_images.parquet")
    pd.DataFrame(image_rows).to_parquet(images_out, index=False)
    print(f"{len(image_rows)} images, {len(out)} image-genus rows -> {args.out}, {images_out}")


if __name__ == "__main__":
    main()
