#!/usr/bin/env python3
"""Genus probabilities at annotated points, for geographic gating (inat/Genus_Geographic_Gating.ipynb).

Same model, data and live-coral rules as ``scripts/taxon_diversity_predict.py`` (the KL05 diversity run), but keeps the
full genus head softmax at every annotated point instead of only its argmax, so genera can be masked afterwards.

Writes to ``--out-dir``:
- ``points.parquet``: one row per annotated point: ``source_id``, ``image_id``, ``gt_live``, ``pred_live``,
  ``gt_genus`` ("" if not given), ``pred_genus`` (ungated argmax, to check against the KL05 parquet);
- ``genus_probs.npy``: float16 (points × genus channels), row-aligned with ``points.parquet``;
- ``genus_names.json``: the genus channel names (column order of ``genus_probs.npy``).

Usage:
    python inat/gating_predict.py --sources 290 1990 ... \
        --checkpoint nbs/model_checkpoints/cbm_scratch100k_kl_w0.5_taxv2_cd/model_epoch0 \
        --concept-id2name eval_out/scratch100k_hier/cbm_scratch100k_kl_w0.5_taxv2/concept_id2name.json \
        --class-to-concepts nbs/model_checkpoints/cbm_scratch100k_kl_w0.5_taxv2/class_to_concepts.csv \
        --out-dir inat/data/gating
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
sys.path[:0] = [str(_REPO_ROOT), str(_REPO_ROOT / "demo"), str(_REPO_ROOT / "scripts")]

from inference import build_model, load_artifacts  # noqa: E402  (demo module)
from taxon_diversity_predict import HARD_CORAL, LIVE, LIVE_THRESHOLD, _NoClassLookup  # noqa: E402

from mermaidseg.datasets.coralnet.coralnet_dataset import CoralNetDataset  # noqa: E402
from mermaidseg.datasets.local_cache import setup_local_cache  # noqa: E402
from mermaidseg.evaluation.gt_mapping import build_concept_lookup  # noqa: E402
from mermaidseg.evaluation.point_datasets import PointEvalDataset, _collate, _worker_init  # noqa: E402
from mermaidseg.evaluation.predictor import CBMPredictor  # noqa: E402


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--split", default="test", choices=["val", "test"])
    p.add_argument("--sources", type=int, nargs="+", required=True)
    p.add_argument("--data-config", type=Path, default=_REPO_ROOT / "configs/data_config_dedup.yaml")
    p.add_argument("--concept-id2name", type=Path, required=True)
    p.add_argument("--class-to-concepts", type=Path, required=True)
    p.add_argument("--resolution", type=int, default=256)
    p.add_argument("--batch-size", type=int, default=16)
    p.add_argument("--num-workers", type=int, default=16)
    p.add_argument("--out-dir", type=Path, required=True)
    args = p.parse_args()

    data = yaml.safe_load(args.data_config.read_text())
    data = data.get("data", data)
    setup_local_cache(data)
    split = (data.get("coralnet") or {}).get(args.split) or {}
    base = CoralNetDataset(whitelist_sources=args.sources, image_blacklist=split.get("image_blacklist"),
                           annotation_overrides=split.get("annotation_overrides"), transform=None, split=args.split)

    id2name = json.loads(args.concept_id2name.read_text())
    concept_names = [id2name[k] for k in sorted(id2name, key=int)]
    order_idx = [i for i, n in enumerate(concept_names) if n.startswith("order__")]
    hard_in_order = [concept_names[i] for i in order_idx].index(HARD_CORAL)
    live_ch = concept_names.index(LIVE)
    hard_ch = concept_names.index(HARD_CORAL)
    g_idx = [i for i, n in enumerate(concept_names) if n.startswith("genus__")]
    g_names = np.array([concept_names[i].removeprefix("genus__") for i in g_idx])

    labels = base.df_annotations["source_label_name"].astype(str).str.lower().unique().tolist()
    concept_lookup = build_concept_lookup("coralnet", labels, concept_names, args.class_to_concepts)
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

    points, probs_out = [], []
    img_idx = 0
    with torch.no_grad():
        for bi, batch in enumerate(loader):
            _, probs = predictor.forward_probs(batch["images"])
            for b in range(probs.shape[0]):
                meta = ds.df_images.iloc[img_idx]
                img_idx += 1
                if not batch["ok"][b] or batch["rows"][b].shape[0] == 0:
                    continue
                rows = np.asarray(batch["concept_rows"][b])
                at = predictor.sample_at_points(probs[b], batch["rows"][b], batch["cols"][b], batch["H"][b], batch["W"][b])
                gp = at[g_idx].T.float().cpu().numpy()  # (points, genera)
                pos = rows[:, g_idx] == 2
                points.append(pd.DataFrame({
                    "source_id": int(meta["source_id"]), "image_id": int(meta["image_id"]),
                    "gt_live": (rows[:, hard_ch] == 2) & (rows[:, live_ch] == 2),
                    "pred_live": ((at[order_idx].argmax(dim=0) == hard_in_order)
                                  & (at[live_ch] > LIVE_THRESHOLD)).cpu().numpy(),
                    "gt_genus": np.where(pos.any(axis=1), g_names[pos.argmax(axis=1)], ""),
                    "pred_genus": g_names[gp.argmax(axis=1)]}))
                probs_out.append(gp.astype(np.float16))
            if bi % 100 == 0:
                print(f"batch {bi}, {img_idx}/{len(ds)} images", flush=True)

    args.out_dir.mkdir(parents=True, exist_ok=True)
    pts = pd.concat(points, ignore_index=True)
    pts.to_parquet(args.out_dir / "points.parquet", index=False)
    np.save(args.out_dir / "genus_probs.npy", np.concatenate(probs_out))
    (args.out_dir / "genus_names.json").write_text(json.dumps(g_names.tolist()))
    print(f"{img_idx} images, {len(pts)} points -> {args.out_dir}")


if __name__ == "__main__":
    main()
