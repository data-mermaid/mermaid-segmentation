#!/usr/bin/env python3
"""Live hard coral genera and families: annotations vs a model, per point and per image, for diversity analyses.

Live hard coral = ``order__scleractinia`` and ``live`` (as in ``coral_cover_predict.py``):
- annotated: both concepts positive in the point's concept row (dead coral, ``live`` negative, is not live coral);
- predicted: order head argmax is scleractinia and ``P(live) > 0.5``.
A live coral point / pixel is assigned the genus (family) of the genus (family) head's argmax; the analysis keeps
only scleractinian genera / families (``order = scleractinia`` in the concept CSV).

Writes four tables:
- ``--out`` (``<stem>.parquet``): one row per annotated point: ``source_id``, ``image_id``, ``gt_live``,
  ``gt_genus`` / ``gt_family`` (the positive genus / family concept, "" if not given), ``pred_live``,
  ``pred_genus`` / ``pred_family`` (head argmax, without the live gate), and ``pred_genus_p1`` / ``pred_genus_p2``
  (likewise family): the head's largest and second-largest probability (each level is a softmax);
- ``<stem>_dense.parquet``: one row per image, level and scleractinian taxon with any predicted pixels: ``share`` of
  all pixels (at the model input resolution) that are predicted live hard coral with that taxon as argmax;
- ``<stem>_hist.parquet``: the same pixels split by confidence, so any threshold on the top probability or on the
  margin to the runner-up can be applied afterwards: one row per image, level, taxon, ``p1_bin`` and ``margin_bin``
  with any pixels; bin k covers [k / N_BINS, (k + 1) / N_BINS) of p1 (top probability) and of p1 − p2. Summing
  ``share`` over the bins gives ``<stem>_dense.parquet``;
- ``<stem>_images.parquet``: one row per image: ``n_points``, ``dense_live`` (share of pixels predicted live coral).

Usage:
    python scripts/taxon_diversity_predict.py --split test --sources 3697 3606 \
        --checkpoint nbs/model_checkpoints/cbm_scratch100k_kl_w0.5_taxv2_cd/model_epoch0 \
        --concept-id2name eval_out/scratch100k_hier/cbm_scratch100k_kl_w0.5_taxv2/concept_id2name.json \
        --class-to-concepts nbs/model_checkpoints/cbm_scratch100k_kl_w0.5_taxv2/class_to_concepts.csv \
        --out eval_out/taxon_diversity/test_kl_w0.5_taxv2_step100000_viable.parquet
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
LIVE_THRESHOLD = 0.5
LEVELS = ("genus", "family")
N_BINS = 20  # confidence bins of width 0.05 for the top probability and the margin


class _NoClassLookup:
    """Class ids are not needed here; the dataset only requires the interface."""

    @staticmethod
    def class_id(_name: str) -> int:
        return 0


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
    order_idx = [i for i, n in enumerate(concept_names) if n.startswith("order__")]
    hard_in_order = [concept_names[i] for i in order_idx].index(HARD_CORAL)
    live_ch = concept_names.index(LIVE)
    hard_ch = concept_names.index(HARD_CORAL)

    csv = pd.read_csv(args.class_to_concepts, dtype=str)
    scler = csv[csv["order"].str.lower() == "scleractinia"]
    idx, names, keep = {}, {}, {}
    for lvl in LEVELS:
        idx[lvl] = [i for i, n in enumerate(concept_names) if n.startswith(f"{lvl}__")]
        names[lvl] = np.array([concept_names[i].removeprefix(f"{lvl}__") for i in idx[lvl]])
        taxa = set(scler[lvl].dropna().str.lower()) - {"not_given", "none"}
        keep[lvl] = np.array([n in taxa for n in names[lvl]])

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

    points, dense_rows, image_rows, hist_parts = [], [], [], []
    img_idx = 0
    with torch.no_grad():
        for batch in loader:
            _, probs = predictor.forward_probs(batch["images"])
            live_mask = (probs[:, order_idx].argmax(dim=1) == hard_in_order) & (probs[:, live_ch] > LIVE_THRESHOLD)
            n_pix = live_mask[0].numel()
            shares, hist = {}, {}
            for lvl in LEVELS:
                n_taxa = len(idx[lvl])
                top = probs[:, idx[lvl]].topk(2, dim=1)  # values / indices (B, 2, H, W)
                am = top.indices[:, 0]  # argmax (B, H, W)
                am = torch.where(live_mask, am, torch.full_like(am, n_taxa))  # not live coral -> extra bin
                counts = torch.stack([torch.bincount(a.flatten(), minlength=n_taxa + 1) for a in am])
                shares[lvl] = (counts[:, :-1].float() / n_pix).cpu().numpy()
                # Pixel counts per (taxon, p1 bin, margin bin); not live coral -> one extra bin at the end.
                p1, p2 = top.values[:, 0], top.values[:, 1]
                b1 = (p1 * N_BINS).long().clamp(0, N_BINS - 1)
                bm = ((p1 - p2) * N_BINS).long().clamp(0, N_BINS - 1)
                code = (am * N_BINS + b1) * N_BINS + bm
                code = torch.where(live_mask, code, torch.full_like(code, n_taxa * N_BINS * N_BINS))
                hist[lvl] = [torch.bincount(c.flatten(), minlength=n_taxa * N_BINS * N_BINS + 1)[:-1] for c in code]
            dense_live = live_mask.float().mean(dim=(1, 2)).cpu().numpy()
            for b in range(probs.shape[0]):
                meta = ds.df_images.iloc[img_idx]
                img_idx += 1
                if not batch["ok"][b] or batch["rows"][b].shape[0] == 0:
                    continue
                sid, iid = int(meta["source_id"]), int(meta["image_id"])
                rows = np.asarray(batch["concept_rows"][b])
                at = predictor.sample_at_points(probs[b], batch["rows"][b], batch["cols"][b], batch["H"][b], batch["W"][b])
                rec = {"source_id": sid, "image_id": iid,
                       "gt_live": (rows[:, hard_ch] == 2) & (rows[:, live_ch] == 2),
                       "pred_live": ((at[order_idx].argmax(dim=0) == hard_in_order)
                                     & (at[live_ch] > LIVE_THRESHOLD)).cpu().numpy()}
                for lvl in LEVELS:
                    pos = rows[:, idx[lvl]] == 2
                    rec[f"gt_{lvl}"] = np.where(pos.any(axis=1), names[lvl][pos.argmax(axis=1)], "")
                    top_pt = at[idx[lvl]].topk(2, dim=0)
                    rec[f"pred_{lvl}"] = names[lvl][top_pt.indices[0].cpu().numpy()]
                    rec[f"pred_{lvl}_p1"] = top_pt.values[0].cpu().numpy().astype(np.float32)
                    rec[f"pred_{lvl}_p2"] = top_pt.values[1].cpu().numpy().astype(np.float32)
                    for g in np.flatnonzero(keep[lvl] & (shares[lvl][b] > 0)):
                        dense_rows.append((sid, iid, lvl, names[lvl][g], float(shares[lvl][b, g])))
                    nz = torch.nonzero(hist[lvl][b]).squeeze(1)
                    cnt = hist[lvl][b][nz].cpu().numpy()
                    nz = nz.cpu().numpy()
                    g = nz // (N_BINS * N_BINS)
                    sel = keep[lvl][g]
                    hist_parts.append(pd.DataFrame({
                        "source_id": sid, "image_id": iid, "level": lvl, "taxon": names[lvl][g[sel]],
                        "p1_bin": ((nz[sel] // N_BINS) % N_BINS).astype(np.int8),
                        "margin_bin": (nz[sel] % N_BINS).astype(np.int8),
                        "share": (cnt[sel] / n_pix).astype(np.float32)}))
                points.append(pd.DataFrame(rec))
                image_rows.append({"source_id": sid, "image_id": iid, "n_points": int(len(rows)),
                                   "dense_live": float(dense_live[b])})

    args.out.parent.mkdir(parents=True, exist_ok=True)
    pts = pd.concat(points, ignore_index=True)
    for c in ("gt_genus", "gt_family", "pred_genus", "pred_family"):
        pts[c] = pts[c].astype("category")
    pts.to_parquet(args.out, index=False)
    stem = args.out.with_suffix("")
    pd.DataFrame(dense_rows, columns=["source_id", "image_id", "level", "taxon", "share"]).to_parquet(
        f"{stem}_dense.parquet", index=False)
    hist_df = pd.concat(hist_parts, ignore_index=True)
    for c in ("level", "taxon"):
        hist_df[c] = hist_df[c].astype("category")
    hist_df.to_parquet(f"{stem}_hist.parquet", index=False)
    pd.DataFrame(image_rows).to_parquet(f"{stem}_images.parquet", index=False)
    print(f"{len(image_rows)} images, {len(pts)} points, {len(dense_rows)} dense rows -> {stem}*.parquet")


if __name__ == "__main__":
    main()
