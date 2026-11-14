#!/usr/bin/env python3
"""Per-pixel geographic gating: live hard coral genus pixel shares per image, base vs gated.

Same model, data and live-coral rule as ``scripts/taxon_diversity_predict.py`` (the KL05 diversity run). For every
pixel predicted live hard coral (order head = scleractinia and P(live) > 0.5), the genus is taken from the genus
head in four variants:

- ``base_argmax``: plain argmax (reproduces KL05's ``_dense.parquet``);
- ``base_thresh``: argmax if its probability > ``--thresh``, else "uncertain" (no genus);
- ``gate_argmax`` / ``gate_thresh``: the same after gating: genera the source's realm doesn't allow get probability
  0 and the rest are renormalized (``--allow``, written from ``inat/outputs/genus_realm_presence_combined.csv``).

Writes to ``--out-dir``:
- with ``--with-points``, also the point outputs of ``inat/gating_predict.py`` (``points.parquet``,
  ``genus_probs.npy``, ``genus_names.json``) from the same forward pass, so a slow (e.g. 768 px) model runs once;
- ``dense.parquet``: one row per image, variant and hard coral genus with any pixels: ``share`` of all pixels;
- ``dense_images.parquet``: one row per image: ``dense_live`` (share of pixels predicted live coral) and, per
  variant, ``unc_<variant>`` (share of pixels that are live coral but uncertain).

Usage:
    python inat/gating_dense_predict.py --sources 290 1990 ... --allow inat/data/gating/gate_allow.json \
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

VARIANTS = ("base_argmax", "base_thresh", "gate_argmax", "gate_thresh")


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--split", default="test", choices=["val", "test"])
    p.add_argument("--sources", type=int, nargs="+", required=True)
    p.add_argument("--allow", type=Path, required=True)
    p.add_argument("--thresh", type=float, default=0.5)
    p.add_argument("--data-config", type=Path, default=_REPO_ROOT / "configs/data_config_dedup.yaml")
    p.add_argument("--concept-id2name", type=Path, required=True)
    p.add_argument("--class-to-concepts", type=Path, required=True)
    p.add_argument("--resolution", type=int, default=256)
    p.add_argument("--batch-size", type=int, default=16)
    p.add_argument("--num-workers", type=int, default=16)
    p.add_argument("--out-dir", type=Path, required=True)
    p.add_argument("--with-points", action="store_true", help="also save genus probabilities at the points")
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
    n_g = len(g_idx)

    gate = json.loads(args.allow.read_text())
    assert gate["genera"] == g_names.tolist(), "gate file genus order differs from the model's genus channels"
    allow_by_source = {int(s): np.array(gate["allow"][r], dtype=bool) for s, r in gate["source_realm"].items()}
    missing = set(args.sources) - set(allow_by_source)
    assert not missing, f"sources without a realm: {missing}"

    csv = pd.read_csv(args.class_to_concepts, dtype=str)
    taxa = set(csv.loc[csv["order"].str.lower() == "scleractinia", "genus"].dropna().str.lower()) - {"not_given", "none"}
    keep = np.isin(g_names, list(taxa))

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
    UNC, NOT_LIVE = n_g, n_g + 1  # extra bins

    dense_rows, image_rows, points, probs_out = [], [], [], []
    img_idx = 0
    with torch.no_grad():
        for bi, batch in enumerate(loader):
            _, probs = predictor.forward_probs(batch["images"])
            nb = probs.shape[0]
            metas = [ds.df_images.iloc[img_idx + b] for b in range(nb)]
            img_idx += nb
            live = (probs[:, order_idx].argmax(dim=1) == hard_in_order) & (probs[:, live_ch] > LIVE_THRESHOLD)
            n_pix = live[0].numel()
            g = probs[:, g_idx].float()  # (B, G, H, W)
            allow = torch.from_numpy(np.stack([allow_by_source[int(m["source_id"])] for m in metas])).to(device)
            labels_v = {}
            for kind, mask in (("base", torch.ones_like(allow)), ("gate", allow)):
                m = mask[:, :, None, None]
                top = torch.where(m, g, torch.full_like(g, -1.0)).max(dim=1)
                conf = top.values / (g * m).sum(dim=1).clamp_min(1e-12)
                labels_v[f"{kind}_argmax"] = top.indices
                labels_v[f"{kind}_thresh"] = torch.where(conf > args.thresh, top.indices, torch.full_like(top.indices, UNC))
            counts = {}
            for v, lab in labels_v.items():
                lab = torch.where(live, lab, torch.full_like(lab, NOT_LIVE))
                counts[v] = torch.stack([torch.bincount(a.flatten(), minlength=n_g + 2) for a in lab]).cpu().numpy()
            dense_live = live.float().mean(dim=(1, 2)).cpu().numpy()
            for b, meta in enumerate(metas):
                if not batch["ok"][b] or batch["rows"][b].shape[0] == 0:
                    continue
                sid, iid = int(meta["source_id"]), int(meta["image_id"])
                row = {"source_id": sid, "image_id": iid, "n_points": int(batch["rows"][b].shape[0]),
                       "dense_live": float(dense_live[b])}
                for v in VARIANTS:
                    c = counts[v][b]
                    row[f"unc_{v}"] = float(c[UNC] / n_pix)
                    for j in np.flatnonzero(keep & (c[:n_g] > 0)):
                        dense_rows.append((sid, iid, v, g_names[j], float(c[j] / n_pix)))
                image_rows.append(row)
                if args.with_points:
                    rows = np.asarray(batch["concept_rows"][b])
                    at = predictor.sample_at_points(probs[b], batch["rows"][b], batch["cols"][b], batch["H"][b],
                                                    batch["W"][b])
                    gp = at[g_idx].T.float().cpu().numpy()  # (points, genera)
                    pos = rows[:, g_idx] == 2
                    points.append(pd.DataFrame({
                        "source_id": sid, "image_id": iid,
                        "gt_live": (rows[:, hard_ch] == 2) & (rows[:, live_ch] == 2),
                        "pred_live": ((at[order_idx].argmax(dim=0) == hard_in_order)
                                      & (at[live_ch] > LIVE_THRESHOLD)).cpu().numpy(),
                        "gt_genus": np.where(pos.any(axis=1), g_names[pos.argmax(axis=1)], ""),
                        "pred_genus": g_names[gp.argmax(axis=1)]}))
                    probs_out.append(gp.astype(np.float16))
            if bi % 100 == 0:
                print(f"batch {bi}, {img_idx}/{len(ds)} images", flush=True)

    args.out_dir.mkdir(parents=True, exist_ok=True)
    d = pd.DataFrame(dense_rows, columns=["source_id", "image_id", "variant", "taxon", "share"])
    for c in ("variant", "taxon"):
        d[c] = d[c].astype("category")
    d.to_parquet(args.out_dir / "dense.parquet", index=False)
    pd.DataFrame(image_rows).to_parquet(args.out_dir / "dense_images.parquet", index=False)
    if args.with_points:
        pd.concat(points, ignore_index=True).to_parquet(args.out_dir / "points.parquet", index=False)
        np.save(args.out_dir / "genus_probs.npy", np.concatenate(probs_out))
        (args.out_dir / "genus_names.json").write_text(json.dumps(g_names.tolist()))
        print(f"{sum(len(x) for x in points)} points -> {args.out_dir}")
    print(f"{len(image_rows)} images, {len(d)} dense rows -> {args.out_dir}")


if __name__ == "__main__":
    main()
