#!/usr/bin/env python3
"""Save model predictions for the point review app (``scripts/review_app.py``), so reviewing needs no GPU.

For each sampled image (same sampling as ``qualitative_overlays.py`` / ``Coral_Cover_Qualitative_Live_10src.ipynb``)
writes to ``--out``:

- ``images/<image_id>.jpg``: the 2048 px resized CoralNet image the annotations refer to;
- ``maps/<image_id>_<map>.png``: probability maps at the model input resolution, 8-bit (value / 255), for
  ``hard`` = P(order = scleractinia), ``live`` = P(live) and ``live_hard`` = their product;
- ``points/<image_id>.json``: one entry per annotated point: position, CoralNet and MERMAID class, annotation code
  (2 positive, 1 negative, 0 not given) for the target, probabilities and each map's call at the point;

plus ``manifest.json`` (run, target, per-image summary). Reviews are kept separately by the app (``reviews.csv``), so
re-exporting does not touch them as long as the images and their points are unchanged.

Usage (the 10-image demo; ``--n-per-source 100`` for the full set):
    python scripts/review_export.py --target live_hard --n-per-source 1 \
        --out eval_out/coral_cover/review/base_taxv2_live
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parent))
import qualitative_overlays as qo  # noqa: E402


def _png(p: np.ndarray, path: Path) -> None:
    Image.fromarray(np.clip(np.rint(p * 255), 0, 255).astype(np.uint8), mode="L").save(path, optimize=True)


def export_image(pred: dict, out: Path, jpeg_quality: int = 90) -> dict:
    s = pred["summary"]
    iid = str(s["image_id"])
    Image.fromarray(pred["image"]).save(out / "images" / f"{iid}.jpg", quality=jpeg_quality)
    for name, p in pred["maps"].items():
        _png(p, out / "maps" / f"{iid}_{name}.png")
    pts = pred["points"]
    records = [{
        "idx": int(i), "row": int(r.row), "col": int(r.col),
        "coralnet": str(r.coralnet_name), "mermaid": str(r.mermaid), "gt_code": int(r.gt_code),
        "hard_code": int(r.hard_code), "live_code": int(r.live_code),
        "p_hard": round(float(r.p_scleractinia), 4), "p_live": round(float(r.p_live), 4),
        "p_live_hard": round(float(r.p_live_hard), 4), "pred_order": str(r.pred_order),
        "call_hard": bool(r.call_hard), "call_live": bool(r.call_live), "call_live_hard": bool(r.call_live_hard),
    } for i, r in enumerate(pts.itertuples())]
    (out / "points" / f"{iid}.json").write_text(json.dumps(records))
    keep = ("source_id", "image_id", "H", "W", "n_points", "gt_cover", "model_points_cover", "model_dense_cover")
    return {**{k: s[k] for k in keep}, **{k: s[k] for k in s if k.startswith(("points_", "dense_"))},
            "image_id": iid}


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--run", default="cbm_scratch100k_base_taxv2")
    p.add_argument("--step-dir", default=None)
    p.add_argument("--split", default="test", choices=["val", "test"])
    p.add_argument("--sources", type=int, nargs="+", default=qo.SOURCES_10)
    p.add_argument("--n-per-source", type=int, default=1)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--resolution", type=int, default=256)
    p.add_argument("--target", default="live_hard", choices=list(qo.TARGETS))
    p.add_argument("--mermaid-map", default="csv", choices=["csv", "json"])
    p.add_argument("--out", type=Path, required=True)
    args = p.parse_args()

    for sub in ("images", "maps", "points"):
        (args.out / sub).mkdir(parents=True, exist_ok=True)
    ctx = qo.load_context(args.run, args.sources, args.split, args.resolution, args.step_dir, args.mermaid_map,
                          target=args.target)
    sample = qo.sample_images(ctx, args.n_per_source, args.seed)
    print(f"{len(sample)} images from {sample.source_id.nunique()} sources -> {args.out}")
    images = []
    for i, meta in sample.iterrows():
        try:
            images.append(export_image(qo.predict_image(ctx, meta), args.out))
        except Exception as e:  # noqa: BLE001  (unreadable image: skip, keep going)
            print(f"skip source {meta.source_id} image {meta.image_id}: {e}")
        if (i + 1) % 25 == 0:
            print(f"  {i + 1}/{len(sample)}")
    manifest = {
        "run": args.run, "step_dir": args.step_dir or f"{args.run}_cd/model_epoch0", "target": args.target,
        "target_name": qo.TARGETS[args.target][0], "resolution": args.resolution, "split": args.split,
        "seed": args.seed, "n_per_source": args.n_per_source, "mermaid_map": args.mermaid_map,
        "maps": {k: v[0] for k, v in qo.PANELS.items()}, "bands": qo.BANDS,
        "created": time.strftime("%Y-%m-%d %H:%M:%S"), "images": images,
    }
    (args.out / "manifest.json").write_text(json.dumps(manifest, indent=1))
    print(f"{len(images)} images -> {args.out / 'manifest.json'}")


if __name__ == "__main__":
    main()
