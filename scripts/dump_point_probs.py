#!/usr/bin/env python3
"""Dump per-point taxonomic concept probabilities on CoralNet val sources.

Uses the same model loading, preprocessing and point sampling as ``scripts/evaluate.py``
(model config at the given resolution, AMP), so accuracies computed from the dump match the
eval's ``metrics.json``. Writes one ``.npz`` with, per annotated point: the source label name,
the image id, the concept label row and, for each checkpoint, the probabilities, restricted to
the channels of ``--ranks`` (default kingdom/phylum/class). ``--data-config`` supplies the val split's
image blacklist and annotation overrides (and the sources, when ``--sources`` is omitted), as in
``evaluate.py``.

Usage:
    python scripts/dump_point_probs.py --sources 4721 \
        --out eval_out/ft50k_hier/point_dumps/coralnet_4721_step50000.npz \
        baseline=/path/to/model_step50000 kl=nbs/model_checkpoints/cbm_ft50k_kl/model_step50000
    python scripts/dump_point_probs.py --data-config configs/data_config_dedup.yaml \
        --concept-id2name eval_out/scratch100k_hier/concept_id2name.json \
        --out eval_out/scratch100k_hier/point_dumps/coralnet_val_step100000.npz \
        base=nbs/model_checkpoints/cbm_scratch100k_base_cd/model_epoch0
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
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

DEFAULT_RANKS = ("kingdom", "phylum", "class")


class _NoClassLookup:
    """Class ids are not needed here; the dataset only requires the interface."""

    @staticmethod
    def class_id(_name: str) -> int:
        return 0


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("checkpoints", nargs="+", help="NAME=CHECKPOINT pairs")
    p.add_argument("--sources", type=int, nargs="+", default=None,
                   help="CoralNet source ids (default: the val whitelist_sources of --data-config)")
    p.add_argument("--data-config", type=Path, default=_REPO_ROOT / "configs/data_config_all.yaml")
    p.add_argument("--concept-id2name", type=Path, default=_REPO_ROOT / "demo/concept_id2name.json")
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--resolution", type=int, default=256)
    p.add_argument("--batch-size", type=int, default=8)
    p.add_argument("--num-workers", type=int, default=16)
    p.add_argument("--ranks", nargs="+", default=list(DEFAULT_RANKS), help="taxonomic ranks to keep")
    args = p.parse_args()
    ckpts = dict(a.split("=", 1) for a in args.checkpoints)

    data = yaml.safe_load(args.data_config.read_text())
    data = data.get("data", data)
    setup_local_cache(data)
    val = (data.get("coralnet") or {}).get("val") or {}
    sources = args.sources or val.get("whitelist_sources")
    if not sources:
        p.error(f"--sources not given and {args.data_config} has no coralnet val whitelist_sources")
    id2name = json.loads(args.concept_id2name.read_text())
    concept_names = [id2name[k] for k in sorted(id2name, key=int)]
    keep = [i for i, n in enumerate(concept_names) if n.split("__")[0] in args.ranks]

    base = CoralNetDataset(whitelist_sources=sources, image_blacklist=val.get("image_blacklist"),
                           annotation_overrides=val.get("annotation_overrides"), transform=None, split="val")
    names = base.df_annotations["source_label_name"].astype(str).str.lower().unique().tolist()
    concept_lookup = build_concept_lookup(
        "coralnet", names, concept_names, _REPO_ROOT / "configs/class_to_concepts.csv"
    )
    size = (args.resolution, args.resolution)
    ds = PointEvalDataset(base, "source_id", ["image_id"], size, _NoClassLookup(), concept_lookup)
    labels_of = {
        key[0] if isinstance(key, tuple) else key: g["source_label_name"].astype(str).str.lower().tolist()
        for key, g in base.df_annotations.groupby(["image_id"])
    }

    device = torch.device("cuda")
    result: dict[str, np.ndarray] = {"channel_names": np.array([concept_names[i] for i in keep])}
    for run, ckpt in ckpts.items():
        artifacts = load_artifacts(
            checkpoint=ckpt,
            model_config=_REPO_ROOT / f"configs/model_config_cbm_dpt_lora_vitl_{args.resolution}.yaml",
            id2label=_REPO_ROOT / "demo/id2label.json",
            concept_id2name=args.concept_id2name,
        )
        predictor = CBMPredictor(build_model(artifacts, device), input_size=size, device=device, amp=True)
        loader = DataLoader(ds, batch_size=args.batch_size, num_workers=args.num_workers,
                            collate_fn=_collate, worker_init_fn=_worker_init)
        probs, rows, labels, images, sources = [], [], [], [], []
        img_idx = 0
        for batch in loader:
            _, concept_probs = predictor.forward_probs(batch["images"])
            for b in range(concept_probs.shape[0]):
                meta = ds.df_images.iloc[img_idx]
                img_idx += 1
                if not batch["ok"][b] or batch["rows"][b].shape[0] == 0:
                    continue
                ks = predictor.sample_at_points(concept_probs[b], batch["rows"][b], batch["cols"][b],
                                                batch["H"][b], batch["W"][b])
                probs.append(ks[keep].T.cpu().numpy().astype(np.float32))
                n = batch["rows"][b].shape[0]
                rows.append(batch["concept_rows"][b][:, keep])
                lab = labels_of[meta["image_id"]]
                # Out-of-bounds points are dropped by the dataset; labels align only if none were.
                labels += lab if len(lab) == n else ["<misaligned>"] * n
                images += [meta["image_id"]] * n
                sources += [int(meta["source_id"])] * n
        result[f"probs_{run}"] = np.concatenate(probs)
        result.setdefault("rows", np.concatenate(rows))
        result.setdefault("labels", np.array(labels))
        result.setdefault("image_id", np.array(images))
        result.setdefault("source_id", np.array(sources))
        print(run, result[f"probs_{run}"].shape, flush=True)
        del predictor
        torch.cuda.empty_cache()

    args.out.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(args.out, **result)
    print("saved", args.out)


if __name__ == "__main__":
    main()
