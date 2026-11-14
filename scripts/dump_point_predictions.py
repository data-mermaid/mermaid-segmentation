#!/usr/bin/env python3
"""Dump per-point predictions of one checkpoint on a CoralNet split, for offline analysis.

Same model loading, preprocessing, label mapping and point sampling as
``scripts/evaluate.py --evals coralnet``, but instead of reducing to metrics it writes one row
per annotated point (see :mod:`mermaidseg.evaluation.point_dump` for the layout). Load it with
``PointDump.load(dir)`` and compute any breakdown with ``summary_by`` / ``class_metrics`` /
``taxon_metrics`` / ``binary_metrics`` / ``confusions``.

Example:
    uv run python scripts/dump_point_predictions.py \
        --checkpoint nbs/model_checkpoints/cbm_scratch100k_base_cd/model_epoch0 \
        --model-config configs/model_config_cbm_dpt_lora_vitl_256.yaml \
        --concept-id2name eval_out/scratch100k_hier/concept_id2name.json \
        --data-config configs/data_config_dedup.yaml --split test \
        --output-dir eval_out/point_dumps/scratch100k_base/test --amp
"""

from __future__ import annotations

import argparse
import logging
import subprocess
import sys
from pathlib import Path

import torch
import yaml

_REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path[:0] = [str(_REPO_ROOT), str(_REPO_ROOT / "demo")]

from inference import build_model, load_artifacts  # noqa: E402  (demo module)

from mermaidseg.datasets.coralnet.coralnet_dataset import CoralNetDataset  # noqa: E402
from mermaidseg.datasets.local_cache import setup_local_cache  # noqa: E402
from mermaidseg.evaluation.gt_mapping import (  # noqa: E402
    build_class_lookup,
    build_concept_lookup,
    load_benthic_hierarchy,
)
from mermaidseg.evaluation.point_dump import dump_point_predictions  # noqa: E402
from mermaidseg.evaluation.predictor import CBMPredictor  # noqa: E402

logger = logging.getLogger("dump_point_predictions")


def _git_commit() -> str | None:
    try:
        return subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=_REPO_ROOT, text=True).strip()
    except Exception:  # noqa: BLE001
        return None


def main(argv: list[str] | None = None) -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--checkpoint", required=True, type=Path)
    p.add_argument("--model-config", type=Path, default=_REPO_ROOT / "configs/model_config_cbm_dpt_lora_vitl.yaml")
    p.add_argument("--id2label", type=Path, default=_REPO_ROOT / "demo/id2label.json")
    p.add_argument("--concept-id2name", type=Path, default=_REPO_ROOT / "demo/concept_id2name.json")
    p.add_argument("--data-config", type=Path, default=_REPO_ROOT / "configs/data_config_dedup.yaml")
    p.add_argument("--split", default="test", choices=["val", "test"])
    p.add_argument("--sources", type=int, nargs="+", default=None, help="Override the split's source ids")
    p.add_argument("--taxonomy-csv", type=Path, default=_REPO_ROOT / "configs/class_to_concepts.csv")
    p.add_argument("--benthic-hierarchy-json", type=Path, default=None)
    p.add_argument("--output-dir", required=True, type=Path)
    p.add_argument("--batch-size", type=int, default=8)
    p.add_argument("--num-workers", type=int, default=16)
    p.add_argument("--device", default=None)
    p.add_argument("--amp", action="store_true", help="bf16 autocast (as evaluate.py --amp)")
    p.add_argument("--no-concept-probs", action="store_true", help="Skip the (N, K) concept_probs.npy")
    p.add_argument("--max-images-per-unit", type=int, default=None, help="Debug limiter")
    args = p.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")

    raw = yaml.safe_load(args.data_config.read_text())
    data = raw.get("data", raw)
    setup_local_cache(data)
    split = (data.get("coralnet") or {}).get(args.split) or {}
    sources = args.sources or split.get("whitelist_sources")
    if not sources:
        p.error(f"{args.data_config} has no coralnet {args.split} whitelist_sources; pass --sources")

    device = torch.device(args.device or ("cuda" if torch.cuda.is_available() else "cpu"))
    artifacts = load_artifacts(
        checkpoint=args.checkpoint,
        model_config=args.model_config,
        id2label=args.id2label,
        concept_id2name=args.concept_id2name,
    )
    cfg = yaml.safe_load(args.model_config.read_text())
    size = tuple(int(x) for x in cfg.get("model", cfg).get("input_size", [512, 512]))
    predictor = CBMPredictor(build_model(artifacts, device), input_size=size, device=device, amp=args.amp)
    concept_names = [n for _, n in sorted(artifacts.concept_id2name.items(), key=lambda kv: int(kv[0]))]
    if len(concept_names) != predictor.num_concepts:
        raise ValueError(
            f"concept_id2name has {len(concept_names)} entries but the model has {predictor.num_concepts} "
            "concept channels. Pass the matching --concept-id2name."
        )

    base = CoralNetDataset(
        whitelist_sources=sources,
        image_blacklist=split.get("image_blacklist"),
        annotation_overrides=split.get("annotation_overrides"),
        transform=None,
        split="val",
    )
    names = base.df_annotations["source_label_name"].astype(str).str.lower().unique().tolist()
    hierarchy = load_benthic_hierarchy(args.benthic_hierarchy_json, output_dir=args.output_dir, fetch_remote=True)
    class_lookup = build_class_lookup("coralnet", names, artifacts.id2label, hierarchy)
    concept_lookup = build_concept_lookup("coralnet", names, concept_names, args.taxonomy_csv)
    logger.info(
        "CoralNet %s: %d sources, %d images, %d annotations, %d labels; input_size=%s",
        args.split, len(sources), len(base.df_images), len(base.df_annotations), len(names), size,
    )

    dump_point_predictions(
        predictor=predictor,
        base_ds=base,
        unit_col="source_id",
        key_cols=["image_id"],
        class_lookup=class_lookup,
        concept_lookup=concept_lookup,
        concept_names=concept_names,
        id2label=artifacts.id2label,
        output_dir=args.output_dir,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        max_images_per_unit=args.max_images_per_unit,
        save_concept_probs=not args.no_concept_probs,
        extra_meta={
            "dataset": "coralnet",
            "split": args.split,
            "sources": list(sources),
            "checkpoint": str(args.checkpoint),
            "model_config": str(args.model_config),
            "concept_id2name": str(args.concept_id2name),
            "data_config": str(args.data_config),
            "input_size": list(size),
            "amp": args.amp,
            "git_commit": _git_commit(),
        },
    )


if __name__ == "__main__":
    main()
