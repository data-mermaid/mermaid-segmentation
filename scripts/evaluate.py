"""Standalone full-scale evaluation suite for the concept-bottleneck coral model.

Runs any/all of four evaluations at *source resolution* using the locally cached
datasets (S3 mirror for CoralNet / Pacific / Benthos; HuggingFace or a local
mirror for CoralscapesV2):

1. CoralNet val / test sources    (per source id + aggregate, ``--coralnet-splits``)
2. Pacific Labeled Corals         (per region + aggregate)
3. Benthos zero-shot segmentation (per orthomosaic + aggregate)
4. CoralscapesV2 linear probe     (dense val / test accuracy / mIoU, ``--coralscapes-splits``)

CoralNet and CoralscapesV2 also report the ``bleached`` and ``anthropogenic``
concepts and hierarchical metrics (consistency, mistake severity, ECE). Each split
is written to its own summary key and output subdirectory, e.g. ``coralnet_test``.

The model is specified exactly like in the demo (checkpoint + model config +
id2label + concept_id2name); the model ``input_size`` from the model config
determines the prediction resolution. CoralscapesV2 images (1024x2048) are
predicted as three 1024x1024 windows (left, center, right), averaged in concept
logit space, then scored at the original resolution.

Example (cluster):
    uv run python scripts/evaluate.py \
        --checkpoint /path/to/model_epoch162 \
        --model-config configs/model_config_cbm_dpt_lora_vitl.yaml \
        --data-config configs/data_config_512.yaml \
        --output-dir eval_out/run162 \
        --evals all --batch-size 8 --num-workers 16 --amp
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from pathlib import Path

import torch
import yaml

_REPO_ROOT = Path(__file__).resolve().parent.parent
_DEMO_DIR = _REPO_ROOT / "demo"
if str(_DEMO_DIR) not in sys.path:
    sys.path.insert(0, str(_DEMO_DIR))

from inference import build_model, load_artifacts  # noqa: E402  (demo module)

from mermaidseg.datasets.local_cache import setup_local_cache  # noqa: E402
from mermaidseg.evaluation.gt_mapping import load_benthic_hierarchy  # noqa: E402
from mermaidseg.evaluation.metrics import REPORTED_BINARY_CONCEPTS  # noqa: E402
from mermaidseg.evaluation.predictor import CBMPredictor  # noqa: E402
from mermaidseg.evaluation.reporting import fmt, write_json  # noqa: E402

logger = logging.getLogger("evaluate")

ALL_EVALS = ["coralnet", "pacific", "benthos", "coralscapes"]
SPLITS = ["val", "test"]


def _load_yaml(path: str | Path) -> dict:
    with Path(path).open() as f:
        return yaml.safe_load(f)


def _data_section(data_config: str | Path) -> dict:
    raw = _load_yaml(data_config)
    return raw.get("data", raw)


def _split_field(data: dict, dataset: str, split: str, field: str):
    ds = data.get(dataset)
    if not isinstance(ds, dict):
        return None
    section = ds.get(split)
    if not isinstance(section, dict):
        return None
    return section.get(field)


def _coralnet_sources(data: dict, split: str) -> list[int]:
    """Whitelisted CoralNet sources for ``split``; without them every source would be scored."""
    sources = _split_field(data, "coralnet", split, "whitelist_sources")
    if not sources:
        raise ValueError(f"data config has no coralnet.{split}.whitelist_sources")
    return sources


def _model_input_size(model_config: str | Path) -> tuple[int, int]:
    cfg = _load_yaml(model_config)
    model_cfg = cfg.get("model", cfg)
    size = model_cfg.get("input_size", [512, 512])
    if isinstance(size, (list, tuple)) and len(size) == 2:
        return int(size[0]), int(size[1])
    return 512, 512


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--checkpoint", required=True, type=Path, help="Model checkpoint path")
    p.add_argument(
        "--model-config",
        type=Path,
        default=_REPO_ROOT / "configs" / "model_config_cbm_dpt_lora_vitl.yaml",
    )
    p.add_argument("--id2label", type=Path, default=_DEMO_DIR / "id2label.json")
    p.add_argument("--concept-id2name", type=Path, default=_DEMO_DIR / "concept_id2name.json")
    p.add_argument("--data-config", type=Path, default=_REPO_ROOT / "configs" / "data_config_512.yaml")
    p.add_argument("--output-dir", required=True, type=Path)
    p.add_argument(
        "--evals",
        nargs="+",
        default=["all"],
        choices=[*ALL_EVALS, "all"],
        help="Which evaluations to run",
    )
    p.add_argument(
        "--coralnet-splits",
        nargs="+",
        default=["val"],
        choices=SPLITS,
        help="CoralNet source splits (whitelist_sources of each split in the data config)",
    )
    p.add_argument(
        "--coralscapes-splits",
        nargs="+",
        default=["test"],
        choices=SPLITS,
        help="CoralscapesV2 splits to score with the one probe fitted on the train probe points",
    )
    p.add_argument("--batch-size", type=int, default=8, help="Point-eval image batch size")
    p.add_argument("--num-workers", type=int, default=8)
    p.add_argument("--device", default=None)
    p.add_argument("--amp", action="store_true", help="bf16 autocast for the encoder")

    # Assets.
    p.add_argument("--taxonomy-csv", type=Path, default=_REPO_ROOT / "configs" / "class_to_concepts.csv")
    p.add_argument("--benthos-spec", type=Path, default=_REPO_ROOT / "configs" / "eval" / "benthos_zero_shot.yaml")
    p.add_argument(
        "--probe-points",
        type=Path,
        default=_REPO_ROOT / "configs" / "eval" / "coralscapes_v2_probe_points.json",
    )
    p.add_argument("--benthic-hierarchy-json", type=Path, default=None)
    p.add_argument(
        "--probe-features",
        nargs="+",
        default=["concepts"],
        choices=["concepts", "dpt", "backbone"],
    )
    p.add_argument("--coralscapes-root", type=Path, default=None, help="Local CoralscapesV2 mirror root")
    p.add_argument("--svm-c", type=float, default=1.0)

    # Debug limiters.
    p.add_argument("--max-images-per-unit", type=int, default=None)
    p.add_argument("--max-tiles-per-site", type=int, default=None)
    p.add_argument("--max-test-images", type=int, default=None, help="Per CoralscapesV2 split")
    return p


def _concept_lines(section: dict) -> list[str]:
    """Explicit binary concept and hierarchical lines for one eval section."""
    lines = []
    for concept in REPORTED_BINARY_CONCEPTS:
        b = section.get(concept)
        if not isinstance(b, dict):
            continue
        unannotated = f" n_unannotated={b['n_unannotated']}" if "n_unannotated" in b else ""
        lines.append(
            f"- {concept} acc/precision/recall/F1: "
            f"{fmt(b.get('accuracy'))}/{fmt(b.get('precision'))}/{fmt(b.get('recall'))}/{fmt(b.get('f1'))} "
            f"(n_true={b.get('n_true')} n_false={b.get('n_false')} "
            f"n_not_given={b.get('n_not_given')}{unannotated})"
        )
    h = section.get("hierarchical")
    if isinstance(h, dict):
        lines.append(
            f"- hierarchical: consistency {fmt(h.get('consistency/overall'))}, "
            f"severity (mean normalized) {fmt(h.get('severity/mean_normalized'))}, "
            f"ECE (mean) {fmt(h.get('ece/mean'))}"
        )
    return lines


def _point_lines(title: str, pooled: dict) -> list[str]:
    b = pooled.get("binary") or {}
    return [
        f"## {title}",
        f"- class accuracy: {fmt(pooled['class_accuracy'])}",
        "- taxonomic acc_all/acc_living: "
        + ", ".join(f"{r}={fmt(v['acc_all'])}/{fmt(v['acc_living'])}" for r, v in pooled["taxonomic"].items()),
        f"- binary macro acc/F1: {fmt(b.get('macro_accuracy'))}/{fmt(b.get('macro_f1'))}",
        *_concept_lines(pooled),
        "",
    ]


def _write_summary(output_dir: Path, summary: dict) -> None:
    write_json(output_dir / "summary.json", summary)
    lines = ["# CBM evaluation summary", ""]
    for key, section in summary.items():
        if not isinstance(section, dict):
            continue
        if "error" in section:
            lines += [f"## {key}: ERROR - {section['error']}", ""]
        elif key.startswith("coralnet_"):
            split = key.removeprefix("coralnet_")
            lines += _point_lines(f"CoralNet (pooled over {split} sources)", section["pooled"])
        elif key == "pacific":
            lines += _point_lines("Pacific Labeled Corals (pooled over regions)", section["pooled"])
        elif key == "benthos":
            pooled = section["benthos"]
            lines.append("## Benthos zero-shot (pooled)")
            lines.append(f"- accuracy: {fmt(pooled['accuracy'])}  mIoU: {fmt(pooled['miou'])}")
            for name, cd in pooled["per_class"].items():
                lines.append(f"  - {name}: IoU {fmt(cd['iou'])}")
            lines.append("")
        elif key.startswith("coralscapes_"):
            split = key.removeprefix("coralscapes_")
            lines.append(f"## CoralscapesV2 linear probe ({split})")
            for kind, kd in section["feature_kinds"].items():
                s = kd.get(split) or {}
                lines.append(
                    f"- {kind}: {split} accuracy {fmt(s.get('accuracy'))}, mIoU {fmt(s.get('miou'))} "
                    f"(probe train acc {fmt(kd['probe']['train_accuracy'])})"
                )
            lines += [*_concept_lines(section), ""]
    (output_dir / "summary.md").write_text("\n".join(lines))


def main(argv: list[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )

    evals = ALL_EVALS if "all" in args.evals else list(dict.fromkeys(args.evals))
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device(args.device or ("cuda" if torch.cuda.is_available() else "cpu"))
    logger.info("Device: %s | evals: %s", device, evals)

    # --- Local S3 cache -----------------------------------------------------
    data = _data_section(args.data_config)
    setup_local_cache(data)

    # --- Model --------------------------------------------------------------
    logger.info("Loading model artifacts (checkpoint=%s) ...", args.checkpoint)
    artifacts = load_artifacts(
        checkpoint=args.checkpoint,
        model_config=args.model_config,
        id2label=args.id2label,
        concept_id2name=args.concept_id2name,
    )
    model = build_model(artifacts, device)
    input_size = _model_input_size(args.model_config)
    predictor = CBMPredictor(model, input_size=input_size, device=device, amp=args.amp)
    id2label = artifacts.id2label
    concept_names = [name for _, name in sorted(artifacts.concept_id2name.items(), key=lambda kv: int(kv[0]))]
    logger.info(
        "Model: input_size=%s num_classes=%d num_concepts=%d (concept_names=%d)",
        input_size,
        predictor.num_classes,
        predictor.num_concepts,
        len(concept_names),
    )
    if len(concept_names) != predictor.num_concepts:
        raise ValueError(
            f"concept_id2name has {len(concept_names)} entries but the model has "
            f"{predictor.num_concepts} concept channels. Provide a matching --concept-id2name."
        )
    missing = [c for c in REPORTED_BINARY_CONCEPTS if c not in concept_names]
    if ("coralnet" in evals or "coralscapes" in evals) and missing:
        raise RuntimeError(
            f"coralnet and coralscapes evals require concept channels named exactly {missing}, "
            f"which concept_id2name ({len(concept_names)} entries) does not have."
        )

    # Benthic hierarchy for CoralNet/Pacific class roll-up.
    hierarchy = None
    if "coralnet" in evals or "pacific" in evals:
        hierarchy = load_benthic_hierarchy(
            hierarchy_json=args.benthic_hierarchy_json,
            output_dir=output_dir,
            fetch_remote=True,
        )

    summary: dict = {
        "checkpoint": str(args.checkpoint),
        "model_config": str(args.model_config),
        "input_size": list(input_size),
        "evals": evals,
    }
    t_start = time.time()

    def _run(keys: list[str], fn) -> None:
        """Run one eval returning ``{key: result}``; record errors so others still proceed."""
        try:
            summary.update(fn())
        except Exception as e:  # noqa: BLE001
            logger.exception("Evaluation %s failed: %s", keys, e)
            summary.update({key: {"error": f"{type(e).__name__}: {e}"} for key in keys})
        _write_summary(output_dir, summary)

    if "coralnet" in evals:
        from mermaidseg.evaluation.coralnet_eval import evaluate_coralnet

        for split in dict.fromkeys(args.coralnet_splits):
            _run(
                [f"coralnet_{split}"],
                lambda split=split: {
                    f"coralnet_{split}": evaluate_coralnet(
                        predictor=predictor,
                        whitelist_sources=_coralnet_sources(data, split),
                        image_blacklist=_split_field(data, "coralnet", split, "image_blacklist"),
                        annotation_overrides=_split_field(
                            data, "coralnet", split, "annotation_overrides"
                        ),
                        id2label=id2label,
                        concept_names=concept_names,
                        taxonomy_csv=args.taxonomy_csv,
                        hierarchy=hierarchy,
                        output_dir=output_dir,
                        split=split,
                        batch_size=args.batch_size,
                        num_workers=args.num_workers,
                        max_images_per_unit=args.max_images_per_unit,
                    )
                },
            )

    if "pacific" in evals:
        from mermaidseg.evaluation.pacific_eval import evaluate_pacific

        _run(
            ["pacific"],
            lambda: {
                "pacific": evaluate_pacific(
                    predictor=predictor,
                    whitelist_subsets=_split_field(data, "pacific_labeled_corals", "val", "whitelist_subsets"),
                    whitelist_sites=_split_field(data, "pacific_labeled_corals", "val", "whitelist_sites"),
                    id2label=id2label,
                    concept_names=concept_names,
                    taxonomy_csv=args.taxonomy_csv,
                    hierarchy=hierarchy,
                    output_dir=output_dir,
                    batch_size=args.batch_size,
                    num_workers=args.num_workers,
                    max_images_per_unit=args.max_images_per_unit,
                )
            },
        )

    if "benthos" in evals:
        from mermaidseg.evaluation.benthos_eval import evaluate_benthos

        _run(
            ["benthos"],
            lambda: {
                "benthos": evaluate_benthos(
                    predictor=predictor,
                    whitelist_sites=_split_field(data, "benthos_yuval", "val", "whitelist_sites"),
                    concept_names=concept_names,
                    spec_path=args.benthos_spec,
                    output_dir=output_dir,
                    max_tiles_per_site=args.max_tiles_per_site,
                )
            },
        )

    if "coralscapes" in evals:
        from mermaidseg.evaluation.coralscapes_probe import evaluate_coralscapes_probe

        splits = list(dict.fromkeys(args.coralscapes_splits))
        _run(
            [f"coralscapes_{split}" for split in splits],
            lambda: {
                f"coralscapes_{split}": result
                for split, result in evaluate_coralscapes_probe(
                    predictor=predictor,
                    probe_json=args.probe_points,
                    feature_kinds=args.probe_features,
                    coralscapes_root=args.coralscapes_root,
                    output_dir=output_dir,
                    concept_names=concept_names,
                    taxonomy_csv=args.taxonomy_csv,
                    splits=splits,
                    svm_c=args.svm_c,
                    max_test_images=args.max_test_images,
                ).items()
            },
        )

    _write_summary(output_dir, summary)
    logger.info("ALL DONE in %.1fs. Outputs in %s", time.time() - t_start, output_dir)
    print(json.dumps({k: v for k, v in summary.items() if isinstance(v, dict)}, default=str)[:2000])


if __name__ == "__main__":
    main()
