"""Unified validation suite for the concept-bottleneck coral model.

Runs, on one checkpoint at one prediction resolution:

1. ``coralnet``       CoralNet ``--split`` sources (per source id + pooled)
2. ``pacific``        Pacific Labeled Corals validation regions (per region + pooled)
3. ``coralscapes``    CoralscapesV2 few-shot linear probe, 95 classes, 10 pixels per class
4. ``coralscapes39``  the same probe on the published 39-class map, 50 pixels per class
5. ``val_loss``       ConceptBottleneckLoss components on the val splits of ``--data-config``
                      (classification, each taxonomic rank, the binary concepts together
                      and one by one), per dataset and overall

``--evals`` selects a subset (default: all five). The probes are fitted on the train
probe points and scored on ``--split`` (default ``val``). CoralNet and Pacific report
class, taxonomic (all / living / coral, accuracy and macro F1), binary-concept and
hierarchical metrics (consistency, mistake severity, ECE), with ``bleached`` and
``anthropogenic`` called out. Each section is written to ``summary.json`` and
``summary.md`` and to its own subdirectory, e.g. ``coralnet_val/``.

Images are predicted at ``--resolution`` (default: the model config ``input_size``).
CoralscapesV2 images (1024x2048) are predicted as three 1024x1024 windows, averaged
in concept logit space, then scored at the original resolution.

The class and concept layouts default to ``id2label.json`` and ``concept_id2name.json``
next to the checkpoint, which the training pipeline writes. Without them, the classes
come from demo/id2label.json and the concepts from configs/class_to_concepts.csv over
the datasets of configs/data_config.yaml.

Example (cluster; slurm/eval.sbatch wraps this):
    uv run python scripts/evaluate.py \
        --checkpoint nbs/model_checkpoints/cbm_lossw_pix0.5_img0.5/model_step100000 \
        --model-config configs/model_config_cbm_dpt_lora_vitl.yaml \
        --resolution 512 --output-dir eval_out/run --batch-size 8 --num-workers 16 --amp
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

from mermaidseg.dataset_reconciliation import ConceptSchema  # noqa: E402
from mermaidseg.datasets import configured_source_names  # noqa: E402
from mermaidseg.datasets.local_cache import setup_local_cache  # noqa: E402
from mermaidseg.evaluation.gt_mapping import load_benthic_hierarchy  # noqa: E402
from mermaidseg.evaluation.metrics import REPORTED_BINARY_CONCEPTS  # noqa: E402
from mermaidseg.evaluation.predictor import CBMPredictor  # noqa: E402
from mermaidseg.evaluation.reporting import fmt, write_json  # noqa: E402

logger = logging.getLogger("evaluate")

ALL_EVALS = ["coralnet", "pacific", "coralscapes", "coralscapes39", "val_loss"]
SPLITS = ["val", "test"]
_CONFIGS = _REPO_ROOT / "configs"


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
    """Whitelisted CoralNet sources for ``split``; without a whitelist every source would be scored."""
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


def resolve_layouts(
    checkpoint: Path,
    id2label: Path | None,
    concept_id2name: Path | None,
    *,
    output_dir: Path,
    taxonomy_csv: Path,
    schema_config: Path,
) -> tuple[Path, Path]:
    """Class and concept layout files for ``checkpoint`` (see the module docstring)."""
    run_dir = checkpoint.parent
    if id2label is None:
        id2label = run_dir / "id2label.json"
        if not id2label.is_file():
            id2label = _DEMO_DIR / "id2label.json"
    if concept_id2name is None:
        concept_id2name = run_dir / "concept_id2name.json"
        if not concept_id2name.is_file():
            sources = configured_source_names(_data_section(schema_config))
            schema = ConceptSchema.from_csv(taxonomy_csv, sources=sources)
            concept_id2name = output_dir / "concept_id2name.json"
            write_json(concept_id2name, {str(k): v for k, v in schema.channel_id2name().items()})
    return id2label, concept_id2name


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument("--checkpoint", required=True, type=Path, help="Model checkpoint path")
    p.add_argument(
        "--model-config", type=Path, default=_CONFIGS / "model_config_cbm_dpt_lora_vitl.yaml"
    )
    p.add_argument(
        "--resolution",
        type=int,
        default=None,
        help="Square prediction resolution in pixels (default: the model config input_size)",
    )
    p.add_argument(
        "--id2label", type=Path, default=None, help="Default: id2label.json next to the checkpoint"
    )
    p.add_argument(
        "--concept-id2name",
        type=Path,
        default=None,
        help="Default: concept_id2name.json next to the checkpoint",
    )
    p.add_argument(
        "--data-config",
        type=Path,
        default=_CONFIGS / "data_config.yaml",
        help="Sources and splits of every eval, and the val splits of val_loss",
    )
    p.add_argument(
        "--training-config",
        type=Path,
        default=_CONFIGS / "training_config_cbm_100k.yaml",
        help="val_loss: class subset, padding and loss settings of the training run",
    )
    p.add_argument(
        "--schema-config",
        type=Path,
        default=_CONFIGS / "data_config.yaml",
        help="Data config whose datasets freeze the concept channels (as in training)",
    )
    p.add_argument("--output-dir", required=True, type=Path)
    p.add_argument(
        "--evals",
        nargs="+",
        default=ALL_EVALS,
        choices=ALL_EVALS,
        help="Which evaluations to run (default: all)",
    )
    p.add_argument(
        "--split",
        default="val",
        choices=SPLITS,
        help="CoralNet source split and CoralscapesV2 split scored by the probes",
    )
    p.add_argument("--batch-size", type=int, default=8, help="Image batch size")
    p.add_argument("--num-workers", type=int, default=8)
    p.add_argument("--device", default=None)
    p.add_argument("--amp", action="store_true", help="bf16 autocast for the encoder")
    p.add_argument(
        "--strict",
        action="store_true",
        help="Exit with status 1 if any requested eval recorded an error, after writing "
        "the summary. Without it a failed eval is only logged and the run exits 0.",
    )

    # Assets.
    p.add_argument("--taxonomy-csv", type=Path, default=_CONFIGS / "class_to_concepts.csv")
    p.add_argument(
        "--probe-points",
        type=Path,
        default=_CONFIGS / "eval" / "coralscapes_v2_probe_points.json",
        help="95-class probe points (10 per class)",
    )
    p.add_argument(
        "--coralscapes39-probe",
        type=Path,
        default=_CONFIGS / "eval" / "coralscapes_v2_39_probe_k50.json",
        help="39-class probe points (50 per class)",
    )
    p.add_argument(
        "--benthic-hierarchy-json",
        type=Path,
        default=None,
        help="Benthic-attribute hierarchy for the CoralNet / Pacific class roll-up "
        "(default: fetched from the MERMAID API and cached in --output-dir)",
    )
    p.add_argument(
        "--coralscapes-root", type=Path, default=None, help="Local CoralscapesV2 mirror root"
    )
    p.add_argument("--svm-c", type=float, default=1.0)
    p.add_argument(
        "--val-loss-max-images",
        type=int,
        default=1000,
        help="val_loss: fixed seeded subset of at most this many val images per dataset",
    )

    # Debug limiters.
    p.add_argument("--max-images-per-unit", type=int, default=None)
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
        + ", ".join(
            f"{r}={fmt(v['acc_all'])}/{fmt(v['acc_living'])}"
            for r, v in pooled["taxonomic"].items()
        ),
        "- taxonomic macro F1 all/living, none-class F1: "
        + ", ".join(
            f"{r}={fmt(v.get('macro_f1_all'))}/{fmt(v.get('macro_f1_living'))}, "
            f"{fmt(v.get('f1_none'))}"
            for r, v in pooled["taxonomic"].items()
        ),
        "- live coral taxonomic acc/macro F1: "
        + ", ".join(
            f"{r}={fmt(v.get('acc_coral'))}/{fmt(v.get('macro_f1_coral'))} (n={v.get('n_coral')})"
            for r, v in pooled["taxonomic"].items()
        ),
        f"- binary macro acc/F1: {fmt(b.get('macro_accuracy'))}/{fmt(b.get('macro_f1'))}",
        *_concept_lines(pooled),
        "",
    ]


def _val_loss_lines(section: dict) -> list[str]:
    overall = section.get("overall") or {}
    focus = [f"multi_hot/{concept}" for concept in REPORTED_BINARY_CONCEPTS]
    headline = ["loss", "classification", "concepts", "multi_hot", *focus]
    ranks = [k for k in overall if "/" not in k and k not in headline]
    lines = ["## Validation loss (overall)"]
    lines += [f"- {key}: {fmt(overall.get(key))}" for key in headline if key in overall]
    if ranks:
        lines.append("- ranks: " + ", ".join(f"{k}={fmt(overall[k])}" for k in ranks))
    for name, block in (section.get("per_dataset") or {}).items():
        parts = [f"{key}={fmt(block.get(key))}" for key in ("loss", *focus) if key in block]
        lines.append(f"- {name}: " + ", ".join(parts))
    lines.append("")
    return lines


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
        elif key == "val_loss":
            lines += _val_loss_lines(section)
        elif key.startswith("coralscapes39_"):
            split = key.removeprefix("coralscapes39_")
            lines.append(f"## CoralscapesV2 39-class linear probe ({split})")
            for k, block in section.get("by_k", {}).items():
                scores = block.get(split) or {}
                probe = block.get("probe") or {}
                lines.append(
                    f"- k={k}: {split} accuracy {fmt(scores.get('accuracy'))}, "
                    f"mIoU {fmt(scores.get('miou'))} "
                    f"(probe train acc {fmt(probe.get('train_accuracy'))}, "
                    f"{probe.get('num_points')} pixels)"
                )
            lines.append("")
        elif key.startswith("coralscapes_"):
            split = key.removeprefix("coralscapes_")
            lines.append(f"## CoralscapesV2 95-class linear probe ({split})")
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

    evals = list(dict.fromkeys(args.evals))
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device(args.device or ("cuda" if torch.cuda.is_available() else "cpu"))
    input_size = (
        (args.resolution, args.resolution)
        if args.resolution is not None
        else _model_input_size(args.model_config)
    )
    logger.info("Device: %s | evals: %s | resolution: %s", device, evals, input_size)

    # --- Local S3 cache -----------------------------------------------------
    data = _data_section(args.data_config)
    setup_local_cache(data)

    # --- Model --------------------------------------------------------------
    id2label_path, concept_path = resolve_layouts(
        Path(args.checkpoint),
        args.id2label,
        args.concept_id2name,
        output_dir=output_dir,
        taxonomy_csv=args.taxonomy_csv,
        schema_config=args.schema_config,
    )
    logger.info(
        "Loading model artifacts (checkpoint=%s, id2label=%s, concept_id2name=%s) ...",
        args.checkpoint,
        id2label_path,
        concept_path,
    )
    artifacts = load_artifacts(
        checkpoint=args.checkpoint,
        model_config=args.model_config,
        id2label=id2label_path,
        concept_id2name=concept_path,
    )
    model = build_model(artifacts, device)
    predictor = CBMPredictor(model, input_size=input_size, device=device, amp=args.amp)
    id2label = artifacts.id2label
    concept_names = [
        name for _, name in sorted(artifacts.concept_id2name.items(), key=lambda kv: int(kv[0]))
    ]
    logger.info(
        "Model: input_size=%s num_classes=%d num_concepts=%d (concept_names=%d)",
        input_size,
        predictor.num_classes,
        predictor.num_concepts,
        len(concept_names),
    )
    if len(concept_names) != predictor.num_concepts:
        raise ValueError(
            f"{concept_path} has {len(concept_names)} entries but the model has "
            f"{predictor.num_concepts} concept channels. Provide a matching --concept-id2name."
        )
    missing = [c for c in REPORTED_BINARY_CONCEPTS if c not in concept_names]
    if any(e in evals for e in ("coralnet", "pacific", "coralscapes")) and missing:
        raise RuntimeError(
            f"coralnet, pacific and coralscapes evals require concept channels named exactly {missing}, "
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
        "split": args.split,
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

    split = args.split
    if "coralnet" in evals:
        from mermaidseg.evaluation.coralnet_eval import evaluate_coralnet

        _run(
            [f"coralnet_{split}"],
            lambda: {
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
                    whitelist_subsets=_split_field(
                        data, "pacific_labeled_corals", "val", "whitelist_subsets"
                    ),
                    whitelist_sites=_split_field(
                        data, "pacific_labeled_corals", "val", "whitelist_sites"
                    ),
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

    if "coralscapes" in evals:
        from mermaidseg.evaluation.coralscapes_probe import evaluate_coralscapes_probe

        _run(
            [f"coralscapes_{split}"],
            lambda: {
                f"coralscapes_{s}": result
                for s, result in evaluate_coralscapes_probe(
                    predictor=predictor,
                    probe_json=args.probe_points,
                    feature_kinds=["concepts"],
                    coralscapes_root=args.coralscapes_root,
                    output_dir=output_dir,
                    concept_names=concept_names,
                    taxonomy_csv=args.taxonomy_csv,
                    splits=[split],
                    svm_c=args.svm_c,
                    max_test_images=args.max_test_images,
                ).items()
            },
        )

    if "coralscapes39" in evals:
        from mermaidseg.evaluation.coralscapes_probe import evaluate_coralscapes39_probe

        _run(
            [f"coralscapes39_{split}"],
            lambda: {
                f"coralscapes39_{s}": result
                for s, result in evaluate_coralscapes39_probe(
                    predictor=predictor,
                    probe_jsons=[args.coralscapes39_probe],
                    feature_kinds=["concepts"],
                    coralscapes_root=args.coralscapes_root,
                    output_dir=output_dir,
                    splits=[split],
                    svm_c=args.svm_c,
                    max_test_images=args.max_test_images,
                ).items()
            },
        )

    if "val_loss" in evals:
        from mermaidseg.evaluation.val_loss import evaluate_val_loss

        _run(
            ["val_loss"],
            lambda: {
                "val_loss": evaluate_val_loss(
                    checkpoint=args.checkpoint,
                    model_config=args.model_config,
                    data_config=args.data_config,
                    training_config=args.training_config,
                    resolution=input_size[0],
                    output_dir=output_dir,
                    device=device,
                    batch_size=args.batch_size,
                    num_workers=args.num_workers,
                    amp=args.amp,
                    schema_config=args.schema_config,
                    max_images_per_dataset=args.val_loss_max_images,
                )
            },
        )

    _write_summary(output_dir, summary)
    logger.info("ALL DONE in %.1fs. Outputs in %s", time.time() - t_start, output_dir)
    print(json.dumps({k: v for k, v in summary.items() if isinstance(v, dict)}, default=str)[:2000])
    failed = failed_sections(summary)
    if failed:
        logger.error("Evaluations failed: %s", ", ".join(failed))
        if args.strict:
            sys.exit(1)


def failed_sections(summary: dict) -> list[str]:
    """Summary keys whose eval recorded ``{"error": ...}``."""
    return [k for k, v in summary.items() if isinstance(v, dict) and "error" in v]


if __name__ == "__main__":
    main()
