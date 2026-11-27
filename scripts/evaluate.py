"""Unified validation suite for the concept-bottleneck coral model.

Runs, on one exported model bundle at the bundle's input size:

1. ``coralnet``       CoralNet ``--split`` sources (per source id + pooled)
2. ``pacific``        Pacific Labeled Corals validation regions (per region + pooled)
3. ``coralscapes``    CoralscapesV2 few-shot linear probe, 95 classes, 10 pixels per class
4. ``coralscapes39``  the same probe on the published 39-class map, 50 pixels per class
5. ``val_loss``       ConceptBottleneckLoss components on the blocks of ``--split`` in ``--data-config``
                      (classification, each taxonomic rank, the binary concepts together
                      and one by one), per dataset and overall

Every dataset is read from the versions ``--data-config`` pins, in the synced ``--cache-dir``.
The CoralNet groups scored are the ``--split`` groups of the data config's CoralNet split file
(``configs/splits/coralnet.csv``); Pacific is the data config's val block.

``summary.json`` records ``coralnet_split_in_training`` and ``coralscapes_split_in_training``
when the data config's train block also selects the scored groups; a warning is logged.

``--evals`` selects a subset (default: all five). The probes are fitted on the train
probe points and scored on ``--split`` (default ``val``). ``--split train`` scores the
CoralNet train sources and allows only ``coralnet`` and ``val_loss``; the loss is then
computed on those train images (val transform, the run's ``concept_source_mask``
applied) and reported as ``train_loss``. ``--images-per-source N`` draws a seeded random
sample of at most N images per CoralNet source, written to
``coralnet_<split>_sample.parquet``; both evals read only those images. CoralNet and Pacific report
class, taxonomic (all / living / coral, accuracy and macro F1), binary-concept and
hierarchical metrics (consistency, mistake severity, ECE), with ``bleached`` and
``anthropogenic`` called out. Each section is written to ``summary.json`` and
``summary.md`` and to its own subdirectory, e.g. ``coralnet_val/``.

``--bundle`` is the only model input: the weights, class and concept layouts, input size,
normalisation and the run settings (autocast dtype, removed concepts, loss config) all come
from the bundle's ``model.json``. CoralscapesV2 images (1024x2048) are predicted as three 1024x1024 windows, averaged
in concept logit space, then scored at the original resolution.

Example (cluster):
    uv run python scripts/evaluate.py \
        --bundle outputs/bundles/cbm_lossw_pix0.5_img0.5 \
        --output-dir eval_out/run --cache-dir /path/to/cache \
        --batch-size 8 --num-workers 16 --amp
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from dataclasses import fields
from pathlib import Path

from mermaidseg.evaluation.run import ALL_EVALS, EvalOptions, failed_sections, run_evaluations
from mermaidseg.io import REPO_ROOT

logger = logging.getLogger("evaluate")

SPLITS = ["train", "val", "test"]
_CONFIGS = REPO_ROOT / "configs"


def build_parser() -> argparse.ArgumentParser:
    """Build the command-line parser."""
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument(
        "--bundle",
        required=True,
        type=Path,
        help="Exported model bundle directory (model.json + weights.safetensors)",
    )
    p.add_argument(
        "--cache-dir",
        required=True,
        type=Path,
        help="Directory scripts/sync_data.py synced --data-config into",
    )
    p.add_argument(
        "--data-config",
        type=Path,
        default=_CONFIGS / "data_config.yaml",
        help="Dataset pins of every eval, and the val blocks of val_loss",
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
        help="CoralNet source split and CoralscapesV2 split scored by the probes "
        "(train: coralnet and val_loss only)",
    )
    p.add_argument(
        "--images-per-source",
        type=int,
        default=None,
        help="Seeded random sample of at most this many images per CoralNet source "
        "(default: every image)",
    )
    p.add_argument("--sample-seed", type=int, default=0, help="Seed of --images-per-source")
    p.add_argument("--batch-size", type=int, default=8, help="Image batch size")
    p.add_argument("--num-workers", type=int, default=8)
    p.add_argument("--device", default=None)
    p.add_argument(
        "--amp",
        action="store_true",
        help="autocast the forward pass in the run's training.mixed_precision_dtype",
    )
    p.add_argument(
        "--strict",
        action="store_true",
        help="Exit with status 1 if any requested eval recorded an error, after writing "
        "the summary. Without it a failed eval is only logged and the run exits 0.",
    )

    # Assets.
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


def main(argv: list[str] | None = None) -> None:
    """Run the selected evaluations and write their outputs and summary."""
    args = build_parser().parse_args(argv)
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    options = EvalOptions(**{f.name: getattr(args, f.name) for f in fields(EvalOptions)})
    summary = run_evaluations(options)
    print(json.dumps({k: v for k, v in summary.items() if isinstance(v, dict)}, default=str)[:2000])
    failed = failed_sections(summary)
    if failed:
        logger.error("Evaluations failed: %s", ", ".join(failed))
        if args.strict:
            sys.exit(1)


if __name__ == "__main__":
    main()
