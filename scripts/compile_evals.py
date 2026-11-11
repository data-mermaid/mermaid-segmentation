#!/usr/bin/env python3
"""Compile every finished eval under eval_out/ into two CSVs.

slurm/launch_evals.sh writes each eval to ``<eval-root>/<family>/<run>/<checkpoint>_res<R>/``
and marks it finished with ``.done``. This script reads every finished cell's
``summary.json`` and writes:

    <eval-root>/compiled.csv
        One row per (run, checkpoint, resolution). Identity columns (family, run,
        checkpoint, step, resolution), the run parameters parsed from the run name, and
        every flattened metric:
            coralnet/pooled/*, coralnet/macro/*   CoralNet sources (macro = mean over sources)
            pacific/pooled/*, pacific/macro/*     Pacific Labeled Corals regions
            coralscapes/*                         95-class probe (accuracy, mIoU, concepts)
            coralscapes39/*                       39-class probe (accuracy, mIoU)
            val_loss/*                            overall loss components, every binary concept
            val_loss/<dataset>/*                  per dataset: ranks, totals, bleached, anthropogenic
    <eval-root>/compiled_coralnet_per_source.csv
        The CoralNet per-source table (coralnet_<split>/per_unit.csv) of every cell, with
        the identity and run-parameter columns in front.

Usage:
    python scripts/compile_evals.py [--eval-root eval_out]
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import re
from collections.abc import Iterable, Mapping
from pathlib import Path
from typing import Any

_REPO_ROOT = Path(__file__).resolve().parents[1]

DEFAULT_EVAL_ROOT = _REPO_ROOT / "eval_out"
COMPILED_CSV = "compiled.csv"
PER_SOURCE_CSV = "compiled_coralnet_per_source.csv"
# Must match iterations_per_train_epoch in configs/training_config_cbm_duration_peak.yaml.
DURATION_ITERS_PER_EPOCH = 25000
DURATION_COOLDOWN_STEPS = 25000
DEFAULT_TRAIN_RESOLUTION = 256
REPORTED_BINARY_CONCEPTS = ("bleached", "anthropogenic")
_TAX_SCALARS = (
    "acc_all",
    "acc_living",
    "acc_coral",
    "macro_f1_all",
    "macro_f1_living",
    "macro_f1_coral",
    "f1_none",
)
_CONCEPT_SCALARS = ("accuracy", "precision", "recall", "f1")
_HIER_KEYS = ("consistency/overall", "severity/mean_normalized", "ece/mean")

IDENTITY_COLUMNS = ["family", "run", "checkpoint", "step", "resolution"]
PARAM_COLUMNS = [
    "pixel_weight",
    "image_weight",
    "tax_loss",
    "tax_weight",
    "ds_experiment",
    "ds_drop",
    "vit_size",
    "train_resolution",
    "duration_phase",
    "duration_branch_step",
    "train_steps",
]

_CELL_RE = re.compile(r"^(?P<checkpoint>model_(?:step|epoch)\d+)_res(?P<res>\d+)$")
_LOSSW_RE = re.compile(r"^cbm_lossw_pix(?P<pix>[\d.]+)_img(?P<img>[\d.]+)$")
_TAX_RE = re.compile(r"^cbm_tax_(?P<loss>kl|partial|tree)$")
_DSIZE_RE = re.compile(r"^cbm_dsize_(?P<exp>exp\d)_drop(?P<drop>\d+)$")
_VITSIZE_RE = re.compile(r"^cbm_vitsize_(?P<size>\w+)$")
_COOLDOWN_RE = re.compile(r"^cbm_duration_cd(?P<step>\d+)$")
_HIRES_RE = re.compile(r"^cbm_hires_res(?P<res>\d+)$")


def checkpoint_step(run: str, checkpoint: str) -> int:
    """Optimizer updates behind ``checkpoint`` (peak epoch checkpoints count (E+1) epochs)."""
    kind, number = re.fullmatch(r"model_(step|epoch)(\d+)", checkpoint).groups()
    if kind == "step":
        return int(number)
    if run != "cbm_duration_peak":
        raise ValueError(
            f"{run}/{checkpoint}: epoch checkpoints are only compiled for the peak run"
        )
    return (int(number) + 1) * DURATION_ITERS_PER_EPOCH


def run_params(run: str, step: int) -> dict[str, Any]:
    """Experiment parameters encoded in a run name (shared recipe values filled in)."""
    params: dict[str, Any] = {
        "pixel_weight": 0.5,
        "image_weight": 0.5,
        "tax_loss": "none",
        "tax_weight": 0.0,
        "ds_experiment": "",
        "ds_drop": "",
        "vit_size": "vitl",
        "train_resolution": DEFAULT_TRAIN_RESOLUTION,
        "duration_phase": "",
        "duration_branch_step": "",
        "train_steps": step,
    }
    if m := _LOSSW_RE.match(run):
        params["pixel_weight"] = float(m["pix"])
        params["image_weight"] = float(m["img"])
    elif m := _TAX_RE.match(run):
        params["tax_loss"] = m["loss"]
        params["tax_weight"] = 0.5
    elif m := _DSIZE_RE.match(run):
        params["ds_experiment"] = m["exp"]
        params["ds_drop"] = int(m["drop"])
    elif run == "cbm_sparse":
        params["ds_experiment"] = "sparse"
    elif m := _VITSIZE_RE.match(run):
        params["vit_size"] = m["size"]
    elif run == "cbm_duration_peak":
        params["duration_phase"] = "peak"
        params["duration_branch_step"] = step
    elif m := _COOLDOWN_RE.match(run):
        params["duration_phase"] = "cooldown"
        params["duration_branch_step"] = int(m["step"])
        params["train_steps"] = int(m["step"]) + step
    elif m := _HIRES_RE.match(run):
        params["train_resolution"] = int(m["res"])
        # 275k peak + 25k cooldown, then the fine-tune.
        params["train_steps"] = 275000 + DURATION_COOLDOWN_STEPS + step
    return params


def _num(value: Any) -> float:
    if value is None:
        return math.nan
    try:
        return float(value)
    except (TypeError, ValueError):
        return math.nan


def _scalars(block: Mapping[str, Any] | None, prefix: str) -> dict[str, float]:
    """Top-level numeric entries of ``block`` (nested dicts and lists are skipped)."""
    out: dict[str, float] = {}
    for key, value in (block or {}).items():
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            continue
        out[f"{prefix}{key}"] = float(value)
    return out


def point_metrics(pooled: Mapping[str, Any]) -> dict[str, float]:
    """The flat point-eval scalars of a pooled dict (as in ``point_datasets.unit_scalars``)."""
    binary = pooled.get("binary") or {}
    out = {
        "class_accuracy": _num(pooled.get("class_accuracy")),
        "class_miou": _num(pooled.get("class_miou")),
        "binary_macro_accuracy": _num(binary.get("macro_accuracy")),
        "binary_macro_f1": _num(binary.get("macro_f1")),
    }
    for rank, rd in (pooled.get("taxonomic") or {}).items():
        for key in _TAX_SCALARS:
            out[f"tax_{rank}_{key}"] = _num(rd.get(key))
    for concept in REPORTED_BINARY_CONCEPTS:
        block = pooled.get(concept) or {}
        for key in _CONCEPT_SCALARS:
            out[f"{concept}_{key}"] = _num(block.get(key))
    hier = pooled.get("hierarchical")
    if isinstance(hier, Mapping):
        for key in _HIER_KEYS:
            out[f"hier/{key}"] = _num(hier.get(key))
    return out


def _section(summary: Mapping[str, Any], prefix: str) -> tuple[str, Mapping[str, Any]] | None:
    """The one summary section named ``<prefix>`` or ``<prefix>_<split>``."""
    for key, value in summary.items():
        if (key == prefix or re.fullmatch(rf"{prefix}_(val|test)", key)) and isinstance(
            value, Mapping
        ):
            return key, value
    return None


def summary_metrics(summary: Mapping[str, Any]) -> dict[str, float]:
    """Every compiled metric column of one eval cell."""
    row: dict[str, float] = {}
    for name in ("coralnet", "pacific"):
        found = _section(summary, name)
        if found is None or "error" in found[1]:
            continue
        section = found[1]
        for key, value in point_metrics(section.get("pooled") or {}).items():
            row[f"{name}/pooled/{key}"] = value
        for key, value in (section.get("macro") or {}).items():
            row[f"{name}/macro/{key}"] = _num(value)
        row[f"{name}/images_processed"] = _num(section.get("images_processed"))
        row[f"{name}/images_failed"] = _num(section.get("images_failed"))

    found = _section(summary, "coralscapes")
    if found is not None and "error" not in found[1]:
        key, section = found
        split = key.removeprefix("coralscapes_")
        concepts = (section.get("feature_kinds") or {}).get("concepts") or {}
        row.update(_scalars(concepts.get(split), "coralscapes/"))
        row.update(_scalars(concepts.get("probe"), "coralscapes/probe_"))
        for concept in REPORTED_BINARY_CONCEPTS:
            block = section.get(concept) or {}
            for metric in _CONCEPT_SCALARS:
                row[f"coralscapes/{concept}_{metric}"] = _num(block.get(metric))
        hier = section.get("hierarchical") or {}
        for metric in _HIER_KEYS:
            row[f"coralscapes/hier/{metric}"] = _num(hier.get(metric))

    found = _section(summary, "coralscapes39")
    if found is not None and "error" not in found[1]:
        key, section = found
        split = key.removeprefix("coralscapes39_")
        for k, block in (section.get("by_k") or {}).items():
            row.update(_scalars(block.get(split), f"coralscapes39/k{k}/"))
            row.update(_scalars(block.get("probe"), f"coralscapes39/k{k}/probe_"))

    val_loss = summary.get("val_loss")
    if isinstance(val_loss, Mapping) and "error" not in val_loss:
        for key, value in (val_loss.get("overall") or {}).items():
            row[f"val_loss/{key}"] = _num(value)
        focus = {f"multi_hot/{concept}" for concept in REPORTED_BINARY_CONCEPTS}
        for dataset, block in (val_loss.get("per_dataset") or {}).items():
            for key, value in block.items():
                if "/" not in key or key in focus:
                    row[f"val_loss/{dataset}/{key}"] = _num(value)
    return row


def iter_cells(eval_root: Path) -> Iterable[tuple[str, str, str, int, Path]]:
    """``(family, run, checkpoint, resolution, cell_dir)`` of every finished eval."""
    for done in sorted(eval_root.glob("*/*/*/.done")):
        cell = done.parent
        match = _CELL_RE.match(cell.name)
        if match is None:
            continue
        yield (
            cell.parent.parent.name,
            cell.parent.name,
            match["checkpoint"],
            int(match["res"]),
            cell,
        )


def build_rows(eval_root: Path) -> tuple[list[dict], list[dict]]:
    """Rows of the compiled table and of the per-source table."""
    rows: list[dict] = []
    source_rows: list[dict] = []
    for family, run, checkpoint, resolution, cell in iter_cells(eval_root):
        summary_path = cell / "summary.json"
        if not summary_path.is_file():
            continue
        summary = json.loads(summary_path.read_text())
        step = checkpoint_step(run, checkpoint)
        ident = {
            "family": family,
            "run": run,
            "checkpoint": checkpoint,
            "step": step,
            "resolution": resolution,
            **run_params(run, step),
        }
        rows.append({**ident, **summary_metrics(summary)})
        for per_unit in sorted(cell.glob("coralnet_*/per_unit.csv")):
            with per_unit.open(newline="") as f:
                for unit_row in csv.DictReader(f):
                    source_rows.append({**ident, "split": per_unit.parent.name, **unit_row})
    return rows, source_rows


def _columns(rows: list[dict], leading: list[str]) -> list[str]:
    seen = dict.fromkeys(leading)
    for row in rows:
        for key in row:
            seen.setdefault(key, None)
    return list(seen)


def write_csv(path: Path, rows: list[dict], leading: list[str]) -> None:
    columns = _columns(rows, leading)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=columns, restval="")
        writer.writeheader()
        writer.writerows(rows)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--eval-root", type=Path, default=DEFAULT_EVAL_ROOT)
    return parser


def main(argv: list[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    rows, source_rows = build_rows(args.eval_root)
    leading = IDENTITY_COLUMNS + PARAM_COLUMNS
    write_csv(args.eval_root / COMPILED_CSV, rows, leading)
    write_csv(args.eval_root / PER_SOURCE_CSV, source_rows, [*leading, "split"])
    print(f"wrote {len(rows)} eval rows to {args.eval_root / COMPILED_CSV}")
    print(f"wrote {len(source_rows)} CoralNet source rows to {args.eval_root / PER_SOURCE_CSV}")


if __name__ == "__main__":
    main()
