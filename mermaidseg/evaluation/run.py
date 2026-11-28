"""Run the selected evaluations on one exported bundle and write their summary.

``REGISTRY`` names each evaluation, what it needs loaded, how it runs and how its summary
section renders. :func:`run_evaluations` loads the bundle and the data section once, runs the
selected entries in registry order, records a failed entry as ``{"error": ...}`` and carries on,
and rewrites ``summary.json`` and ``summary.md`` after each entry.
"""

from __future__ import annotations

import logging
import time
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch

from mermaidseg.data.config import DataSection, Split, SplitSelection, entry_split_groups
from mermaidseg.data.factory import load_benthic_hierarchy, open_dataset_version
from mermaidseg.evaluation.coralscapes_probe import (
    CORALSCAPES39,
    CORALSCAPES_V2,
    ProbeConcepts,
    evaluate_coralscapes_probe,
    split_group,
)
from mermaidseg.evaluation.metrics import require_reported_concepts
from mermaidseg.evaluation.point_datasets import evaluate_point_dataset
from mermaidseg.evaluation.point_sources import (
    CORALNET_SOURCE,
    PACIFIC_SOURCE,
    coralnet_reader,
    pacific_reader,
    split_groups,
    write_coralnet_sample,
)
from mermaidseg.evaluation.summary import (
    TEST_LOSS_KEY,
    TRAIN_LOSS_KEY,
    VAL_LOSS_KEY,
    summarize_coralnet,
    summarize_coralscapes,
    summarize_coralscapes39,
    summarize_loss,
    summarize_pacific,
    write_summary,
)
from mermaidseg.evaluation.val_loss import evaluate_val_loss
from mermaidseg.inference.bundle import MODEL_JSON, ModelBundle, load_bundle
from mermaidseg.inference.predictor import CBMPredictor
from mermaidseg.io import REPO_ROOT, load_section, resolve_repo_path
from mermaidseg.model.meta import resolve_amp_dtype
from mermaidseg.run_setup import concept_removal_list

logger = logging.getLogger(__name__)

CORALNET = "coralnet"
CORALSCAPES = "coralscapes_v2"
TRAIN_SPLIT_EVALS = ("coralnet", "val_loss")


@dataclass(frozen=True)
class EvalOptions:
    """The command-line settings of one evaluation run (see ``scripts/evaluate.py``)."""

    bundle: Path
    cache_dir: Path
    data_config: Path
    schema_config: Path
    output_dir: Path
    evals: list[str]
    split: Split
    images_per_source: int | None
    sample_seed: int
    batch_size: int
    num_workers: int
    device: str | None
    amp: bool
    probe_points: Path
    coralscapes39_probe: Path
    svm_c: float
    val_loss_max_images: int | None
    max_images_per_unit: int | None
    max_test_images: int | None


@dataclass(frozen=True)
class EvalContext:
    """What every evaluation reads: the loaded bundle, the data section and the run options."""

    bundle: ModelBundle
    predictor: CBMPredictor
    section: DataSection
    cache_dir: Path
    output_dir: Path
    split: Split
    concept_names: list[str]
    hierarchy: dict[str, str | None] | None
    removed_concepts: list[str]
    sample_ids: list[str] | None
    options: EvalOptions


@dataclass(frozen=True)
class Evaluation:
    """One selectable evaluation.

    ``run`` returns ``{summary key: result}``; ``summarize`` renders one of its sections as
    Markdown lines.
    """

    name: str
    needs_concepts: bool
    needs_hierarchy: bool
    run: Callable[[EvalContext], dict[str, dict]]
    summarize: Callable[[str, dict], list[str]]


def coralnet_groups(section: DataSection, split: str, repo_root: Path = REPO_ROOT) -> list[str]:
    """The groups of the data config's CoralNet split file whose split is ``split``.

    Raises:
        ValueError: The coralnet entry has no ``split_file``, or no group has the split.
    """
    split_file = section.datasets[CORALNET].split_file
    if split_file is None:
        raise ValueError("the data config's coralnet entry has no split_file")
    return split_groups(repo_root / split_file, split)


def trained_on_scored_groups(section: DataSection, scored: list[str]) -> list[str]:
    """The scored CoralNet groups that the data config's ``train`` block also selects, sorted.

    A ``None`` selection (every group) overlaps all of ``scored``; an entry with no ``train``
    block overlaps none.
    """
    entry = section.datasets[CORALNET]
    if entry.train is None:
        return []
    train = entry_split_groups(entry, "train", REPO_ROOT)
    return sorted(scored if train is None else set(scored) & train)


def coralscapes_trained_on_scored(section: DataSection, split: str) -> bool:
    """Whether the data config's coralscapes_v2 ``train`` block selects the group the probes score."""
    entry = section.datasets[CORALSCAPES]
    if entry.train is None:
        return False
    train = entry_split_groups(entry, "train", REPO_ROOT)
    return train is None or split_group(split) in train


def check_split_evals(split: str, evals: list[str]) -> None:
    """``--split train`` only has CoralNet sources and the loss to score."""
    if split != "train":
        return
    unsupported = [e for e in evals if e not in TRAIN_SPLIT_EVALS]
    if unsupported:
        raise ValueError(
            f"--split train supports only --evals {' '.join(TRAIN_SPLIT_EVALS)}, got {unsupported}"
        )


def eval_amp_dtype(run_config: dict, amp: bool) -> torch.dtype | None:
    """Autocast dtype from the run's ``training.mixed_precision_dtype`` when ``amp`` is set, else ``None``."""
    if not amp:
        return None
    return resolve_amp_dtype(run_config["training"]["mixed_precision_dtype"])


def concept_mapping_path_of(run_config: dict) -> Path:
    """The run's ``training.concept_mapping_path``, resolved against the repository root."""
    return resolve_repo_path(run_config["training"]["concept_mapping_path"])


def run_removed_concepts(run_config: dict) -> list[str]:
    """Channels removed by the run's ``training.concept_removal``, if any."""
    return concept_removal_list(run_config["training"])


def concept_names_of(bundle: ModelBundle) -> list[str]:
    """The bundle's concept channel names in channel order."""
    return [name for _, name in sorted(bundle.concept_id2name.items())]


def failed_sections(summary: dict) -> list[str]:
    """Summary keys whose eval recorded ``{"error": ...}``."""
    return [k for k, v in summary.items() if isinstance(v, dict) and "error" in v]


def _section_key(name: str, split: str) -> str:
    """The summary key an evaluation's result is recorded under."""
    if name == "pacific":
        return name
    if name == VAL_LOSS_KEY:
        return {"train": TRAIN_LOSS_KEY, "test": TEST_LOSS_KEY}.get(split, VAL_LOSS_KEY)
    return f"{name}_{split}"


def _point_kwargs(ctx: EvalContext) -> dict[str, Any]:
    """The keyword arguments the CoralNet and Pacific point evaluations share."""
    options = ctx.options
    return {
        "predictor": ctx.predictor,
        "id2label": ctx.bundle.id2label,
        "concept_names": ctx.concept_names,
        "taxonomy_csv": concept_mapping_path_of(ctx.bundle.config),
        "hierarchy": ctx.hierarchy,
        "output_dir": ctx.output_dir,
        "batch_size": options.batch_size,
        "num_workers": options.num_workers,
        "max_images_per_unit": options.max_images_per_unit,
        "removed_concepts": ctx.removed_concepts,
    }


def _run_coralnet(ctx: EvalContext) -> dict[str, dict]:
    key = _section_key("coralnet", ctx.split)
    reader = coralnet_reader(
        ctx.section,
        ctx.cache_dir,
        groups=coralnet_groups(ctx.section, ctx.split),
        split=ctx.split,
        image_ids=ctx.sample_ids,
    )
    return {
        key: evaluate_point_dataset(key, reader, source_name=CORALNET_SOURCE, **_point_kwargs(ctx))
    }


def _run_pacific(ctx: EvalContext) -> dict[str, dict]:
    reader = pacific_reader(ctx.section, ctx.cache_dir, REPO_ROOT)
    result = evaluate_point_dataset(
        PACIFIC_SOURCE, reader, source_name=PACIFIC_SOURCE, **_point_kwargs(ctx)
    )
    return {_section_key("pacific", ctx.split): result}


def _run_coralscapes(ctx: EvalContext) -> dict[str, dict]:
    return evaluate_coralscapes_probe(
        CORALSCAPES_V2,
        [ctx.options.probe_points],
        open_dataset_version(ctx.section, CORALSCAPES, ctx.cache_dir),
        predictor=ctx.predictor,
        output_dir=ctx.output_dir,
        concepts=ProbeConcepts(
            ctx.concept_names, concept_mapping_path_of(ctx.bundle.config), ctx.removed_concepts
        ),
        splits=[ctx.split],
        svm_c=ctx.options.svm_c,
        max_test_images=ctx.options.max_test_images,
    )


def _run_coralscapes39(ctx: EvalContext) -> dict[str, dict]:
    return evaluate_coralscapes_probe(
        CORALSCAPES39,
        [ctx.options.coralscapes39_probe],
        open_dataset_version(ctx.section, CORALSCAPES, ctx.cache_dir),
        predictor=ctx.predictor,
        output_dir=ctx.output_dir,
        splits=[ctx.split],
        svm_c=ctx.options.svm_c,
        max_test_images=ctx.options.max_test_images,
    )


def _run_val_loss(ctx: EvalContext) -> dict[str, dict]:
    """The loss on the blocks of ``ctx.split``: val, test, or the CoralNet split-file train groups.

    The train loss applies the run's concept source mask; on a CoralNet sample it scores
    every sampled image.
    """
    options = ctx.options
    section = ctx.section
    max_images = options.val_loss_max_images
    image_ids = None
    if ctx.split == "train":
        entry = section.datasets[CORALNET].model_copy(
            update={"train": SplitSelection(groups=coralnet_groups(section, "train"))}
        )
        section = section.model_copy(update={"datasets": {CORALNET: entry}})
        if ctx.sample_ids is not None:
            image_ids = {(CORALNET, "val"): ctx.sample_ids}
            max_images = None
    key = _section_key(VAL_LOSS_KEY, ctx.split)
    result = evaluate_val_loss(
        bundle=ctx.bundle,
        section=section,
        split=ctx.split,
        cache_dir=ctx.cache_dir,
        output_dir=ctx.output_dir,
        device=ctx.predictor.device,
        batch_size=options.batch_size,
        num_workers=options.num_workers,
        amp_dtype=ctx.predictor.amp_dtype,
        schema_config=options.schema_config,
        max_images_per_dataset=max_images,
        name=key,
        apply_source_mask=ctx.split == "train",
        image_ids=image_ids,
    )
    return {key: result}


REGISTRY: dict[str, Evaluation] = {
    e.name: e
    for e in (
        Evaluation("coralnet", True, True, _run_coralnet, summarize_coralnet),
        Evaluation("pacific", True, True, _run_pacific, summarize_pacific),
        Evaluation("coralscapes", True, False, _run_coralscapes, summarize_coralscapes),
        Evaluation("coralscapes39", False, False, _run_coralscapes39, summarize_coralscapes39),
        Evaluation(VAL_LOSS_KEY, False, False, _run_val_loss, summarize_loss),
    )
}
ALL_EVALS = list(REGISTRY)


def _check_concepts(
    bundle_dir: Path, concept_names: list[str], num_concepts: int, needs_concepts: bool
) -> None:
    """Check the bundle's concept names against the model, and the reported concepts if needed.

    Raises:
        ValueError: The bundle names a different number of concepts than the model has.
        RuntimeError: ``needs_concepts`` is set and a reported binary concept is missing.
    """
    if len(concept_names) != num_concepts:
        raise ValueError(
            f"{bundle_dir / MODEL_JSON} has {len(concept_names)} concept names but the model has "
            f"{num_concepts} concept channels."
        )
    if needs_concepts:
        require_reported_concepts(concept_names)


def _draw_sample(
    section: DataSection,
    cache_dir: Path,
    path: Path,
    split: Split,
    images_per_source: int,
    seed: int,
) -> list[str]:
    """Write the seeded per-source sample of the CoralNet ``split`` groups to ``path``."""
    reader = coralnet_reader(
        section, cache_dir, groups=coralnet_groups(section, split), split=split
    )
    return write_coralnet_sample(
        path=path, reader=reader, images_per_source=images_per_source, seed=seed
    )


def _record_training_overlap(
    summary: dict, section: DataSection, split: str, evals: list[str]
) -> None:
    """Record, and warn about, scored CoralscapesV2 and CoralNet groups the config trains on."""
    if ("coralscapes" in evals or "coralscapes39" in evals) and CORALSCAPES in section.datasets:
        in_training = coralscapes_trained_on_scored(section, split)
        summary["coralscapes_split_in_training"] = in_training
        if in_training:
            logger.warning(
                "The data config trains on CoralscapesV2 groups scored as --split %s: %s",
                split,
                split_group(split),
            )
    if "coralnet" in evals:
        overlap = trained_on_scored_groups(section, coralnet_groups(section, split))
        summary["coralnet_split_in_training"] = bool(overlap)
        if overlap:
            logger.warning(
                "The data config trains on CoralNet groups scored as --split %s: %s",
                split,
                overlap,
            )


def _run_entry(entry: Evaluation, ctx: EvalContext, summary: dict) -> None:
    """Run one evaluation into ``summary``; a failure is recorded under its key, not raised."""
    try:
        summary.update(entry.run(ctx))
    except Exception as e:  # noqa: BLE001
        logger.exception("Evaluation %s failed: %s", entry.name, e)
        summary[_section_key(entry.name, ctx.split)] = {"error": f"{type(e).__name__}: {e}"}
    write_summary(summary, ctx.output_dir, REGISTRY)


def run_evaluations(options: EvalOptions) -> dict:
    """Run the evaluations ``options`` selects and write their outputs and summary.

    Returns:
        The summary, as written to ``<output_dir>/summary.json``.

    Raises:
        ValueError: ``--split train`` selects an evaluation other than ``coralnet`` and
            ``val_loss``, or the bundle's concept names do not match the model.
        RuntimeError: A selected evaluation needs a reported binary concept the bundle lacks.
    """
    evals = list(dict.fromkeys(options.evals))
    check_split_evals(options.split, evals)
    selected = [entry for name, entry in REGISTRY.items() if name in evals]
    output_dir = Path(options.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device(options.device or ("cuda" if torch.cuda.is_available() else "cpu"))
    section = DataSection.model_validate(load_section(options.data_config, "data"))
    cache_dir = Path(options.cache_dir)

    logger.info("Loading bundle %s ...", options.bundle)
    bundle = load_bundle(options.bundle, device)
    predictor = CBMPredictor.from_bundle(
        bundle, amp_dtype=eval_amp_dtype(bundle.config, options.amp)
    )
    concept_names = concept_names_of(bundle)
    logger.info(
        "Device: %s | evals: %s | input_size: %s | num_classes=%d num_concepts=%d",
        device,
        evals,
        bundle.input_size,
        predictor.num_classes,
        predictor.num_concepts,
    )
    needs_concepts = any(entry.needs_concepts for entry in selected)
    _check_concepts(options.bundle, concept_names, predictor.num_concepts, needs_concepts)

    removed_concepts: list[str] = []
    if needs_concepts:
        removed_concepts = run_removed_concepts(bundle.config)
        logger.info("Concept removal of the bundle's run: %d channels", len(removed_concepts))
    hierarchy = None
    if any(entry.needs_hierarchy for entry in selected):
        hierarchy = load_benthic_hierarchy(section, cache_dir)

    summary: dict = {
        "bundle": str(options.bundle),
        "input_size": list(bundle.input_size),
        "split": options.split,
        "data_config": str(options.data_config),
        "evals": evals,
        "num_removed_concepts": len(removed_concepts),
    }
    t_start = time.time()
    sample_ids = None
    uses_sample = "coralnet" in evals or (options.split == "train" and "val_loss" in evals)
    if options.images_per_source is not None and uses_sample:
        sample_path = output_dir / f"coralnet_{options.split}_sample.parquet"
        sample_ids = _draw_sample(
            section,
            cache_dir,
            sample_path,
            options.split,
            options.images_per_source,
            options.sample_seed,
        )
        summary["images_per_source"] = options.images_per_source
        summary["sample_seed"] = options.sample_seed
        summary["sample"] = str(sample_path)
    _record_training_overlap(summary, section, options.split, evals)

    ctx = EvalContext(
        bundle=bundle,
        predictor=predictor,
        section=section,
        cache_dir=cache_dir,
        output_dir=output_dir,
        split=options.split,
        concept_names=concept_names,
        hierarchy=hierarchy,
        removed_concepts=removed_concepts,
        sample_ids=sample_ids,
        options=options,
    )
    for entry in selected:
        _run_entry(entry, ctx, summary)

    write_summary(summary, output_dir, REGISTRY)
    logger.info("ALL DONE in %.1fs. Outputs in %s", time.time() - t_start, output_dir)
    return summary
