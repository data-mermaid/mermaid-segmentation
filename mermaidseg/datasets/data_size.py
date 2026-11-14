"""Choose which training images a data-size profile keeps.

The four experiments share this module so the drop rule, the config written for it, and the
allowlist applied at load time cannot drift apart.

Populations are the train split of ``configs/data_config.yaml`` only. Val and test blocks
are copied through unchanged. Image drops are an allowlist (``image_keep``) applied after each
dataset's own train filters. Source drops remove a whole CoralNet source or a whole other
dataset; they never split one.

Fractions are 10%, 20% and 50% of images. Experiment 1 also has a 0% cell: every CoralNet
training image, and no other dataset's train split. Image drops share one permutation (seed 0).
Source drops (experiments 2 and 4) share one ranking, most coral genera first, and remove the
whole-unit prefix whose image count is closest to the fraction, so the larger drop contains
the smaller one. Experiment 2 ranks CoralNet sources only. Experiment 4 ranks those sources
plus one unit per other dataset.
"""

from __future__ import annotations

import copy
import math
import re
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from fractions import Fraction
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

FRACTIONS: tuple[float, ...] = (0.1, 0.2, 0.5)
SEED = 0
RUN_PREFIX = "cbm_dsize"
METRIC = "n_coral_genera"
SKIP_SPLIT = "None"

MISSING_TAXON = frozenset({"", "none", "not_given", "nan"})
CORAL_CLASSES = frozenset({"hexacorallia", "octocorallia"})

# Identity columns of a training image, after that dataset's own train filters.
POINT_KEYS: dict[str, tuple[str, ...]] = {
    "coralnet": ("source_id", "image_id"),
    "mermaid": ("image_id",),
    "catlin_seaview": ("image_id", "region"),
    "moorea_labeled_corals": ("image_id", "year"),
}
UCSD_KEYS: tuple[str, ...] = ("site", "filename")
# First match on a Coralscapes split wins. Otherwise the keep file stores the train-split
# row index and the load pins the HF revision those indexes came from.
CORALSCAPES_ID_COLUMNS: tuple[str, ...] = ("filename", "file_name", "image_id", "id")
HF_DATASETS: tuple[str, ...] = ("coralscapes_v2", "ucsd_mosaics")

DATASET_ORDER: tuple[str, ...] = (
    "coralnet",
    "mermaid",
    "coralscapes_v2",
    "catlin_seaview",
    "moorea_labeled_corals",
    "ucsd_mosaics",
)
META_DATASETS: tuple[str, ...] = tuple(name for name in DATASET_ORDER if name != "coralnet")

STATS_COLUMNS: tuple[str, ...] = (
    "unit_id",
    "dataset",
    "n_images",
    "n_genera",
    "n_coral_genera",
    "n_labels",
    "n_unmapped_labels",
)

_PROFILE_RE = re.compile(r"^(exp[1-4])_drop(0|10|20|50)$")
_CORALNET_UNIT_RE = re.compile(r"^coralnet:(\d+)$")
_DATASET_UNIT_PREFIX = "dataset:"


@dataclass(frozen=True)
class Experiment:
    """One of the four size sweeps."""

    name: str
    population: str  # "coralnet" or "meta"
    mode: str  # "iid_images" or "source_images"
    metric: str | None


EXPERIMENTS: dict[str, Experiment] = {
    "exp1": Experiment("exp1", "coralnet", "iid_images", None),
    "exp2": Experiment("exp2", "coralnet", "source_images", METRIC),
    "exp3": Experiment("exp3", "meta", "iid_images", None),
    "exp4": Experiment("exp4", "meta", "source_images", METRIC),
}


@dataclass(frozen=True)
class GenusCount:
    """Distinct genera among the labels a unit actually annotates."""

    n_genera: int
    n_coral_genera: int
    n_labels: int
    n_unmapped_labels: int


@dataclass(frozen=True)
class SizeUnit:
    """One atomic source: a CoralNet source, or a whole non-CoralNet dataset."""

    unit_id: str
    dataset: str
    n_images: int
    n_coral_genera: int
    n_genera: int = 0


@dataclass(frozen=True)
class DropPlan:
    """Images and units removed for one experiment cell."""

    experiment: str
    fraction: float
    dropped_uids: tuple[int, ...]
    dropped_units: tuple[str, ...]
    n_images_total: int
    n_images_dropped: int
    n_units_total: int
    n_units_dropped: int

    @property
    def fraction_images_achieved(self) -> float:
        return self.n_images_dropped / self.n_images_total

    @property
    def fraction_units_achieved(self) -> float:
        return self.n_units_dropped / self.n_units_total

    @property
    def n_images_kept(self) -> int:
        return self.n_images_total - self.n_images_dropped


def fractions_for(experiment: str) -> tuple[float, ...]:
    """Drop fractions written for ``experiment``.

    The shared grid is 10%, 20% and 50%. Experiment 1 also writes 0%: CoralNet train
    images only, with none of them removed.
    """
    if experiment not in EXPERIMENTS:
        raise ValueError(f"unknown experiment {experiment!r}")
    if experiment == "exp1":
        return (0.0, *FRACTIONS)
    return FRACTIONS


def profile_name(experiment: str, fraction: float) -> str:
    """``exp1_drop10`` for a 10% drop. Experiment 1 also accepts ``exp1_drop0``."""
    if experiment not in EXPERIMENTS:
        raise ValueError(f"unknown experiment {experiment!r}")
    percent = int(round(float(fraction) * 100))
    allowed = fractions_for(experiment)
    recognized = percent in {int(round(item * 100)) for item in allowed}
    if not recognized or abs(float(fraction) - percent / 100) > 1e-9:
        listed = ", ".join(str(item) for item in allowed)
        raise ValueError(f"fraction for {experiment} must be one of {listed}, got {fraction}")
    return f"{experiment}_drop{percent}"


def parse_profile_name(name: str) -> tuple[str, float]:
    """Return ``(experiment, fraction)`` or raise ``ValueError``."""
    match = _PROFILE_RE.match(name)
    if match is None:
        raise ValueError(f"not a data-size profile: {name}")
    experiment = match.group(1)
    fraction = int(match.group(2)) / 100
    if profile_name(experiment, fraction) != name:
        raise ValueError(f"not a data-size profile: {name}")
    return experiment, fraction


def all_profile_names() -> list[str]:
    """The twelve 10/20/50 cells, experiment then fraction.

    ``exp1_drop0`` is not in this list. :func:`fractions_for` includes it.
    """
    return [profile_name(name, fraction) for name in EXPERIMENTS for fraction in FRACTIONS]


def run_name(profile: str) -> str:
    """Checkpoint directory name. Rejects anything that is not a data-size profile."""
    parse_profile_name(profile)
    return f"{RUN_PREFIX}_{profile}"


def coralnet_unit_id(source_id: int) -> str:
    """Zero-padded so a metric tie removes the smaller source id first."""
    return f"coralnet:{int(source_id):07d}"


def dataset_unit_id(dataset: str) -> str:
    return f"{_DATASET_UNIT_PREFIX}{dataset}"


def source_id_from_unit(unit_id: str) -> int:
    match = _CORALNET_UNIT_RE.match(unit_id)
    if match is None:
        raise ValueError(f"not a CoralNet unit id: {unit_id}")
    return int(match.group(1))


def dataset_from_unit(unit_id: str) -> str:
    if not unit_id.startswith(_DATASET_UNIT_PREFIX):
        raise ValueError(f"not a dataset unit id: {unit_id}")
    return unit_id[len(_DATASET_UNIT_PREFIX) :]


def drop_count(n: int, fraction: float) -> int:
    """Closest integer to ``fraction * n``. Halves round away from zero (``floor(x + 0.5)``)."""
    if n < 0:
        raise ValueError(f"n must be non-negative, got {n}")
    if not 0.0 <= fraction <= 1.0:
        raise ValueError(f"fraction must be in [0, 1], got {fraction}")
    return int(math.floor(fraction * n + 0.5))


def permute_uids(uids: Sequence[int], seed: int) -> np.ndarray:
    """Shuffle ``uids`` with a seed. The input order does not matter; values are sorted first."""
    ordered = np.array(sorted(int(uid) for uid in uids), dtype=np.int64)
    if len(ordered) != len(set(ordered.tolist())):
        raise ValueError("image uids are not unique")
    rng = np.random.Generator(np.random.PCG64(seed))
    return ordered[rng.permutation(len(ordered))]


def _fraction_score(running: int, total: int, fraction: float) -> int:
    """Integer distance of ``running`` from ``fraction * total``, scaled so ties stay exact."""
    frac = Fraction(fraction).limit_denominator(1000)
    return abs(frac.denominator * running - frac.numerator * total)


def rank_units(units: Sequence[SizeUnit], metric: str = METRIC) -> list[SizeUnit]:
    """Highest ``metric`` first. Ties break on ``unit_id`` ascending (smaller source id first)."""
    if metric != METRIC:
        raise ValueError(f"only {METRIC} is a ranking metric, got {metric}")

    def key(unit: SizeUnit) -> tuple[int, str]:
        return (-unit.n_coral_genera, unit.unit_id)

    return sorted(units, key=key)


def closest_prefix(ranked: Sequence[SizeUnit], fraction: float, *, weight: str) -> list[SizeUnit]:
    """Prefix of ``ranked`` whose weight is closest to ``fraction`` of the total.

    ``weight="n_images"`` matches an image fraction (experiments 2 and 4). ``weight="units"``
    counts each source once. An exact distance tie keeps the shorter prefix, which removes
    fewer images or units. Weights must be positive.
    """
    if weight == "units":
        weights = [1] * len(ranked)
    elif weight == "n_images":
        weights = [unit.n_images for unit in ranked]
    else:
        raise ValueError(f"weight must be 'n_images' or 'units', got {weight}")
    if any(w <= 0 for w in weights):
        raise ValueError("prefix weights must be positive")
    if not 0.0 <= fraction <= 1.0:
        raise ValueError(f"fraction must be in [0, 1], got {fraction}")
    total = sum(weights)
    best_k = 0
    best_score = _fraction_score(0, total, fraction)
    running = 0
    for k, w in enumerate(weights, start=1):
        running += w
        score = _fraction_score(running, total, fraction)
        if score < best_score:
            best_score = score
            best_k = k
    return list(ranked[:best_k])


def _concepts_for(concepts: pd.DataFrame, dataset_source: str) -> pd.DataFrame:
    needed = {"source_label_class_name", "source_dataset_source", "genus", "class"}
    missing = needed - set(concepts.columns)
    if missing:
        raise ValueError(f"class_to_concepts is missing columns {sorted(missing)}")
    source = dataset_source.strip().lower()
    rows = concepts[concepts["source_dataset_source"].astype(str).str.strip().str.lower() == source]
    if rows.empty:
        raise ValueError(f"class_to_concepts.csv has no rows for {dataset_source}")
    frame = rows.assign(
        label=rows["source_label_class_name"].astype(str).str.strip().str.lower(),
        genus=rows["genus"].astype(str).str.strip().str.lower(),
        cls=rows["class"].astype(str).str.strip().str.lower(),
    )
    return frame.drop_duplicates("label").set_index("label")[["genus", "cls"]]


def genus_counts(labels: Iterable[str], concepts: pd.DataFrame, dataset_source: str) -> GenusCount:
    """Count distinct genus values for labels that occur in this unit.

    The join is that dataset's own ``class_to_concepts`` rows, on the lowercased label.
    Blank, ``none``, ``not_given`` and ``nan`` genera are ignored. A genus is a coral genus
    only when the row's ``class`` is ``hexacorallia`` or ``octocorallia``. The same genus on
    two labels counts once. Labels with no row, and rows whose label never occurs, add nothing.
    """
    indexed = _concepts_for(concepts, dataset_source)
    genus_by_label = indexed["genus"].astype(str).to_dict()
    class_by_label = indexed["cls"].astype(str).to_dict()
    genera: set[str] = set()
    coral: set[str] = set()
    seen: set[str] = set()
    unmapped = 0
    for label in labels:
        key = str(label).strip().lower()
        if key in seen:
            continue
        seen.add(key)
        genus = genus_by_label.get(key)
        if genus is None:
            unmapped += 1
            continue
        if genus in MISSING_TAXON:
            continue
        genera.add(genus)
        if class_by_label[key] in CORAL_CLASSES:
            coral.add(genus)
    return GenusCount(
        n_genera=len(genera),
        n_coral_genera=len(coral),
        n_labels=len(seen),
        n_unmapped_labels=unmapped,
    )


def coralscapes_keys(column_names: Sequence[str]) -> tuple[str, ...]:
    """Stable id columns, or ``("row_index",)`` when the split has none."""
    names = set(column_names)
    for name in CORALSCAPES_ID_COLUMNS:
        if name in names:
            return (name,)
    return ("row_index",)


def _population(inventory: pd.DataFrame, population: str) -> pd.DataFrame:
    if population == "coralnet":
        return inventory.loc[inventory["dataset"] == "coralnet"].reset_index(drop=True)
    if population == "meta":
        return inventory.reset_index(drop=True)
    raise ValueError(f"unknown population {population}")


def _units_for(stats: pd.DataFrame, population: pd.DataFrame) -> list[SizeUnit]:
    counts = population.groupby(population["unit_id"].astype(str)).size()
    present = set(counts.index)
    rows = stats.loc[stats["unit_id"].astype(str).isin(present)]
    found = set(rows["unit_id"].astype(str))
    if found != present:
        missing = sorted(present - found)
        raise ValueError(f"unit stats are missing {missing[:5]}")
    units: list[SizeUnit] = []
    for row in rows.itertuples(index=False):
        unit_id = str(row.unit_id)
        n_images = int(counts[unit_id])
        if int(row.n_images) != n_images:
            raise ValueError(
                f"{unit_id} stats say {int(row.n_images)} images but the inventory has {n_images}"
            )
        units.append(
            SizeUnit(
                unit_id=unit_id,
                dataset=str(row.dataset),
                n_images=n_images,
                n_coral_genera=int(row.n_coral_genera),
                n_genera=int(row.n_genera),
            )
        )
    return units


def plan_drop(
    experiment: str,
    fraction: float,
    inventory: pd.DataFrame,
    stats: pd.DataFrame,
    permutation: np.ndarray | None = None,
) -> DropPlan:
    """Decide the drop for one cell.

    ``permutation`` is the saved uid order for an IID experiment (the first ``k`` are dropped).
    Source experiments ignore it and take a prefix of the ``n_coral_genera`` ranking.
    """
    spec = EXPERIMENTS[experiment]
    profile_name(experiment, fraction)
    population = _population(inventory, spec.population)
    if population.empty:
        raise ValueError(f"{experiment} population has no training images")
    if population["image_uid"].duplicated().any():
        raise ValueError("inventory image_uid values are not unique")
    units = _units_for(stats, population)
    n_images_total = int(len(population))
    n_units_total = len(units)

    if spec.mode == "iid_images":
        if permutation is None:
            raise ValueError(f"{experiment} needs the saved image permutation")
        perm = np.asarray(permutation, dtype=np.int64)
        uids = population["image_uid"].to_numpy(dtype=np.int64)
        if sorted(perm.tolist()) != sorted(int(uid) for uid in uids):
            raise ValueError("permutation does not match the population")
        k = drop_count(len(perm), fraction)
        dropped_uid_list = tuple(int(uid) for uid in perm[:k])
        dropped_uid_set = set(dropped_uid_list)
        fully_dropped = []
        for unit_id, group in population.groupby("unit_id"):
            if {int(uid) for uid in group["image_uid"]} <= dropped_uid_set:
                fully_dropped.append(str(unit_id))
        dropped_units = tuple(sorted(fully_dropped))
    elif spec.mode == "source_images":
        dropped = closest_prefix(
            rank_units(units, spec.metric or METRIC), fraction, weight="n_images"
        )
        dropped_units = tuple(unit.unit_id for unit in dropped)
        dropped_set = set(dropped_units)
        mask = population["unit_id"].astype(str).isin(dropped_set)
        dropped_uid_list = tuple(int(uid) for uid in population.loc[mask, "image_uid"])
        # All-or-nothing: every image of a dropped unit is dropped, and no other image is.
        kept_units = set(population.loc[~mask, "unit_id"].astype(str))
        if kept_units & dropped_set:
            raise ValueError("a dropped unit still has a kept image")
    else:
        raise ValueError(f"unknown mode {spec.mode}")

    return DropPlan(
        experiment=experiment,
        fraction=fraction,
        dropped_uids=dropped_uid_list,
        dropped_units=dropped_units,
        n_images_total=n_images_total,
        n_images_dropped=len(dropped_uid_list),
        n_units_total=n_units_total,
        n_units_dropped=len(dropped_units),
    )


def per_dataset_counts(plan: DropPlan, inventory: pd.DataFrame) -> dict[str, dict[str, int]]:
    """Images before and after the drop, for every dataset present in ``inventory``.

    A CoralNet-only experiment reports the other datasets as kept 0: their train splits are
    turned off, so a non-zero keep count would mean those images were still in the population.
    """
    spec = EXPERIMENTS[plan.experiment]
    dropped = set(plan.dropped_uids)
    counts: dict[str, dict[str, int]] = {}
    for dataset in DATASET_ORDER:
        sub = inventory.loc[inventory["dataset"] == dataset]
        if sub.empty:
            continue
        if spec.population == "coralnet" and dataset != "coralnet":
            kept = 0
        else:
            kept = int((~sub["image_uid"].isin(dropped)).sum())
        counts[dataset] = {"n_total": int(len(sub)), "n_kept": kept}
    return counts


def keep_frames(
    plan: DropPlan,
    inventory: pd.DataFrame,
    key_columns: Mapping[str, Sequence[str]],
) -> dict[str, pd.DataFrame]:
    """Per-dataset allowlists of images that remain. Empty datasets are omitted.

    Source-level plans return no frames: the config expresses them with whitelists and
    ``train: None`` instead of an image allowlist.
    """
    spec = EXPERIMENTS[plan.experiment]
    if spec.mode != "iid_images":
        return {}
    population = _population(inventory, spec.population)
    dropped = set(plan.dropped_uids)
    kept = population.loc[~population["image_uid"].isin(dropped)]
    frames: dict[str, pd.DataFrame] = {}
    for dataset, group in kept.groupby("dataset", sort=False):
        keys = list(key_columns[str(dataset)])
        missing = [key for key in keys if key not in group.columns]
        if missing:
            raise ValueError(f"{dataset} inventory is missing keep columns {missing}")
        frame = group.loc[:, ["dataset", *keys]].reset_index(drop=True)
        if frame.empty:
            continue
        frames[str(dataset)] = frame
    return frames


def _train_block(config: dict, dataset: str) -> Any:
    section = config["data"].get(dataset)
    if not isinstance(section, dict):
        raise ValueError(f"base config has no data.{dataset} section")
    return section.get("train")


def _is_skip(block: Any) -> bool:
    return block is None or block == SKIP_SPLIT


def assemble_config(
    base: dict,
    experiment: str,
    *,
    kept_coralnet_sources: Sequence[int] | None = None,
    image_keep: Mapping[str, str] | None = None,
    hf_pins: Mapping[str, Mapping[str, Any]] | None = None,
    dropped_datasets: Sequence[str] | None = None,
) -> dict:
    """Copy ``base`` and apply one experiment's training-split edits.

    Val and test blocks are not touched. ``hf_repo_id`` is not touched, so a dense dataset that
    remains stays on the repo the base config names.
    """
    spec = EXPERIMENTS[experiment]
    config = copy.deepcopy(base)
    image_keep = dict(image_keep or {})
    hf_pins = dict(hf_pins or {})
    dropped_datasets = list(dropped_datasets or [])

    if spec.population == "coralnet":
        for dataset in META_DATASETS:
            config["data"][dataset]["train"] = SKIP_SPLIT
        image_keep = {name: path for name, path in image_keep.items() if name == "coralnet"}

    if spec.mode == "iid_images":
        if kept_coralnet_sources is not None or dropped_datasets:
            raise ValueError(f"{experiment} drops images, not sources")
        for dataset, path in image_keep.items():
            block = _train_block(config, dataset)
            if _is_skip(block):
                raise ValueError(f"{experiment} keep list for {dataset} but its train split is off")
            block["image_keep"] = str(path)
            if dataset in HF_DATASETS:
                pin = hf_pins.get(dataset)
                if not pin or "n_rows" not in pin:
                    raise ValueError(
                        f"{dataset} image_keep needs the split row count it was built from"
                    )
                revision = pin.get("revision") or None
                repo = block.get("hf_repo_id")
                local_repo = isinstance(repo, str) and Path(repo).is_dir()
                if revision:
                    block["hf_revision"] = str(revision)
                elif not local_repo:
                    raise ValueError(f"{dataset} image_keep needs an HF revision")
                block["image_keep_n_rows"] = int(pin["n_rows"])
        if spec.population == "meta":
            for dataset in DATASET_ORDER:
                if dataset not in image_keep and not _is_skip(_train_block(config, dataset)):
                    config["data"][dataset]["train"] = SKIP_SPLIT
    else:
        if image_keep or hf_pins:
            raise ValueError(f"{experiment} removes whole sources and must not set image_keep")
        if kept_coralnet_sources is None:
            raise ValueError(f"{experiment} needs the CoralNet sources that stay")
        kept = [int(source) for source in kept_coralnet_sources]
        block = _train_block(config, "coralnet")
        if _is_skip(block):
            raise ValueError("base config has no CoralNet train split")
        if kept:
            block["whitelist_sources"] = kept
        else:
            config["data"]["coralnet"]["train"] = SKIP_SPLIT
        for dataset in dropped_datasets:
            if dataset == "coralnet":
                raise ValueError("drop CoralNet by source id, not as one dataset")
            if dataset not in META_DATASETS:
                raise ValueError(f"cannot drop unknown dataset {dataset}")
            config["data"][dataset]["train"] = SKIP_SPLIT

    assert_profile_config(base, config, experiment)
    return config


def config_for_plan(
    base: dict,
    plan: DropPlan,
    inventory: pd.DataFrame,
    key_columns: Mapping[str, Sequence[str]],
    *,
    keep_dir: Path,
    hf_pins: Mapping[str, Mapping[str, Any]] | None = None,
) -> tuple[dict, dict[str, pd.DataFrame]]:
    """Build the data config and the allowlist frames for ``plan``.

    A 0% CoralNet drop leaves that train block unchanged, so the run loads every CoralNet
    training image the base config would. It does not write an allowlist of those images.
    """
    spec = EXPERIMENTS[plan.experiment]
    if spec.mode == "iid_images" and spec.population == "coralnet" and plan.fraction == 0.0:
        if plan.n_images_dropped != 0:
            raise ValueError(f"{plan.experiment} drop 0 removed {plan.n_images_dropped} images")
        return assemble_config(base, plan.experiment), {}
    frames = keep_frames(plan, inventory, key_columns)
    if spec.mode == "iid_images":
        paths = {dataset: str(keep_dir / f"{dataset}.parquet") for dataset in frames}
        config = assemble_config(base, plan.experiment, image_keep=paths, hf_pins=hf_pins)
        return config, frames

    base_sources = [
        int(source) for source in base["data"]["coralnet"]["train"]["whitelist_sources"]
    ]
    dropped_sources = {
        source_id_from_unit(unit_id)
        for unit_id in plan.dropped_units
        if unit_id.startswith("coralnet:")
    }
    kept_sources = [source for source in base_sources if source not in dropped_sources]
    dropped_datasets = [
        dataset_from_unit(unit_id)
        for unit_id in plan.dropped_units
        if unit_id.startswith(_DATASET_UNIT_PREFIX)
    ]
    config = assemble_config(
        base,
        plan.experiment,
        kept_coralnet_sources=kept_sources,
        dropped_datasets=dropped_datasets,
    )
    return config, {}


def _coralnet_sources(config: dict, split: str) -> set[int]:
    section = config.get("data", {}).get("coralnet", {})
    block = section.get(split) if isinstance(section, dict) else None
    if not isinstance(block, dict):
        return set()
    return {int(source) for source in block.get("whitelist_sources") or []}


def assert_profile_config(base: dict, config: dict, experiment: str) -> None:
    """Reject a config that could train on val/test or change dense masks.

    Raises ``ValueError`` on the first problem. A config that passes trains on a subset of the
    base train population and evaluates against the original val and test blocks.
    """
    spec = EXPERIMENTS[experiment]
    for dataset, section in base["data"].items():
        if not isinstance(section, dict) or dataset == "default":
            continue
        out = config["data"].get(dataset)
        if not isinstance(out, dict):
            raise ValueError(f"profile dropped the data.{dataset} section")
        for split in ("val", "test"):
            if out.get(split) != section.get(split):
                raise ValueError(f"data.{dataset}.{split} differs from the base config")
            block = out.get(split)
            if isinstance(block, dict) and ("image_keep" in block or "hf_revision" in block):
                raise ValueError(f"data.{dataset}.{split} must not carry an image allowlist")

    base_train = _coralnet_sources(base, "train")
    held_out = _coralnet_sources(base, "val") | _coralnet_sources(base, "test")
    if not base_train.isdisjoint(held_out):
        raise ValueError("base CoralNet train sources overlap val or test")
    out_train = _coralnet_sources(config, "train")
    if not out_train <= base_train:
        raise ValueError("profile CoralNet train sources are not a subset of the base train list")
    if not out_train.isdisjoint(held_out):
        raise ValueError("profile CoralNet train sources overlap val or test")

    for dataset, section in base["data"].items():
        if not isinstance(section, dict):
            continue
        base_train_block = section.get("train")
        out_train_block = config["data"][dataset].get("train")
        if isinstance(base_train_block, dict) and isinstance(out_train_block, dict):
            if out_train_block.get("hf_repo_id") != base_train_block.get("hf_repo_id"):
                raise ValueError(f"data.{dataset}.train.hf_repo_id changed")
            repo = out_train_block.get("hf_repo_id")
            if isinstance(repo, str) and "sparse" in repo:
                raise ValueError(f"data.{dataset}.train points at a sparse copy")

    if spec.population == "coralnet":
        for dataset in META_DATASETS:
            if not _is_skip(config["data"][dataset].get("train")):
                raise ValueError(f"{experiment} must turn off {dataset} training")

    if spec.mode != "iid_images":
        for dataset, section in config["data"].items():
            block = section.get("train") if isinstance(section, dict) else None
            if isinstance(block, dict) and ("image_keep" in block or "hf_revision" in block):
                raise ValueError(f"{experiment} must not set image_keep on {dataset}")

    if experiment == "exp4":
        for dataset in HF_DATASETS:
            out_block = config["data"][dataset].get("train")
            base_block = base["data"][dataset].get("train")
            if isinstance(out_block, dict) and out_block != base_block:
                raise ValueError(f"{dataset} train split must stay dense and unchanged when kept")

    if spec.mode == "iid_images" and spec.population == "coralnet":
        base_list = base["data"]["coralnet"]["train"]["whitelist_sources"]
        out_block = config["data"]["coralnet"].get("train")
        if isinstance(out_block, dict) and list(out_block["whitelist_sources"]) != list(base_list):
            raise ValueError("an IID CoralNet drop must keep the full train source whitelist")


def _normalize_value(value: object) -> str:
    """String form of one key. Real numbers drop a trailing ``.0``; strings are kept as written."""
    if isinstance(value, bool | np.bool_):
        return str(value)
    if isinstance(value, int | np.integer):
        return str(int(value))
    if isinstance(value, float | np.floating):
        if not np.isfinite(value) or not np.isclose(value, round(float(value))):
            raise ValueError(f"image key {value!r} is not an integer")
        return str(int(round(float(value))))
    return str(value)


def _normalize_key(series: pd.Series) -> pd.Series:
    """String keys so ``109``, ``\"109\"`` and ``109.0`` name the same image."""
    if series.isna().any():
        raise ValueError("an image key column has missing values")
    return series.map(_normalize_value)


def _row_keys(frame: pd.DataFrame, keys: Sequence[str]) -> pd.Series:
    parts = [_normalize_key(frame[key]) for key in keys]
    key = parts[0]
    for part in parts[1:]:
        key = key.str.cat(part, sep="\0")
    return key


def filter_point_frames(
    df_annotations: pd.DataFrame,
    df_images: pd.DataFrame,
    keep: pd.DataFrame | str | Path,
    dataset: str,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Restrict point-annotation frames to an ``image_keep`` allowlist.

    Every keep key must already be in ``df_images`` (this split, after the dataset's own
    filters). Keys are the allowlist columns other than ``dataset``, and the ``dataset``
    column must be exactly ``dataset``. Images not listed are dropped.
    """
    if not isinstance(keep, pd.DataFrame):
        keep = pd.read_parquet(keep)
    if keep.empty:
        raise ValueError(f"image_keep for {dataset} is empty")
    if "dataset" not in keep.columns:
        raise ValueError("image_keep is missing a dataset column")
    names = set(keep["dataset"].astype(str))
    if names != {dataset}:
        raise ValueError(f"image_keep dataset column is {sorted(names)}, expected {dataset}")
    keys = [column for column in keep.columns if column != "dataset"]
    if not keys:
        raise ValueError("image_keep has no key columns")
    for frame_name, frame in (("annotations", df_annotations), ("images", df_images)):
        missing = [key for key in keys if key not in frame.columns]
        if missing:
            raise ValueError(f"image_keep columns {missing} are not in {dataset} {frame_name}")

    keep_keys = set(_row_keys(keep, keys))
    image_keys = _row_keys(df_images, keys)
    absent = keep_keys - set(image_keys)
    if absent:
        sample = sorted(absent)[:5]
        raise ValueError(
            f"image_keep for {dataset} has {len(absent)} keys that are not in this split, "
            f"e.g. {sample}"
        )
    images = df_images.loc[image_keys.isin(keep_keys)].reset_index(drop=True)
    ann_keys = _row_keys(df_annotations, keys)
    annotations = df_annotations.loc[ann_keys.isin(keep_keys)].reset_index(drop=True)
    if images.empty:
        raise ValueError(f"image_keep for {dataset} matched no images")
    return annotations, images


def apply_hf_image_keep(dataset: Any, path: str | Path, dataset_name: str, n_rows: int) -> Any:
    """Return ``dataset`` restricted to an allowlist, without rewriting masks.

    ``n_rows`` is the split length the allowlist was built from. A different length means the
    HF revision (or the site / original-split filters) drifted, so this raises instead of
    selecting look-alike indexes.
    """
    if int(n_rows) != len(dataset):
        raise ValueError(
            f"{dataset_name} has {len(dataset)} rows after its train filters, "
            f"but image_keep was built for {int(n_rows)}"
        )
    keep = pd.read_parquet(path)
    if keep.empty:
        raise ValueError(f"image_keep for {dataset_name} is empty")
    if "dataset" not in keep.columns:
        raise ValueError("image_keep is missing a dataset column")
    names = set(keep["dataset"].astype(str))
    if names != {dataset_name}:
        raise ValueError(f"image_keep dataset column is {sorted(names)}, expected {dataset_name}")
    keys = [column for column in keep.columns if column != "dataset"]
    if keys == ["row_index"]:
        indices = sorted({int(value) for value in keep["row_index"]})
        if any(index < 0 or index >= len(dataset) for index in indices):
            raise ValueError(f"image_keep row indexes for {dataset_name} fall outside the split")
        if not indices:
            raise ValueError(f"image_keep for {dataset_name} is empty")
        return dataset.select(indices)

    missing = [key for key in keys if key not in dataset.column_names]
    if missing:
        raise ValueError(f"{dataset_name} has no columns {missing} for image_keep")
    frame = dataset.remove_columns(
        [column for column in dataset.column_names if column not in keys]
    ).to_pandas()
    if frame.duplicated(subset=keys).any():
        raise ValueError(f"{dataset_name} image keys are not unique; refusing to subsample")
    keep_keys = set(_row_keys(keep, keys))
    row_keys = _row_keys(frame, keys)
    absent = keep_keys - set(row_keys)
    if absent:
        sample = sorted(absent)[:5]
        raise ValueError(
            f"image_keep for {dataset_name} has {len(absent)} keys that are not in this split, "
            f"e.g. {sample}"
        )
    indices = [index for index, key in enumerate(row_keys.tolist()) if key in keep_keys]
    if len(indices) != len(keep_keys):
        raise ValueError(f"image_keep for {dataset_name} did not match one row per key")
    return dataset.select(indices)
