#!/usr/bin/env python3
"""Build the training profiles for the four data-size experiments.

The population is the train split of ``--base-config`` (default
``configs/data_config.yaml``) after each dataset's own filters. Val and test are copied
through unchanged. One inventory is shared by every experiment, under ``--ds-root``:

    <ds-root>/
      components/
        inventory.parquet
        unit_stats.parquet
        revisions.json
        permutation_coralnet.parquet
        permutation_meta.parquet
        run.json
      profiles/<experiment>_drop<0|10|20|50>/
        data_config.yaml
        profile.json
        keep/<dataset>.parquet    # IID experiments only, and not the 0% cell

``exp1`` and ``exp3`` keep an image allowlist from one permutation (seed 0), so the 10% drop is
inside the 20% drop and that is inside the 50% drop. ``exp1`` also writes ``exp1_drop0``: every
CoralNet training image and no other dataset's train split, with the CoralNet train block left
as the base config wrote it. ``exp2`` and ``exp4`` drop the whole-unit prefix of the
``n_coral_genera`` ranking (most coral genera first) whose image count is closest to that
fraction. Experiment 4 counts each CoralNet source and each other dataset as one unit. Dense
datasets that remain point at the original HF repo.

The profile is checked by loading its train splits back and comparing them to the inventory
before it is published. Build with ``slurm/launch_build_profiles.sh`` and train with
``slurm/launch_data_experiments.sh``.

Usage:
    python scripts/build_data_size_profiles.py exp1
    python scripts/build_data_size_profiles.py exp2 exp4 --no-verify
"""

from __future__ import annotations

import argparse
import fcntl
import inspect
import json
import logging
import shutil
import subprocess
import sys
from collections.abc import Iterator, Mapping, Sequence
from contextlib import contextmanager
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import yaml

_SCRIPTS_DIR = Path(__file__).resolve().parent
_REPO_ROOT = _SCRIPTS_DIR.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from mermaidseg.datasets.data_size import (  # noqa: E402
    DATASET_ORDER,
    EXPERIMENTS,
    HF_DATASETS,
    POINT_KEYS,
    SEED,
    STATS_COLUMNS,
    UCSD_KEYS,
    DropPlan,
    config_for_plan,
    coralscapes_keys,
    dataset_unit_id,
    fractions_for,
    genus_counts,
    per_dataset_counts,
    permute_uids,
    plan_drop,
    profile_name,
    source_id_from_unit,
)

logger = logging.getLogger("build_data_size_profiles")

DEFAULT_DS_ROOT = Path("/data/vision/beery/scratch/sauder/mermaid-segmentation/data_size")
DEFAULT_BASE_CONFIG = _REPO_ROOT / "configs" / "data_config.yaml"
DEFAULT_CONCEPTS = _REPO_ROOT / "configs" / "class_to_concepts.csv"
COMPONENTS_DIR = "components"
PROFILES_DIR = "profiles"
DATA_CONFIG_FILE = "data_config.yaml"
PROFILE_FILE = "profile.json"
PADDING = 3

_INVENTORY = "inventory.parquet"
_STATS = "unit_stats.parquet"
_REVISIONS = "revisions.json"
_RUN = "run.json"
_PERM = {"coralnet": "permutation_coralnet.parquet", "meta": "permutation_meta.parquet"}


def _git_commit() -> str | None:
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=_REPO_ROOT,
            capture_output=True,
            text=True,
            check=True,
        )
    except (OSError, subprocess.CalledProcessError):
        return None
    return result.stdout.strip()


@contextmanager
def _locked(path: Path) -> Iterator[None]:
    """Hold an exclusive lock on ``path``; the kernel drops it if the job dies."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a") as handle:
        fcntl.flock(handle, fcntl.LOCK_EX)
        try:
            yield
        finally:
            fcntl.flock(handle, fcntl.LOCK_UN)


class _ConfigDumper(yaml.SafeDumper):
    """Block mappings; lists of scalars inline, as in the hand-written configs."""


def _represent_list(dumper: yaml.SafeDumper, data: list) -> yaml.Node:
    flow = not any(isinstance(item, (dict, list)) for item in data)
    return dumper.represent_sequence("tag:yaml.org,2002:seq", data, flow_style=flow)


_ConfigDumper.add_representer(list, _represent_list)


def _write_yaml(path: Path, name: str, config: dict, experiment: str, fraction: float) -> None:
    spec = EXPERIMENTS[experiment]
    header = [
        f"# Data-size profile {name}, written by scripts/build_data_size_profiles.py",
        (
            f"# {experiment}: drop {fraction:.0%} ({spec.mode}, population {spec.population}"
            + (f", ranked by {spec.metric}" if spec.metric else "")
            + ")."
        ),
        "# Val and test splits are unchanged. Eval uses configs/data_config.yaml.",
        "# Dense datasets that remain are not sparsified.",
    ]
    body = yaml.dump(config, Dumper=_ConfigDumper, sort_keys=False, width=100)
    path.write_text("\n".join(header) + "\n" + body)


def _train_cfg(base: dict, dataset: str) -> dict | None:
    section = base.get("data", {}).get(dataset)
    if not isinstance(section, dict):
        raise ValueError(f"base config has no data.{dataset} section")
    train = section.get("train")
    if train is None or train == "None":
        return None
    if not isinstance(train, dict):
        raise ValueError(f"data.{dataset}.train is {train!r}")
    return train


def _constructor_kwargs(train: dict, cls: type) -> dict[str, Any]:
    params = inspect.signature(cls.__init__).parameters
    kwargs: dict[str, Any] = {}
    for key, value in train.items():
        if key in ("transform", "augmentation"):
            continue
        if key not in params:
            raise ValueError(f"{cls.__name__} has no argument {key} (from the data config)")
        kwargs[key] = value
    return kwargs


def _hub_revision(repo_id: str) -> str | None:
    if Path(repo_id).expanduser().is_dir():
        return None
    from huggingface_hub import HfApi

    return HfApi().dataset_info(repo_id).sha


def _point_frame(dataset: str, ds: Any) -> pd.DataFrame:
    keys = list(POINT_KEYS[dataset])
    missing = [key for key in keys if key not in ds.df_images.columns]
    if missing:
        raise RuntimeError(f"{dataset} images are missing {missing}")
    frame = ds.df_images.loc[:, keys].reset_index(drop=True)
    if frame.duplicated().any():
        raise RuntimeError(f"{dataset} has duplicate training-image keys")
    frame.insert(0, "dataset", dataset)
    if dataset == "coralnet":
        from mermaidseg.datasets.data_size import coralnet_unit_id

        frame["unit_id"] = frame["source_id"].map(lambda source: coralnet_unit_id(int(source)))
    else:
        frame["unit_id"] = dataset_unit_id(dataset)
    return frame


def _point_labels(dataset: str, ds: Any) -> dict[str, set[str]]:
    annotations = ds.df_annotations
    labels = annotations["source_label_name"].astype(str)
    if dataset == "coralnet":
        from mermaidseg.datasets.data_size import coralnet_unit_id

        units = annotations["source_id"].map(lambda source: coralnet_unit_id(int(source)))
        grouped: dict[str, set[str]] = {}
        for unit, label in zip(units.tolist(), labels.tolist(), strict=True):
            grouped.setdefault(unit, set()).add(label)
        return grouped
    return {dataset_unit_id(dataset): set(labels.tolist())}


def _mask_labels(hf_dataset: Any, id2name: Mapping[int, str]) -> tuple[set[str], set[int]]:
    """Distinct label names in ``hf_dataset`` masks. Images are not decoded."""
    if "label" not in hf_dataset.column_names:
        raise RuntimeError("dense dataset has no label column")
    slim = hf_dataset.remove_columns(
        [column for column in hf_dataset.column_names if column != "label"]
    )
    found: set[str] = set()
    unknown: set[int] = set()
    for batch in slim.iter(batch_size=32):
        for mask in batch["label"]:
            for value in np.unique(np.asarray(mask)):
                index = int(value)
                if index == 0:
                    continue
                name = id2name.get(index)
                if name is None:
                    unknown.add(index)
                else:
                    found.add(str(name))
    return found, unknown


def _dense_frame(dataset: str, ds: Any, keys: Sequence[str]) -> pd.DataFrame:
    if keys == ("row_index",):
        frame = pd.DataFrame({"row_index": np.arange(len(ds), dtype=np.int64)})
    else:
        missing = [key for key in keys if key not in ds.dataset.column_names]
        if missing:
            raise RuntimeError(f"{dataset} is missing image-key columns {missing}")
        slim = ds.dataset.remove_columns(
            [column for column in ds.dataset.column_names if column not in keys]
        )
        frame = slim.to_pandas()
        if frame.duplicated(subset=list(keys)).any():
            raise RuntimeError(f"{dataset} image keys are not unique")
        if len(frame) != len(ds):
            raise RuntimeError(f"{dataset} key table has {len(frame)} rows, dataset has {len(ds)}")
    frame.insert(0, "dataset", dataset)
    frame["unit_id"] = dataset_unit_id(dataset)
    return frame


def _build_population(
    base: dict, concepts: pd.DataFrame
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    """Load every train split once and return inventory, unit stats, and HF pins."""
    from mermaidseg.datasets.coralscapes_v2.coralscapes_v2_dataset import (
        CORALSCAPES_V2_HF_REPO,
        CORALSCAPES_V2_ID2NAME,
    )
    from mermaidseg.datasets.factory import DATASET_CLASSES
    from mermaidseg.datasets.local_cache import setup_local_cache

    setup_local_cache(base["data"])
    frames: list[pd.DataFrame] = []
    label_sets: dict[str, set[str]] = {}
    extra_unmapped: dict[str, int] = {}
    revisions: dict[str, Any] = {}

    for dataset in DATASET_ORDER:
        train = _train_cfg(base, dataset)
        if train is None:
            logger.info("%s has no train split", dataset)
            continue
        cls = DATASET_CLASSES[dataset]
        kwargs = _constructor_kwargs(train, cls)
        if dataset in HF_DATASETS:
            default_repo = getattr(cls, "HF_REPO_ID", CORALSCAPES_V2_HF_REPO)
            repo = str(kwargs.get("hf_repo_id") or default_repo)
            revision = _hub_revision(repo)
            if revision:
                kwargs["hf_revision"] = revision
            logger.info("loading %s (%s @ %s)", dataset, repo, revision or "local")
            ds = cls(**kwargs)
            if dataset == "ucsd_mosaics":
                keys: tuple[str, ...] = UCSD_KEYS
                id2name = {int(entry["id"]): str(entry["name"]) for entry in ds.class_table}
            else:
                keys = coralscapes_keys(ds.dataset.column_names)
                id2name = CORALSCAPES_V2_ID2NAME
            logger.info("%s keep keys %s, %d images", dataset, ",".join(keys), len(ds))
            frame = _dense_frame(dataset, ds, keys)
            logger.info("scanning %s masks for annotated labels", dataset)
            names, unknown = _mask_labels(ds.dataset, id2name)
            unit = dataset_unit_id(dataset)
            label_sets[unit] = names
            extra_unmapped[unit] = len(unknown)
            if unknown:
                logger.warning("%s has mask ids with no class name: %s", dataset, sorted(unknown))
            revisions[dataset] = {
                "keys": list(keys),
                "revision": revision,
                "n_rows": int(len(ds)),
                "repo": repo,
            }
        else:
            logger.info("loading %s", dataset)
            ds = cls(**kwargs, padding=PADDING)
            frame = _point_frame(dataset, ds)
            label_sets.update(_point_labels(dataset, ds))
            logger.info("%s: %d images", dataset, len(frame))
        frames.append(frame)
        del ds

    if not frames:
        raise RuntimeError("base config has no training images")
    inventory = pd.concat(frames, ignore_index=True)
    inventory.insert(0, "image_uid", np.arange(len(inventory), dtype=np.int64))
    if inventory["image_uid"].duplicated().any() or inventory.duplicated("image_uid").any():
        raise RuntimeError("failed to assign unique image uids")

    rows: list[dict[str, Any]] = []
    for dataset in DATASET_ORDER:
        sub = inventory.loc[inventory["dataset"] == dataset]
        if sub.empty:
            continue
        for unit_id, group in sub.groupby("unit_id", sort=False):
            counts = genus_counts(label_sets.get(str(unit_id), ()), concepts, dataset)
            rows.append(
                {
                    "unit_id": str(unit_id),
                    "dataset": dataset,
                    "n_images": int(len(group)),
                    "n_genera": counts.n_genera,
                    "n_coral_genera": counts.n_coral_genera,
                    "n_labels": counts.n_labels,
                    "n_unmapped_labels": counts.n_unmapped_labels
                    + extra_unmapped.get(str(unit_id), 0),
                }
            )
    stats = pd.DataFrame(rows, columns=list(STATS_COLUMNS))
    if stats.empty:
        raise RuntimeError("unit stats are empty")
    return inventory, stats, revisions


def _population_uids(inventory: pd.DataFrame, population: str) -> list[int]:
    if population == "coralnet":
        frame = inventory.loc[inventory["dataset"] == "coralnet"]
    elif population == "meta":
        frame = inventory
    else:
        raise ValueError(population)
    return [int(uid) for uid in frame["image_uid"]]


def _read_permutation(path: Path, uids: Sequence[int]) -> np.ndarray:
    perm = pd.read_parquet(path)["image_uid"].to_numpy(dtype=np.int64)
    if sorted(int(uid) for uid in perm) != sorted(uids):
        raise RuntimeError(f"{path} does not match the inventory; pass --overwrite-inventory")
    return perm


def _write_permutation(path: Path, uids: Sequence[int]) -> np.ndarray:
    perm = permute_uids(uids, SEED)
    path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame({"image_uid": perm}).to_parquet(path, index=False)
    return perm


def ensure_components(
    ds_root: Path,
    base: dict,
    concepts: pd.DataFrame,
    *,
    base_config: Path,
    overwrite_inventory: bool,
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, Any], dict[str, np.ndarray]]:
    """Return the shared inventory, stats, HF pins, and the two saved permutations."""
    comp = ds_root / COMPONENTS_DIR
    required = [comp / _INVENTORY, comp / _STATS, comp / _REVISIONS, comp / _RUN]
    with _locked(comp.with_name(".components.lock")):
        if overwrite_inventory or not all(path.is_file() for path in required):
            logger.info("building training inventory under %s", comp)
            inventory, stats, revisions = _build_population(base, concepts)
            comp.mkdir(parents=True, exist_ok=True)
            inventory.to_parquet(comp / _INVENTORY, index=False)
            stats.to_parquet(comp / _STATS, index=False)
            (comp / _REVISIONS).write_text(json.dumps(revisions, indent=2) + "\n")
            run = {
                "seed": SEED,
                "n_images": int(len(inventory)),
                "n_units": int(stats["unit_id"].nunique()),
                "base_config": str(base_config),
                "git_commit": _git_commit(),
            }
            (comp / _RUN).write_text(json.dumps(run, indent=2) + "\n")
            for path in (comp / name for name in _PERM.values()):
                path.unlink(missing_ok=True)
        else:
            logger.info("reusing %s", comp)
            inventory = pd.read_parquet(comp / _INVENTORY)
            stats = pd.read_parquet(comp / _STATS)
            revisions = json.loads((comp / _REVISIONS).read_text())

        permutations: dict[str, np.ndarray] = {}
        for population, filename in _PERM.items():
            path = comp / filename
            uids = _population_uids(inventory, population)
            if not uids:
                raise RuntimeError(f"{population} population has no images")
            if path.is_file():
                permutations[population] = _read_permutation(path, uids)
            else:
                logger.info(
                    "writing %s permutation (%d images, seed %d)", population, len(uids), SEED
                )
                permutations[population] = _write_permutation(path, uids)
    return inventory, stats, revisions, permutations


def _key_columns(revisions: Mapping[str, Any]) -> dict[str, tuple[str, ...]]:
    keys: dict[str, tuple[str, ...]] = dict(POINT_KEYS)
    keys["ucsd_mosaics"] = UCSD_KEYS
    coralscapes = revisions.get("coralscapes_v2")
    keys["coralscapes_v2"] = tuple(coralscapes["keys"]) if coralscapes else ("row_index",)
    return keys


def _hf_pins(revisions: Mapping[str, Any]) -> dict[str, dict[str, Any]]:
    return {
        dataset: {"revision": info.get("revision"), "n_rows": int(info["n_rows"])}
        for dataset, info in revisions.items()
    }


def _profile_record(
    plan: DropPlan,
    per_dataset: Mapping[str, Mapping[str, int]],
    *,
    base_config: Path,
) -> dict[str, Any]:
    spec = EXPERIMENTS[plan.experiment]
    return {
        "name": profile_name(plan.experiment, plan.fraction),
        "experiment": plan.experiment,
        "fraction": plan.fraction,
        "seed": SEED,
        "metric": spec.metric,
        "population": spec.population,
        "mode": spec.mode,
        "n_images_total": plan.n_images_total,
        "n_images_dropped": plan.n_images_dropped,
        "n_images_kept": plan.n_images_kept,
        "fraction_images_achieved": plan.fraction_images_achieved,
        "n_units_total": plan.n_units_total,
        "n_units_dropped": plan.n_units_dropped,
        "fraction_units_achieved": plan.fraction_units_achieved,
        "dropped_units": list(plan.dropped_units),
        "per_dataset": {dataset: dict(counts) for dataset, counts in per_dataset.items()},
        "base_config": str(base_config),
        "git_commit": _git_commit(),
    }


def _source_counts(frame: pd.DataFrame) -> pd.Series:
    sources = pd.to_numeric(frame["source_id"]).round().astype(int)
    return sources.groupby(sources).size()


def verify_profile(
    data_config: Path,
    info: Mapping[str, Any],
    inventory: pd.DataFrame,
    base: dict,
    key_columns: Mapping[str, Sequence[str]],
) -> None:
    """Load the profile's train splits and require them to match the recorded keep counts."""
    from mermaidseg.datasets.data_size import _row_keys
    from mermaidseg.datasets.factory import build_datasets
    from mermaidseg.datasets.local_cache import setup_local_cache
    from mermaidseg.io import setup_config

    cfg = setup_config({"data": str(data_config)})
    setup_local_cache(cfg.data)
    expected: Mapping[str, Mapping[str, int]] = info["per_dataset"]
    names = [name for name, counts in expected.items() if int(counts["n_kept"]) > 0]
    datasets = build_datasets(cfg.data, padding=PADDING, names=names, splits=["train"])
    built = {name for name, _split in datasets}
    if built != set(names):
        raise RuntimeError(
            f"{data_config} built {sorted(built)}, expected train splits {sorted(names)}"
        )

    written = yaml.safe_load(data_config.read_text())
    held_out = {
        int(source)
        for split in ("val", "test")
        for source in base["data"]["coralnet"][split]["whitelist_sources"]
    }
    experiment = str(info["experiment"])
    for (name, _split), ds in datasets.items():
        got = len(ds)
        want = int(expected[name]["n_kept"])
        if got != want:
            raise RuntimeError(f"{name} loaded {got} images, profile records {want}")
        _image, mask = ds[0]
        if mask is None:
            raise RuntimeError(f"{name} failed to load its first kept image")
        train = written["data"][name]["train"]
        if isinstance(train, dict) and train.get("image_keep") and hasattr(ds, "df_images"):
            keep = pd.read_parquet(train["image_keep"])
            keys = list(key_columns[name])
            if set(_row_keys(ds.df_images, keys)) != set(_row_keys(keep, keys)):
                raise RuntimeError(f"{name} loaded images do not match its image_keep file")

    coralnet = datasets.get(("coralnet", "train"))
    if coralnet is not None:
        loaded = _source_counts(coralnet.df_images)
        if set(loaded.index) & held_out:
            raise RuntimeError("loaded CoralNet train sources overlap val or test")
        if experiment in ("exp2", "exp4"):
            dropped = {
                source_id_from_unit(unit)
                for unit in info["dropped_units"]
                if str(unit).startswith("coralnet:")
            }
            inventory_counts = _source_counts(inventory.loc[inventory["dataset"] == "coralnet"])
            for source, n_images in inventory_counts.items():
                got = int(loaded[source]) if source in loaded.index else 0
                if source in dropped and got != 0:
                    raise RuntimeError(f"dropped CoralNet source {source} still has {got} images")
                if source not in dropped and got != int(n_images):
                    raise RuntimeError(
                        f"CoralNet source {source} loaded {got} images, inventory has {n_images}"
                    )
    logger.info("verified %s", info["name"])


def write_profile(
    ds_root: Path,
    experiment: str,
    fraction: float,
    inventory: pd.DataFrame,
    stats: pd.DataFrame,
    permutations: Mapping[str, np.ndarray],
    revisions: Mapping[str, Any],
    base: dict,
    *,
    base_config: Path,
    overwrite: bool,
    verify: bool,
) -> Path:
    """Write one profile directory. An existing profile is kept unless ``overwrite``."""
    name = profile_name(experiment, fraction)
    final = ds_root / PROFILES_DIR / name
    if (final / DATA_CONFIG_FILE).is_file() and (final / PROFILE_FILE).is_file() and not overwrite:
        logger.info("reusing profile %s", name)
        return final

    spec = EXPERIMENTS[experiment]
    permutation = permutations[spec.population] if spec.mode == "iid_images" else None
    plan = plan_drop(experiment, fraction, inventory, stats, permutation)
    counts = per_dataset_counts(plan, inventory)
    logger.info(
        "%s: drop %d/%d images (%.2f%%), %d/%d units (%.2f%%)",
        name,
        plan.n_images_dropped,
        plan.n_images_total,
        100 * plan.fraction_images_achieved,
        plan.n_units_dropped,
        plan.n_units_total,
        100 * plan.fraction_units_achieved,
    )
    keys = _key_columns(revisions)
    pins = _hf_pins(revisions)
    partial = final.with_name(f".{final.name}.partial")
    with _locked(final.with_name(f".{final.name}.lock")):
        if (final / DATA_CONFIG_FILE).is_file() and not overwrite:
            logger.info("reusing profile %s", name)
            return final
        if partial.exists():
            shutil.rmtree(partial)
        partial.mkdir(parents=True)
        try:
            _published, frames = config_for_plan(
                base, plan, inventory, keys, keep_dir=final / "keep", hf_pins=pins
            )
            verify_config, _frames = config_for_plan(
                base, plan, inventory, keys, keep_dir=partial / "keep", hf_pins=pins
            )
            if set(frames) != set(_frames):
                raise RuntimeError(f"{name} keep lists changed between the two config builds")
            for dataset, frame in frames.items():
                path = partial / "keep" / f"{dataset}.parquet"
                path.parent.mkdir(parents=True, exist_ok=True)
                frame.to_parquet(path, index=False)
            info = _profile_record(plan, counts, base_config=base_config)
            if verify:
                verify_yaml = partial / "verify_data_config.yaml"
                _write_yaml(verify_yaml, name, verify_config, experiment, fraction)
                verify_profile(verify_yaml, info, inventory, base, keys)
                verify_yaml.unlink()
            _write_yaml(partial / DATA_CONFIG_FILE, name, _published, experiment, fraction)
            (partial / PROFILE_FILE).write_text(json.dumps(info, indent=2) + "\n")
        except BaseException:
            shutil.rmtree(partial, ignore_errors=True)
            raise
        if final.exists():
            shutil.rmtree(final)
        partial.rename(final)
    logger.info("wrote profile %s to %s", name, final)
    return final


def build(args: argparse.Namespace) -> list[Path]:
    base_config = Path(args.base_config).resolve()
    base = yaml.safe_load(base_config.read_text())
    concepts = pd.read_csv(args.concepts, dtype=str, keep_default_na=False)
    inventory, stats, revisions, permutations = ensure_components(
        args.ds_root.resolve(),
        base,
        concepts,
        base_config=base_config,
        overwrite_inventory=args.overwrite_inventory,
    )
    written: list[Path] = []
    for experiment in args.experiments:
        for fraction in fractions_for(experiment):
            written.append(
                write_profile(
                    args.ds_root.resolve(),
                    experiment,
                    fraction,
                    inventory,
                    stats,
                    permutations,
                    revisions,
                    base,
                    base_config=base_config,
                    overwrite=args.overwrite,
                    verify=args.verify,
                )
            )
    return written


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        "experiments",
        nargs="+",
        choices=tuple(EXPERIMENTS),
        help="Which experiments to build. Each one writes drop10, drop20 and drop50. exp1 also writes drop0.",
    )
    parser.add_argument("--ds-root", type=Path, default=DEFAULT_DS_ROOT)
    parser.add_argument("--base-config", type=Path, default=DEFAULT_BASE_CONFIG)
    parser.add_argument("--concepts", type=Path, default=DEFAULT_CONCEPTS)
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Replace existing profiles. The shared inventory is still reused.",
    )
    parser.add_argument(
        "--overwrite-inventory",
        action="store_true",
        help="Reload every training image and rewrite components/, including the permutations.",
    )
    parser.add_argument(
        "--no-verify",
        dest="verify",
        action="store_false",
        help="Skip loading the new train splits back and comparing them to the inventory.",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s"
    )
    build(_parse_args(argv))


if __name__ == "__main__":
    main()
