#!/usr/bin/env python3
"""Build a named data-quality training profile: degraded training data plus its data config.

A profile applies one or both degradations to the training data of ``--base-config``:

    sparse  Coralscapes V2 train and UCSD mosaics keep 25 random points per image as 7x7
            squares (scripts/build_sparse_dense_datasets.py).
    phylum  CoralNet training sources flagged by ``--quantiles`` keep only phylum-level
            labels (scripts/build_coralnet_phylum_manifest.py).

Its name lists the degradations it applies, with the quantiles in ``METRIC_KEYS`` order
formatted with ``%g``, so one setup always maps to one name:

    sparse                                    -> sparse
    phylum --quantiles 0 0 0 0 0 0.1          -> phylum_0_0_0_0_0_0.1
    sparse phylum --quantiles 0 0 0 0 0 0.1   -> sparse_phylum_0_0_0_0_0_0.1

The degraded data are components shared between profiles: an existing component is reused,
a missing one is built (one job at a time per component, under a lock file). To rebuild a
component, delete its directory.

    <dq-root>/
      components/
        coralscapes_v2_sparse25_pad3_seed0/
        ucsd_mosaics_sparse25_pad3_seed0/
        coralnet_phylum_<quantiles>/
      profiles/<name>/
        data_config.yaml   --base-config with the degraded training splits pointed at components
        profile.json       degradations, quantiles, components, base config, git commit

A phylum profile drops ``coralnet.train.annotation_overrides``. Those points are already in the
phylum manifest, and applying the overrides again would replace the coarse labels.

The profile is written next to its final directory, checked by building the degraded train
splits from its data config, and only then moved into place. Train on it with
``--data-config <dq-root>/profiles/<name>/data_config.yaml``.

Usage:
    python scripts/build_data_quality_profile.py sparse --num-proc 32
    python scripts/build_data_quality_profile.py sparse phylum --quantiles 0 0 0 0 0 0.1
"""

from __future__ import annotations

import argparse
import copy
import fcntl
import json
import logging
import shutil
import subprocess
import sys
from collections.abc import Callable, Iterator, Sequence
from contextlib import contextmanager
from pathlib import Path

import yaml

_SCRIPTS_DIR = Path(__file__).resolve().parent
_REPO_ROOT = _SCRIPTS_DIR.parent
if str(_SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(_SCRIPTS_DIR))

from build_coralnet_phylum_manifest import MANIFEST_FILE  # noqa: E402
from coralnet_quantile_exploration import METRIC_KEYS  # noqa: E402

logger = logging.getLogger("build_data_quality_profile")

DEFAULT_DQ_ROOT = Path("/data/vision/beery/scratch/sauder/mermaid-segmentation/data_quality")
DEFAULT_BASE_CONFIG = _REPO_ROOT / "configs" / "data_config_dedup.yaml"
COMPONENTS_DIR = "components"
PROFILES_DIR = "profiles"
DATA_CONFIG_FILE = "data_config.yaml"
PROFILE_FILE = "profile.json"

SPARSE = "sparse"
PHYLUM = "phylum"
DEGRADATIONS: tuple[str, ...] = (SPARSE, PHYLUM)
SPARSE_DATASETS: tuple[str, ...] = ("coralscapes_v2", "ucsd_mosaics")
SPARSE_N_POINTS = 25
# training.padding, so a sparse point has the footprint of a CoralNet point.
PADDING = 3
SPARSE_SEED = 0


def format_quantile(q: float) -> str:
    return f"{q:g}"


def profile_name(sparse: bool, quantiles: Sequence[float] | None) -> str:
    """Name of the profile applying ``sparse`` and, unless ``quantiles`` is None, ``phylum``."""
    parts = [SPARSE] if sparse else []
    if quantiles is not None:
        if len(quantiles) != len(METRIC_KEYS):
            raise ValueError(f"expected {len(METRIC_KEYS)} quantiles, got {len(quantiles)}")
        if not all(0.0 <= q <= 1.0 for q in quantiles):
            raise ValueError(f"quantiles must be in [0, 1], got {list(quantiles)}")
        if not any(q > 0 for q in quantiles):
            raise ValueError("at least one quantile must be > 0, or no source is flagged")
        inexact = [q for q in quantiles if float(format_quantile(q)) != q]
        if inexact:
            raise ValueError(f"quantiles {inexact} need more than 6 significant digits")
        parts.append("_".join([PHYLUM, *map(format_quantile, quantiles)]))
    if not parts:
        raise ValueError(f"a profile needs at least one of {', '.join(DEGRADATIONS)}")
    return "_".join(parts)


def parse_profile_name(name: str) -> tuple[bool, tuple[float, ...] | None]:
    """``(sparse, quantiles)`` of a profile name; the inverse of :func:`profile_name`."""
    parts = name.split("_")
    sparse = parts[0] == SPARSE
    rest = parts[1:] if sparse else parts
    quantiles = None
    try:
        if rest:
            if rest[0] != PHYLUM:
                raise ValueError
            quantiles = tuple(float(token) for token in rest[1:])
        canonical = profile_name(sparse, quantiles)
    except ValueError:
        canonical = None
    if canonical != name:
        raise ValueError(f"{name!r} is not a data-quality profile name")
    return sparse, quantiles


def sparse_component_dir(dq_root: Path, dataset: str) -> Path:
    name = f"{dataset}_sparse{SPARSE_N_POINTS}_pad{PADDING}_seed{SPARSE_SEED}"
    return dq_root / COMPONENTS_DIR / name


def phylum_component_dir(dq_root: Path, quantiles: Sequence[float]) -> Path:
    return dq_root / COMPONENTS_DIR / f"coralnet_{profile_name(False, quantiles)}"


def profile_dir(dq_root: Path, name: str) -> Path:
    return dq_root / PROFILES_DIR / name


def component_dirs(dq_root: Path, sparse: bool, quantiles: Sequence[float] | None) -> list[Path]:
    dirs = [sparse_component_dir(dq_root, d) for d in SPARSE_DATASETS] if sparse else []
    if quantiles is not None:
        dirs.append(phylum_component_dir(dq_root, quantiles))
    return dirs


def overrides(
    dq_root: Path, sparse: bool, quantiles: Sequence[float] | None
) -> dict[tuple[str, ...], str]:
    """Data-config keys a profile changes, each pointing a training split at its component."""
    out: dict[tuple[str, ...], str] = {}
    if sparse:
        for dataset in SPARSE_DATASETS:
            out[("data", dataset, "train", "hf_repo_id")] = str(
                sparse_component_dir(dq_root, dataset)
            )
    if quantiles is not None:
        manifest = phylum_component_dir(dq_root, quantiles) / MANIFEST_FILE
        out[("data", "coralnet", "train", "manifest_path")] = str(manifest)
    return out


def profile_config(base: dict, changes: dict[tuple[str, ...], str]) -> dict:
    """``base`` with every key in ``changes`` set; each key's section must already exist."""
    config = copy.deepcopy(base)
    for path, value in changes.items():
        section = config
        for depth, key in enumerate(path[:-1], start=1):
            section = section.get(key)
            if not isinstance(section, dict):
                raise ValueError(f"base config has no section {'.'.join(path[:depth])}")
        section[path[-1]] = value
    # The phylum manifest is built from the post-override points. Leaving
    # annotation_overrides on the train split would replace those coarse labels
    # with the fine dedup points when the dataset loads.
    if ("data", "coralnet", "train", "manifest_path") in changes:
        config["data"]["coralnet"]["train"].pop("annotation_overrides", None)
    return config


def build_sparse_component(out_dir: Path, dataset: str, num_proc: int) -> None:
    import build_sparse_dense_datasets as builder

    argv = [
        f"--dataset={dataset}",
        f"--output-dir={out_dir}",
        f"--n-points={SPARSE_N_POINTS}",
        f"--padding={PADDING}",
        f"--seed={SPARSE_SEED}",
        f"--num-proc={num_proc}",
    ]
    builder.build(builder._parse_args(argv))


def build_phylum_component(
    out_dir: Path, quantiles: Sequence[float], base_config: Path, stats_csv: Path | None
) -> None:
    import build_coralnet_phylum_manifest as builder

    argv = [
        "--quantiles",
        *map(format_quantile, quantiles),
        f"--output-dir={out_dir}",
        f"--data-config={base_config}",
    ]
    if stats_csv is not None:
        argv.append(f"--stats-csv={stats_csv}")
    builder.build(builder._parse_args(argv))


def verify_profile(data_config: Path, names: Sequence[str]) -> None:
    """Build the train splits of ``names`` from ``data_config`` and read one item of each."""
    from mermaidseg.datasets import build_datasets, setup_local_cache
    from mermaidseg.io import setup_config

    cfg = setup_config({"data": str(data_config)})
    setup_local_cache(cfg.data)
    datasets = build_datasets(cfg.data, padding=PADDING, names=list(names), splits=["train"])
    missing = sorted(set(names) - {name for name, _ in datasets})
    if missing:
        raise RuntimeError(f"{data_config} builds no train split for {missing}")
    for (name, split), ds in datasets.items():
        _, mask = ds[0]
        logger.info(
            "verified %s/%s: %d items; item 0 has %d labelled pixels",
            name,
            split,
            len(ds),
            int((mask > 0).sum()),
        )


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


def _lock_path(directory: Path) -> Path:
    return directory.with_name(f".{directory.name}.lock")


def ensure_component(out_dir: Path, build: Callable[..., None], *build_args: object) -> bool:
    """Run ``build(out_dir, *build_args)`` unless ``out_dir`` exists; True if it was built."""
    with _locked(_lock_path(out_dir)):
        if out_dir.is_dir():
            logger.info("reusing %s", out_dir)
            return False
        logger.info("building %s", out_dir)
        build(out_dir, *build_args)
        if not out_dir.is_dir():
            raise RuntimeError(f"building {out_dir} did not create it")
        return True


def _git_commit() -> str | None:
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"], cwd=_REPO_ROOT, capture_output=True, text=True, check=True
        )
    except (OSError, subprocess.CalledProcessError):
        return None
    return result.stdout.strip()


class _ConfigDumper(yaml.SafeDumper):
    """Block mappings; lists of scalars inline, as in the hand-written configs."""


def _represent_list(dumper: yaml.SafeDumper, data: list) -> yaml.Node:
    flow = not any(isinstance(item, (dict, list)) for item in data)
    return dumper.represent_sequence("tag:yaml.org,2002:seq", data, flow_style=flow)


_ConfigDumper.add_representer(list, _represent_list)


def _write_profile(
    out_dir: Path,
    name: str,
    config: dict,
    info: dict,
    changes: dict[tuple[str, ...], str],
    base_config: Path,
) -> None:
    header = [
        f"# Data-quality profile {name}, written by scripts/build_data_quality_profile.py",
        f"# (see {PROFILE_FILE}): {base_config} with these keys changed:",
        *(f"#   {'.'.join(path)}" for path in changes),
    ]
    if ("data", "coralnet", "train", "manifest_path") in changes:
        header.append(
            "# data.coralnet.train.annotation_overrides is omitted; those points are already "
            "in the phylum manifest."
        )
    body = yaml.dump(config, Dumper=_ConfigDumper, sort_keys=False, width=100)
    (out_dir / DATA_CONFIG_FILE).write_text("\n".join(header) + "\n" + body)
    (out_dir / PROFILE_FILE).write_text(json.dumps(info, indent=2))


def build(args: argparse.Namespace) -> Path:
    sparse = SPARSE in args.degradations
    quantiles = tuple(args.quantiles) if PHYLUM in args.degradations else None
    name = profile_name(sparse, quantiles)
    dq_root: Path = args.dq_root.resolve()
    base_config: Path = args.base_config.resolve()
    out_dir = profile_dir(dq_root, name)
    if out_dir.exists() and not args.overwrite:
        raise SystemExit(f"profile {name} exists at {out_dir}; pass --overwrite to replace it")

    changes = overrides(dq_root, sparse, quantiles)
    config = profile_config(yaml.safe_load(base_config.read_text()), changes)

    if sparse:
        for dataset in SPARSE_DATASETS:
            component = sparse_component_dir(dq_root, dataset)
            ensure_component(component, build_sparse_component, dataset, args.num_proc)
    if quantiles is not None:
        component = phylum_component_dir(dq_root, quantiles)
        ensure_component(component, build_phylum_component, quantiles, base_config, args.stats_csv)

    info = {
        "name": name,
        "sparse": sparse,
        "quantiles": (
            dict(zip(METRIC_KEYS, quantiles, strict=True)) if quantiles is not None else None
        ),
        "sparse_settings": (
            {"n_points": SPARSE_N_POINTS, "padding": PADDING, "seed": SPARSE_SEED}
            if sparse
            else None
        ),
        "components": [str(d) for d in component_dirs(dq_root, sparse, quantiles)],
        "overrides": {".".join(path): value for path, value in changes.items()},
        "base_config": str(base_config),
        "git_commit": _git_commit(),
    }
    with _locked(_lock_path(out_dir)):
        if out_dir.exists() and not args.overwrite:
            raise SystemExit(f"profile {name} exists at {out_dir}; pass --overwrite to replace it")
        partial = out_dir.with_name(f".{out_dir.name}.partial")
        if partial.exists():
            shutil.rmtree(partial)
        partial.mkdir(parents=True)
        try:
            _write_profile(partial, name, config, info, changes, base_config)
            if args.verify:
                verify_profile(partial / DATA_CONFIG_FILE, sorted({p[1] for p in changes}))
        except BaseException:
            shutil.rmtree(partial, ignore_errors=True)
            raise
        if out_dir.exists():
            shutil.rmtree(out_dir)
        partial.rename(out_dir)
    logger.info("wrote profile %s to %s", name, out_dir)
    return out_dir


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        "degradations",
        nargs="+",
        choices=DEGRADATIONS,
        metavar="DEGRADATION",
        help=f"One or both of: {' '.join(DEGRADATIONS)}.",
    )
    parser.add_argument(
        "--quantiles",
        type=float,
        nargs=len(METRIC_KEYS),
        default=None,
        metavar="Q",
        help=f"phylum only. Upper-tail fraction per metric, in order: {' '.join(METRIC_KEYS)}.",
    )
    parser.add_argument("--dq-root", type=Path, default=DEFAULT_DQ_ROOT)
    parser.add_argument("--base-config", type=Path, default=DEFAULT_BASE_CONFIG)
    parser.add_argument(
        "--stats-csv",
        type=Path,
        default=None,
        help="phylum only. Read per-source metrics from this CSV instead of the manifest.",
    )
    parser.add_argument("--num-proc", type=int, default=1, help="Processes for the sparse build.")
    parser.add_argument(
        "--overwrite", action="store_true", help="Replace the profile (components are reused)."
    )
    parser.add_argument(
        "--no-verify",
        dest="verify",
        action="store_false",
        help="Skip building the degraded train splits from the new data config.",
    )
    args = parser.parse_args(argv)
    if len(set(args.degradations)) != len(args.degradations):
        parser.error(f"degradations repeat: {' '.join(args.degradations)}")
    phylum = PHYLUM in args.degradations
    if phylum and args.quantiles is None:
        parser.error("phylum needs --quantiles")
    if not phylum and (args.quantiles is not None or args.stats_csv is not None):
        parser.error("--quantiles and --stats-csv only apply to phylum")
    try:
        profile_name(SPARSE in args.degradations, args.quantiles if phylum else None)
    except ValueError as e:
        parser.error(str(e))
    return args


def main(argv: list[str] | None = None) -> None:
    args = _parse_args(argv)
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s"
    )
    build(args)


if __name__ == "__main__":
    main()
