#!/usr/bin/env python3
"""Build the sparse training profile: sparsified dense data plus its data config.

Coralscapes V2 train and UCSD mosaics keep 25 random points per image as 7x7 squares
(scripts/build_sparse_dense_datasets.py); every other dataset is unchanged.

    <sparse-root>/
      components/
        coralscapes_v2_sparse25_pad3_seed0/
        ucsd_mosaics_sparse25_pad3_seed0/
      profile/
        data_config.yaml   --base-config with the two training splits pointed at components
        profile.json       settings, components, base config, git commit

An existing component is reused, a missing one is built (one job at a time per
component, under a lock file). To rebuild a component, delete its directory. The
profile is written next to its final directory, checked by building the sparse train
splits from its data config, and only then moved into place. Train on it with
``--data-config <sparse-root>/profile/data_config.yaml``.

Usage:
    python scripts/build_sparse_profile.py --num-proc 32
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

logger = logging.getLogger("build_sparse_profile")

DEFAULT_SPARSE_ROOT = Path("/data/vision/beery/scratch/sauder/mermaid-segmentation/data_sparse")
DEFAULT_BASE_CONFIG = _REPO_ROOT / "configs" / "data_config.yaml"
COMPONENTS_DIR = "components"
PROFILE_DIR = "profile"
DATA_CONFIG_FILE = "data_config.yaml"
PROFILE_FILE = "profile.json"

SPARSE_DATASETS: tuple[str, ...] = ("coralscapes_v2", "ucsd_mosaics")
SPARSE_N_POINTS = 25
# training.padding, so a sparse point has the footprint of a CoralNet point.
PADDING = 3
SPARSE_SEED = 0


def sparse_component_dir(root: Path, dataset: str) -> Path:
    name = f"{dataset}_sparse{SPARSE_N_POINTS}_pad{PADDING}_seed{SPARSE_SEED}"
    return root / COMPONENTS_DIR / name


def profile_dir(root: Path) -> Path:
    return root / PROFILE_DIR


def overrides(root: Path) -> dict[tuple[str, ...], str]:
    """Data-config keys the profile changes, each pointing a training split at its component."""
    return {
        ("data", dataset, "train", "hf_repo_id"): str(sparse_component_dir(root, dataset))
        for dataset in SPARSE_DATASETS
    }


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
    out_dir: Path, config: dict, info: dict, changes: dict[tuple[str, ...], str], base_config: Path
) -> None:
    header = [
        "# Sparse profile, written by scripts/build_sparse_profile.py",
        f"# (see {PROFILE_FILE}): {base_config} with these keys changed:",
        *(f"#   {'.'.join(path)}" for path in changes),
    ]
    body = yaml.dump(config, Dumper=_ConfigDumper, sort_keys=False, width=100)
    (out_dir / DATA_CONFIG_FILE).write_text("\n".join(header) + "\n" + body)
    (out_dir / PROFILE_FILE).write_text(json.dumps(info, indent=2))


def build(args: argparse.Namespace) -> Path:
    root: Path = args.sparse_root.resolve()
    base_config: Path = args.base_config.resolve()
    out_dir = profile_dir(root)
    if out_dir.exists() and not args.overwrite:
        raise SystemExit(f"profile exists at {out_dir}; pass --overwrite to replace it")

    changes = overrides(root)
    config = profile_config(yaml.safe_load(base_config.read_text()), changes)
    for dataset in SPARSE_DATASETS:
        ensure_component(
            sparse_component_dir(root, dataset), build_sparse_component, dataset, args.num_proc
        )

    info = {
        "sparse_settings": {"n_points": SPARSE_N_POINTS, "padding": PADDING, "seed": SPARSE_SEED},
        "components": [str(sparse_component_dir(root, d)) for d in SPARSE_DATASETS],
        "overrides": {".".join(path): value for path, value in changes.items()},
        "base_config": str(base_config),
        "git_commit": _git_commit(),
    }
    with _locked(_lock_path(out_dir)):
        if out_dir.exists() and not args.overwrite:
            raise SystemExit(f"profile exists at {out_dir}; pass --overwrite to replace it")
        partial = out_dir.with_name(f".{out_dir.name}.partial")
        if partial.exists():
            shutil.rmtree(partial)
        partial.mkdir(parents=True)
        try:
            _write_profile(partial, config, info, changes, base_config)
            if args.verify:
                verify_profile(partial / DATA_CONFIG_FILE, SPARSE_DATASETS)
        except BaseException:
            shutil.rmtree(partial, ignore_errors=True)
            raise
        if out_dir.exists():
            shutil.rmtree(out_dir)
        partial.rename(out_dir)
    logger.info("wrote sparse profile to %s", out_dir)
    return out_dir


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--sparse-root", type=Path, default=DEFAULT_SPARSE_ROOT)
    parser.add_argument("--base-config", type=Path, default=DEFAULT_BASE_CONFIG)
    parser.add_argument("--num-proc", type=int, default=1, help="Processes for the sparse build.")
    parser.add_argument(
        "--overwrite", action="store_true", help="Replace the profile (components are reused)."
    )
    parser.add_argument(
        "--no-verify",
        dest="verify",
        action="store_false",
        help="Skip building the sparse train splits from the new data config.",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    args = _parse_args(argv)
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s"
    )
    build(args)


if __name__ == "__main__":
    main()
