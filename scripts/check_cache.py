#!/usr/bin/env python3
"""Check that a local cache holds the dataset versions a data config pins, without syncing it.

Reads only the local cache: no S3 call, and no re-hash of images or tables. Each dataset is opened
by the same ``build_datasets`` call that training makes, so the check fails exactly when job
start-up would: a missing or corrupt marker, a pin mismatch, a missing table, a selected group that
is absent or unsynced, or a missing image or mask file. The pinned taxonomy must be in the cache too.
Prints one line per dataset and exits 1 if any check failed. A failed check names its cause and
the step that fixes it: usually ``scripts/sync_data.py``, or a fix to the data config.

The config's ``data:`` section is the one ``scripts/sync_data.py`` reads.

Run it with:

    python scripts/check_cache.py --data-config configs/my_data_config.yaml --cache-dir /scratch/cache
"""

from __future__ import annotations

import argparse
from pathlib import Path

from mermaidseg.data.config import DataSection
from mermaidseg.data.factory import build_datasets, load_benthic_hierarchy
from mermaidseg.io import REPO_ROOT, load_section
from segdata_spec.errors import SegdataError

# Failures of the cache or the config. Anything else is a bug and propagates as a traceback.
_CHECK_ERRORS = (SegdataError, OSError, ValueError)
_SYNC = "run scripts/sync_data.py with the same --data-config and --cache-dir"
# Padding and input size only shape transforms and masks, so any valid value gives the same check.
_PADDING = 0
_INPUT_SIZE = (1, 1)


def _check_dataset(section: DataSection, name: str, cache_dir: Path) -> str | None:
    """Return the failure of dataset ``name`` in the cache, or None if training can open it."""
    one = section.model_copy(update={"datasets": {name: section.datasets[name]}})
    try:
        build_datasets(
            one, cache_dir, repo_root=REPO_ROOT, padding=_PADDING, input_size=_INPUT_SIZE
        )
    except _CHECK_ERRORS as e:
        return _failure(e)
    return None


def _check_taxonomy(section: DataSection, cache_dir: Path) -> str | None:
    """Return the failure of the pinned taxonomy in the cache, or None if it is usable."""
    try:
        load_benthic_hierarchy(section, cache_dir)
    except _CHECK_ERRORS as e:
        return _failure(e)
    return None


def _failure(e: Exception) -> str:
    """Return the printed failure of ``e``, with the step that fixes it."""
    if isinstance(e, (SegdataError, FileNotFoundError)):
        hint = _SYNC
    elif isinstance(e, OSError):
        hint = "check that the cache files are readable"
    else:
        hint = f"check the data config; if it is right, {_SYNC}"
    return f"{type(e).__name__}: {e}; {hint}"


def _line(name: str, ok_detail: str, failure: str | None) -> tuple[str, bool]:
    """Return the printed line of one check and whether it passed."""
    if failure is None:
        return f"{name}: {ok_detail}, ok", True
    return f"{name}: {failure}", False


def check_cache(section: DataSection, cache_dir: Path) -> list[tuple[str, bool]]:
    """Check every pinned dataset and the taxonomy against the cache.

    Returns:
        One ``(line, passed)`` pair per check, to print as is.
    """
    results: list[tuple[str, bool]] = []
    for name, entry in section.datasets.items():
        resize = entry.resize.id if entry.resize else "original"
        failure = _check_dataset(section, name, cache_dir)
        results.append(_line(name, f"version {entry.version}, resize {resize}", failure))
    failure = _check_taxonomy(section, cache_dir)
    results.append(_line("taxonomy", f"version {section.taxonomy.version}", failure))
    return results


def main(argv: list[str] | None = None) -> int:
    """Parse arguments, check the cache and print one line per dataset."""
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument("--data-config", type=Path, required=True, help="YAML file with a data: section")
    p.add_argument("--cache-dir", type=Path, required=True, help="directory the versions are in")
    args = p.parse_args(argv)
    data_cfg = load_section(args.data_config, "data")
    section = DataSection.model_validate(data_cfg)
    results = check_cache(section, args.cache_dir)
    for line, _ in results:
        print(line)
    return 0 if all(passed for _, passed in results) else 1


if __name__ == "__main__":
    raise SystemExit(main())
