"""Lineage of a training run: code revision, environment lock, config, seed, data."""

from __future__ import annotations

import hashlib
import json
import subprocess
from collections.abc import Mapping
from importlib.metadata import version
from pathlib import Path
from typing import Any

from mermaidseg.io import REPO_ROOT

GIT_TIMEOUT_SEC = 10
UV_LOCK = "uv.lock"
VERSIONED_PACKAGES = ("torch", "transformers", "peft")


def _git_output(*args: str) -> str | None:
    """Stdout of ``git -C <repo> <args>``; None when git is missing, fails, or hangs."""
    try:
        result = subprocess.run(
            ["git", "-C", str(REPO_ROOT), *args],
            capture_output=True,
            text=True,
            check=True,
            timeout=GIT_TIMEOUT_SEC,
        )
    except (OSError, subprocess.SubprocessError):
        return None
    return result.stdout


def _git_sha() -> str | None:
    out = _git_output("rev-parse", "HEAD")
    return out.strip() if out is not None else None


def _git_dirty() -> bool | None:
    """Whether tracked files differ from HEAD; untracked files do not count."""
    out = _git_output("status", "--porcelain", "--untracked-files=no")
    return None if out is None else bool(out.strip())


def _uv_lock_sha256() -> str | None:
    lock = Path(REPO_ROOT) / UV_LOCK
    if not lock.is_file():
        return None
    return hashlib.sha256(lock.read_bytes()).hexdigest()


def data_version(cfg_data_plain: dict, datasets: Mapping[tuple[str, str], Any]) -> dict:
    """Hash of the data config plus the manifest path of each dataset that has one.

    Args:
        cfg_data_plain: The ``data`` config section as plain containers.
        datasets: Built datasets keyed by ``(name, split)``.

    Returns:
        ``{"data_config_sha1": ..., "manifests": {"<name>.<split>": manifest_path}}``.
    """
    config_json = json.dumps(cfg_data_plain, sort_keys=True, default=str)
    return {
        "data_config_sha1": hashlib.sha1(config_json.encode()).hexdigest(),
        "manifests": {
            f"{name}.{split}": ds.manifest_path
            for (name, split), ds in datasets.items()
            if hasattr(ds, "manifest_path")
        },
    }


def collect_lineage(*, resolved_config: dict, seed: int, data_version: dict) -> dict:
    """Everything needed to trace a run back to its code, environment, config and data.

    ``git_dirty`` counts tracked changes only; None (git unavailable) is as untrusted as True.
    """
    return {
        "git_sha": _git_sha(),
        "git_dirty": _git_dirty(),
        "uv_lock_sha256": _uv_lock_sha256(),
        "seed": seed,
        "data_version": data_version,
        "resolved_config": resolved_config,
        "versions": {name: version(name) for name in VERSIONED_PACKAGES},
    }
