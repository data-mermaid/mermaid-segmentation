"""In-container entrypoint for a mermaid-segmentation TrainingJob.

Reads the run YAML from /opt/ml/input/data/config/, applies any top-level `env:` keys, then invokes
the existing training pipeline (scripts/train.py) with the seg-specific `config:` block translated
into CLI flags.

This is the **launcher-facing entrypoint** — the surface the launcher script names via
CONTAINER_ENTRYPOINT_SCRIPT. The training pipeline itself stays in mermaidseg/ and scripts/train.py;
this file is a thin adapter that bridges the run-YAML world to the existing CLI-driven entrypoint.
"""

from __future__ import annotations

import logging
import os
import subprocess
import sys
from pathlib import Path

import yaml

log = logging.getLogger("seg_train_entrypoint")
CONFIG_DIR = Path("/opt/ml/input/data/config")


def _configure_logging():
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
        stream=sys.stdout,
    )


def _find_run_yaml() -> Path:
    yamls = sorted(CONFIG_DIR.rglob("*.yaml")) + sorted(CONFIG_DIR.rglob("*.yml"))
    candidates = [p for p in yamls if "job:" in p.read_text()[:200]]
    if not candidates:
        raise SystemExit(f"No run YAML with a job: block under {CONFIG_DIR}")
    if len(candidates) > 1:
        raise SystemExit(f"Multiple candidate run YAMLs under {CONFIG_DIR}: {candidates}")
    return candidates[0]


def main():
    _configure_logging()
    run_yaml = _find_run_yaml()
    log.info("Loaded run YAML: %s", run_yaml)
    data = yaml.safe_load(run_yaml.read_text())

    # Apply env: from job block (launcher already injected these into
    # the container env, but if a user lands here via `docker run`
    # directly the YAML is the source of truth.)
    for k, v in (data.get("job", {}).get("env") or {}).items():
        os.environ.setdefault(k, str(v))

    # The seg-specific `config:` block names split-config paths (same as
    # scripts/train.py) plus optional CLI overrides as a flat dict.
    seg = data.get("config")
    if seg is None:
        raise SystemExit("Run YAML missing top-level `config:` block.")

    config_flags = {
        "config_data": seg.get("config_data", "configs/data_config.yaml"),
        "config_model": seg.get("config_model", "configs/model_config_cbm.yaml"),
        "config_training": seg.get("config_training", "configs/training_config_cbm.yaml"),
        "config_logger": seg.get("config_logger", "configs/logger_config.yaml"),
    }
    overrides = seg.get("overrides", {})

    if os.getenv("SMOKE_TEST"):
        log.info("Smoke test OK — skipping train.py invocation")
        return

    cmd = [sys.executable, "-u", "scripts/train.py"]
    for flag, path in config_flags.items():
        cmd += [f"--{flag.replace('_', '-')}", path]
    for k, v in overrides.items():
        cmd += [f"--{k}", str(v)]

    log.info("Invoking: %s", " ".join(cmd))
    rc = subprocess.call(cmd)
    sys.exit(rc)


if __name__ == "__main__":
    main()
