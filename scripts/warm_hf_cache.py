#!/usr/bin/env python3
"""Download the pretrained encoder a model config names into HF_HOME.

Run it where the hub is reachable (a login node) so jobs on offline compute nodes find the
encoder in HF_HOME. HF_TOKEN, when set, is used for the download.

    python scripts/warm_hf_cache.py --model-config configs/model_config_cbm_dpt_lora_vitl.yaml
"""

from __future__ import annotations

import argparse
from pathlib import Path

import yaml
from huggingface_hub import snapshot_download


def main(argv: list[str] | None = None) -> int:
    """Snapshot-download the ``model.encoder_name`` repo of ``--model-config``."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-config", type=Path, required=True)
    args = parser.parse_args(argv)
    config = yaml.safe_load(args.model_config.read_text())
    encoder_name = config["model"]["encoder_name"]
    print(f"Downloading {encoder_name} into HF_HOME ...", flush=True)
    snapshot_download(repo_id=encoder_name)
    print(f"  done: {encoder_name}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
