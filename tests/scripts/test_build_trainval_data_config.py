"""build_trainval_data_config: the committed trainval config is the base config, rebuilt."""

from __future__ import annotations

from pathlib import Path

from scripts.build_trainval_data_config import HEADER, build_final_data_config

_CONFIGS = Path(__file__).resolve().parents[2] / "configs"


def test_trainval_config_is_rebuilt_from_the_base_config():
    rebuilt = build_final_data_config((_CONFIGS / "data_config.yaml").read_text(), header=HEADER)
    assert rebuilt == (_CONFIGS / "data_config_trainval.yaml").read_text()
