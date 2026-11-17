"""Training entrypoint parser."""

from __future__ import annotations

from mermaidseg.run_setup import resolve_repo_path
from scripts.train import build_parser

CONFIG_FLAGS = (
    "training_config",
    "model_config",
    "data_config",
    "concept_schema_config",
    "logger_config",
)


def test_config_defaults_point_at_existing_repo_files():
    args = build_parser().parse_args([])
    for flag in CONFIG_FLAGS:
        assert resolve_repo_path(getattr(args, flag)).is_file(), flag
