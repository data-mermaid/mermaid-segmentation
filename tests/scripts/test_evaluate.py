"""Evaluation entrypoint: the parser, and one call into ``run_evaluations``."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

import scripts.evaluate as evaluate_script
from mermaidseg.evaluation.coralscapes_probe import CORALSCAPES39, load_probes
from mermaidseg.evaluation.run import EvalOptions

_REQUIRED = ["--bundle", "b", "--output-dir", "out", "--cache-dir", "c"]


def _parse_defaults() -> Any:
    return evaluate_script.build_parser().parse_args(_REQUIRED)


def test_default_probe_points_file_loads():
    path = Path(_parse_defaults().probe_points)

    probe = json.loads(path.read_text())

    assert probe["num_classes"] == 95
    assert probe["images"]


def test_default_coralscapes39_probe_file_loads():
    path = _parse_defaults().coralscapes39_probe

    probes = load_probes([path])

    assert probes.label_space is CORALSCAPES39
    assert 50 in probes.by_k


def _stub_run_evaluations(monkeypatch: pytest.MonkeyPatch, summary: dict) -> list[EvalOptions]:
    """Replace ``run_evaluations`` with one returning ``summary``; return the options it got."""
    seen: list[EvalOptions] = []

    def _run(options: EvalOptions) -> dict:
        seen.append(options)
        return summary

    monkeypatch.setattr(evaluate_script, "run_evaluations", _run)
    return seen


def test_main_hands_the_parsed_options_to_run_evaluations(monkeypatch):
    seen = _stub_run_evaluations(monkeypatch, {})

    evaluate_script.main([*_REQUIRED, "--evals", "pacific", "--split", "test", "--amp"])

    (options,) = seen
    assert (options.bundle, options.output_dir, options.cache_dir) == (
        Path("b"),
        Path("out"),
        Path("c"),
    )
    assert (options.evals, options.split, options.amp) == (["pacific"], "test", True)


def test_main_strict_exits_1_when_a_section_failed(monkeypatch):
    _stub_run_evaluations(monkeypatch, {"val_loss": {"error": "RuntimeError: boom"}})

    with pytest.raises(SystemExit) as exit_info:
        evaluate_script.main([*_REQUIRED, "--strict"])

    assert exit_info.value.code == 1


def test_main_without_strict_returns_when_a_section_failed(monkeypatch):
    _stub_run_evaluations(monkeypatch, {"val_loss": {"error": "RuntimeError: boom"}})

    evaluate_script.main(_REQUIRED)


def test_main_strict_returns_when_every_section_succeeds(monkeypatch):
    _stub_run_evaluations(monkeypatch, {"val_loss": {"overall": {"loss": 1.0}}})

    evaluate_script.main([*_REQUIRED, "--strict"])
