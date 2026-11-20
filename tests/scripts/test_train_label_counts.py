"""train_label_counts: main counts the train splits of the data config and writes the parquet."""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pandas as pd
import pytest

import scripts.train_label_counts as script

_DATA_CONFIG = "configs/data_config_trainval.yaml"

_ANNOTATIONS = pd.DataFrame(
    {
        "image_id": ["i1"],
        "source_label_name": ["Acropora"],
    }
)


@pytest.fixture
def seen(monkeypatch: pytest.MonkeyPatch) -> dict[str, Any]:
    """Stub dataset building and cache setup; record how the script calls them."""
    seen: dict[str, Any] = {}
    dataset = SimpleNamespace(df_annotations=_ANNOTATIONS)

    def _build_datasets(data: Any, *, input_size: tuple[int, int], **kwargs: Any) -> dict:
        seen["build"] = kwargs
        seen["input_size"] = input_size
        return {("mermaid", "train"): dataset}

    def _setup_local_cache(data: Any) -> None:
        seen["cache_dir"] = data.get("local_cache_dir")

    monkeypatch.setattr(script, "build_datasets", _build_datasets)
    monkeypatch.setattr(script, "setup_local_cache", _setup_local_cache)
    return seen


def _run(monkeypatch: pytest.MonkeyPatch, out: Path, *extra: str) -> None:
    argv = ["train_label_counts.py", "--data-config", _DATA_CONFIG, "--output", str(out), *extra]
    monkeypatch.setattr("sys.argv", argv)
    script.main()


def test_main_writes_the_counts_frame_under_a_new_output_subdirectory(
    monkeypatch, seen, tmp_path, capsys
):
    out = tmp_path / "nested" / "counts.parquet"

    _run(monkeypatch, out)

    expected = pd.DataFrame(
        {
            "dataset": ["mermaid"],
            "image_id": ["i1"],
            "label": ["acropora"],
            "n": [1],
        }
    )
    pd.testing.assert_frame_equal(pd.read_parquet(out), expected, check_dtype=False)
    assert f"Wrote {out} (1 rows)" in capsys.readouterr().out


@pytest.mark.parametrize(
    ("extra", "names"),
    [([], None), (["--datasets", "mermaid"], ["mermaid"])],
)
def test_main_builds_only_the_train_split_of_the_requested_datasets(
    monkeypatch, seen, tmp_path, extra, names
):
    _run(monkeypatch, tmp_path / "counts.parquet", *extra)

    assert seen["build"]["splits"] == ["train"]
    assert seen["build"]["names"] == names


def test_main_sets_the_cache_dir_on_the_data_config_before_setting_up_the_cache(
    monkeypatch, seen, tmp_path
):
    _run(monkeypatch, tmp_path / "counts.parquet", "--cache-dir", "/tmp/x")

    assert seen["cache_dir"] == "/tmp/x"


def test_main_builds_datasets_with_a_two_dimensional_positive_input_size(
    monkeypatch, seen, tmp_path
):
    _run(monkeypatch, tmp_path / "counts.parquet")

    size = seen["input_size"]
    assert isinstance(size, tuple)
    assert len(size) == 2
    assert all(isinstance(side, int) and side > 0 for side in size)
