"""Strided Coralscapes window geometry and bleached code table."""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest
import torch

from mermaidseg.datasets.coralscapes_v2.coralscapes_v2_dataset import CORALSCAPES_V2_ID2NAME
from mermaidseg.evaluation.coralscapes_probe import (
    average_window_logits,
    build_coralscapes_bleached_codes,
    coralscapes_windows,
    evaluate_coralscapes_probe,
)

_CSV = Path(__file__).resolve().parents[2] / "configs" / "class_to_concepts.csv"


def test_windows_and_overlap_average():
    windows = coralscapes_windows()
    assert [name for name, *_box in windows] == ["left", "center", "right"]
    assert windows[0][1:] == (0, 0, 1024, 1024)
    assert windows[1][1:] == (0, 512, 1024, 1536)
    assert windows[2][1:] == (0, 1024, 1024, 2048)

    # A constant map stays constant under bilinear upsampling, so the overlap
    # bands are the unweighted means of the windows that cover them.
    values = (1.0, 3.0, 5.0)
    logits = torch.stack([torch.full((1, 2, 2), value) for value in values], dim=0)
    boxes = [window[1:] for window in windows]
    averaged = average_window_logits(logits, boxes, 1024, 2048)
    assert averaged.shape == (1, 1024, 2048)
    assert torch.allclose(averaged[:, :, 0:512], torch.full((1, 1024, 512), 1.0))
    assert torch.allclose(averaged[:, :, 512:1024], torch.full((1, 1024, 512), 2.0))
    assert torch.allclose(averaged[:, :, 1024:1536], torch.full((1, 1024, 512), 4.0))
    assert torch.allclose(averaged[:, :, 1536:2048], torch.full((1, 1024, 512), 5.0))


def test_uncovered_pixel_raises():
    logits = torch.zeros(1, 1, 2, 2)
    with pytest.raises(RuntimeError, match="no window coverage"):
        average_window_logits(logits, [(0, 0, 4, 4)], height=8, width=8)


def test_non_concept_feature_kind_raises(tmp_path: Path):
    probe = tmp_path / "probe.json"
    probe.write_text('{"images": [], "num_images": 0, "total_points": 0, "points_per_class_target": 10}')
    with pytest.raises(RuntimeError, match="only supports feature kind"):
        evaluate_coralscapes_probe(
            predictor=None,  # type: ignore[arg-type]
            probe_json=probe,
            feature_kinds=["dpt"],
            coralscapes_root=None,
            output_dir=tmp_path / "out",
            concept_names=["bleached"],
            taxonomy_csv=_CSV,
        )


def test_coralscapes_bleached_codes_match_the_concept_table():
    codes = build_coralscapes_bleached_codes(_CSV)
    assert codes.shape == (96,)
    assert int(codes[0]) == 0
    assert int((codes[1:] == 2).sum()) == 19
    assert int((codes[1:] == 1).sum()) == 76
    assert int((codes[1:] == 0).sum()) == 0
    by_name = {name: int(codes[class_id]) for class_id, name in CORALSCAPES_V2_ID2NAME.items()}
    assert by_name["acropora bleached"] == 2
    assert by_name["acropora alive"] == 1
    assert by_name["acropora dead"] == 1
    assert by_name["background"] == 1
    assert by_name["sand"] == 1


def _csv_with_sand_cell(tmp_path: Path, cell: str | None, drop_sand: bool = False) -> Path:
    frame = pd.read_csv(_CSV)
    name = frame["source_label_class_name"].astype(str).str.lower()
    source = frame["source_dataset_source"].astype(str).str.lower()
    sand = (source == "coralscapes_v2") & (name == "sand")
    if drop_sand:
        frame = frame.loc[~sand].copy()
    else:
        frame.loc[sand, "bleached"] = cell
    path = tmp_path / "class_to_concepts.csv"
    frame.to_csv(path, index=False)
    return path


def test_missing_coralscapes_class_is_not_treated_as_not_given(tmp_path: Path):
    path = _csv_with_sand_cell(tmp_path, cell=None, drop_sand=True)
    with pytest.raises(RuntimeError, match="sand"):
        build_coralscapes_bleached_codes(path)


def test_unexpected_bleached_cell_raises(tmp_path: Path):
    path = _csv_with_sand_cell(tmp_path, cell="maybe")
    with pytest.raises(RuntimeError, match="maybe"):
        build_coralscapes_bleached_codes(path)
