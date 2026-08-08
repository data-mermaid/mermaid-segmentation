"""Tests for the CBM video demo concept expression DSL."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

_DEMO_DIR = Path(__file__).resolve().parents[1] / "demo"
if str(_DEMO_DIR) not in sys.path:
    sys.path.insert(0, str(_DEMO_DIR))

from concept_expr import (  # noqa: E402
    CLASSES_SENTINEL,
    ConceptExpressionError,
    ConceptResolver,
    evaluate,
    is_classes_sentinel,
    parse,
    tokenize,
)


@pytest.fixture
def concept_names() -> list[str]:
    return [
        "class__hexacorallia",
        "class__octocorallia",
        "genus__acropora",
        "live",
        "bleached",
        "branching",
        "tabular",
        "background",
        "dark",
        "transect",
    ]


@pytest.fixture
def concept_probs(concept_names: list[str]) -> np.ndarray:
    c = len(concept_names)
    base = np.linspace(0.1, 0.9, c, dtype=np.float32)
    return base[:, None, None] * np.ones((c, 2, 2), dtype=np.float32)


@pytest.fixture
def resolver(concept_names: list[str]) -> ConceptResolver:
    return ConceptResolver(concept_names)


def test_tokenize_rank_value_and_bare_atoms() -> None:
    tokens = tokenize("live * (class:hexacorallia + class:octocorallia)")
    values = [t.value for t in tokens]
    assert "live" in values
    assert "class:hexacorallia" in values
    assert "class:octocorallia" in values
    assert "(" in values
    assert ")" in values


def test_parse_to_rpn(concept_names: list[str]) -> None:
    rpn = parse("1 - genus:acropora")
    assert [t.value for t in rpn] == ["1", "genus:acropora", "-"]


def test_evaluate_bare_atom(concept_probs, resolver) -> None:
    out = evaluate("branching", concept_probs, resolver)
    expected = concept_probs[5]
    np.testing.assert_allclose(out, expected)


def test_evaluate_rank_value_atom(concept_probs, resolver) -> None:
    out = evaluate("class:hexacorallia", concept_probs, resolver)
    np.testing.assert_allclose(out, concept_probs[0])


def test_evaluate_multiply_and_add(concept_probs, resolver) -> None:
    out = evaluate(
        "live * (class:hexacorallia + class:octocorallia)",
        concept_probs,
        resolver,
    )
    expected = np.clip(
        concept_probs[3] * (concept_probs[0] + concept_probs[1]),
        0.0,
        1.0,
    )
    np.testing.assert_allclose(out, expected)


def test_evaluate_subtract(concept_probs, resolver) -> None:
    out = evaluate("branching * (1 - genus:acropora)", concept_probs, resolver)
    one = np.ones(concept_probs.shape[1:], dtype=np.float32)
    expected = np.clip(concept_probs[5] * (one - concept_probs[2]), 0.0, 1.0)
    np.testing.assert_allclose(out, expected)


def test_unknown_atom_raises(resolver, concept_probs) -> None:
    with pytest.raises(ConceptExpressionError, match="Unknown concept atom"):
        evaluate("genus:missing", concept_probs, resolver)


def test_classes_sentinel_helpers() -> None:
    assert is_classes_sentinel("@classes")
    assert not is_classes_sentinel("background")
    with pytest.raises(ConceptExpressionError, match="@classes"):
        evaluate(CLASSES_SENTINEL, np.zeros((1, 1, 1), dtype=np.float32), ConceptResolver(["live"]))


def test_resolver_rejects_classes_inside_expression(concept_names: list[str]) -> None:
    resolver = ConceptResolver(concept_names)
    with pytest.raises(ConceptExpressionError, match="sentinel"):
        resolver.resolve("@classes")


def test_mismatched_parentheses() -> None:
    with pytest.raises(ConceptExpressionError, match="parentheses"):
        parse("(live + branching")


def test_tile_starts_overlap() -> None:
    from inference import tile_starts

    starts = tile_starts(1000, 512, min_overlap=0.2)
    assert starts[0] == 0
    assert starts[-1] == 488
    stride = starts[1] - starts[0]
    overlap = (512 - stride) / 512
    assert overlap >= 0.2 - 1e-6


def test_tile_blend_weights_shape_and_center() -> None:
    from inference import tile_blend_weights

    w = tile_blend_weights(8, 6)
    assert w.shape == (8, 6)
    assert w.dtype == np.float32
    assert np.all(w > 0)
    center = w[3:5, 2:4].mean()
    assert center >= w[0, 0]
    assert center >= w[-1, -1]


def test_tile_blend_weights_linear_ramp_in_overlap() -> None:
    from inference import tile_blend_weights, tile_starts

    tile = 8
    length = 12
    starts = tile_starts(length, tile, min_overlap=0.2)
    assert len(starts) == 2

    blend = tile_blend_weights(tile, tile)
    row = blend[0]  # 1D weights along the width axis (single row is enough)

    acc = np.zeros(length, dtype=np.float32)
    weight = np.zeros(length, dtype=np.float32)
    preds = [0.0, 1.0]
    for start, value in zip(starts, preds):
        acc[start : start + tile] += value * row
        weight[start : start + tile] += row
    blended = acc / np.maximum(weight, 1e-6)

    overlap_lo = starts[1]
    overlap_hi = starts[0] + tile
    overlap = blended[overlap_lo:overlap_hi]
    assert overlap[0] < 0.5 < overlap[-1]
    assert np.all(np.diff(overlap) > 0)
    mid = (overlap_lo + overlap_hi) / 2.0 - 0.5
    lo_i = int(np.floor(mid))
    interp = blended[lo_i] + (mid - lo_i) * (blended[lo_i + 1] - blended[lo_i])
    assert abs(interp - 0.5) < 1e-6


def test_parse_processing_resolution() -> None:
    from inference import parse_processing_resolution

    assert parse_processing_resolution("1080x1920") == (1080, 1920)
    assert parse_processing_resolution("1080,1920") == (1080, 1920)


def test_render_inferno_mask_panel() -> None:
    from video_demo import render_inferno_mask_panel

    rgb = np.full((2, 2, 3), 200, dtype=np.uint8)
    values = np.array([[0.2, 0.8], [0.39, 0.41]], dtype=np.float32)
    out = render_inferno_mask_panel(rgb, values, threshold=0.4)
    assert out[0, 0, 1] == 200
    assert out[0, 1, 1] != 200
    assert out[1, 0, 1] == 200
    assert out[1, 1, 1] != 200


def test_banner_layout_scales_with_frame_height() -> None:
    from video_demo import _banner_layout

    banner_h, title_scale, subtitle_scale, title_y, subtitle_y, thickness = _banner_layout(1440)
    assert banner_h >= 120
    assert title_scale > 1.0
    assert subtitle_y > title_y
    assert thickness >= 2


def test_normalize_banner_text_replaces_unicode_math_symbols() -> None:
    from video_demo import _normalize_banner_text

    assert _normalize_banner_text("live \u00d7 hex") == "live x hex"
    assert _normalize_banner_text("1 \u2212 acropora") == "1 - acropora"


def test_load_schedule_and_active_row() -> None:
    from video_demo import active_schedule_row, load_schedule

    csv_path = _DEMO_DIR / "video_concepts_example.csv"
    rows = load_schedule(csv_path)
    assert rows[0].display_title == "Live Hard Coral"
    assert active_schedule_row(rows, 0.0).display_title == "Live Hard Coral"
    assert active_schedule_row(rows, 3.0).display_title == "Live Hard Coral"
    assert active_schedule_row(rows, 4.5).display_title == "Bleached Hard Coral"
    assert active_schedule_row(rows, 22.0).expression == "phylum:chordata"
