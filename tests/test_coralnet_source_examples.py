"""Layout of the one-row CoralNet source figures. No cache or cluster access."""

from __future__ import annotations

import numpy as np
from PIL import Image

from scripts import make_appendix_figures as figs
from scripts.make_coralnet_source_examples import (
    FIG_WIDTH_IN,
    LEGEND_WIDTH_IN,
    N_IMAGES,
    POINT_SCALE,
    REQUIRED_SOURCE_IDS,
    choose_sources,
    compose_source_figure,
    fitted_size,
    pdf_media_box,
    row_layout,
    sample_order,
    save_fullres,
)


def test_point_scale_is_twice_the_appendix():
    assert POINT_SCALE == 2 * figs.DEFAULT_POINT_SCALE


def test_required_sources_are_kept_and_choice_is_stable():
    eligible = list(REQUIRED_SOURCE_IDS) + list(range(1000, 1100))
    first = choose_sources(eligible, n=50, required=REQUIRED_SOURCE_IDS, seed=0)
    again = choose_sources(eligible, n=50, required=REQUIRED_SOURCE_IDS, seed=0)
    assert first == again
    assert len(first) == 50
    assert set(REQUIRED_SOURCE_IDS) <= set(first)
    assert first == sorted(first)


def test_choose_sources_rejects_a_missing_required_source():
    try:
        choose_sources([1, 2, 3, 4], n=3, required=(1, 9), seed=0)
    except RuntimeError as exc:
        assert "9" in str(exc)
    else:
        raise AssertionError("expected RuntimeError")


def test_save_fullres_keeps_figure_order(tmp_path):
    cache = tmp_path / "cache"
    (cache / "s800").mkdir(parents=True)
    (cache / "s800" / "second.jpg").write_bytes(b"second")
    (cache / "s800" / "first.jpg").write_bytes(b"first")
    dest = save_fullres(
        800,
        ["111", "222"],
        {"111": "s800/first.jpg", "222": "s800/second.jpg"},
        cache,
        tmp_path / "out",
    )
    assert (dest / "00_111.jpg").read_bytes() == b"first"
    assert (dest / "01_222.jpg").read_bytes() == b"second"


def test_sample_order_is_deterministic_and_covers_the_ids():
    ids = [str(i) for i in range(20)]
    order = sample_order(ids, seed=0, source_id=4906)
    assert order == sample_order(ids, seed=0, source_id=4906)
    assert sorted(order, key=int) == ids
    assert order != sample_order(ids, seed=0, source_id=3443)


def test_row_is_the_page_width_with_five_wider_cells_and_a_narrower_legend():
    layout = row_layout()
    assert layout.fig_width_in == FIG_WIDTH_IN == figs.TEXT_WIDTH_IN
    assert layout.n_images == N_IMAGES == 5
    cells = layout.n_images * layout.cell_w_in + (layout.n_images - 1) * layout.image_gap_in
    assert abs(cells - layout.strip_w_in) < 1e-9
    assert layout.strip_x_in + layout.strip_w_in <= layout.legend_x_in + 1e-9
    assert abs(layout.legend_w_in - LEGEND_WIDTH_IN) < 1e-9
    assert layout.legend_w_in < 0.30 * FIG_WIDTH_IN
    assert layout.strip_w_in > 0.68 * FIG_WIDTH_IN
    assert abs(layout.cell_h_in / layout.cell_w_in - 0.7) < 1e-9


def test_fitted_size_keeps_aspect_inside_the_cell():
    cell_w, cell_h = 1.4, 0.7
    wide_w, wide_h = fitted_size((1000, 500), cell_w, cell_h)
    tall_w, tall_h = fitted_size((400, 800), cell_w, cell_h)
    assert wide_w <= cell_w + 1e-9 and wide_h <= cell_h + 1e-9
    assert tall_w <= cell_w + 1e-9 and tall_h <= cell_h + 1e-9
    assert abs(wide_w / wide_h - 2) < 1e-9
    assert abs(tall_w / tall_h - 0.5) < 1e-9


def _solid_images(seed: int, size: tuple[int, int]) -> list[Image.Image]:
    rng = np.random.default_rng(seed)
    return [
        Image.fromarray(rng.integers(0, 255, (size[1], size[0], 3), dtype=np.uint8))
        for _ in range(N_IMAGES)
    ]


def test_composed_figures_share_a_page_width(tmp_path):
    figs.setup_style()
    short_labels = ["coral", "sand"]
    tall_labels = [f"label {i:02d}" for i in range(40)]
    colors = {
        name: ((40 + i * 5) % 256, 80, 140) for i, name in enumerate(short_labels + tall_labels)
    }

    short = tmp_path / "s1.pdf"
    tall = tmp_path / "s2.pdf"
    compose_source_figure(_solid_images(0, (32, 24)), short_labels, colors, 3443, short)
    compose_source_figure(_solid_images(1, (24, 32)), tall_labels, colors, 6695, tall)
    short_w, short_h = pdf_media_box(short)
    tall_w, tall_h = pdf_media_box(tall)
    assert abs(short_w - FIG_WIDTH_IN * 72) < 0.2
    assert abs(tall_w - short_w) < 0.2
    assert tall_h > short_h
