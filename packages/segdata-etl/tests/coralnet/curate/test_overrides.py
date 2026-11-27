"""Override replacement and the resized-to-original coordinate mapping.

Numbers come from image 1924169 (3486 wide x 3469 high) of the 20260623_nogit build, whose resized
grid is 2047 wide x 2038 high (the float order in ``scaled_size`` truncates the longest edge to 2047).
"""

from __future__ import annotations

import pandas as pd
import pyarrow as pa
import pytest

from segdata_etl.datasets.coralnet.curate.overrides import (
    OverrideStats,
    _invert,
    apply_overrides,
)
from segdata_etl.datasets.coralnet.curate.overrides import (
    _forward as forward_coord,
)
from segdata_spec.resize import ResizeSpec, scale_points

IMG = "1924169"


def _images(*rows: tuple[str, int, int]) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "image_id": [r[0] for r in rows],
            "width": pd.array([r[1] for r in rows], dtype="int32"),
            "height": pd.array([r[2] for r in rows], dtype="int32"),
        }
    )


def _points(*rows: tuple[str, int, int, int]) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "image_id": [r[0] for r in rows],
            "row": pd.array([r[1] for r in rows], dtype="int32"),
            "col": pd.array([r[2] for r in rows], dtype="int32"),
            "label_id": pd.array([r[3] for r in rows], dtype="int32"),
            "annotator": "default",
        }
    )


def _overrides(*rows: tuple[str, int, int, int]) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "image_id": [r[0] for r in rows],
            "row": pd.array([r[1] for r in rows], dtype="int16"),
            "col": pd.array([r[2] for r in rows], dtype="int16"),
            "coralnet_id": [r[3] for r in rows],
        }
    )


def _tuples(df: pd.DataFrame) -> list[tuple[str, int, int, int]]:
    rows = zip(df["image_id"], df["row"], df["col"], df["label_id"], strict=True)
    return sorted((str(i), int(r), int(c), int(label)) for i, r, c, label in rows)


def _forward(images: pd.DataFrame, points: pd.DataFrame) -> list[tuple[int, int]]:
    scaled = scale_points(
        pa.Table.from_pandas(points[["image_id", "row", "col"]], preserve_index=False),
        pa.Table.from_pandas(images, preserve_index=False),
        ResizeSpec(max_edge=2048),
    ).to_pandas()
    return [(int(r), int(c)) for r, c in zip(scaled["row"], scaled["col"], strict=True)]


def test_override_matching_a_raw_point_takes_that_points_exact_original_coords() -> None:
    images = _images((IMG, 3486, 3469))
    raw = _points((IMG, 97, 3159, 141))  # scales to (56, 1854); the smallest preimage is (96, 3158)
    out, _ = apply_overrides(raw, images, _overrides((IMG, 56, 1854, 141)))
    assert _tuples(out) == [(IMG, 97, 3159, 141)]


def test_override_without_a_raw_match_inverts_to_the_smallest_original_coords() -> None:
    images = _images((IMG, 3486, 3469))
    raw = _points((IMG, 5, 5, 1))
    out, _ = apply_overrides(raw, images, _overrides((IMG, 81, 342, 3522)))
    assert _tuples(out) == [(IMG, 138, 583, 3522)]
    assert _forward(images, out) == [(81, 342)]


def test_override_on_an_image_within_the_edge_is_copied_unchanged() -> None:
    images = _images(("small", 2000, 1500))
    raw = _points(("small", 10, 10, 1))
    out, stats = apply_overrides(raw, images, _overrides(("small", 1499, 7, 5), ("small", 3, 4, 6)))
    assert _tuples(out) == [("small", 3, 4, 6), ("small", 1499, 7, 5)]
    assert stats.points_unscaled == 2


def test_overridden_image_is_fully_replaced_and_others_are_untouched() -> None:
    images = _images((IMG, 3486, 3469), ("other", 3486, 3469))
    raw = _points((IMG, 96, 3158, 141), (IMG, 200, 200, 7), ("other", 50, 60, 9))
    overrides = _overrides(
        (IMG, 56, 1854, 141),  # matches the first raw point
        (IMG, 81, 342, 3522),  # no raw match
    )
    out, stats = apply_overrides(raw, images, overrides)
    assert _tuples(out) == [(IMG, 96, 3158, 141), (IMG, 138, 583, 3522), ("other", 50, 60, 9)]
    assert stats == OverrideStats(
        images_overridden=1,
        points_exact=1,
        points_inverted=1,
        points_unscaled=0,
        override_images_not_in_raw=0,
    )
    assert out.dtypes.to_dict() == raw.dtypes.to_dict()
    assert set(out["annotator"]) == {"default"}


def test_override_equal_in_position_but_different_in_label_is_inverted_not_matched() -> None:
    images = _images((IMG, 3486, 3469))
    raw = _points((IMG, 97, 3158, 141))  # scales to (56, 1854); the smallest preimage is (96, 3158)
    out, stats = apply_overrides(raw, images, _overrides((IMG, 56, 1854, 999)))
    assert _tuples(out) == [(IMG, 96, 3158, 999)]
    assert (stats.points_exact, stats.points_inverted) == (0, 1)


def test_several_matching_raw_points_resolve_to_the_smallest_coords() -> None:
    images = _images((IMG, 3486, 3469))
    raw = _points((IMG, 97, 3159, 141), (IMG, 97, 3158, 141))  # both scale to (56, 1854)
    assert _forward(images, raw) == [(56, 1854), (56, 1854)]
    out, _ = apply_overrides(raw, images, _overrides((IMG, 56, 1854, 141)))
    assert _tuples(out) == [(IMG, 97, 3158, 141)]


def test_override_for_an_image_not_in_raw_is_dropped_and_counted() -> None:
    images = _images((IMG, 3486, 3469))
    raw = _points((IMG, 96, 3158, 141))
    out, stats = apply_overrides(raw, images, _overrides(("ghost", 1, 1, 5)))
    assert _tuples(out) == [(IMG, 96, 3158, 141)]
    assert stats.override_images_not_in_raw == 1
    assert stats.images_overridden == 0


def test_override_image_ids_are_compared_as_strings() -> None:
    images = _images((IMG, 2000, 1500))
    overrides = _overrides((IMG, 3, 4, 6))
    overrides["image_id"] = overrides["image_id"].astype("int64")
    out, _ = apply_overrides(_points((IMG, 1, 1, 1)), images, overrides)
    assert _tuples(out) == [(IMG, 3, 4, 6)]


def test_image_exactly_at_the_load_edge_is_copied_unchanged() -> None:
    images = _images(("edge", 2048, 1500))
    out, stats = apply_overrides(
        _points(("edge", 1, 1, 1)), images, _overrides(("edge", 1000, 1000, 4))
    )
    assert _tuples(out) == [("edge", 1000, 1000, 4)]
    assert (stats.points_unscaled, stats.points_inverted, stats.points_exact) == (1, 0, 0)


@pytest.mark.parametrize(
    ("orig", "new"),
    [(4096, 2048), (3000, 2048), (2049, 2048), (1500, 1096), (3469, 2038), (2576, 2047)],
)
def test_invert_is_the_smallest_original_that_scales_forward_to_each_resized_coordinate(
    orig: int, new: int
) -> None:
    for x in range(new):
        c = _invert(x, orig, new)
        assert forward_coord(c, orig, new) == x
        if c > 0:
            assert forward_coord(c - 1, orig, new) < x
