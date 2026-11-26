"""Concept helpers: label map to per-pixel concept vectors, and the one concept encoding."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
import torch

from mermaidseg.dataset_reconciliation.concept_ranks import binary_offset, rank_slices
from mermaidseg.dataset_reconciliation.concepts import (
    encode_concept_row,
    generate_binary_concept_id_mapping,
    parse_concept_cell,
    read_class_to_concepts,
    source_labels_to_concepts,
)

# Row i is the concept vector of label i: label 0 (background) is all zeros.
LOOKUP = torch.tensor([[0.0, 0.0, 0.0], [1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])


def test_non_square_label_map_becomes_channels_first_concepts():
    labels = torch.tensor([[[1, 0, 2], [2, 2, 1]]])  # (B=1, H=2, W=3)

    out = source_labels_to_concepts(labels, LOOKUP)

    assert out.shape == (1, 3, 2, 3)
    assert out[0, :, 0, 0].tolist() == [1.0, 2.0, 3.0]  # label 1 at (0, 0)
    assert out[0, :, 0, 1].tolist() == [0.0, 0.0, 0.0]  # background at (0, 1)
    assert out[0, :, 1, 0].tolist() == [4.0, 5.0, 6.0]  # label 2 at (1, 0)
    assert out[0, :, 1, 2].tolist() == [1.0, 2.0, 3.0]  # label 1 at (1, 2)


@pytest.mark.parametrize(
    ("cell", "expected"),
    [("TRUE", "true"), ("true", "true"), (" FALSE ", "false"), ("not_given", "not_given")],
)
def test_binary_cells_parse_case_and_whitespace_insensitively(cell, expected):
    assert parse_concept_cell(cell, rank=False) == expected


@pytest.mark.parametrize(
    ("cell", "expected"),
    [("Acropora", "acropora"), ("none", "none"), ("not_given", "not_given")],
)
def test_rank_cells_parse_to_lower_case_and_keep_none_distinct_from_not_given(cell, expected):
    assert parse_concept_cell(cell, rank=True) == expected


@pytest.mark.parametrize("cell", ["not given", "", "nan", "yes", True])
def test_a_binary_cell_that_is_not_true_false_or_not_given_raises(cell):
    with pytest.raises(ValueError, match="concept cell"):
        parse_concept_cell(cell, rank=False)


@pytest.mark.parametrize("cell", ["", "not given", "nan", None])
def test_a_rank_cell_that_is_a_placeholder_other_than_not_given_raises(cell):
    with pytest.raises(ValueError, match="concept cell"):
        parse_concept_cell(cell, rank=True)


VALUE2ID = {
    "kingdom": {"not_given": [0, 0], "animalia": [2, 1], "plantae": [1, 2]},
    "genus": {"not_given": [0, 0, 0], "a": [2, 1, 1], "b": [1, 2, 1], "c": [1, 1, 2]},
    "live": {"not_given": 0, "FALSE": 1, "TRUE": 2},
}


def test_rank_slices_skip_absent_ranks_and_exclude_not_given_from_the_width():
    assert rank_slices(VALUE2ID) == [("kingdom", 0, 2), ("genus", 2, 3)]
    assert binary_offset(VALUE2ID) == 5


def test_without_taxonomic_ranks_there_are_no_slices_and_the_binary_tail_starts_at_zero():
    assert rank_slices({"live": VALUE2ID["live"]}) == []
    assert binary_offset({"live": VALUE2ID["live"]}) == 0


CHANNELS = ["genus__a", "genus__b", "live", "fish"]


@pytest.mark.parametrize(
    ("row", "expected"),
    [
        pytest.param({"genus": "b", "live": "TRUE", "fish": "FALSE"}, [1, 2, 2, 1], id="coded"),
        pytest.param({"genus": "zzz", "live": "TRUE", "fish": "TRUE"}, [0, 0, 2, 2], id="no_channel"),
        pytest.param({"genus": "not_given", "live": "FALSE"}, [0, 0, 1, 0], id="not_given_and_absent"),
    ],
)  # fmt: skip
def test_encode_concept_row_codes_ranks_by_channel_and_zeroes_a_value_without_one(row, expected):
    encoded = encode_concept_row(row, CHANNELS)

    assert encoded.tolist() == expected
    assert encoded.dtype == np.int8


def test_binary_mapping_has_the_three_string_keys_only():
    df = pd.DataFrame({"live": ["TRUE", "FALSE", "not_given"]})

    assert generate_binary_concept_id_mapping(df, "live") == {"not_given": 0, "FALSE": 1, "TRUE": 2}


def test_reading_the_table_keeps_strings_empty_cells_and_lower_cases_the_keys(tmp_path):
    path = tmp_path / "c.csv"
    path.write_text(
        "source_label_class_name,source_dataset_source,live,genus\n"
        "Acro,CoralNet,TRUE,\n"
        "Sand,CoralNet,FALSE,not_given\n"
    )

    df = read_class_to_concepts(path)

    assert df["live"].tolist() == ["TRUE", "FALSE"]
    assert df["genus"].tolist() == ["", "not_given"]
    assert df["source_dataset_source"].tolist() == ["coralnet", "coralnet"]
    assert df["source_label_class_name"].tolist() == ["acro", "sand"]
