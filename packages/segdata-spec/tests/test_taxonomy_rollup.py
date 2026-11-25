import pyarrow as pa
import pytest

from segdata_spec.taxonomy import hierarchy_from_table, roll_up

HIERARCHY = {"acropora": "hard coral", "hard coral": None, "porites": "hard coral", "sand": None}


@pytest.mark.parametrize(
    ("label", "subset", "expected"),
    [
        ("hard coral", {"hard coral"}, "hard coral"),
        ("acropora", {"hard coral"}, "hard coral"),
        ("sand", {"hard coral"}, None),
        ("unknown", {"hard coral"}, None),
        (None, {"hard coral"}, None),
        ("outside", {"outside"}, "outside"),
    ],
)
def test_roll_up(label, subset, expected):
    assert roll_up(label, HIERARCHY, subset) == expected


def test_roll_up_returns_nearest_ancestor():
    chain = {"a": "b", "b": "c", "c": None}

    assert roll_up("a", chain, {"b", "c"}) == "b"


def test_hierarchy_from_table_lowercases_and_resolves_parent_ids():
    table = pa.table(
        {
            "id": ["1", "2", "3"],
            "name": ["Hard coral", "Acropora", "Orphan"],
            "parent_id": [None, "1", "99"],
        }
    )

    assert hierarchy_from_table(table) == {
        "hard coral": None,
        "acropora": "hard coral",
        "orphan": None,
    }
