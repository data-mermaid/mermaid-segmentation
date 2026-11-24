"""resolve_taxonomy_name looks up a benthic-attribute name in a taxonomy, ignoring case."""

from __future__ import annotations

import pyarrow as pa
import pytest

from segdata_etl.common.labels import (
    AmbiguousTaxonomyName,
    resolve_taxonomy_name,
    taxonomy_name_index,
)

ATTRIBUTES = pa.table(
    {
        "id": ["ba-hc", "ba-acro", "ba-por-1", "ba-por-2"],
        "name": ["Hard coral", "Acropora", "Porites", "PORITES"],
        "parent_id": [None, "ba-hc", "ba-hc", "ba-hc"],
    }
)


@pytest.mark.parametrize("name", ["Acropora", "acropora", "ACROPORA"])
def test_a_name_resolves_to_its_id_and_canonical_spelling_ignoring_case(name: str) -> None:
    index = taxonomy_name_index(ATTRIBUTES)
    assert resolve_taxonomy_name(name, index) == ("ba-acro", "Acropora")


def test_a_name_no_attribute_carries_resolves_to_none() -> None:
    index = taxonomy_name_index(ATTRIBUTES)
    assert resolve_taxonomy_name("Sand", index) is None


def test_a_name_matching_several_attributes_raises_naming_them() -> None:
    index = taxonomy_name_index(ATTRIBUTES)
    with pytest.raises(AmbiguousTaxonomyName, match=r"'porites'.*'Porites'.*'PORITES'") as info:
        resolve_taxonomy_name("porites", index)
    assert isinstance(info.value, ValueError)
