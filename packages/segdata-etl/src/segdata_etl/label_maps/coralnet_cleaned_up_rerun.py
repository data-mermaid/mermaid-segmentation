"""CoralNet labels mapped to MERMAID benthic attributes by the cleaned_up_rerun JSON.

The JSON keys lowercased CoralNet label names; its values are MERMAID benthic-attribute names,
or null for a label with no target. The file is a byte copy of the mapping the cleaned_up_rerun
branch trained with.
"""

from __future__ import annotations

import hashlib
import json
from collections import defaultdict
from collections.abc import Mapping
from importlib import resources
from typing import Final, Literal

import pyarrow as pa

from segdata_spec.manifest import LabelMapping
from segdata_spec.schema import LABELS_SCHEMA, conform

MAPPING_SOURCE: Final[Literal["cleaned_up_rerun_json"]] = "cleaned_up_rerun_json"
GIT_REF = "e0d3876"
SOURCE_PATH = "configs/coralnet_to_mermaid_mapping_temporary.json"
GIT_BLOB = "4f37f59ec8f57e26540f57adfa093830c8305f64"
_JSON_NAME = "coralnet_cleaned_up_rerun.json"


def _target_name(name: str | None, name_to_target: Mapping[str, str | None]) -> str | None:
    """The JSON target for a CoralNet label name; None when unnamed, absent or null."""
    if not isinstance(name, str):
        return None
    return name_to_target.get(name.lower())


def _ids_by_lower_name(taxonomy: pa.Table) -> dict[str, list[str]]:
    ids: dict[str, list[str]] = defaultdict(list)
    for tax_id, tax_name in zip(
        taxonomy.column("id").to_pylist(), taxonomy.column("name").to_pylist(), strict=True
    ):
        ids[tax_name.lower()].append(tax_id)
    return ids


def _resolve_ids(targets: set[str], taxonomy: pa.Table) -> dict[str, str | None]:
    """Map each target name to its taxonomy id by case-insensitive name.

    Raises:
        ValueError: A target matches more than one taxonomy row.
    """
    by_name = _ids_by_lower_name(taxonomy)
    ambiguous = sorted(t for t in targets if len(by_name.get(t.lower(), ())) > 1)
    if ambiguous:
        raise ValueError(
            f"taxonomy has several benthic attributes named (ignoring case): {ambiguous}"
        )
    resolved: dict[str, str | None] = {}
    for target in targets:
        matches = by_name.get(target.lower(), [])
        resolved[target] = matches[0] if matches else None
    return resolved


def cleaned_up_rerun_labels(
    id2name: Mapping[int, str | None], taxonomy: pa.Table
) -> tuple[pa.Table, LabelMapping]:
    """Build ``labels.parquet`` rows and the mapping provenance for the labels in ``id2name``.

    A label is unmapped when its name is None, absent from the JSON, or null there. A None name
    is stored as an empty string because ``name`` is not nullable.

    Args:
        id2name: CoralNet label id -> name, for the ids the dataset uses.
        taxonomy: The ``benthic_attributes`` table (``id``, ``name``, ``parent_id``).

    Raises:
        ValueError: A mapped target name matches more than one taxonomy row ignoring case.
    """
    json_bytes = resources.files(__package__).joinpath(_JSON_NAME).read_bytes()
    name_to_target: dict[str, str | None] = json.loads(json_bytes)
    targets = {label_id: _target_name(name, name_to_target) for label_id, name in id2name.items()}
    ids = _resolve_ids({t for t in targets.values() if t is not None}, taxonomy)

    label_ids = list(id2name)
    mapped = [targets[i] for i in label_ids]
    table = conform(
        pa.table(
            {
                "label_id": label_ids,
                "name": [id2name[i] or "" for i in label_ids],
                "mermaid_benthic_attribute_id": [None if t is None else ids[t] for t in mapped],
                "mermaid_benthic_attribute": mapped,
                "mermaid_growth_form": [None] * len(label_ids),
                "mapping_source": [MAPPING_SOURCE] * len(label_ids),
            }
        ),
        LABELS_SCHEMA,
    )
    n_mapped = sum(t is not None for t in mapped)
    n_resolved = sum(t is not None and ids[t] is not None for t in mapped)
    mapping = LabelMapping(
        source=MAPPING_SOURCE,
        git_ref=GIT_REF,
        path=SOURCE_PATH,
        blob=GIT_BLOB,
        sha256=hashlib.sha256(json_bytes).hexdigest(),
        n_labels=len(label_ids),
        n_mapped=n_mapped,
        n_unresolved=n_mapped - n_resolved,
    )
    return table, mapping
