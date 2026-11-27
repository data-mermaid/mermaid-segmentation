"""Reading a concept-removal file and removing concepts from the concept table."""

from __future__ import annotations

from typing import Any

import pandas as pd
import pytest
import yaml

from mermaidseg.data.config import DataSection
from mermaidseg.dataset_reconciliation.artifact_meta import meta_path
from mermaidseg.dataset_reconciliation.concept_removal import (
    concept_coverage,
    load_concept_removal,
    section_train_label_counts,
    train_label_counts,
)
from mermaidseg.dataset_reconciliation.concept_schema import ConceptSchema
from segdata_spec.manifest import Pin
from tests.mermaidseg.conftest import concept_table
from tests.mermaidseg.data.conftest import ImageSpec, write_version


def _write_removal(tmp_path, concepts: list[str]):
    path = tmp_path / "removal.csv"
    pd.DataFrame({"concept": concepts, "kind": "x"}).to_csv(path, index=False)
    meta_path(path).write_text(yaml.safe_dump({}))
    return path


# ---------- load_concept_removal ----------


def test_load_returns_the_listed_concepts_in_file_order(tmp_path):
    path = _write_removal(tmp_path, ["tubular", "genus__porites", "bleached"])

    assert load_concept_removal(path) == ["tubular", "genus__porites", "bleached"]


def test_load_without_a_sidecar_raises_file_not_found(tmp_path):
    path = tmp_path / "removal.csv"
    pd.DataFrame({"concept": ["tubular"]}).to_csv(path, index=False)

    with pytest.raises(FileNotFoundError, match="no sidecar"):
        load_concept_removal(path)


# ---------- removed_concepts ----------


# ---------- train_label_counts ----------

IMAGES = pd.DataFrame({"image_id": ["a1", "a2", "v1"], "group": ["s7", "s7", "s9"]})
LABELS = pd.DataFrame(
    {
        "label_id": [1, 2, 3, 4],
        "name": ["Acropora", " Sand ", "Unlabeled", "Porites"],
        "mermaid_benthic_attribute": [None] * 4,
    }
)
LABEL_COUNTS = pd.DataFrame(
    {
        "image_id": ["a1", "a1", "a2", "a2", "a2", "v1"],
        "label_id": [1, 2, 1, 3, 0, 4],
        "n": [2, 1, 3, 5, 9, 4],
    }
)


def test_coralnet_counts_cover_the_train_groups_without_unlabeled_with_the_source_id():
    out = train_label_counts("coralnet", LABEL_COUNTS, IMAGES, LABELS, {"s7"})

    assert out.to_numpy().tolist() == [
        ["coralnet", "a1", "acropora", 2, "7"],
        ["coralnet", "a1", "sand", 1, "7"],
        ["coralnet", "a2", "acropora", 3, "7"],
    ]


def test_another_datasets_source_is_its_name_and_no_groups_counts_every_image():
    out = train_label_counts("mermaid", LABEL_COUNTS, IMAGES, LABELS, None)

    assert out[["image_id", "label", "source"]].to_numpy().tolist() == [
        ["a1", "acropora", "mermaid"],
        ["a1", "sand", "mermaid"],
        ["a2", "acropora", "mermaid"],
        ["v1", "porites", "mermaid"],
    ]


def test_mermaid_labels_count_under_the_concept_row_of_their_benthic_attribute(tmp_path):
    labels = pd.DataFrame(
        {
            "label_id": [1, 2],
            "name": ["Acropora::Branching", "Acropora::"],
            "mermaid_benthic_attribute": ["Acropora", "Acropora"],
        }
    )
    counts = pd.DataFrame({"image_id": ["a1", "a2"], "label_id": [1, 2], "n": [2, 3]})
    concepts = tmp_path / "concepts.csv"
    concept_table().assign(source_dataset_source="mermaid").to_csv(concepts, index=False)
    schema = ConceptSchema.from_csv(concepts, sources={"mermaid"})

    out = train_label_counts("mermaid", counts, IMAGES, labels, None)
    coverage = concept_coverage(out, schema)

    assert out["label"].tolist() == ["acropora", "acropora"]
    assert coverage.loc["genus__acropora", ["n_annotations", "n_images"]].tolist() == [5, 2]


def _section(pin: Pin, entry: dict[str, Any]) -> DataSection:
    """A data section pinning ``pin`` as ``coralnet`` with the split blocks of ``entry``."""
    return DataSection.model_validate(
        {
            "taxonomy": {"version": "20261121-ab12cd3", "manifest_sha256": "c" * 64},
            "datasets": {
                "coralnet": {
                    "version": pin.version,
                    "manifest_sha256": pin.manifest_sha256,
                    **entry,
                }
            },
        }
    )


ONE_ANNOTATOR = [("a1", 0, 0, 1)]


@pytest.mark.parametrize(
    ("entry", "points", "message"),
    [
        ({"val": {"groups": "all"}}, ONE_ANNOTATOR, "no dataset has a train block to count"),
        (
            {"train": {"groups": "all"}},
            [("a1", 0, 0, 1, "a"), ("a1", 1, 1, 1, "b")],
            r"coralnet: several annotators \['a', 'b'\]",
        ),
    ],
    ids=["no-train-block", "several-annotators-none-named"],
)
def test_section_train_label_counts_guards_raise(tmp_path, entry, points, message):
    cache_dir, _, pin = write_version(
        tmp_path,
        kind="points",
        images=[ImageSpec("a1", group="s7")],
        labels={1: "Acropora"},
        points=points,
        name="coralnet",
    )

    with pytest.raises(ValueError, match=message):
        section_train_label_counts(_section(pin, entry), cache_dir)
