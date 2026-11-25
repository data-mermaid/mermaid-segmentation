"""build_concept_removal: train label counts from tiny versions' stats, and the removal file."""

from __future__ import annotations

from typing import Any

import pandas as pd
import pytest
import yaml

import scripts.build_concept_removal as script
from mermaidseg.data.config import DataSection
from mermaidseg.dataset_reconciliation.concept_removal import REMOVAL_COLUMNS, concept_coverage
from mermaidseg.dataset_reconciliation.concept_schema import ConceptSchema
from segdata_spec.manifest import Pin
from tests.mermaidseg.conftest import concept_table
from tests.mermaidseg.data.conftest import ImageSpec, write_version

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
    out = script.train_label_counts("coralnet", LABEL_COUNTS, IMAGES, LABELS, {"s7"})

    assert out.to_numpy().tolist() == [
        ["coralnet", "a1", "acropora", 2, "7"],
        ["coralnet", "a1", "sand", 1, "7"],
        ["coralnet", "a2", "acropora", 3, "7"],
    ]


def test_another_datasets_source_is_its_name_and_no_groups_counts_every_image():
    out = script.train_label_counts("mermaid", LABEL_COUNTS, IMAGES, LABELS, None)

    assert out[["image_id", "label", "source"]].to_numpy().tolist() == [
        ["a1", "acropora", "mermaid"],
        ["a1", "sand", "mermaid"],
        ["a2", "acropora", "mermaid"],
        ["v1", "porites", "mermaid"],
    ]


def test_a_counted_label_id_missing_from_labels_raises():
    counts = pd.DataFrame({"image_id": ["a1"], "label_id": [8], "n": [1]})
    with pytest.raises(ValueError, match=r"label ids missing from labels.parquet: \[8\]"):
        script.train_label_counts("coralnet", counts, IMAGES, LABELS, None)


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

    out = script.train_label_counts("mermaid", counts, IMAGES, labels, None)
    coverage = concept_coverage(out, schema)

    assert out["label"].tolist() == ["acropora", "acropora"]
    assert coverage.loc["genus__acropora", ["n_annotations", "n_images"]].tolist() == [5, 2]


def _section(pin: Pin, entry: dict[str, Any]) -> DataSection:
    """A data section pinning ``pin`` as ``coralnet`` with the split blocks of ``entry``."""
    return DataSection.model_validate(
        {
            "datasets": {
                "coralnet": {
                    "version": pin.version,
                    "manifest_sha256": pin.manifest_sha256,
                    **entry,
                }
            }
        }
    )


ONE_ANNOTATOR = [("a1", 0, 0, 1)]


@pytest.mark.parametrize(
    ("entry", "points", "message"),
    [
        (
            {"train": {"groups": ["s99"]}},
            ONE_ANNOTATOR,
            r"groups not in the dataset version: \['s99'\]",
        ),
        ({"val": {"groups": "all"}}, ONE_ANNOTATOR, "no dataset has a train block to count"),
        (
            {"train": {"groups": "all"}},
            [("a1", 0, 0, 1, "a"), ("a1", 1, 1, 1, "b")],
            r"coralnet: several annotators \['a', 'b'\]",
        ),
        (
            {"train": {"groups": "all", "annotators": ["a"]}},
            [("a1", 0, 0, 1, "a"), ("a1", 1, 1, 1, "b")],
            r"coralnet: several annotators \['a', 'b'\]",
        ),
    ],
    ids=[
        "unknown-group",
        "no-train-block",
        "several-annotators-none-named",
        "several-annotators-some-named",
    ],
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
        script.section_train_label_counts(_section(pin, entry), cache_dir)


def test_main_writes_the_channels_that_fail_the_rule_over_the_train_groups(tmp_path):
    cache_dir, _, pin = write_version(
        tmp_path,
        kind="points",
        images=[
            ImageSpec(i, group=g) for i, g in zip(IMAGES["image_id"], IMAGES["group"], strict=True)
        ],
        labels=dict(zip(LABELS["label_id"], LABELS["name"], strict=True)),
        points=[("a1", 0, 0, 1), ("a1", 1, 1, 2), ("a2", 0, 0, 1), ("v1", 0, 0, 4)],
        name="coralnet",
    )
    split_file = tmp_path / "coralnet.csv"
    split_file.write_text("group,split\ns7,train\ns9,val\n")
    data_config = tmp_path / "data_config.yaml"
    data_config.write_text(
        yaml.safe_dump(
            {
                "data": {
                    "datasets": {
                        "coralnet": {
                            "version": pin.version,
                            "manifest_sha256": pin.manifest_sha256,
                            "split_file": str(split_file),
                            "train": {"use": ["train"]},
                        }
                    }
                }
            }
        )
    )
    concepts = tmp_path / "concepts.csv"
    concept_table().to_csv(concepts, index=False)
    out = tmp_path / "removal.csv"

    script.main(
        [
            "--cache-dir",
            str(cache_dir),
            "--data-config",
            str(data_config),
            "--schema-config",
            str(data_config),
            "--concepts",
            str(concepts),
            "--out",
            str(out),
            "--min-images",
            "2",
            "--min-noncoralnet-images",
            "99",
            "--min-annotations",
            "1",
            "--min-coralnet-sources",
            "1",
        ]
    )

    # Acropora is in both train images and is kept; sand is in one. The val image's porites
    # is not counted, so the porites channels have no support.
    written = pd.read_csv(out)
    assert list(written.columns) == REMOVAL_COLUMNS
    assert written["concept"].tolist() == [
        "kingdom__none",
        "phylum__none",
        "class__none",
        "order__none",
        "family__none",
        "family__poritidae",
        "genus__none",
        "genus__porites",
        "tubular",
        "bleached",
    ]
    assert written.set_index("concept").loc["genus__none"].tolist() == [
        "genus",
        1,
        1,
        0,
        1,
        0,
        True,
        False,
        False,
    ]
    meta = yaml.safe_load((tmp_path / "removal.meta.yaml").read_text())
    assert meta["datasets_counted"] == ["coralnet"]
    assert meta["dataset_versions"] == {"coralnet": pin.version}
