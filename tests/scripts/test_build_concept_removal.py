"""build_concept_removal: train label counts from tiny versions' stats, and the removal file."""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest
import yaml

import scripts.build_concept_removal as script
from mermaidseg.dataset_reconciliation.concept_removal import REMOVAL_COLUMNS
from segdata_spec.errors import DirtyTree
from segdata_spec.git import GitState
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


@pytest.fixture(autouse=True)
def clean_tree(monkeypatch) -> list[tuple[Path, ...]]:
    """The builders record the sha of a clean tree; return the ``inputs`` each check received."""
    checked: list[tuple[Path, ...]] = []

    def clean(_start, *, inputs=()):
        checked.append(tuple(inputs))
        return GitState("abc", False, True)

    monkeypatch.setattr(script, "clean_git_state", clean)
    return checked


def _write_inputs(tmp_path):
    """Write the cache, data config and concepts; return ``(cache_dir, data_config, concepts, pin)``."""
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
                    "taxonomy": {"version": "20261121-ab12cd3", "manifest_sha256": "c" * 64},
                    "datasets": {
                        "coralnet": {
                            "version": pin.version,
                            "manifest_sha256": pin.manifest_sha256,
                            "split_file": str(split_file),
                            "train": {"use": ["train"]},
                        }
                    },
                }
            }
        )
    )
    concepts = tmp_path / "concepts.csv"
    concept_table().to_csv(concepts, index=False)
    return cache_dir, data_config, concepts, pin


def test_main_writes_the_channels_that_fail_the_rule_over_the_train_groups(tmp_path):
    cache_dir, data_config, concepts, pin = _write_inputs(tmp_path)
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
    assert meta["git_sha"] == "abc"
    assert not {"git_dirty", "data_config_sha1"} & set(meta)
    assert "git_commit" not in meta


def test_a_dirty_tree_refuses_before_any_output_is_written(tmp_path, monkeypatch):
    cache_dir, data_config, concepts, _ = _write_inputs(tmp_path)
    out = tmp_path / "new_dir" / "removal.csv"

    def refuse(_start, *, inputs=()):
        raise DirtyTree("dirty")

    monkeypatch.setattr(script, "clean_git_state", refuse)

    with pytest.raises(DirtyTree):
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
            ]
        )

    assert not out.parent.exists()


def test_the_data_and_schema_configs_are_checked_as_tracked_inputs(tmp_path, clean_tree):
    cache_dir, data_config, concepts, _ = _write_inputs(tmp_path)

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
            str(tmp_path / "removal.csv"),
        ]
    )

    assert clean_tree == [(data_config, data_config)]
