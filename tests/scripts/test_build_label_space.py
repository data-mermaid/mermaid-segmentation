"""build_label_space: the four files and their meta from a tables-only cache, and the refusal."""

from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Any

import pandas as pd
import pyarrow as pa
import pytest
import yaml

import scripts.build_label_space as script
from mermaidseg.dataset_reconciliation.label_space import label_keys_sha1
from mermaidseg.io import REPO_ROOT, ConfigDict
from segdata_spec.git import GitState
from segdata_spec.manifest import Pin
from tests.mermaidseg.conftest import concept_table
from tests.mermaidseg.data.conftest import ImageSpec, write_taxonomy, write_version

_TAXONOMY = pa.table(
    {
        "id": ["1", "3", "5", "6"],
        "name": ["Hard coral", "Acropora", "Porites", "Sand"],
        "parent_id": [None, "1", "1", None],
    }
)
# s7 is train, s8 is val: its point must not be counted.
_IMAGES = [ImageSpec("i1", group="s7"), ImageSpec("i2", group="s8")]
_POINTS = [("i1", 0, 0, 1), ("i1", 0, 1, 3), ("i1", 1, 0, 3), ("i2", 0, 0, 3)]
_LABELS = [
    {"label_id": 1, "name": "Acropora", "mermaid_benthic_attribute": "Acropora"},
    {"label_id": 3, "name": "Sand", "mermaid_benthic_attribute": "Sand"},
    {"label_id": 4, "name": "Unknown", "mermaid_benthic_attribute": None},
]
# No concept row is named "Mystery", and it reaches the Porites class.
_MYSTERY = {"label_id": 5, "name": "Mystery", "mermaid_benthic_attribute": "Porites"}


@pytest.fixture(autouse=True)
def fixed_git(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        script, "clean_git_state", lambda _start, *, inputs=(): GitState("abc1234", False, True)
    )


def _inputs(tmp_path: Path, labels: list[dict[str, Any]], *, train: bool = True) -> dict[str, Any]:
    cache_dir, _, pin = write_version(
        tmp_path,
        kind="points",
        images=_IMAGES,
        labels=labels,
        points=_POINTS,
        name="coralnet",
    )
    taxonomy_pin = write_taxonomy(cache_dir, tmp_path, _TAXONOMY)
    split_file = tmp_path / "coralnet.csv"
    split_file.write_text("group,split\ns7,train\ns8,val\n")
    data_config = tmp_path / "data_config.yaml"
    data_config.write_text(
        yaml.safe_dump(
            {
                "data": {
                    "taxonomy": {
                        "version": taxonomy_pin.version,
                        "manifest_sha256": taxonomy_pin.manifest_sha256,
                    },
                    "datasets": {
                        "coralnet": {
                            "version": pin.version,
                            "manifest_sha256": pin.manifest_sha256,
                            "split_file": str(split_file),
                            **({"train": {"use": ["train"]}} if train else {}),
                        }
                    },
                }
            }
        )
    )
    concepts = tmp_path / "concepts.csv"
    # "other" holds the same rows under a second source, for a schema config that names it.
    table = concept_table()
    pd.concat([table, table.assign(source_dataset_source="other")]).to_csv(concepts, index=False)
    training = yaml.safe_load((REPO_ROOT / "configs/training_config_cbm_final.yaml").read_text())
    training["training"].update(
        concept_mapping_path=str(concepts),
        concept_removal="",
        concept_source_mask="",
        class_subset=["Acropora", "Porites", "Sand"],
        label_roll_up=False,
    )
    training_config = tmp_path / "training_config.yaml"
    training_config.write_text(yaml.safe_dump(training))
    return {
        "cache_dir": cache_dir,
        "data_config": data_config,
        "training_config": training_config,
        "concepts": concepts,
        "pin": pin,
        "taxonomy_pin": taxonomy_pin,
    }


def _run(inputs: dict[str, Any], out: Path, *extra: str) -> None:
    script.main(
        [
            "--cache-dir",
            str(inputs["cache_dir"]),
            "--data-config",
            str(inputs["data_config"]),
            "--schema-config",
            str(inputs["data_config"]),
            "--training-config",
            str(inputs["training_config"]),
            "--out",
            str(out),
            *extra,
        ]
    )


def test_the_build_writes_the_four_files_with_the_pins_in_the_meta(tmp_path):
    inputs = _inputs(tmp_path, _LABELS)
    out = tmp_path / "space"

    _run(inputs, out)

    assert sorted(p.name for p in out.iterdir()) == [
        "classes.csv",
        "concepts.csv",
        "label_space.meta.yaml",
        "source_labels.csv",
    ]
    meta = yaml.safe_load((out / "label_space.meta.yaml").read_text())
    pin: Pin = inputs["pin"]
    taxonomy_pin: Pin = inputs["taxonomy_pin"]
    coralnet = meta["data_version"]["datasets"]["coralnet"]
    assert (coralnet["version"], coralnet["manifest_sha256"]) == (
        pin.version,
        pin.manifest_sha256,
    )
    assert meta["data_version"]["taxonomy"] == {
        "version": taxonomy_pin.version,
        "manifest_sha256": taxonomy_pin.manifest_sha256,
    }
    training = ConfigDict(yaml.safe_load(inputs["training_config"].read_text())["training"])
    assert meta["label_keys_sha1"] == label_keys_sha1(training)
    assert meta["concepts_csv_sha1"] == hashlib.sha1(inputs["concepts"].read_bytes()).hexdigest()
    assert meta["git_sha"] == "abc1234"
    assert meta["acknowledged"] is None
    assert (meta["n_classes"], meta["n_extra_classes"], meta["n_source_labels"]) == (3, 0, 3)
    assert meta["drops"] == {"unmapped": 1}


def test_source_labels_record_a_kept_label_and_a_dropped_one(tmp_path):
    inputs = _inputs(tmp_path, _LABELS)
    out = tmp_path / "space"

    _run(inputs, out)

    rows = pd.read_csv(out / "source_labels.csv", keep_default_na=False).set_index("native_name")
    # Classes sort to 1 Acropora, 2 Porites, 3 Sand; Sand has two train points (the val one is out).
    assert rows.loc["Sand", ["class_id", "drop_reason", "n"]].tolist() == [3, "", 2]
    assert rows.loc["Unknown", ["class_id", "drop_reason", "n"]].tolist() == [0, "unmapped", 0]


def test_a_label_that_trains_a_class_without_a_concept_row_refuses_and_writes_nothing(tmp_path):
    inputs = _inputs(tmp_path, [*_LABELS, _MYSTERY])
    out = tmp_path / "space"

    with pytest.raises(ValueError, match=r"native_name=Mystery kind=missing_concepts"):
        _run(inputs, out)

    assert not out.exists()


def test_an_acknowledged_label_does_not_refuse(tmp_path):
    inputs = _inputs(tmp_path, [*_LABELS, _MYSTERY])
    out = tmp_path / "space"
    acknowledged = tmp_path / "ack.csv"
    acknowledged.write_text(
        "dataset,native_name,kind,reason\ncoralnet,Mystery,missing_concepts,no concept row yet\n"
    )

    _run(inputs, out, "--acknowledge", str(acknowledged))

    meta = yaml.safe_load((out / "label_space.meta.yaml").read_text())
    assert meta["acknowledged_sha1"] == hashlib.sha1(acknowledged.read_bytes()).hexdigest()
    assert (out / "source_labels.csv").is_file()


def test_a_dataset_without_a_train_block_counts_zero_for_every_label(tmp_path):
    inputs = _inputs(tmp_path, _LABELS, train=False)
    out = tmp_path / "space"

    _run(inputs, out)

    rows = pd.read_csv(out / "source_labels.csv", keep_default_na=False)
    assert rows["n"].tolist() == [0, 0, 0]


@pytest.mark.parametrize(
    ("schema_dataset", "status"),
    [("coralnet", "ok"), ("other", "missing")],
    ids=["schema-names-the-dataset", "schema-names-another-dataset"],
)
def test_the_schema_configs_datasets_decide_which_concept_rows_exist(
    tmp_path, schema_dataset, status
):
    # Class_subset is Porites only, so every label drops and a missing row refuses nothing.
    inputs = _inputs(tmp_path, _LABELS)
    training = yaml.safe_load(inputs["training_config"].read_text())
    training["training"]["class_subset"] = ["Porites"]
    inputs["training_config"].write_text(yaml.safe_dump(training))
    data = yaml.safe_load(inputs["data_config"].read_text())
    schema_config = tmp_path / "schema_config.yaml"
    schema_config.write_text(
        yaml.safe_dump(
            {
                "data": {
                    "taxonomy": data["data"]["taxonomy"],
                    "datasets": {schema_dataset: data["data"]["datasets"]["coralnet"]},
                }
            }
        )
    )
    out = tmp_path / "space"

    _run(inputs, out, "--schema-config", str(schema_config))

    rows = pd.read_csv(out / "source_labels.csv", keep_default_na=False).set_index("native_name")
    assert rows.loc["Sand", "concept_status"] == status
