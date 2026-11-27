"""Run-setup helpers: checkpoint reads, weight loading and the resolved config."""

from __future__ import annotations

import pickle
import re
import shutil
from pathlib import Path
from types import SimpleNamespace

import albumentations as A
import pytest
import torch
import yaml

from mermaidseg.dataset_reconciliation.artifact_meta import file_sha1, meta_path
from mermaidseg.dataset_reconciliation.concept_schema import ConceptSchema
from mermaidseg.dataset_reconciliation.label_space import META_YAML
from mermaidseg.io import ConfigDict
from mermaidseg.run_setup import (
    MODEL_CHECKPOINT_KEYS,
    build_registry,
    concept_removal_list,
    load_run_concept_source_mask,
    read_checkpoint,
    write_resolved_config,
)
from tests.mermaidseg.conftest import (
    CORALSCAPES_LABELS,
    PACIFIC_LABELS,
    label_space_reader,
    label_space_section,
)
from tests.mermaidseg.data.conftest import VERSION


def test_resolved_config_yaml_holds_the_overridden_values(tmp_path):
    cfg = ConfigDict(
        {
            "run_name": "r1",
            "training": {"grad_accum_steps": 4, "concept_mapping_path": "/abs/map.csv"},
            "model": {"input_size": (56, 56)},
        }
    )
    write_resolved_config(cfg, tmp_path)
    assert yaml.safe_load((tmp_path / "config.yaml").read_text()) == {
        "run_name": "r1",
        "training": {"grad_accum_steps": 4, "concept_mapping_path": "/abs/map.csv"},
        "model": {"input_size": [56, 56]},
    }


def _save_checkpoint(path: Path, state_dict: dict, config: dict | None = None) -> Path:
    torch.save(
        {"config": config or {}, "model_state_dict": state_dict, "lineage": {"git_sha": "abc"}},
        path,
    )
    return path


def test_read_checkpoint_refuses_a_checkpoint_that_pickles_a_compose_naming_the_file(tmp_path):
    path = _save_checkpoint(
        tmp_path / "ckpt.pt",
        torch.nn.Linear(2, 2).state_dict(),
        {"transform": A.Compose([A.HorizontalFlip()])},
    )
    with pytest.raises(RuntimeError, match="ckpt.pt") as excinfo:
        read_checkpoint(path, MODEL_CHECKPOINT_KEYS)
    assert isinstance(excinfo.value.__cause__, pickle.UnpicklingError)


def test_read_checkpoint_names_the_file_and_the_missing_key(tmp_path):
    path = tmp_path / "ckpt.pt"
    torch.save({"config": {}}, path)
    with pytest.raises(RuntimeError, match="ckpt.pt.*model_state_dict"):
        read_checkpoint(path, MODEL_CHECKPOINT_KEYS)


@pytest.mark.parametrize("path", [None])
def test_concept_removal_list_is_empty_without_a_removal_file(path):
    assert concept_removal_list({"concept_removal": path}, "unused.csv") == []


@pytest.mark.parametrize("path", [None])
def test_no_concept_source_mask_path_loads_no_mask(path):
    training_cfg = ConfigDict({"concept_source_mask": path})
    assert load_run_concept_source_mask(training_cfg, schema=None) is None  # type: ignore[arg-type]


# ---------- build_registry ----------


def _build(run: SimpleNamespace, readers=None, section=None):
    return build_registry(
        run.readers if readers is None else readers,
        run.training_cfg,
        section=run.section if section is None else section,
        schema_config=run.schema_config,
    )


def test_build_registry_refuses_a_run_config_without_a_label_space(label_space_run):
    del label_space_run.training_cfg["label_space"]

    with pytest.raises(ValueError, match=r"no training\.label_space.*retrain it"):
        _build(label_space_run)


def test_build_registry_exits_naming_a_dataset_missing_from_the_concept_schema(
    label_space_run, tmp_path
):
    readers = {
        ("coralnet", "train"): label_space_reader(tmp_path, "coralnet", {1: ("sand", "Sand")})
    }
    with pytest.raises(SystemExit, match=r"\['coralnet'\].*missing from the frozen concept schema"):
        _build(label_space_run, readers=readers)


def test_build_registry_registers_one_reader_per_source(label_space_run, tmp_path):
    val = label_space_reader(tmp_path / "v", "coralscapes_v2", CORALSCAPES_LABELS, split="val")
    readers = {("coralscapes_v2", "val"): val, **label_space_run.readers}

    registry, _ = _build(label_space_run, readers=readers)

    # pacific has no train split, so its only split stands in.
    assert len(registry.datasets) == 2
    assert registry.datasets[0] is label_space_run.readers[("coralscapes_v2", "train")]
    assert registry.datasets[1] is label_space_run.readers[("pacific_labeled_corals", "val")]


def test_build_registry_builds_a_run_over_part_of_the_label_space(label_space_run):
    coralscapes = ("coralscapes_v2", "train")
    readers = {coralscapes: label_space_run.readers[coralscapes]}
    section = label_space_section({"coralscapes_v2": VERSION})

    registry, _ = _build(label_space_run, readers=readers, section=section)

    assert registry.source_to_target.tolist() == [0, 1, 2]


def _other_pin(run: SimpleNamespace) -> None:
    run.section = label_space_section(
        {"coralscapes_v2": "20990101-fffffff", "pacific_labeled_corals": VERSION}
    )


def _other_taxonomy_manifest(run: SimpleNamespace) -> None:
    run.section = run.section.model_copy(
        update={"taxonomy": run.section.taxonomy.model_copy(update={"manifest_sha256": "other"})}
    )


def _other_dataset_manifest(run: SimpleNamespace) -> None:
    datasets = dict(run.section.datasets)
    datasets["coralscapes_v2"] = datasets["coralscapes_v2"].model_copy(
        update={"manifest_sha256": "other"}
    )
    run.section = run.section.model_copy(update={"datasets": datasets})


def _other_subset(run: SimpleNamespace) -> None:
    run.training_cfg.class_subset = ["Acropora"]


def _other_concepts(run: SimpleNamespace) -> None:
    with run.concepts_csv.open("a") as f:
        f.write("\n")


def _unknown_reader_label(run: SimpleNamespace) -> None:
    run.readers[("pacific_labeled_corals", "val")] = label_space_reader(
        run.directory.parent / "extra",
        "pacific_labeled_corals",
        {**PACIFIC_LABELS, 5: ("sand", "Sand")},
        split="val",
        global_offset=2,
    )


def _pin_and_subset(run: SimpleNamespace) -> None:
    _other_pin(run)
    _other_subset(run)


_PIN = rf"coralscapes_v2.*20990101-fffffff.*{VERSION}"
_LABEL_KEYS = r"label_keys_sha1"
_CONCEPTS = r"concept_mapping_path"
_LABEL = r"pacific_labeled_corals.*\[5\]"


@pytest.mark.parametrize(
    ("mutate", "patterns"),
    [
        (_other_pin, [_PIN]),
        (_other_taxonomy_manifest, [r"taxonomy.*other"]),
        (_other_dataset_manifest, [r"dataset coralscapes_v2.*other"]),
        (_other_subset, [_LABEL_KEYS]),
        (_other_concepts, [_CONCEPTS]),
        (_unknown_reader_label, [_LABEL]),
        (_pin_and_subset, [_PIN, _LABEL_KEYS]),
    ],
    ids=[
        "dataset-pin",
        "taxonomy-manifest",
        "dataset-manifest",
        "label-keys",
        "concepts-csv",
        "reader-label",
        "several",
    ],
)
def test_build_registry_refuses_a_stale_label_space_naming_each_mismatch(
    label_space_run, mutate, patterns
):
    mutate(label_space_run)

    with pytest.raises(ValueError) as excinfo:
        _build(label_space_run)

    message = str(excinfo.value)
    for pattern in patterns:
        assert re.search(pattern, message), (pattern, message)


def _repo_removal_list(run: SimpleNamespace) -> None:
    run.training_cfg.concept_removal = str(run.concepts_csv)


def _repo_label_space_gone(run: SimpleNamespace) -> None:
    run.training_cfg.label_space = str(run.directory.parent / "gone")


@pytest.mark.parametrize(
    "mutate",
    [_other_subset, _other_concepts, _repo_removal_list, _repo_label_space_gone],
    ids=["label-keys", "concepts-csv", "removal-list", "repo-label-space-gone"],
)
def test_build_registry_with_a_label_space_dir_ignores_the_repo_inputs(label_space_run, mutate):
    mutate(label_space_run)

    registry, _ = build_registry(
        label_space_run.readers,
        label_space_run.training_cfg,
        section=label_space_run.section,
        schema_config=label_space_run.schema_config,
        label_space_dir=label_space_run.directory,
    )

    assert registry.source_to_target.tolist() == [0, 1, 2, 0]


@pytest.mark.parametrize(
    ("mutate", "pattern"),
    [
        (_other_pin, _PIN),
        (_other_dataset_manifest, r"dataset coralscapes_v2.*other"),
        (_other_taxonomy_manifest, r"taxonomy.*other"),
        (_unknown_reader_label, _LABEL),
    ],
    ids=["dataset-pin", "dataset-manifest", "taxonomy-manifest", "reader-label"],
)
def test_build_registry_with_a_label_space_dir_still_refuses_pin_and_label_mismatches(
    label_space_run, mutate, pattern
):
    mutate(label_space_run)

    with pytest.raises(ValueError, match=pattern):
        build_registry(
            label_space_run.readers,
            label_space_run.training_cfg,
            section=label_space_run.section,
            schema_config=label_space_run.schema_config,
            label_space_dir=label_space_run.directory,
        )


def _label_space_copy_recording_inputs(run: SimpleNamespace, **recorded: str) -> Path:
    """A copy of the run's label space whose meta records the sha1s of a mask and a removal list.

    The cfg names both files and the mask sidecar records the copy's concept table;
    ``recorded`` overrides what the copy's meta records. The repo label space is gone.
    """
    root = run.directory.parent
    copy = root / "bundle_label_space"
    shutil.copytree(run.directory, copy)
    mask, removal = root / "mask.csv", root / "removal.csv"
    mask.write_text("source_id,concept\n")
    removal.write_text("concept\n")
    meta_file = copy / META_YAML
    meta = yaml.safe_load(meta_file.read_text())
    meta["concept_source_mask_sha1"] = file_sha1(mask)
    meta["concept_removal_sha1"] = file_sha1(removal)
    meta.update(recorded)
    meta_file.write_text(yaml.safe_dump(meta))
    meta_path(mask).write_text(
        yaml.safe_dump({"source_ids": ["1"], "concepts_csv_sha1": meta["concepts_csv_sha1"]})
    )
    meta_path(removal).write_text("{}\n")
    cfg = run.training_cfg
    cfg.concept_source_mask, cfg.concept_removal = str(mask), str(removal)
    cfg.label_space = str(root / "gone")
    return copy


@pytest.mark.parametrize(
    ("recorded", "stale_key"),
    [
        ({}, None),
        ({"concept_source_mask_sha1": "stale"}, "concept_source_mask_sha1"),
        ({"concept_removal_sha1": "stale"}, "concept_removal_sha1"),
    ],
    ids=["matches", "mask-differs", "removal-differs"],
)
def test_source_mask_with_a_label_space_dir_checks_the_mask_and_removal_against_its_meta(
    label_space_run, recorded, stale_key
):
    copy = _label_space_copy_recording_inputs(label_space_run, **recorded)
    schema = ConceptSchema.from_csv(
        label_space_run.concepts_csv, sources={"coralscapes_v2", "pacific_labeled_corals"}
    )

    def load():
        return load_run_concept_source_mask(
            label_space_run.training_cfg, schema, label_space_dir=copy
        )

    if stale_key is None:
        assert load().path == label_space_run.training_cfg.concept_source_mask
        return
    with pytest.raises(ValueError) as excinfo:
        load()
    assert f"{stale_key} stale" in str(excinfo.value)
