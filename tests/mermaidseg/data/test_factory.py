"""build_datasets: readers, transforms, groups, resize and image ids from a DataSection."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pyarrow as pa
import pytest

from mermaidseg.data import DenseDataset, PointDataset
from mermaidseg.data.config import DataSection
from mermaidseg.data.factory import (
    build_datasets,
    load_benthic_hierarchy,
    open_dataset_version,
    source_offsets,
)
from segdata_spec.errors import NotSynced
from segdata_spec.manifest import Pin
from tests.mermaidseg.data.conftest import ImageSpec, open_written, write_taxonomy, write_version

LABELS = {5: "five"}


def _section(pin: Pin, **entry: Any) -> DataSection:
    cfg = {"version": pin.version, "manifest_sha256": pin.manifest_sha256, **entry}
    return DataSection.model_validate({"datasets": {pin.name: cfg}})


def _points(tmp_path: Path, images: list[ImageSpec], points: list[Any]) -> tuple[Path, Pin]:
    cache_dir, _, pin = write_version(
        tmp_path, kind="points", images=images, labels=LABELS, points=points
    )
    return cache_dir, pin


def _build(
    section: DataSection,
    cache_dir: Path,
    tmp_path: Path,
    *,
    padding: int = 0,
    input_size: tuple[int, int] = (4, 6),
    image_ids: Any = None,
) -> Any:
    return build_datasets(
        section,
        cache_dir,
        repo_root=tmp_path,
        padding=padding,
        input_size=input_size,
        image_ids=image_ids,
    )


def test_point_version_gives_a_point_dataset_with_padding_and_annotators(tmp_path: Path) -> None:
    cache_dir, pin = _points(
        tmp_path, [ImageSpec("i1")], [("i1", 1, 1, 5, "a"), ("i1", 2, 2, 5, "b")]
    )
    section = _section(pin, val={"groups": "all", "annotators": ["b"]})
    ds = _build(section, cache_dir, tmp_path, padding=2)[("ds", "val")]
    assert isinstance(ds, PointDataset)
    assert ds.padding == 2
    assert list(ds.points["annotator"]) == ["b"]
    assert ds.split == "val"


def test_dense_version_gives_a_dense_dataset(tmp_path: Path) -> None:
    mask = np.zeros((8, 8), np.uint8)
    mask[0, 0] = 5
    cache_dir, _, pin = write_version(
        tmp_path, kind="dense", images=[ImageSpec("i1", mask=mask)], labels=LABELS
    )
    ds = _build(_section(pin, test={"groups": "all"}), cache_dir, tmp_path)[("ds", "test")]
    assert isinstance(ds, DenseDataset)


def test_annotators_on_a_dense_version_raise(tmp_path: Path) -> None:
    cache_dir, _, pin = write_version(
        tmp_path,
        kind="dense",
        images=[ImageSpec("i1", mask=np.zeros((8, 8), np.uint8))],
        labels=LABELS,
    )
    section = _section(pin, val={"groups": "all", "annotators": ["a"]})
    with pytest.raises(ValueError, match="annotators"):
        _build(section, cache_dir, tmp_path)


def test_only_defined_blocks_are_built_and_keyed_by_dataset_and_split(tmp_path: Path) -> None:
    cache_dir, pin = _points(tmp_path, [ImageSpec("i1")], [("i1", 1, 1, 5)])
    section = _section(pin, train={"groups": "all"}, test={"groups": "all"})
    assert sorted(_build(section, cache_dir, tmp_path)) == [("ds", "test"), ("ds", "train")]


def test_train_gets_the_compiled_transform_and_eval_splits_the_resize(tmp_path: Path) -> None:
    cache_dir, pin = _points(tmp_path, [ImageSpec("i1")], [("i1", 1, 1, 5)])
    section = _section(
        pin,
        train={"groups": "all", "transform": {"SmallestMaxSize": {"max_size": 16}}},
        val={"groups": "all"},
        test={"groups": "all"},
    )
    built = _build(section, cache_dir, tmp_path, input_size=(4, 6))
    assert built[("ds", "train")][0][0].shape == (3, 16, 16)
    assert built[("ds", "val")][0][0].shape == (3, 4, 6)
    assert built[("ds", "test")][0][0].shape == (3, 4, 6)
    assert built[("ds", "val")][0][0].dtype == np.float32


def test_image_ids_subset_a_dataset_and_split(tmp_path: Path) -> None:
    cache_dir, pin = _points(
        tmp_path, [ImageSpec("i1"), ImageSpec("i2")], [("i1", 1, 1, 5), ("i2", 1, 1, 5)]
    )
    section = _section(pin, train={"groups": "all"}, val={"groups": "all"})
    built = _build(section, cache_dir, tmp_path, image_ids={("ds", "val"): ["i2"]})
    assert list(built[("ds", "val")].images["image_id"]) == ["i2"]
    assert len(built[("ds", "train")]) == 2


def _labelled_version(tmp_path: Path, name: str, labels: dict[int, str]) -> Any:
    first = min(labels)
    return open_written(
        tmp_path / name,
        kind="points",
        images=[ImageSpec("i1")],
        labels=labels,
        points=[("i1", 1, 1, first)],
        name=name,
    )


def test_source_offsets_step_by_the_previous_sources_max_label_id(tmp_path: Path) -> None:
    # A holds two labels, so a count-based offset would give B 2 rather than 100.
    versions = {
        "a": _labelled_version(tmp_path, "a", {58: "x", 100: "y"}),
        "b": _labelled_version(tmp_path, "b", {1: "p", 2: "q"}),
    }
    assert source_offsets(versions) == {"a": 0, "b": 100}


@pytest.mark.parametrize("bad_id", [0])
def test_source_offsets_reject_a_label_id_below_one_naming_the_source(
    tmp_path: Path, bad_id: int
) -> None:
    versions = {
        "a": _labelled_version(tmp_path, "a", {1: "x"}),
        "b": _labelled_version(tmp_path, "b", {bad_id: "zero", 3: "y"}),
    }
    with pytest.raises(ValueError, match="'b'"):
        source_offsets(versions)


def test_every_split_of_a_source_shares_its_offset(tmp_path: Path) -> None:
    _, _, pin_a = write_version(
        tmp_path,
        kind="points",
        images=[ImageSpec("a1", group="g1"), ImageSpec("a2", group="g2")],
        labels={58: "x", 100: "y"},
        points=[("a1", 1, 1, 58), ("a2", 1, 1, 100)],
        name="a",
    )
    cache_dir, _, pin_b = write_version(
        tmp_path,
        kind="points",
        images=[ImageSpec("b1")],
        labels={1: "p"},
        points=[("b1", 1, 1, 1)],
        name="b",
    )
    section = DataSection.model_validate(
        {
            "datasets": {
                "a": {
                    "version": pin_a.version,
                    "manifest_sha256": pin_a.manifest_sha256,
                    "train": {"groups": ["g1"]},
                    "test": {"groups": ["g2"]},
                },
                "b": {
                    "version": pin_b.version,
                    "manifest_sha256": pin_b.manifest_sha256,
                    "val": {"groups": "all"},
                },
            }
        }
    )
    built = _build(section, cache_dir, tmp_path)
    offsets = {key: ds.global_offset for key, ds in built.items()}
    assert offsets == {("a", "train"): 0, ("a", "test"): 0, ("b", "val"): 100}


def test_open_dataset_version_requires_the_entrys_resize_synced(tmp_path: Path) -> None:
    cache_dir, pin = _points(tmp_path, [ImageSpec("i1")], [("i1", 1, 1, 5)])
    section = _section(pin, resize={"max_edge": 4}, val={"groups": "all"})
    with pytest.raises(NotSynced):
        open_dataset_version(section, "ds", cache_dir)


def _taxonomy_section(pin: Pin, manifest_sha256: str) -> DataSection:
    return DataSection.model_validate(
        {
            "datasets": {},
            "taxonomy": {"version": pin.version, "manifest_sha256": manifest_sha256},
        }
    )


def _write_hard_coral_taxonomy(tmp_path: Path) -> tuple[Path, Pin]:
    cache_dir = tmp_path / "cache"
    table = pa.table(
        {"id": ["1", "2"], "name": ["Hard coral", "Acropora"], "parent_id": [None, "1"]}
    )
    return cache_dir, write_taxonomy(cache_dir, tmp_path, table)


def test_pinned_taxonomy_gives_the_lower_cased_child_to_parent_hierarchy(tmp_path: Path) -> None:
    cache_dir, pin = _write_hard_coral_taxonomy(tmp_path)
    section = _taxonomy_section(pin, pin.manifest_sha256)
    assert load_benthic_hierarchy(section, cache_dir) == {
        "hard coral": None,
        "acropora": "hard coral",
    }
