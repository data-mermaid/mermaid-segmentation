from __future__ import annotations

import hashlib
import json
from collections import Counter
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import pytest
from PIL import Image

import mermaidseg.data.parity as parity_module
import mermaidseg.dataset_reconciliation.registry as registry_module
from mermaidseg.data.config import DATASET_BUCKET
from mermaidseg.data.parity import (
    DenseParityReport,
    ParityReport,
    VocabMismatch,
    compare,
    compare_dense,
    format_dense_report,
    format_report,
    new_dense_counts,
    new_points,
    old_dense_counts,
    old_dense_rows,
    old_points,
    run_dense_parity,
    run_parity,
)
from segdata_spec.errors import ChecksumMismatch, PinMismatch
from segdata_spec.manifest import FileEntry, Manifest, Pin, dump_manifest, manifest_sha256
from segdata_spec.read import open_version
from segdata_spec.resize import ResizeSpec
from tests.mermaidseg.data.conftest import (
    BUCKET,
    IMAGE_BUCKET,
    IMAGE_PREFIX,
    ImageSpec,
    open_written,
    publish_points,
    write_version,
)

_COLUMNS = ["image_id", "row", "col", "native_name", "mapped"]


def _frame(rows: Sequence[tuple[Any, int, int, str, str | None]]) -> pd.DataFrame:
    return pd.DataFrame(rows, columns=_COLUMNS)


_BASE = [("a", 1, 1, "x", "hard"), ("a", 2, 2, "y", "soft"), ("b", 3, 3, "x", "hard")]
_GROUPS = {"a": "g1", "b": "g1"}


def test_compare_equal_frames_is_clean() -> None:
    report = compare(_frame(_BASE), _frame(_BASE), _GROUPS)

    assert report.ok
    assert report == ParityReport(points_matched=3, groups_compared=1)


@pytest.mark.parametrize(
    ("new_rows", "field", "expected"),
    [
        (
            [("a", 1, 9, "x", "hard"), *_BASE[1:]],
            "old_only_points",
            Counter({("a", 1, 1, "hard"): 1}),
        ),
        (
            [("a", 1, 9, "x", "hard"), *_BASE[1:]],
            "new_only_points",
            Counter({("a", 1, 9, "hard"): 1}),
        ),
        (
            [("a", 1, 1, "x", "other"), *_BASE[1:]],
            "new_only_points",
            Counter({("a", 1, 1, "other"): 1}),
        ),
        (
            [*_BASE, ("a", 1, 1, "x", "hard")],
            "new_only_points",
            Counter({("a", 1, 1, "hard"): 1}),
        ),
        (_BASE[:2], "old_only_images", ["b"]),
        ([*_BASE, ("c", 5, 5, "x", "hard")], "new_only_images", ["c"]),
    ],
    ids=["moved-old", "moved-new", "relabel", "duplicate", "old-only-image", "new-only-image"],
)
def test_compare_reports_each_difference_in_its_field(
    new_rows: list[tuple[Any, int, int, str, str | None]], field: str, expected: object
) -> None:
    groups = {**_GROUPS, "c": "g1"}

    report = compare(_frame(_BASE), _frame(new_rows), groups)

    assert not report.ok
    assert getattr(report, field) == expected


def test_compare_lists_new_groups_without_old_images_apart_from_the_diff() -> None:
    groups = {"a": "g1", "b": "g1", "z": "g2", "y": "g2", "w": "g3"}
    new = _frame(
        [*_BASE, ("z", 8, 8, "x", "hard"), ("z", 9, 9, "x", "hard"), ("y", 1, 1, "x", "soft")]
        + [("w", 2, 2, "y", "soft")]
    )

    report = compare(_frame(_BASE), new, groups)

    assert report.ok
    assert report.excluded_groups == {"g2": (2, 3), "g3": (1, 1)}
    assert report.groups_compared == 1


def test_compare_ignores_new_groups_without_old_images() -> None:
    groups = {"a": "g1", "b": "g1", "extra": "g1", "z": "g2"}
    new = _frame([*_BASE, ("extra", 7, 7, "x", "hard"), ("z", 8, 8, "x", "hard")])

    report = compare(_frame(_BASE), new, groups)

    assert report.new_only_images == ["extra"]


def test_compare_reports_label_count_diffs() -> None:
    old = _frame([("a", i, i, "x", "m") for i in range(3)])
    new = _frame([("a", i, i, "x", "m") for i in range(2)])

    report = compare(old, new, {"a": "g1"})

    assert report.native_diff == {"x": (3, 2)}
    assert report.mapped_diff == {"m": (3, 2)}


def test_format_report_names_the_verdict() -> None:
    assert format_report(ParityReport()).endswith("PARITY OK")
    failed = format_report(ParityReport(old_only_images=["a"], native_diff={"x": (2, 1)}))
    assert failed.endswith("PARITY FAILED")
    assert "old-only images: 1" in failed
    assert "x: old 2, new 1" in failed


def test_format_report_states_what_was_compared_and_the_excluded_groups() -> None:
    text = format_report(
        ParityReport(points_matched=40, groups_compared=3, excluded_groups={"g9": (2, 7)})
    )

    lines = text.splitlines()
    assert "points matched: 40" in lines
    assert "groups compared: 3" in lines
    assert "new groups excluded: 1" in lines
    assert "  g9: 2 images, 7 points" in lines
    assert text.endswith("PARITY OK")


class _StubOld:
    SOURCE_NAME = "catlin_seaview"

    def __init__(self, annotations: pd.DataFrame | None = None) -> None:
        self.df_annotations = (
            annotations
            if annotations is not None
            else pd.DataFrame(
                {
                    "image_id": [10, 10, 11],
                    "row": [1, 2, 3],
                    "col": [4, 5, 6],
                    "source_label_name": ["Hard", "Soft", "Hard"],
                }
            )
        )
        self.source_id2name = {1: "Hard", 2: "Soft"}
        self.source_name2id = {"Hard": 1, "Soft": 2}
        self.offset: int | None = None

    def set_global_offset(self, offset: int) -> None:
        self.offset = offset


def test_old_points_maps_through_the_registry_and_leaves_unmapped_none(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setitem(
        registry_module._REMOTE_FETCHERS,
        "catlin_seaview",
        lambda: {"Hard": "Hard Coral", "Soft": None},
    )

    points = old_points(_StubOld())

    assert list(points.columns) == _COLUMNS
    assert points["image_id"].tolist() == ["10", "10", "11"]
    assert points["native_name"].tolist() == ["hard", "soft", "hard"]
    assert points["mapped"].tolist() == ["hard coral", None, "hard coral"]


_LABELS = [
    {"label_id": 1, "name": "Acropora", "mermaid_benthic_attribute": "Hard Coral"},
    {"label_id": 2, "name": "Rubble", "mermaid_benthic_attribute": None},
]
_IMAGES = [ImageSpec("i1"), ImageSpec("i2")]


def test_new_points_lowercases_names_and_attribute_and_keeps_null_none(tmp_path: Path) -> None:
    version = open_written(
        tmp_path,
        kind="points",
        images=_IMAGES,
        labels=_LABELS,
        points=[("i1", 1, 1, 1), ("i1", 2, 2, 2), ("i2", 3, 3, 1)],
    )

    assert version.points is not None
    points = new_points(version, version.points.to_pandas())

    assert list(points.columns) == _COLUMNS
    assert points["native_name"].tolist() == ["acropora", "rubble", "acropora"]
    assert points["mapped"].tolist() == ["hard coral", None, "hard coral"]
    assert points["image_id"].tolist() == ["i1", "i1", "i2"]


def _old_stub(rows: list[tuple[str, int, int]]) -> _StubOld:
    return _StubOld(
        pd.DataFrame(
            {
                "image_id": [r[0] for r in rows],
                "row": [r[1] for r in rows],
                "col": [r[2] for r in rows],
                "source_label_name": ["Acropora"] * len(rows),
            }
        )
    )


def _run(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    old_rows: list[tuple[str, int, int]],
    *,
    images: Sequence[ImageSpec] = _IMAGES,
    points: Sequence[tuple[str, int, int, int]] = (("i1", 1, 1, 1), ("i2", 3, 3, 1)),
    resize: ResizeSpec | None = None,
) -> tuple[ParityReport, Path]:
    fake, pin = publish_points(tmp_path / "pub", images=images, labels=_LABELS, points=points)
    monkeypatch.setitem(
        registry_module._REMOTE_FETCHERS, "catlin_seaview", lambda: {"Acropora": "Hard Coral"}
    )
    stub = _old_stub(old_rows)
    stub.source_name2id = {"Acropora": 1}
    stub.source_id2name = {1: "Acropora"}
    monkeypatch.setattr(parity_module, "build_datasets", lambda *a, **k: {("ds", "train"): stub})
    cache_dir = tmp_path / "cache"
    report = run_parity(
        {},
        dataset="ds",
        split="train",
        version=pin.version,
        manifest_sha256=pin.manifest_sha256,
        cache_dir=cache_dir,
        client=fake,
        annotators=None,
        resize=resize,
    )
    assert isinstance(report, ParityReport)
    return report, cache_dir


def test_run_parity_passes_on_matching_points_without_downloading_images(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    report, cache_dir = _run(monkeypatch, tmp_path, [("i1", 1, 1), ("i2", 3, 3)])

    assert report.ok
    assert {p.suffix for p in cache_dir.rglob("*") if p.is_file()} == {".json", ".parquet"}


def test_run_parity_reports_a_point_moved_by_one_row(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    report, _ = _run(monkeypatch, tmp_path, [("i1", 2, 1), ("i2", 3, 3)])

    assert not report.ok
    assert report.old_only_points == Counter({("i1", 2, 1, "hard coral"): 1})
    assert report.new_only_points == Counter({("i1", 1, 1, "hard coral"): 1})


class _NoClient:
    """A client whose every attribute access fails, proving S3 is never reached."""

    def __getattribute__(self, name: str) -> Any:
        raise AssertionError(f"client.{name} was used")


def test_run_parity_without_sync_reads_the_cache_and_never_uses_the_client(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    cache_dir, _, pin = write_version(
        tmp_path,
        kind="points",
        images=_IMAGES,
        labels=_LABELS,
        points=[("i1", 1, 1, 1), ("i2", 3, 3, 1)],
    )
    monkeypatch.setitem(
        registry_module._REMOTE_FETCHERS, "catlin_seaview", lambda: {"Acropora": "Hard Coral"}
    )
    stub = _old_stub([("i1", 1, 1), ("i2", 3, 3)])
    stub.source_name2id = {"Acropora": 1}
    stub.source_id2name = {1: "Acropora"}
    monkeypatch.setattr(parity_module, "build_datasets", lambda *a, **k: {("ds", "train"): stub})

    report = run_parity(
        {},
        dataset="ds",
        split="train",
        version=pin.version,
        manifest_sha256=pin.manifest_sha256,
        cache_dir=cache_dir,
        client=_NoClient(),
        annotators=None,
        sync=False,
    )

    assert report.ok


_DEFAULT_CFG: dict[str, Any] = {
    "local_cache_dir": "/cache",
    "default": {"train": {"transform": "t"}, "val": {"transform": "v"}},
    "ds": {"train": {"sites": ["a"]}},
}


def _strict_builder(seen: list[Any], built: Any) -> Any:
    """A ``build_datasets`` stand-in that rejects ``default`` the way the real one does."""

    def build(data_cfg: Any, **_: Any) -> Any:
        if "default" in data_cfg:
            raise ValueError("unknown data keys ['default']")
        seen.append(data_cfg)
        return {("ds", "train"): built}

    return build


def _assert_default_expanded(seen: list[Any]) -> None:
    (cfg,) = seen
    assert cfg["ds"]["train"] == {"transform": "t", "sites": ["a"]}
    assert cfg["ds"]["val"] == {"transform": "v"}
    assert cfg["local_cache_dir"] == "/cache"


def test_run_parity_builds_the_old_side_from_the_expanded_data_config(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    cache_dir, _, pin = write_version(
        tmp_path,
        kind="points",
        images=_IMAGES,
        labels=_LABELS,
        points=[("i1", 1, 1, 1), ("i2", 3, 3, 1)],
    )
    monkeypatch.setitem(
        registry_module._REMOTE_FETCHERS, "catlin_seaview", lambda: {"Acropora": "Hard Coral"}
    )
    stub = _old_stub([("i1", 1, 1), ("i2", 3, 3)])
    stub.source_name2id = {"Acropora": 1}
    stub.source_id2name = {1: "Acropora"}
    seen: list[Any] = []
    monkeypatch.setattr(parity_module, "build_datasets", _strict_builder(seen, stub))

    run_parity(
        _DEFAULT_CFG,
        dataset="ds",
        split="train",
        version=pin.version,
        manifest_sha256=pin.manifest_sha256,
        cache_dir=cache_dir,
        client=_NoClient(),
        annotators=None,
        sync=False,
    )

    _assert_default_expanded(seen)
    assert "default" in _DEFAULT_CFG


def _stage_points(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> tuple[Path, Pin, _StubOld]:
    cache_dir, bucket, pin = write_version(
        tmp_path,
        kind="points",
        images=_IMAGES,
        labels=_LABELS,
        points=[("i1", 1, 1, 1), ("i2", 3, 3, 1)],
    )
    monkeypatch.setitem(
        registry_module._REMOTE_FETCHERS, "catlin_seaview", lambda: {"Acropora": "Hard Coral"}
    )
    stub = _old_stub([("i1", 1, 1), ("i2", 3, 3)])
    stub.source_name2id = {"Acropora": 1}
    stub.source_id2name = {1: "Acropora"}
    return cache_dir, pin, stub


def test_run_parity_expands_the_default_block_before_building(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    cache_dir, pin, stub = _stage_points(monkeypatch, tmp_path)
    cfg_seen: list[Any] = []

    def _build(cfg: Any, **_: Any) -> dict[tuple[str, str], Any]:
        cfg_seen.append(cfg)
        return {("ds", "train"): stub}

    monkeypatch.setattr(parity_module, "build_datasets", _build)

    run_parity(
        {"default": {"train": {"k": 1}}, "ds": {"val": {"v": 2}}},
        dataset="ds",
        split="train",
        version=pin.version,
        manifest_sha256=pin.manifest_sha256,
        cache_dir=cache_dir,
        client=_NoClient(),
        annotators=None,
        sync=False,
    )

    assert cfg_seen == [{"ds": {"train": {"k": 1}, "val": {"v": 2}}}]


def test_run_parity_without_sync_checks_the_staged_manifest_against_the_pin(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    cache_dir, pin, stub = _stage_points(monkeypatch, tmp_path)
    monkeypatch.setattr(parity_module, "build_datasets", lambda *a, **k: {("ds", "train"): stub})
    args: dict[str, Any] = {
        "dataset": "ds",
        "split": "train",
        "version": pin.version,
        "cache_dir": cache_dir,
        "client": _NoClient(),
        "annotators": None,
        "sync": False,
    }

    assert run_parity({}, manifest_sha256=pin.manifest_sha256, **args).ok
    with pytest.raises(PinMismatch):
        run_parity({}, manifest_sha256="0" * 64, **args)


# --- dense ---------------------------------------------------------------------------------

_RNG = np.random.default_rng(0)
_PIXELS = {name: _RNG.integers(0, 255, (4, 4, 3), dtype=np.uint8) for name in ("p1", "p2", "p3")}
_MASK1 = np.array([[1, 1, 2, 2], [1, 1, 2, 2], [3, 3, 0, 0], [3, 3, 0, 0]], dtype=np.uint8)
_MASK2 = np.array([[2, 2, 2, 2], [1, 1, 1, 1], [3, 3, 3, 3], [0, 0, 0, 0]], dtype=np.uint8)
_NAMES = {1: "a", 2: "b", 3: "c"}
_ATTRIBUTES = {"a": "Hard Coral", "b": "Rubble", "c": None}


class _StubDense:
    SOURCE_NAME = "catlin_seaview"
    class_subset: list[str] | None = None

    def __init__(
        self,
        rows: list[tuple[str, np.ndarray, np.ndarray]],
        mapping: dict[str, str | None] | None = None,
    ) -> None:
        self.dataset = [{"filename": f, "image": i, "label": m} for f, i, m in rows]
        self.split = "train"
        self.class_table = [{"id": i, "name": n} for i, n in _NAMES.items()]
        self.source_id2name = dict(_NAMES)
        self.source_name2id = {n: i for i, n in _NAMES.items()}
        self.mapping = _ATTRIBUTES if mapping is None else mapping

    def set_global_offset(self, offset: int) -> None:
        self.offset = offset


class _StubUcsd(_StubDense):
    def __init__(self, *args: Any, whitelist: set[str] | None = None, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self.whitelist = whitelist

    def _resolved_original_splits(self) -> set[str] | None:
        return self.whitelist


def _set_attrs(cache_dir: Path, pin: Pin, attrs: Mapping[str, str]) -> Pin:
    """Rewrite ``attrs`` of the version's images and re-pin the manifest."""
    prefix = cache_dir / DATASET_BUCKET / "datasets" / pin.name / pin.version
    table = pq.read_table(prefix / "images.parquet")
    column = pa.array([attrs.get(i) for i in table["image_id"].to_pylist()], pa.string())
    table = table.set_column(table.schema.get_field_index("attrs"), "attrs", column)
    pq.write_table(table, prefix / "images.parquet")
    data = (prefix / "images.parquet").read_bytes()
    manifest = Manifest.model_validate_json((prefix / "manifest.json").read_bytes())
    files = {
        **manifest.files,
        "images.parquet": FileEntry(
            sha256=hashlib.sha256(data).hexdigest(), bytes=len(data), rows=table.num_rows
        ),
    }
    manifest_bytes = dump_manifest(manifest.model_copy(update={"files": files}))
    (prefix / "manifest.json").write_bytes(manifest_bytes)
    return Pin(name=pin.name, version=pin.version, manifest_sha256=manifest_sha256(manifest_bytes))


def _old_rows() -> list[tuple[str, np.ndarray, np.ndarray]]:
    return [("o1", _PIXELS["p1"], _MASK1), ("o2", _PIXELS["p2"], _MASK2)]


def _run_dense(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    old: _StubDense,
    new_images: Sequence[ImageSpec],
    new_labels: Sequence[dict[str, Any]] | None = None,
    probe: Any = lambda: None,
    attrs: Mapping[str, str] | None = None,
) -> DenseParityReport:
    labels = new_labels or [
        {"label_id": i, "name": n, "mermaid_benthic_attribute": _ATTRIBUTES[n]}
        for i, n in _NAMES.items()
    ]
    cache_dir, _, pin = write_version(tmp_path, kind="dense", images=new_images, labels=labels)
    if attrs is not None:
        pin = _set_attrs(cache_dir, pin, attrs)
    monkeypatch.setitem(
        registry_module._REMOTE_FETCHERS, "catlin_seaview", lambda: dict(old.mapping)
    )
    monkeypatch.setattr(parity_module, "build_datasets", lambda *a, **k: {("ds", "train"): old})
    return run_dense_parity(
        {},
        dataset="ds",
        split="train",
        version=pin.version,
        manifest_sha256=pin.manifest_sha256,
        cache_dir=cache_dir,
        client=_NoClient(),
        sync=False,
        hf_revision_probe=probe,
    )


def _new_images(mask1: np.ndarray = _MASK1) -> list[ImageSpec]:
    return [
        ImageSpec("n1", group="train", mask=mask1, pixels=_PIXELS["p1"]),
        ImageSpec("n2", group="train", mask=_MASK2, pixels=_PIXELS["p2"]),
    ]


def test_dense_identical_sides_are_ok(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    report = _run_dense(monkeypatch, tmp_path, _StubDense(_old_rows()), _new_images())

    assert report.ok
    assert report.matched == 2
    assert report == DenseParityReport(matched=2)


def test_dense_one_differing_mask_pixel_names_the_image_and_shifts_label_counts(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    changed = _MASK1.copy()
    changed[0, 0] = 2  # label 1 -> 2

    report = _run_dense(monkeypatch, tmp_path, _StubDense(_old_rows()), _new_images(changed))

    assert not report.ok
    assert report.mask_diffs == {"n1": 1}
    assert report.label_pixel_diff == {1: (8, 7), 2: (8, 9)}


def test_dense_unmatched_images_are_listed_on_each_side(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    report = _run_dense(
        monkeypatch,
        tmp_path,
        _StubDense([_old_rows()[0], ("o3", _PIXELS["p3"], _MASK2)]),
        _new_images(),
    )

    assert not report.ok
    assert report.matched == 1
    assert report.old_only == ["o3"]
    assert report.new_only == ["n2"]


def test_dense_mapping_difference_is_a_vocabulary_mismatch(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    old = _StubDense(_old_rows(), mapping={"a": "Hard Coral", "b": "Sand", "c": None})

    report = _run_dense(monkeypatch, tmp_path, old, _new_images())

    assert not report.ok
    assert report.vocab_mismatches == [
        VocabMismatch(
            label_id=2, old_name="b", new_name="b", old_mapped="sand", new_mapped="rubble"
        )
    ]


def test_dense_name_difference_alone_is_a_vocabulary_mismatch(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    labels = [
        {"label_id": 1, "name": "a", "mermaid_benthic_attribute": "Hard Coral"},
        {"label_id": 2, "name": "b2", "mermaid_benthic_attribute": "Rubble"},
        {"label_id": 3, "name": "c", "mermaid_benthic_attribute": None},
    ]

    report = _run_dense(
        monkeypatch, tmp_path, _StubDense(_old_rows()), _new_images(), new_labels=labels
    )

    assert not report.ok
    assert report.vocab_mismatches == [
        VocabMismatch(
            label_id=2, old_name="b", new_name="b2", old_mapped="rubble", new_mapped="rubble"
        )
    ]


def _split_attrs(**splits: str) -> dict[str, str]:
    return {i: json.dumps({"original_split": split}) for i, split in splits.items()}


def test_dense_original_split_whitelist_drops_new_images_the_old_loader_dropped(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    old = _StubUcsd(_old_rows()[:1], whitelist={"train"})

    report = _run_dense(
        monkeypatch, tmp_path, old, _new_images(), attrs=_split_attrs(n1="train", n2="test")
    )

    assert report.new_only == []
    assert report.ok


def test_dense_without_an_original_split_filter_keeps_every_new_image(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    old = _StubUcsd(_old_rows()[:1], whitelist=None)

    report = _run_dense(
        monkeypatch, tmp_path, old, _new_images(), attrs=_split_attrs(n1="train", n2="test")
    )

    assert report.new_only == ["n2"]


def _subset_old(mapping: dict[str, str | None] | None = None) -> _StubDense:
    old = _StubDense(_old_rows(), mapping=mapping)
    old.class_subset = ["a"]
    old.source_id2name = {1: "a"}
    old.source_name2id = {"a": 1}
    return old


def test_dense_class_subset_still_reports_a_mismatch_inside_the_subset(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    old = _subset_old(mapping={"a": "Sand", "b": "Rubble", "c": None})

    report = _run_dense(monkeypatch, tmp_path, old, _new_images())

    assert report.vocab_mismatches == [
        VocabMismatch(
            label_id=1, old_name="a", new_name="a", old_mapped="sand", new_mapped="hard coral"
        )
    ]


def test_old_dense_rows_name_unnamed_rows_by_hf_split_and_index_within_it(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    image, mask = _PIXELS["p1"], _MASK1
    old: Any = _StubDense([])
    old.split = ["train", "validation"]
    old.hf_repo_id = "repo"
    old.dataset = [{"image": image, "label": mask} for _ in range(3)]
    monkeypatch.setattr(
        parity_module,
        "load_dataset",
        lambda repo: {"train": [0, 1], "validation": [0]} if repo == "repo" else {},
    )

    ids = [row_id for row_id, _, _ in old_dense_rows(old)]

    assert ids == ["train row 0", "train row 1", "validation row 0"]


def test_old_dense_rows_keep_the_filename_when_the_row_has_one() -> None:
    old = _StubDense([("f1", _PIXELS["p1"], _MASK1)])

    assert [row_id for row_id, _, _ in old_dense_rows(old)] == ["f1"]


def test_dense_records_the_probe_result(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    report = _run_dense(
        monkeypatch, tmp_path, _StubDense(_old_rows()), _new_images(), probe=lambda: "abc123"
    )

    assert report.hf_revision_at_run == "abc123"
    assert report.ok


def test_format_dense_report_lists_each_difference() -> None:
    report = DenseParityReport(
        matched=1,
        old_only=["o3"],
        new_only=["n9"],
        mask_diffs={"n1": 4},
        label_pixel_diff={1: (8, 7)},
        vocab_mismatches=[VocabMismatch(2, "b", "b", "sand", "rubble")],
        hf_revision_at_run="abc123",
    )

    text = format_dense_report(report)

    assert text.endswith("PARITY FAILED")
    for expected in (
        "matched images: 1",
        "o3",
        "n9",
        "n1: 4 pixels",
        "1: old 8, new 7",
        "id 2",
        "sand",
        "rubble",
        "abc123",
    ):
        assert expected in text
    assert format_dense_report(DenseParityReport()).endswith("PARITY OK")


def test_run_dense_parity_builds_the_old_side_from_the_expanded_data_config(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    old = _StubDense(_old_rows())
    cache_dir, _, pin = write_version(
        tmp_path,
        kind="dense",
        images=_new_images(),
        labels=[
            {"label_id": i, "name": n, "mermaid_benthic_attribute": _ATTRIBUTES[n]}
            for i, n in _NAMES.items()
        ],
    )
    monkeypatch.setitem(
        registry_module._REMOTE_FETCHERS, "catlin_seaview", lambda: dict(old.mapping)
    )
    seen: list[Any] = []
    monkeypatch.setattr(parity_module, "build_datasets", _strict_builder(seen, old))

    run_dense_parity(
        _DEFAULT_CFG,
        dataset="ds",
        split="train",
        version=pin.version,
        manifest_sha256=pin.manifest_sha256,
        cache_dir=cache_dir,
        client=_NoClient(),
        sync=False,
    )

    _assert_default_expanded(seen)


# --- dense: Benthos -----------------------------------------------------------------------

_TILE_MASK_A = np.array([[0, 1], [2, 2]], np.uint8)
_TILE_MASK_B = np.array([[3, 3], [3, 0]], np.uint8)
_DENSE_LABELS = [
    {"label_id": 1, "name": "Coral", "mermaid_benthic_attribute": "Hard Coral"},
    {"label_id": 2, "name": "Sand", "mermaid_benthic_attribute": None},
    {"label_id": 3, "name": "Turf", "mermaid_benthic_attribute": "Turf Algae"},
]
_BY_MAPPING = {"Coral": "Hard Coral", "Sand": None, "Turf": "Turf Algae"}


def _dense_version(tmp_path: Path) -> tuple[Path, str, Pin]:
    return write_version(
        tmp_path,
        kind="dense",
        images=[
            ImageSpec("RS24_0_5", group="RS24", mask=_TILE_MASK_A, source_name="0_5.png"),
            ImageSpec("RS24_7_9", group="RS24", mask=_TILE_MASK_B, source_name="7_9.png"),
        ],
        labels=_DENSE_LABELS,
    )


def test_new_dense_counts_keys_tiles_by_group_and_source_name_stem(tmp_path: Path) -> None:
    cache_dir, bucket, pin = _dense_version(tmp_path)

    counts, mapped = new_dense_counts(open_version(cache_dir, bucket, pin))

    assert counts == {
        ("RS24", "0_5"): Counter({1: 1, 2: 2}),
        ("RS24", "7_9"): Counter({3: 3}),
    }
    assert mapped == {1: "hard coral", 2: None, 3: "turf algae"}


def test_new_dense_counts_rejects_a_tile_without_source_name(tmp_path: Path) -> None:
    version = open_written(
        tmp_path, kind="dense", images=[ImageSpec("t", mask=_TILE_MASK_A)], labels=_DENSE_LABELS
    )

    with pytest.raises(ValueError, match="source_name"):
        new_dense_counts(version)


class _StubBenthos:
    SOURCE_NAME = "benthos_yuval"

    def __init__(self, masks: dict[tuple[str, str], np.ndarray[Any, Any]]) -> None:
        self._masks = masks
        self.df_images = pd.DataFrame(
            {"site": [k[0] for k in masks], "image_id": [k[1] for k in masks]}
        )
        self._classes_global = {"Coral": 1, "Sand": 2, "Turf": 3, "Dropped": 4}
        self.source_id2name = {1: "Coral", 2: "Sand", 3: "Turf"}
        self.source_name2id = {"Coral": 1, "Sand": 2, "Turf": 3}

    def set_global_offset(self, offset: int) -> None:
        pass

    def read_label(self, image_id: str, site: str, **_: Any) -> np.ndarray[Any, Any]:
        return self._masks[(site, image_id)]


def test_old_dense_counts_drop_the_unlabeled_class_and_map_through_the_registry(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setitem(registry_module._REMOTE_FETCHERS, "benthos_yuval", lambda: _BY_MAPPING)
    ds = _StubBenthos({("RS24", "0_5"): _TILE_MASK_A, ("RS24", "7_9"): _TILE_MASK_B})

    counts, mapped = old_dense_counts(ds)

    assert counts == {
        ("RS24", "0_5"): Counter({1: 1, 2: 2}),
        ("RS24", "7_9"): Counter({3: 3}),
    }
    assert mapped == {1: "hard coral", 2: None, 3: "turf algae", 4: None}


_MAPPED = {1: "hard coral", 2: None}
_TILES = {("s", "0_1"): Counter({1: 4, 2: 6}), ("s", "2_3"): Counter({1: 5})}


def test_compare_dense_equal_counts_is_clean() -> None:
    report = compare_dense(_TILES, _TILES, _MAPPED, _MAPPED)

    assert report == DenseParityReport(tiles_matched=2, sites_compared=1)
    assert report.ok


def test_compare_dense_reports_a_tile_whose_label_count_differs() -> None:
    new = {**_TILES, ("s", "0_1"): Counter({1: 3, 2: 6})}

    report = compare_dense(_TILES, new, _MAPPED, _MAPPED)

    assert not report.ok
    assert report.tile_diff == {("s", "0_1"): {1: (4, 3)}}
    assert report.mapped_diff == {"hard coral": (9, 8)}
    assert report.old_only_tiles == report.new_only_tiles == []


def test_compare_dense_reports_tiles_on_one_side_and_ignores_unrelated_sites() -> None:
    new = {
        ("s", "0_1"): _TILES[("s", "0_1")],
        ("s", "9_9"): Counter({1: 1}),
        ("other", "0_1"): Counter({1: 7}),
    }

    report = compare_dense(_TILES, new, _MAPPED, _MAPPED)

    assert report.old_only_tiles == [("s", "2_3")]
    assert report.new_only_tiles == [("s", "9_9")]
    assert report.tile_diff == {}
    assert report.tiles_matched == 1
    assert report.excluded_sites == {"other": (1, 7)}


def test_compare_dense_maps_each_side_with_its_own_table() -> None:
    new_mapped = {1: "soft coral", 2: None}

    report = compare_dense(_TILES, _TILES, _MAPPED, new_mapped)

    assert report.tile_diff == {}
    assert report.mapped_diff == {"hard coral": (9, 0), "soft coral": (0, 9)}


def test_compare_dense_reports_swapped_label_mappings_whose_totals_agree() -> None:
    tiles = {("s", "0_1"): Counter({1: 5, 2: 5})}
    old_mapped = {1: "a", 2: "b", 3: "c"}
    new_mapped = {1: "b", 2: "a", 3: "d"}

    report = compare_dense(tiles, tiles, old_mapped, new_mapped)

    assert report.mapped_diff == {}
    assert report.label_map_diff == {1: ("a", "b"), 2: ("b", "a")}
    assert not report.ok


def test_format_dense_report_lists_ten_tile_diffs_then_the_remainder() -> None:
    old = {("s", f"{i:02d}"): Counter({1: 2}) for i in range(12)}
    new = {("s", f"{i:02d}"): Counter({1: 1}) for i in range(12)}

    text = format_dense_report(compare_dense(old, new, _MAPPED, _MAPPED))

    assert text.endswith("PARITY FAILED")
    assert "s/00: label 1 old 2, new 1" in text
    assert "s/09: label 1 old 2, new 1" in text
    assert "s/10:" not in text
    assert "+2 more" in text
    assert format_dense_report(DenseParityReport()).endswith("PARITY OK")


def test_run_dense_parity_compares_a_benthos_version_tile_by_tile(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    cache_dir, _, pin = _dense_version(tmp_path)
    monkeypatch.setitem(registry_module._REMOTE_FETCHERS, "benthos_yuval", lambda: _BY_MAPPING)
    masks = {("RS24", "0_5"): _TILE_MASK_A, ("RS24", "7_9"): _TILE_MASK_B}
    monkeypatch.setattr(
        parity_module, "build_datasets", lambda *a, **k: {("ds", "val"): _StubBenthos(masks)}
    )

    assert run_dense_parity({}, **_dense_args(cache_dir, pin)).ok

    masks[("RS24", "7_9")] = np.array([[3, 3], [3, 3]], np.uint8)
    report = run_dense_parity({}, **_dense_args(cache_dir, pin))
    assert report.tile_diff == {("RS24", "7_9"): {3: (4, 3)}}


def test_format_dense_report_lists_every_finding_and_what_was_compared() -> None:
    report = DenseParityReport(
        old_only_tiles=[("s", "1_1")],
        new_only_tiles=[("s", "2_2")],
        mask_mismatches=[("s", "3_3")],
        label_count_mismatches={("t", "4_4"): {3: (3, 4)}},
        label_map_diff={1: ("a", None)},
        tiles_matched=12,
        sites_compared=2,
        excluded_sites={"t": (5, 900)},
    )

    lines = format_dense_report(report).splitlines()

    for line in [
        "old-only tiles: 1",
        "  s/1_1",
        "new-only tiles: 1",
        "  s/2_2",
        "mask pixel mismatches: 1",
        "  s/3_3",
        "label_counts vs mask: 1 tiles",
        "  t/4_4: label 3 table 3, mask 4",
        "label id mapping:",
        "  1: old a, new None",
        "tiles matched: 12",
        "sites compared: 2",
        "new sites excluded: 1",
        "  t: 5 tiles, 900 pixels",
    ]:
        assert line in lines
    assert lines[-1] == "PARITY FAILED"


def test_new_dense_counts_ignores_a_label_zero_row_in_label_counts(tmp_path: Path) -> None:
    version = open_written(
        tmp_path,
        kind="dense",
        images=[ImageSpec("t", group="g", mask=_TILE_MASK_B, source_name="7_9.png")],
        labels=_DENSE_LABELS,
        label_counts=[("t", 0, 1), ("t", 3, 3)],
    )

    counts, _ = new_dense_counts(version)

    assert counts == {("g", "7_9"): Counter({3: 3})}


def test_new_dense_counts_rejects_two_tiles_with_the_same_key(tmp_path: Path) -> None:
    version = open_written(
        tmp_path,
        kind="dense",
        images=[
            ImageSpec("a", group="RS24", mask=_TILE_MASK_A, source_name="0_5.png"),
            ImageSpec("b", group="RS24", mask=_TILE_MASK_B, source_name="0_5.jpg"),
        ],
        labels=_DENSE_LABELS,
    )

    with pytest.raises(ValueError, match=r"\('RS24', '0_5'\)"):
        new_dense_counts(version)


def _points_args(cache_dir: Path, pin: Pin) -> dict[str, Any]:
    return {
        "dataset": "ds",
        "split": "train",
        "version": pin.version,
        "manifest_sha256": pin.manifest_sha256,
        "cache_dir": cache_dir,
        "client": _NoClient(),
        "annotators": None,
        "sync": False,
    }


def _dense_args(cache_dir: Path, pin: Pin) -> dict[str, Any]:
    return {
        "dataset": "ds",
        "split": "val",
        "version": pin.version,
        "manifest_sha256": pin.manifest_sha256,
        "cache_dir": cache_dir,
        "client": _NoClient(),
        "sync": False,
    }


def _run_benthos(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    images: Sequence[ImageSpec],
    old_masks: dict[tuple[str, str], np.ndarray[Any, Any]],
    label_counts: Sequence[tuple[str, int, int]] | None = None,
    labels: Sequence[dict[str, Any]] = _DENSE_LABELS,
) -> DenseParityReport:
    cache_dir, bucket, pin = write_version(
        tmp_path, kind="dense", images=images, labels=labels, label_counts=label_counts
    )
    monkeypatch.setitem(registry_module._REMOTE_FETCHERS, "benthos_yuval", lambda: _BY_MAPPING)
    monkeypatch.setattr(
        parity_module, "build_datasets", lambda *a, **k: {("ds", "val"): _StubBenthos(old_masks)}
    )
    return run_dense_parity({}, **_dense_args(cache_dir, pin))


_TILE_MASK_B_MOVED = np.array([[3, 3], [0, 3]], np.uint8)


def test_run_dense_parity_reports_a_stored_mask_whose_pixels_moved(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    images = [
        ImageSpec("RS24_0_5", group="RS24", mask=_TILE_MASK_A, source_name="0_5.png"),
        ImageSpec("RS24_7_9", group="RS24", mask=_TILE_MASK_B_MOVED, source_name="7_9.png"),
    ]
    old = {("RS24", "0_5"): _TILE_MASK_A, ("RS24", "7_9"): _TILE_MASK_B}

    report = _run_benthos(monkeypatch, tmp_path, images, old)

    assert report.tile_diff == {}
    assert report.mask_mismatches == [("RS24", "7_9")]
    assert not report.ok


def test_run_dense_parity_reports_label_counts_that_disagree_with_the_stored_mask(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    images = [
        ImageSpec("RS24_0_5", group="RS24", mask=_TILE_MASK_A, source_name="0_5.png"),
        ImageSpec("X_7_9", group="X", mask=_TILE_MASK_B, source_name="7_9.png"),
    ]
    counts = [("RS24_0_5", 1, 1), ("RS24_0_5", 2, 2), ("X_7_9", 3, 4)]

    report = _run_benthos(monkeypatch, tmp_path, images, {("RS24", "0_5"): _TILE_MASK_A}, counts)

    assert report.label_count_mismatches == {("X", "7_9"): {3: (4, 3)}}
    assert report.excluded_sites == {"X": (1, 4)}
    assert not report.ok


_PARITY_IMAGES = [
    ImageSpec("RS24_0_5", group="RS24", mask=_TILE_MASK_A, source_name="0_5.png"),
    ImageSpec("RS24_7_9", group="RS24", mask=_TILE_MASK_B, source_name="7_9.png"),
]
_PARITY_OLD_MASKS = {("RS24", "0_5"): _TILE_MASK_A, ("RS24", "7_9"): _TILE_MASK_B}


def test_run_dense_parity_reports_label_names_swapped_under_the_same_attributes(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    swapped = [
        {**_DENSE_LABELS[0], "name": "Turf"},
        _DENSE_LABELS[1],
        {**_DENSE_LABELS[2], "name": "Coral"},
    ]

    report = _run_benthos(monkeypatch, tmp_path, _PARITY_IMAGES, _PARITY_OLD_MASKS, labels=swapped)

    assert report.native_name_diff == {1: ("coral", "turf"), 3: ("turf", "coral")}
    assert report.tile_diff == {}
    assert report.mapped_diff == {}
    assert not report.ok


def test_run_dense_parity_ignores_a_label_name_that_differs_only_in_case(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    recased = [{**_DENSE_LABELS[0], "name": "coral"}, *_DENSE_LABELS[1:]]

    report = _run_benthos(monkeypatch, tmp_path, _PARITY_IMAGES, _PARITY_OLD_MASKS, labels=recased)

    assert report.native_name_diff == {}
    assert report.ok


def test_dense_report_with_only_excluded_sites_is_ok() -> None:
    assert DenseParityReport(excluded_sites={"X": (1, 4)}).ok


def test_format_dense_report_lists_ten_label_name_diffs_then_the_remainder() -> None:
    diff: dict[int, tuple[str | None, str | None]] = {i: (f"a{i}", f"b{i}") for i in range(1, 13)}

    text = format_dense_report(DenseParityReport(native_name_diff=diff))

    assert "label id native names:" in text
    assert "  1: old a1, new b1" in text
    assert "  10: old a10, new b10" in text
    assert "  11:" not in text
    assert "+2 more" in text
    assert text.endswith("PARITY FAILED")


def _mask_file(cache_dir: Path, image_id: str) -> Path:
    return cache_dir / IMAGE_BUCKET / IMAGE_PREFIX / f"{image_id}.png"


def _stage_dense(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> tuple[Path, Pin]:
    cache_dir, _, pin = _dense_version(tmp_path)
    monkeypatch.setitem(registry_module._REMOTE_FETCHERS, "benthos_yuval", lambda: _BY_MAPPING)
    old = {("RS24", "0_5"): _TILE_MASK_A, ("RS24", "7_9"): _TILE_MASK_B}
    monkeypatch.setattr(
        parity_module, "build_datasets", lambda *a, **k: {("ds", "val"): _StubBenthos(old)}
    )
    return cache_dir, pin


def test_run_dense_parity_without_sync_reads_a_staged_mask_as_is(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    cache_dir, pin = _stage_dense(monkeypatch, tmp_path)
    Image.fromarray(_TILE_MASK_B_MOVED).save(_mask_file(cache_dir, "RS24_7_9"), "PNG")

    report = run_dense_parity({}, **_dense_args(cache_dir, pin))

    assert report.mask_mismatches == [("RS24", "7_9")]
    assert not report.ok


def test_run_dense_parity_without_sync_fails_on_a_mask_that_is_not_staged(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    cache_dir, pin = _stage_dense(monkeypatch, tmp_path)
    _mask_file(cache_dir, "RS24_7_9").unlink()

    with pytest.raises(FileNotFoundError, match="RS24_7_9.png"):
        run_dense_parity({}, **_dense_args(cache_dir, pin))


def _rewrite_table(cache_dir: Path, pin: Pin, file_name: str, column: str, value: Any) -> None:
    """Change the first value of ``column`` in a staged table, leaving the manifest as is."""
    path = cache_dir / BUCKET / "datasets" / pin.name / pin.version / file_name
    table = pq.read_table(path)
    values = table.column(column).to_pylist()
    values[0] = value
    index = table.schema.get_field_index(column)
    kind = table.schema.field(column).type
    pq.write_table(table.set_column(index, column, pa.array(values, kind)), path)


def test_run_parity_without_sync_verifies_the_staged_tables(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    cache_dir, pin, stub = _stage_points(monkeypatch, tmp_path)
    stub.df_annotations.loc[0, "row"] = 2
    monkeypatch.setattr(parity_module, "build_datasets", lambda *a, **k: {("ds", "train"): stub})
    _rewrite_table(cache_dir, pin, "points.parquet", "row", 2)

    with pytest.raises(ChecksumMismatch, match="points.parquet"):
        run_parity({}, **_points_args(cache_dir, pin))


def test_run_dense_parity_without_sync_verifies_the_staged_tables(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    cache_dir, pin = _stage_dense(monkeypatch, tmp_path)
    _rewrite_table(cache_dir, pin, "labels.parquet", "name", "Coral")

    with pytest.raises(ChecksumMismatch, match="labels.parquet"):
        run_dense_parity({}, **_dense_args(cache_dir, pin))


def test_run_parity_rejects_an_old_split_without_points(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    cache_dir, pin, stub = _stage_points(monkeypatch, tmp_path)
    stub.df_annotations = stub.df_annotations.iloc[0:0]
    monkeypatch.setattr(parity_module, "build_datasets", lambda *a, **k: {("ds", "train"): stub})

    with pytest.raises(ValueError, match="no points"):
        run_parity({}, **_points_args(cache_dir, pin))


def test_run_dense_parity_rejects_an_old_split_without_tiles(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    cache_dir, pin = _stage_dense(monkeypatch, tmp_path)
    monkeypatch.setattr(
        parity_module, "build_datasets", lambda *a, **k: {("ds", "val"): _StubBenthos({})}
    )

    with pytest.raises(ValueError, match="no tiles"):
        run_dense_parity({}, **_dense_args(cache_dir, pin))


def test_run_dense_parity_rejects_an_old_hf_split_without_images(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    with pytest.raises(ValueError, match="no images"):
        _run_dense(monkeypatch, tmp_path, _StubDense([]), _new_images())


def test_run_dense_parity_fills_the_benthos_fields_for_a_benthos_loader(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    cache_dir, pin = _stage_dense(monkeypatch, tmp_path)

    report = run_dense_parity({}, **_dense_args(cache_dir, pin), hf_revision_probe=lambda: "abc123")

    assert report == DenseParityReport(tiles_matched=2, sites_compared=1)


def test_run_dense_parity_fills_the_hf_fields_for_an_hf_loader(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    report = _run_dense(
        monkeypatch, tmp_path, _StubDense(_old_rows()), _new_images(), probe=lambda: "abc123"
    )

    assert report == DenseParityReport(matched=2, hf_revision_at_run="abc123")


_WIDE = [ImageSpec("w1", width=4096, height=2048)]
_WIDE_POINT = (("w1", 1000, 3000, 1),)


def test_run_parity_scales_new_points_into_the_resized_space(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    report, cache_dir = _run(
        monkeypatch,
        tmp_path,
        [("w1", 500, 1500)],
        images=_WIDE,
        points=_WIDE_POINT,
        resize=ResizeSpec(max_edge=2048),
    )

    assert report.ok
    assert {p.suffix for p in cache_dir.rglob("*") if p.is_file()} == {".json", ".parquet"}


def test_run_parity_without_resize_reports_the_scaled_pair_as_different(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    report, _ = _run(monkeypatch, tmp_path, [("w1", 500, 1500)], images=_WIDE, points=_WIDE_POINT)

    assert report.old_only_points == Counter({("w1", 500, 1500, "hard coral"): 1})
    assert report.new_only_points == Counter({("w1", 1000, 3000, "hard coral"): 1})
