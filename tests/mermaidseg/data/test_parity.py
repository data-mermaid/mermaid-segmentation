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

import mermaidseg.data.parity as parity_module
import mermaidseg.dataset_reconciliation.registry as registry_module
from mermaidseg.data.config import DATASET_BUCKET
from mermaidseg.data.parity import (
    DenseParityReport,
    ParityReport,
    VocabMismatch,
    compare,
    format_dense_report,
    format_report,
    new_points,
    old_dense_rows,
    old_points,
    run_dense_parity,
    run_parity,
)
from segdata_spec.manifest import FileEntry, Manifest, Pin, dump_manifest, manifest_sha256
from tests.mermaidseg.data.conftest import (
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
    assert report == ParityReport()


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
    assert points["native_name"].tolist() == ["Hard", "Soft", "Hard"]
    assert points["mapped"].tolist() == ["hard coral", None, "hard coral"]


_LABELS = [
    {"label_id": 1, "name": "Acropora", "mermaid_benthic_attribute": "Hard Coral"},
    {"label_id": 2, "name": "Rubble", "mermaid_benthic_attribute": None},
]
_IMAGES = [ImageSpec("i1"), ImageSpec("i2")]


def test_new_points_lowercases_the_attribute_and_keeps_null_none(tmp_path: Path) -> None:
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
    assert points["native_name"].tolist() == ["Acropora", "Rubble", "Acropora"]
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
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, old_rows: list[tuple[str, int, int]]
) -> tuple[ParityReport, Path]:
    fake, pin = publish_points(
        tmp_path / "pub",
        images=_IMAGES,
        labels=_LABELS,
        points=[("i1", 1, 1, 1), ("i2", 3, 3, 1)],
    )
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
    )
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


# --- dense ---------------------------------------------------------------------------------

_RNG = np.random.default_rng(0)
_PIXELS = {name: _RNG.integers(0, 255, (4, 4, 3), dtype=np.uint8) for name in ("p1", "p2", "p3")}
_MASK1 = np.array([[1, 1, 2, 2], [1, 1, 2, 2], [3, 3, 0, 0], [3, 3, 0, 0]], dtype=np.uint8)
_MASK2 = np.array([[2, 2, 2, 2], [1, 1, 1, 1], [3, 3, 3, 3], [0, 0, 0, 0]], dtype=np.uint8)
_NAMES = {1: "a", 2: "b", 3: "c"}
_ATTRIBUTES = {"a": "Hard Coral", "b": "Rubble", "c": None}


class _StubDense:
    SOURCE_NAME = "catlin_seaview"

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


def _new_images(mask1: np.ndarray = _MASK1, pixels2: str = "p2") -> list[ImageSpec]:
    return [
        ImageSpec("n1", group="train", mask=mask1, pixels=_PIXELS["p1"]),
        ImageSpec("n2", group="train", mask=_MASK2, pixels=_PIXELS[pixels2]),
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


def test_dense_original_split_filter_still_reports_a_kept_image_that_differs(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    old = _StubUcsd(_old_rows()[:1], whitelist={"train"})

    report = _run_dense(
        monkeypatch,
        tmp_path,
        old,
        _new_images(pixels2="p3"),
        attrs=_split_attrs(n1="train", n2="train"),
    )

    assert report.new_only == ["n2"]
    assert not report.ok


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


def test_dense_class_subset_compares_the_vocabulary_of_the_subset_only(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    report = _run_dense(monkeypatch, tmp_path, _subset_old(), _new_images())

    assert report.vocab_mismatches == []


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
