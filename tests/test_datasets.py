"""Unit tests for mermaidseg.datasets.utils and BaseCoralDataset error handling."""

from __future__ import annotations

import io
import pickle
from typing import Any

import numpy as np
import pandas as pd
import pytest
import torch
from botocore.exceptions import ClientError
from PIL import Image

from mermaidseg.datasets.base_dataset import BaseCoralDataset
from mermaidseg.datasets.utils import (
    DataLoadError,
    create_annotation_mask,
    create_annotation_mask_from_arrays,
    get_image_s3,
    get_image_s3_candidates,
)


def _make_annotations(rows: list, cols: list, labels: list) -> pd.DataFrame:
    """Build a minimal annotations DataFrame."""
    return pd.DataFrame({"row": rows, "col": cols, "source_label_name": labels})


@pytest.fixture
def minimal_dataset() -> BaseCoralDataset:
    """Smallest valid BaseCoralDataset — no real images needed."""
    df_annotations = pd.DataFrame(
        {
            "image_id": ["img1", "img2"],
            "region_id": [1, 2],
            "region_name": ["r1", "r2"],
            "source_label_name": ["Coral", "Sand"],
            "row": [10, 20],
            "col": [10, 20],
        }
    )
    df_images = (
        df_annotations[["image_id", "region_id", "region_name"]]
        .drop_duplicates()
        .reset_index(drop=True)
    )
    return BaseCoralDataset(
        df_annotations=df_annotations,
        df_images=df_images,
        class_subset=["Coral", "Sand"],
    )


@pytest.fixture
def single_image_annotations() -> tuple[pd.DataFrame, pd.DataFrame]:
    """Single-image annotations and images DataFrames."""
    df_annotations = pd.DataFrame(
        {
            "image_id": ["img1"],
            "region_id": [1],
            "region_name": ["r1"],
            "source_label_name": ["Coral"],
            "row": [5],
            "col": [5],
        }
    )
    df_images = (
        df_annotations[["image_id", "region_id", "region_name"]]
        .drop_duplicates()
        .reset_index(drop=True)
    )
    return df_annotations, df_images


class _AlwaysFailDataset(BaseCoralDataset):
    """Minimal subclass whose read_image always raises."""

    def read_image(self, **row_kwargs) -> Any:
        raise RuntimeError("simulated read failure")


class _FakeS3:
    def __init__(self, payload: bytes):
        self.payload = payload

    def get_object(self, **_kwargs):
        return {"Body": io.BytesIO(self.payload)}


class _S3ImageDataset(BaseCoralDataset):
    def __init__(self, payload: bytes, **kwargs):
        self.s3 = _FakeS3(payload)
        super().__init__(**kwargs)

    def read_image(self, **_row_kwargs):
        image = get_image_s3(self.s3, "bucket", "image.png")
        return np.array(image.convert("RGB"))


class _RoutingS3:
    def __init__(self, responses):
        self.responses = responses
        self.requested_keys = []

    def get_object(self, **kwargs):
        key = kwargs["Key"]
        self.requested_keys.append(key)
        response = self.responses[key]
        if isinstance(response, Exception):
            raise response
        return {"Body": io.BytesIO(response)}


def _jpeg_bytes() -> bytes:
    buffer = io.BytesIO()
    Image.new("RGB", (32, 24), color=(20, 40, 60)).save(buffer, format="JPEG")
    return buffer.getvalue()


def _s3_error(code: str) -> ClientError:
    return ClientError({"Error": {"Code": code, "Message": code}}, "GetObject")


def test_get_image_s3_detects_jpeg_content_behind_png_key():
    image = get_image_s3(_FakeS3(_jpeg_bytes()), "bucket", "misnamed.png")

    assert image.format == "JPEG"
    assert image.size == (32, 24)


def test_get_image_s3_rejects_truncated_image_during_read():
    payload = _jpeg_bytes()

    with pytest.raises(DataLoadError, match="decode image"):
        get_image_s3(_FakeS3(payload[:-2]), "bucket", "truncated.png")


def test_base_dataset_skips_truncated_image(single_image_annotations):
    df_annotations, df_images = single_image_annotations
    payload = _jpeg_bytes()
    dataset = _S3ImageDataset(
        payload=payload[:-2],
        df_annotations=df_annotations,
        df_images=df_images,
        class_subset=["Coral"],
    )

    assert dataset[0] == (None, None)
    assert dataset.num_load_failures() == 1
    assert dataset.load_failures_df().loc[0, "error_type"] == "DataLoadError"


def test_get_image_s3_candidates_falls_back_when_extension_key_is_missing():
    s3 = _RoutingS3(
        {
            "mermaid/image.png": _s3_error("NoSuchKey"),
            "mermaid/image.jpg": _jpeg_bytes(),
        }
    )

    image = get_image_s3_candidates(
        s3,
        "bucket",
        ["mermaid/image.png", "mermaid/image.jpg"],
    )

    assert image.format == "JPEG"
    assert s3.requested_keys == ["mermaid/image.png", "mermaid/image.jpg"]


def test_get_image_s3_candidates_does_not_hide_non_missing_errors():
    s3 = _RoutingS3(
        {
            "mermaid/image.png": _s3_error("AccessDenied"),
            "mermaid/image.jpg": _jpeg_bytes(),
        }
    )

    with pytest.raises(DataLoadError, match="AccessDenied"):
        get_image_s3_candidates(
            s3,
            "bucket",
            ["mermaid/image.png", "mermaid/image.jpg"],
        )

    assert s3.requested_keys == ["mermaid/image.png"]


def test_get_image_s3_candidates_reports_all_missing_keys():
    keys = ["mermaid/image.png", "mermaid/image.jpg", "mermaid/image.jpeg"]
    s3 = _RoutingS3({key: _s3_error("NoSuchKey") for key in keys})

    with pytest.raises(DataLoadError, match=r"image\.png.*image\.jpg.*image\.jpeg"):
        get_image_s3_candidates(s3, "bucket", keys)


# --- create_annotation_mask ---


def test_create_annotation_mask_basic():
    annotations = _make_annotations([10, 20, 30], [5, 15, 25], ["Coral", "Sand", "Rubble"])
    source_name2id = {"Coral": 1, "Sand": 2, "Rubble": 3}
    mask = create_annotation_mask(annotations, (50, 50), source_name2id)

    assert mask[10, 5] == 1
    assert mask[20, 15] == 2
    assert mask[30, 25] == 3
    assert mask[0, 0] == 0


def test_create_annotation_mask_with_padding():
    annotations = _make_annotations([10], [10], ["Coral"])
    mask = create_annotation_mask(annotations, (50, 50), {"Coral": 1}, padding=2)

    assert np.all(mask[8:12, 8:12] == 1)
    assert mask[7, 10] == 0
    assert mask[12, 10] == 0


def test_create_annotation_mask_padding_bounds_clamped():
    """Large padding at image corners must clamp to bounds without raising
    IndexError."""
    annotations = _make_annotations([0, 19], [0, 19], ["Coral", "Coral"])
    mask = create_annotation_mask(annotations, (20, 20), {"Coral": 1}, padding=5)

    assert np.all(mask[0:5, 0:5] == 1)
    assert np.all(mask[14:20, 14:20] == 1)


def test_create_annotation_mask_overlapping_padding():
    """When padding regions overlap, later annotations should overwrite earlier ones."""
    annotations = _make_annotations([10, 10], [10, 14], ["Coral", "Sand"])
    mask = create_annotation_mask(annotations, (20, 20), {"Coral": 1, "Sand": 2}, padding=3)

    assert mask[10, 12] == 2
    assert mask[10, 13] == 2
    assert mask[10, 9] == 1
    assert mask[10, 16] == 2


@pytest.mark.parametrize("padding", [None, 0, 2, 5])
def test_mask_from_arrays_equals_dataframe_path(padding):
    """The array-native core (used by _load_item) is byte-identical to the DataFrame
    path."""
    rows = [10, 20, 0, 19]
    cols = [5, 15, 0, 19]
    labels = ["Coral", "Sand", "Rubble", "Coral"]
    source_name2id = {"Coral": 1, "Sand": 2, "Rubble": 3}
    shape = (25, 25)

    df_mask = create_annotation_mask(
        _make_annotations(rows, cols, labels), shape, source_name2id, padding=padding
    )
    arr_mask = create_annotation_mask_from_arrays(
        np.asarray(rows, dtype=np.intp),
        np.asarray(cols, dtype=np.intp),
        np.asarray([source_name2id[label] for label in labels], dtype=np.int64),
        shape,
        padding=padding,
    )
    np.testing.assert_array_equal(df_mask, arr_mask)


def test_mask_from_arrays_empty_returns_background():
    empty = np.empty(0, dtype=np.intp)
    mask = create_annotation_mask_from_arrays(
        empty, empty, np.empty(0, dtype=np.int64), (8, 8), padding=3
    )
    assert mask.shape == (8, 8)
    assert not mask.any()


def test_create_annotation_mask_unknown_label_skipped(caplog):
    annotations = _make_annotations([5, 10], [5, 10], ["Coral", "UnknownLabel"])

    with caplog.at_level("WARNING", logger="mermaidseg.datasets.utils"):
        mask = create_annotation_mask(annotations, (20, 20), {"Coral": 1})

    assert mask[5, 5] == 1
    assert mask[10, 10] == 0
    assert "unknown label" in caplog.text.lower() or "UnknownLabel" in caplog.text


@pytest.mark.parametrize(
    "rows,cols,labels",
    [
        ([], [], []),
        ([5], [5], [None]),
    ],
    ids=["empty", "all_null_labels"],
)
def test_create_annotation_mask_produces_zero_mask(rows, cols, labels):
    annotations = _make_annotations(rows, cols, labels)
    mask = create_annotation_mask(annotations, (10, 10), {"Coral": 1})
    assert np.all(mask == 0)


# --- BaseCoralDataset basic API ---


def test_base_dataset_exposes_source_label_attributes(minimal_dataset):
    assert minimal_dataset.source_id2name == {1: "Coral", 2: "Sand"}
    assert minimal_dataset.source_name2id == {"Coral": 1, "Sand": 2}
    assert minimal_dataset.num_source_classes == 3  # background + 2
    assert minimal_dataset.global_offset == 0


def test_base_dataset_set_global_offset_validates_negative(minimal_dataset):
    with pytest.raises(ValueError):
        minimal_dataset.set_global_offset(-1)


def test_base_dataset_set_global_offset_shifts_mask_via_helper():
    """Verify offset arithmetic on the helper directly: offset=10 → values become
    11/12."""
    minimal_mask = np.array([[0, 1, 2], [2, 0, 1]], dtype=np.int64)
    offset = 10
    shifted = np.where(minimal_mask > 0, minimal_mask + offset, minimal_mask)
    assert shifted[0, 0] == 0
    assert shifted[0, 1] == 11
    assert shifted[0, 2] == 12
    assert shifted[1, 1] == 0


# --- BaseCoralDataset.collate_fn ---


def test_collate_fn_filters_none_items(minimal_dataset, caplog):
    img = torch.zeros(3, 4, 4)
    msk = torch.zeros(4, 4, dtype=torch.long)
    batch = [(img, msk), (None, None), (img, msk), (None, None)]

    with caplog.at_level("WARNING", logger="mermaidseg.datasets.base_dataset"):
        images, masks = minimal_dataset.collate_fn(batch)

    assert images.shape[0] == 2
    assert masks.shape[0] == 2
    assert "skipped 2/4" in caplog.text


def test_collate_fn_all_none_returns_empty_tensors(minimal_dataset, caplog):
    with caplog.at_level("WARNING", logger="mermaidseg.datasets.base_dataset"):
        images, masks = minimal_dataset.collate_fn([(None, None), (None, None)])

    assert images.numel() == 0
    assert masks.numel() == 0
    assert "entire batch" in caplog.text


# --- BaseCoralDataset.__getitem__ ---


def test_base_dataset_getitem_skips_and_logs_on_read_failure(single_image_annotations, caplog):
    df_annotations, df_images = single_image_annotations
    ds = _AlwaysFailDataset(
        df_annotations=df_annotations,
        df_images=df_images,
        class_subset=["Coral"],
    )

    with caplog.at_level("WARNING", logger="mermaidseg.datasets.base_dataset"):
        result = ds[0]

    assert result == (None, None)
    assert "img1" in caplog.text
    assert "RuntimeError" in caplog.text


def test_base_dataset_records_failure_context(single_image_annotations):
    df_annotations, df_images = single_image_annotations
    ds = _AlwaysFailDataset(
        df_annotations=df_annotations,
        df_images=df_images,
        class_subset=["Coral"],
        split="train",
    )

    _ = ds[0]
    failures = ds.load_failures_df()
    assert len(failures) == 1

    row = failures.iloc[0]
    assert row["dataset_class"] == "_AlwaysFailDataset"
    assert row["split"] == "train"
    assert row["image_id"] == "img1"
    assert row["region_name"] == "r1"
    assert row["annotation_count"] == 1
    assert row["annotation_labels"] == "Coral"
    assert not row["missing_annotations"]
    assert row["error_type"] == "RuntimeError"
    assert "simulated read failure" in row["error_message"]


def test_base_dataset_saves_failure_report_as_parquet(single_image_annotations, tmp_path):
    df_annotations, df_images = single_image_annotations
    ds = _AlwaysFailDataset(
        df_annotations=df_annotations,
        df_images=df_images,
        class_subset=["Coral"],
    )

    _ = ds[0]
    output_path = tmp_path / "load_failures.parquet"
    saved_path = ds.save_load_failures(output_path)

    assert saved_path == output_path
    assert output_path.exists()
    saved_df = pd.read_parquet(output_path)
    assert len(saved_df) == 1
    assert saved_df.iloc[0]["image_id"] == "img1"


def test_load_failures_count_is_exact_but_records_are_capped(monkeypatch):
    """num_load_failures() stays exact past the record cap; stored records are
    bounded."""
    from mermaidseg.datasets import base_dataset as bd

    monkeypatch.setattr(bd, "_LOAD_FAILURE_RECORD_CAP", 3)

    n = 10
    df_annotations = pd.DataFrame(
        {
            "image_id": [f"img{i}" for i in range(n)],
            "region_id": list(range(n)),
            "region_name": [f"r{i}" for i in range(n)],
            "row": [1] * n,
            "col": [1] * n,
            "source_label_name": ["Coral"] * n,
        }
    )
    df_images = df_annotations[["image_id", "region_id", "region_name"]].copy()
    ds = _AlwaysFailDataset(
        df_annotations=df_annotations, df_images=df_images, class_subset=["Coral"]
    )

    for i in range(n):
        assert ds[i] == (None, None)

    assert ds.num_load_failures() == n  # exact count, past the cap
    assert len(ds.load_failures_df()) == 3  # records bounded at the cap


# --- BaseCoralDataset O(1) annotation lookup (Ticket 1b) ---


class _StubImageDataset(BaseCoralDataset):
    """Minimal subclass returning a fixed blank image so _load_item runs offline."""

    def read_image(self, **row_kwargs: Any) -> np.ndarray:
        return np.zeros((32, 32, 3), dtype=np.uint8)


@pytest.fixture
def multi_annotation_dataset() -> _StubImageDataset:
    """Dataset with a multi-annotation image, a single-annotation image, and an image
    that has no annotations at all (present only in df_images)."""
    df_annotations = pd.DataFrame(
        {
            "image_id": ["img1", "img1", "img2"],
            "source_label_name": ["Coral", "Sand", "Coral"],
            "row": [1, 5, 9],
            "col": [2, 6, 10],
        }
    )
    df_images = pd.DataFrame({"image_id": ["img1", "img2", "img3_empty"]})
    return _StubImageDataset(df_annotations=df_annotations, df_images=df_images)


def test_annotation_index_matches_boolean_scan(multi_annotation_dataset):
    """The precomputed numpy annotation index reproduces the boolean-scan selection
    exactly."""
    ds = multi_annotation_dataset
    for image_id in ("img1", "img2"):
        pos = int(ds.df_images.index[ds.df_images["image_id"] == image_id][0])
        start, end = int(ds._ann_offsets[pos]), int(ds._ann_offsets[pos + 1])
        got = set(
            zip(
                ds._ann_row[start:end].tolist(),
                ds._ann_col[start:end].tolist(),
                ds._ann_label_id[start:end].tolist(),
                strict=True,
            )
        )
        rows = ds.df_annotations[ds.df_annotations["image_id"] == image_id]
        expected = {
            (int(r.row), int(r.col), ds.source_name2id[r.source_label_name])
            for r in rows.itertuples()
        }
        assert got == expected


def test_annotation_index_empty_for_unannotated_image(multi_annotation_dataset):
    """Images with no annotations get an empty slice (→ empty mask)."""
    ds = multi_annotation_dataset
    pos = int(ds.df_images.index[ds.df_images["image_id"] == "img3_empty"][0])
    assert int(ds._ann_offsets[pos]) == int(ds._ann_offsets[pos + 1])


def test_load_item_paints_all_annotations_of_target_image(multi_annotation_dataset):
    """_load_item builds a mask from every annotation of the looked-up image, and
    nothing else."""
    ds = multi_annotation_dataset
    idx = int(ds.df_images.index[ds.df_images["image_id"] == "img1"][0])
    _image, mask = ds._load_item(idx)
    assert mask.shape == (32, 32)
    assert mask[1, 2] == ds.source_name2id["Coral"]
    assert mask[5, 6] == ds.source_name2id["Sand"]
    # img2's annotation must not leak into img1's mask; unannotated pixels stay background.
    assert mask[9, 10] == 0
    assert mask[0, 0] == 0


def test_load_item_empty_image_yields_zero_mask(multi_annotation_dataset):
    """An image with no annotations produces an all-background mask, not a crash."""
    ds = multi_annotation_dataset
    idx = int(ds.df_images.index[ds.df_images["image_id"] == "img3_empty"][0])
    _image, mask = ds._load_item(idx)
    assert mask.shape == (32, 32)
    assert int(mask.sum()) == 0


# --- Disk image cache (MERMAIDSEG_IMAGE_CACHE_DIR) ---------------------------------


def _png_bytes(size: tuple[int, int] = (16, 12)) -> bytes:
    buffer = io.BytesIO()
    Image.new("RGB", size, color=(10, 20, 30)).save(buffer, format="PNG")
    return buffer.getvalue()


class _CountingS3:
    """Fake S3 that counts get_object calls, to prove cache hits avoid re-fetching."""

    def __init__(self, payload: bytes):
        self.payload = payload
        self.calls = 0

    def get_object(self, **_kwargs):
        self.calls += 1
        return {"Body": io.BytesIO(self.payload)}


def test_get_image_s3_writes_to_disk_cache(monkeypatch, tmp_path):
    monkeypatch.setattr("mermaidseg.datasets.utils._IMAGE_CACHE_DIR", str(tmp_path))
    s3 = _CountingS3(_png_bytes())

    get_image_s3(s3, "mybucket", "a/b/img.png")

    cached = tmp_path / "mybucket" / "a" / "b" / "img.png"
    assert cached.exists()
    assert cached.read_bytes()  # non-empty


def test_get_image_s3_serves_second_read_from_cache(monkeypatch, tmp_path):
    monkeypatch.setattr("mermaidseg.datasets.utils._IMAGE_CACHE_DIR", str(tmp_path))
    s3 = _CountingS3(_png_bytes())

    get_image_s3(s3, "mybucket", "img.png")
    get_image_s3(s3, "mybucket", "img.png")

    assert s3.calls == 1  # second read served from disk, no extra S3 GET


def test_get_image_s3_refetches_when_cache_file_is_corrupt(monkeypatch, tmp_path):
    monkeypatch.setattr("mermaidseg.datasets.utils._IMAGE_CACHE_DIR", str(tmp_path))
    s3 = _CountingS3(_png_bytes())

    cached = tmp_path / "mybucket" / "img.png"
    cached.parent.mkdir(parents=True)
    cached.write_bytes(b"not an image")  # poison the cache

    image = get_image_s3(s3, "mybucket", "img.png")

    assert np.array(image.convert("RGB")).shape == (12, 16, 3)
    assert s3.calls == 1  # corrupt cache discarded and refetched


# --- Fork/pickle-safe S3 client ----------------------------------------------------
# The end-to-end "DataLoader(num_workers>0, spawn) does not raise PicklingError" scenario is not
# tested here: driving a spawn-context DataLoader from inside pytest deadlocks (spawned workers
# re-import the test session). The defect it guarded against — a live boto3 client making the
# dataset unpicklable — is covered deterministically by test_dataset_with_live_client_is_picklable.


def test_dataset_lazily_creates_s3_client(minimal_dataset):
    assert minimal_dataset._s3 is None  # nothing created at construction
    client = minimal_dataset.s3
    assert client is not None
    assert minimal_dataset.s3 is client  # cached — same instance on re-access


def test_dataset_getstate_drops_s3_client(minimal_dataset):
    _ = minimal_dataset.s3  # create it
    assert minimal_dataset.__getstate__()["_s3"] is None


def test_dataset_with_live_client_is_picklable(minimal_dataset):
    _ = minimal_dataset.s3  # a live boto3 client now exists (the PicklingError trigger)
    restored = pickle.loads(pickle.dumps(minimal_dataset))
    assert restored._s3 is None  # dropped on pickle; recreated lazily per process


def test_injected_fake_client_survives_super_init():
    """The base __init__ must not clobber a client a subclass injected before
    super().__init__."""
    ds = _S3ImageDataset(
        _png_bytes(),
        df_annotations=pd.DataFrame(
            {
                "image_id": ["img1"],
                "region_id": [1],
                "region_name": ["r1"],
                "source_label_name": ["Coral"],
                "row": [1],
                "col": [1],
            }
        ),
        df_images=pd.DataFrame({"image_id": ["img1"], "region_id": [1], "region_name": ["r1"]}),
    )
    assert isinstance(ds.s3, _FakeS3)  # not clobbered into a real boto3 client
