from __future__ import annotations

import argparse
import importlib
import io
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.parquet as pq
import pytest
from PIL import Image, UnidentifiedImageError

from external_datasets import benthos_yuval, images

B = "bkt"
P = "ds"
# RS24 uses the default dictionary, CR_DoubleWreck the alt one; ids differ per variant.
DEFAULT_DICT = "{'Coral': 1, 'Sand': 2}"
ALT_DICT = "{'Coral': 5, 'Algae': 6}"


def _rgb(h: int, w: int) -> np.ndarray:
    return np.full((h, w, 3), 7, dtype=np.uint8)


def _save_png(path: Path, arr: np.ndarray) -> None:
    Image.fromarray(arr).save(path)


def _write_raw(
    root: Path,
    *,
    rs24_mask: np.ndarray,
    rs24_image: np.ndarray | None = None,
    dw_mask: np.ndarray | None = None,
    default_dict: str = DEFAULT_DICT,
) -> Path:
    root.mkdir(parents=True, exist_ok=True)
    h, w = rs24_mask.shape
    _save_png(root / "RS24.png", rs24_image if rs24_image is not None else _rgb(h, w))
    _save_png(root / "ManualRS24.png", rs24_mask)
    dw = dw_mask if dw_mask is not None else np.full((2, 2), 5, dtype=np.uint8)
    _save_png(root / "CR_DoubleWreck.png", _rgb(*dw.shape))
    _save_png(root / "ManualDW.png", dw)
    (root / "dictionary_labels.txt").write_text(default_dict)
    (root / "dictionary_labelsDW.txt").write_text(ALT_DICT)
    return root


def _args(tmp_path: Path, data_dir: Path, **overrides) -> argparse.Namespace:
    values = {
        "bucket": B,
        "prefix": P,
        "workers": 2,
        "dry_run": False,
        "skip_existing_s3": True,
        "overwrite": False,
        "data_dir": data_dir,
        "workdir": tmp_path / "work",
    }
    values.update(overrides)
    return argparse.Namespace(**values)


def _parquet(fake_s3) -> pd.DataFrame:
    body = fake_s3.objects[(B, f"{P}/benthos_yuval_annotations.parquet")]
    return pq.read_table(io.BytesIO(body)).to_pandas()


def _png(fake_s3, key: str) -> Image.Image:
    return Image.open(io.BytesIO(fake_s3.objects[(B, key)]))


def test_be1_wide_mosaic_tiles_into_full_and_remainder_tile(tmp_path, fake_s3):
    mask = np.full((2, 2049), 1, dtype=np.uint8)
    root = _write_raw(tmp_path / "raw", rs24_mask=mask)

    assert benthos_yuval.run(_args(tmp_path, root), fake_s3) == 0

    rs24_keys = {k for (_, k) in fake_s3.objects if "/RS24/" in k}
    assert rs24_keys == {
        f"{P}/images/RS24/0_0.png",
        f"{P}/images/RS24/0_2048.png",
        f"{P}/labels/RS24/0_0.png",
        f"{P}/labels/RS24/0_2048.png",
    }
    ingestion = json.loads(fake_s3.objects[(B, f"{P}/manifest.json")])["ingestion"]
    assert ingestion["num_tiles"] == {"RS24": 2, "CR_DoubleWreck": 1}
    assert ingestion["images_uploaded"] == 3
    assert ingestion["labels_uploaded"] == 3
    assert _png(fake_s3, f"{P}/images/RS24/0_0.png").size == (2048, 2)
    assert _png(fake_s3, f"{P}/images/RS24/0_2048.png").size == (1, 2)
    assert _png(fake_s3, f"{P}/labels/RS24/0_0.png").mode == "L"


def test_be2_raw_ids_map_to_global_ids_and_unknown_to_zero(tmp_path, fake_s3):
    # classes.json: unlabeled 0, Algae 1, Coral 2, Sand 3. Raw 9 is in no dictionary.
    mask = np.array([[1, 2, 9, 0]], dtype=np.uint8)
    dw = np.array([[5, 6]], dtype=np.uint8)
    root = _write_raw(tmp_path / "raw", rs24_mask=mask, dw_mask=dw)

    benthos_yuval.run(_args(tmp_path, root), fake_s3)

    assert json.loads(fake_s3.objects[(B, f"{P}/classes.json")]) == {
        "unlabeled": 0,
        "Algae": 1,
        "Coral": 2,
        "Sand": 3,
    }
    rs24 = np.asarray(_png(fake_s3, f"{P}/labels/RS24/0_0.png"))
    assert rs24.tolist() == [[2, 3, 0, 0]]
    cr = np.asarray(_png(fake_s3, f"{P}/labels/CR_DoubleWreck/0_0.png"))
    assert cr.tolist() == [[2, 1]]


def test_be3_mismatched_image_and_mask_crop_to_smaller_size(tmp_path, fake_s3):
    mask = np.full((4, 6), 1, dtype=np.uint8)
    root = _write_raw(tmp_path / "raw", rs24_mask=mask, rs24_image=_rgb(5, 3))

    benthos_yuval.run(_args(tmp_path, root), fake_s3)

    assert _png(fake_s3, f"{P}/images/RS24/0_0.png").size == (3, 4)
    assert _png(fake_s3, f"{P}/labels/RS24/0_0.png").size == (3, 4)


def test_be4_missing_required_file_is_named(tmp_path, fake_s3):
    root = _write_raw(tmp_path / "raw", rs24_mask=np.ones((2, 2), dtype=np.uint8))
    (root / "ManualDW.png").unlink()

    with pytest.raises(FileNotFoundError, match="ManualDW.png"):
        benthos_yuval.run(_args(tmp_path, root), fake_s3)


def test_be5_parquet_has_one_row_per_tile_and_present_class(tmp_path, fake_s3):
    mask = np.array([[1, 2, 0]], dtype=np.uint8)
    dw = np.full((1, 2), 5, dtype=np.uint8)
    root = _write_raw(tmp_path / "raw", rs24_mask=mask, dw_mask=dw)

    benthos_yuval.run(_args(tmp_path, root), fake_s3)

    df = _parquet(fake_s3)
    assert list(df.columns) == [
        "site",
        "image_id",
        "source_label_name",
        "tile_height",
        "tile_width",
    ]
    assert df["tile_height"].dtype == np.int32
    assert df["tile_width"].dtype == np.int32
    assert df.to_numpy().tolist() == [
        ["CR_DoubleWreck", "0_0", "Coral", 1, 2],
        ["RS24", "0_0", "Coral", 1, 3],
        ["RS24", "0_0", "Sand", 1, 3],
        ["RS24", "0_0", "unlabeled", 1, 3],
    ]


def test_be6_label_dictionary_that_is_not_a_dict_raises(tmp_path, fake_s3):
    root = _write_raw(
        tmp_path / "raw", rs24_mask=np.ones((2, 2), dtype=np.uint8), default_dict="[1, 2]"
    )

    with pytest.raises(RuntimeError, match="did not parse to a dict"):
        benthos_yuval.run(_args(tmp_path, root), fake_s3)


def test_overwrite_replaces_stale_label_tiles_but_not_stale_images(tmp_path, fake_s3):
    root = _write_raw(tmp_path / "raw", rs24_mask=np.ones((2, 2), dtype=np.uint8))
    fake_s3.put(B, f"{P}/labels/RS24/0_0.png", b"stale")
    fake_s3.put(B, f"{P}/images/RS24/0_0.png", b"stale")

    benthos_yuval.run(_args(tmp_path, root, overwrite=True), fake_s3)

    ingestion = json.loads(fake_s3.objects[(B, f"{P}/manifest.json")])["ingestion"]
    assert ingestion["images_uploaded"] == 1
    assert ingestion["labels_uploaded"] == 2
    assert fake_s3.objects[(B, f"{P}/labels/RS24/0_0.png")] != b"stale"
    assert fake_s3.objects[(B, f"{P}/images/RS24/0_0.png")] == b"stale"


def test_existing_manifest_without_overwrite_raises_before_any_write(tmp_path, fake_s3):
    root = _write_raw(tmp_path / "raw", rs24_mask=np.ones((2, 2), dtype=np.uint8))
    fake_s3.put(B, f"{P}/manifest.json", b"{}")

    with pytest.raises(FileExistsError):
        benthos_yuval.run(_args(tmp_path, root), fake_s3)

    assert list(fake_s3.objects) == [(B, f"{P}/manifest.json")]


def test_run_lifts_pixel_limit_only_while_decoding_mosaics(tmp_path, fake_s3, monkeypatch):
    # A 1-pixel limit makes any mosaic a decompression bomb unless the limit is lifted.
    monkeypatch.setattr(Image, "MAX_IMAGE_PIXELS", 1)
    root = _write_raw(tmp_path / "raw", rs24_mask=np.full((2, 2), 1, dtype=np.uint8))

    assert benthos_yuval.run(_args(tmp_path, root), fake_s3) == 0
    assert Image.MAX_IMAGE_PIXELS == 1


def test_run_restores_pixel_limit_when_decoding_a_mosaic_fails(tmp_path, fake_s3, monkeypatch):
    monkeypatch.setattr(Image, "MAX_IMAGE_PIXELS", 1)
    root = _write_raw(tmp_path / "raw", rs24_mask=np.full((2, 2), 1, dtype=np.uint8))
    (root / "RS24.png").write_bytes(b"not a png")

    with pytest.raises(UnidentifiedImageError):
        benthos_yuval.run(_args(tmp_path, root), fake_s3)

    assert Image.MAX_IMAGE_PIXELS == 1


@pytest.mark.parametrize("module", [benthos_yuval, images])
def test_importing_a_module_leaves_pixel_limit_unchanged(module, monkeypatch):
    monkeypatch.setattr(Image, "MAX_IMAGE_PIXELS", 12345)

    importlib.reload(module)

    assert Image.MAX_IMAGE_PIXELS == 12345
