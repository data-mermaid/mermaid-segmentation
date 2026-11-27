from __future__ import annotations

import argparse
import io
import json
import zipfile
from pathlib import Path

import pandas as pd
import pyarrow.parquet as pq
import pytest

from segdata_etl.ingest import catlin_seaview

from .raw_trees import CATLIN_CSV_HEADER, jpg_bytes, write_catlin_tree

B = "bkt"
P = "ds"


def _args(tmp_path: Path, data_dir: Path | None, **overrides) -> argparse.Namespace:
    values = {
        "bucket": B,
        "prefix": P,
        "workers": 2,
        "dry_run": False,
        "skip_existing_s3": True,
        "overwrite": False,
        "data_dir": data_dir,
        "workdir": tmp_path / "work",
        "force_download": False,
    }
    values.update(overrides)
    return argparse.Namespace(**values)


def _parquet(fake_s3) -> pd.DataFrame:
    body = fake_s3.objects[(B, f"{P}/catlin_seaview_annotations.parquet")]
    return pq.read_table(io.BytesIO(body)).to_pandas()


def _json(fake_s3, name: str):
    return json.loads(fake_s3.objects[(B, f"{P}/{name}")])


def _image_keys(fake_s3) -> list[str]:
    return sorted(k for (_, k) in fake_s3.objects if k.startswith(f"{P}/images/"))


def test_ca1_two_regions_publish_parquet_images_and_classes(tmp_path, fake_s3):
    jpg = jpg_bytes(64)
    root = write_catlin_tree(
        tmp_path / "raw",
        {
            "ATL": f"{CATLIN_CSV_HEADER}\n1,10.6,3.4,Sand\n1,20,30,Coral\n2,5,6,Sand\n",
            "PAC_AUS": f"{CATLIN_CSV_HEADER}\n7,1,2,Algae\n",
        },
        {"ATL/1": jpg, "ATL/2": jpg, "PAC_AUS/7": jpg},
    )

    assert catlin_seaview.run(_args(tmp_path, root), fake_s3) == 0

    df = _parquet(fake_s3)
    assert list(df.columns) == ["region", "image_id", "row", "col", "source_label_name"]
    assert df.to_dict("records") == [
        {"region": "ATL", "image_id": "1", "row": 3, "col": 11, "source_label_name": "Sand"},
        {"region": "ATL", "image_id": "1", "row": 30, "col": 20, "source_label_name": "Coral"},
        {"region": "ATL", "image_id": "2", "row": 6, "col": 5, "source_label_name": "Sand"},
        {"region": "PAC_AUS", "image_id": "7", "row": 2, "col": 1, "source_label_name": "Algae"},
    ]
    assert [str(t) for t in df.dtypes] == ["object", "object", "int64", "int64", "object"]
    assert _image_keys(fake_s3) == [
        "ds/images/ATL/1.jpg",
        "ds/images/ATL/2.jpg",
        "ds/images/PAC_AUS/7.jpg",
    ]
    assert _json(fake_s3, "classes.json") == {"unlabeled": 0, "Algae": 1, "Coral": 2, "Sand": 3}
    manifest = _json(fake_s3, "manifest.json")
    assert manifest["source"]["citation"]
    assert manifest["source"]["raw_data_url"] == "http://data.qld.edu.au/public/Q1281"
    assert manifest["source"]["license"] == "CC BY-NC-SA 4.0"
    assert manifest["s3"]["image_key_template"] == "ds/images/{region}/{image_id}.jpg"
    assert manifest["ingestion"]["images_uploaded"] == 3
    assert manifest["ingestion"]["dropped"] == {"damaged_images": 0}


def test_ca2_truncated_jpg_is_dropped_recorded_and_not_uploaded(tmp_path, fake_s3):
    jpg = jpg_bytes(64)
    root = write_catlin_tree(
        tmp_path / "raw",
        {"ATL": f"{CATLIN_CSV_HEADER}\n1,1,1,Sand\n2,2,2,Coral\n"},
        {"ATL/1": jpg, "ATL/2": jpg[: len(jpg) // 2]},
    )

    catlin_seaview.run(_args(tmp_path, root), fake_s3)

    assert _parquet(fake_s3)["image_id"].tolist() == ["1"]
    ingestion = _json(fake_s3, "manifest.json")["ingestion"]
    assert ingestion["dropped"] == {"damaged_images": 1}
    assert ingestion["details"]["damaged_images"]["ATL"] == ["2"]
    assert _image_keys(fake_s3) == ["ds/images/ATL/1.jpg"]
    assert _json(fake_s3, "classes.json") == {"unlabeled": 0, "Sand": 1}


def test_ca3_rows_with_nan_fields_are_dropped(tmp_path, fake_s3):
    root = write_catlin_tree(
        tmp_path / "raw",
        {"ATL": (f"{CATLIN_CSV_HEADER}\n1,1,1,Sand\n,2,2,Sand\n1,,3,Sand\n1,4,,Sand\n1,5,5,\n")},
        {"ATL/1": jpg_bytes(64)},
    )

    catlin_seaview.run(_args(tmp_path, root), fake_s3)

    assert len(_parquet(fake_s3)) == 1


def test_ca4_missing_region_csv_is_skipped_others_ingest(tmp_path, fake_s3):
    root = write_catlin_tree(
        tmp_path / "raw",
        {"IND_CHA": f"{CATLIN_CSV_HEADER}\n3,1,1,Sand\n"},
        {"IND_CHA/3": jpg_bytes(64)},
    )

    catlin_seaview.run(_args(tmp_path, root), fake_s3)

    assert _parquet(fake_s3)["region"].tolist() == ["IND_CHA"]


def test_ca4_no_region_csvs_raises_runtime_error(tmp_path, fake_s3):
    root = write_catlin_tree(tmp_path / "raw", {}, {})

    with pytest.raises(RuntimeError, match="nothing to ingest"):
        catlin_seaview.run(_args(tmp_path, root), fake_s3)
    assert fake_s3.objects == {}


def test_ca5_csv_missing_required_column_raises_naming_it(tmp_path, fake_s3):
    root = write_catlin_tree(tmp_path / "raw", {"ATL": "quadratid,x,label_name\n1,1,Sand\n"}, {})

    with pytest.raises(ValueError, match=r"annotations_ATL\.csv.*\['y'\]"):
        catlin_seaview.run(_args(tmp_path, root), fake_s3)


def test_ca7_data_dir_without_tabular_data_raises(tmp_path, fake_s3):
    empty = tmp_path / "empty"
    empty.mkdir()

    with pytest.raises(FileNotFoundError, match="tabular-data"):
        catlin_seaview.run(_args(tmp_path, empty), fake_s3)


def test_data_dir_that_does_not_exist_raises(tmp_path, fake_s3):
    with pytest.raises(FileNotFoundError, match="--data-dir does not exist"):
        catlin_seaview.run(_args(tmp_path, tmp_path / "nope"), fake_s3)


def test_skip_existing_s3_skips_present_images_and_no_flag_reuploads(tmp_path, fake_s3):
    root = write_catlin_tree(
        tmp_path / "raw",
        {"ATL": f"{CATLIN_CSV_HEADER}\n1,1,1,Sand\n"},
        {"ATL/1": jpg_bytes(64)},
    )
    fake_s3.put(B, "ds/images/ATL/1.jpg", b"old")

    catlin_seaview.run(_args(tmp_path, root), fake_s3)
    assert fake_s3.objects[(B, "ds/images/ATL/1.jpg")] == b"old"

    catlin_seaview.run(_args(tmp_path, root, skip_existing_s3=False, overwrite=True), fake_s3)
    assert fake_s3.objects[(B, "ds/images/ATL/1.jpg")] != b"old"


def test_stale_manifest_without_overwrite_raises_before_any_image_upload(tmp_path, fake_s3):
    root = write_catlin_tree(
        tmp_path / "raw",
        {"ATL": f"{CATLIN_CSV_HEADER}\n1,1,1,Sand\n"},
        {"ATL/1": jpg_bytes(64)},
    )
    fake_s3.put(B, "ds/manifest.json", b"old")

    with pytest.raises(FileExistsError, match="manifest.json.*--overwrite"):
        catlin_seaview.run(_args(tmp_path, root), fake_s3)

    assert _image_keys(fake_s3) == []
    assert list(fake_s3.objects) == [(B, "ds/manifest.json")]


def test_dry_run_writes_nothing(tmp_path, fake_s3):
    root = write_catlin_tree(
        tmp_path / "raw",
        {"ATL": f"{CATLIN_CSV_HEADER}\n1,1,1,Sand\n"},
        {"ATL/1": jpg_bytes(64)},
    )

    catlin_seaview.run(_args(tmp_path, root, dry_run=True), fake_s3)

    assert fake_s3.objects == {}


class _FakeResponse:
    def __init__(self, body: bytes):
        self._body = body
        self.headers = {"Content-Length": str(len(body))}

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False

    def raise_for_status(self) -> None:
        return None

    def iter_content(self, chunk_size: int):
        yield self._body


def _zip_bytes(members: dict[str, bytes | str]) -> bytes:
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w") as zf:
        for name, data in members.items():
            zf.writestr(name, data)
    return buf.getvalue()


def _download_bodies() -> dict[str, bytes]:
    base = catlin_seaview.BASE_URL
    bodies = {
        f"{base}/tabular-data.zip": _zip_bytes(
            {"tabular-data/annotations_ATL.csv": f"{CATLIN_CSV_HEADER}\n1,1,1,Sand\n"}
        )
    }
    for region in catlin_seaview.REGIONS:
        members = {f"{region}/1.jpg": jpg_bytes(64)} if region == "ATL" else {}
        bodies[f"{base}/annotated-images/{region}.zip"] = _zip_bytes(
            members or {f"{region}/readme.txt": "x"}
        )
    return bodies


def test_ca6_download_extracts_into_workdir_and_second_run_is_a_noop(
    tmp_path, fake_s3, monkeypatch
):
    bodies = _download_bodies()
    calls: list[str] = []

    def fake_get(url, **kwargs):
        calls.append(url)
        return _FakeResponse(bodies[url])

    monkeypatch.setattr(catlin_seaview.requests, "get", fake_get)
    args = _args(tmp_path, None)

    catlin_seaview.run(args, fake_s3)

    work = tmp_path / "work"
    assert len(calls) == 10
    assert (work / "ATL" / "1.jpg").exists()
    assert (work / ".tabular-data.zip.extracted").exists()
    assert (work / ".ATL.zip.extracted").exists()
    assert _image_keys(fake_s3) == ["ds/images/ATL/1.jpg"]

    def no_extract(*a, **k):
        raise AssertionError("extracted again")

    monkeypatch.setattr(catlin_seaview.zipfile, "ZipFile", no_extract)
    catlin_seaview.run(_args(tmp_path, None, overwrite=True), fake_s3)

    assert len(calls) == 10


def test_force_download_fetches_again(tmp_path, fake_s3, monkeypatch):
    bodies = _download_bodies()
    calls: list[str] = []

    def fake_get(url, **kwargs):
        calls.append(url)
        return _FakeResponse(bodies[url])

    monkeypatch.setattr(catlin_seaview.requests, "get", fake_get)
    catlin_seaview.run(_args(tmp_path, None), fake_s3)
    catlin_seaview.run(_args(tmp_path, None, force_download=True, overwrite=True), fake_s3)

    assert len(calls) == 20
