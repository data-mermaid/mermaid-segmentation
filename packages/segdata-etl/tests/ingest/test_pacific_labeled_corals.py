from __future__ import annotations

import argparse
import io
import json
from pathlib import Path

import pandas as pd
import pyarrow.parquet as pq
import pytest
from PIL import Image

from segdata_etl.ingest import pacific_labeled_corals as plc

B = "bkt"
P = "ds"
SITES = ("heron_reef", "line_islands", "nanwan_bay")
LABELMAP = "labelid,name\n1,Sand\n2,Acropora\n3,Turf\n"
EVAL_ANN = "imageid,row,col,archived,host,visitor1,visitor2,visitor3,visitor4,visitor5\n"
REF_ANN = "imageid,row,col,archived\n"


def _jpg(path: Path, *, valid: bool = True) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if valid:
        buf = io.BytesIO()
        Image.effect_noise((32, 32), 80).convert("RGB").save(buf, format="JPEG")
        path.write_bytes(buf.getvalue())
    else:
        path.write_bytes(b"not a jpeg")


def _write_subset(
    root: Path,
    site: str,
    subset: str,
    *,
    imagemap: str = "imageid,imagefile\n1,img1.JPG\n",
    annotations: str | None = None,
    valid: bool = True,
) -> None:
    subset_dir = root / site / subset
    subset_dir.mkdir(parents=True, exist_ok=True)
    (subset_dir / "imagemap.txt").write_text(imagemap)
    default = REF_ANN + "1,10,20,1\n" if subset == "reference" else EVAL_ANN + "1,10,20,1,2,3,,,,\n"
    (subset_dir / "annotations.txt").write_text(annotations or default)
    for line in imagemap.splitlines()[1:]:
        _jpg(subset_dir / "imgs" / line.split(",")[1], valid=valid)


def _write_tree(root: Path, **kwargs) -> Path:
    for site in SITES:
        (root / site).mkdir(parents=True, exist_ok=True)
        (root / site / "labelmap.txt").write_text(LABELMAP)
        for subset in ("reference", "evaluation"):
            _write_subset(root, site, subset, **kwargs)
    return root


def _args(data_dir: Path, **overrides) -> argparse.Namespace:
    values = {
        "bucket": B,
        "prefix": P,
        "workers": 2,
        "dry_run": False,
        "skip_existing_s3": True,
        "overwrite": False,
        "data_dir": data_dir,
    }
    values.update(overrides)
    return argparse.Namespace(**values)


def _parquet(fake_s3) -> pd.DataFrame:
    body = fake_s3.objects[(B, f"{P}/pacific_labeled_corals_annotations.parquet")]
    return pq.read_table(io.BytesIO(body)).to_pandas()


def test_pa1_three_sites_publish_parquet_images_classes_and_manifest(tmp_path, fake_s3):
    root = _write_tree(tmp_path / "raw")
    fake_s3.put(B, f"{P}/images/heron_reef/evaluation/img1.JPG", b"already-there")

    assert plc.run(_args(root), fake_s3) == 0

    df = _parquet(fake_s3)
    assert list(df.columns) == [
        "site",
        "subset",
        "raw_imageid",
        "image_id",
        "image_ext",
        "row",
        "col",
        "archived",
        "host",
        "visitor1",
        "visitor2",
        "visitor3",
        "visitor4",
        "visitor5",
    ]
    assert len(df) == 6
    first = df.iloc[0]
    assert (first["site"], first["subset"], first["image_id"], first["image_ext"]) == (
        "heron_reef",
        "evaluation",
        "img1",
        ".JPG",
    )
    assert (first["archived"], first["host"], first["visitor1"]) == ("Sand", "Acropora", "Turf")
    assert pd.isna(first["visitor2"])
    ref = df[(df["site"] == "heron_reef") & (df["subset"] == "reference")].iloc[0]
    assert ref["archived"] == "Sand" and pd.isna(ref["host"]) and pd.isna(ref["visitor5"])
    assert df["raw_imageid"].dtype == "int64"
    assert df["row"].dtype == "int64" and df["col"].dtype == "int64"

    assert sorted(k for (_, k) in fake_s3.objects if k.startswith(f"{P}/images/")) == [
        f"{P}/images/{site}/{subset}/img1.JPG"
        for site in SITES
        for subset in ("evaluation", "reference")
    ]
    classes = json.loads(fake_s3.objects[(B, f"{P}/classes.json")])
    assert classes == {"unlabeled": 0, "Acropora": 1, "Sand": 2, "Turf": 3}
    assert (B, f"{P}/colors.json") in fake_s3.objects
    manifest = json.loads(fake_s3.objects[(B, f"{P}/manifest.json")])
    assert manifest["dataset_name"] == "pacific_labeled_corals"
    assert manifest["source"]["license"] == "CC0 1.0 (Dryad)"
    assert manifest["ingestion"] == {
        "excluded_sites": ["moorea"],
        "images_uploaded": 5,
        "annotations_dropped_unknown_archived": 0,
        "annotations_dropped_unmapped_imageid": 0,
    }
    assert (
        manifest["s3"]["image_key_template"]
        == f"{P}/images/{{site}}/{{subset}}/{{image_id}}{{image_ext}}"
    )


def test_pa2_undecodable_image_raises_and_writes_nothing(tmp_path, fake_s3):
    root = _write_tree(tmp_path / "raw")
    _jpg(root / "nanwan_bay" / "evaluation" / "imgs" / "img1.JPG", valid=False)

    with pytest.raises(RuntimeError, match="img1.JPG"):
        plc.run(_args(root), fake_s3)

    assert fake_s3.objects == {}


def test_pa3_differing_labelmaps_raise_value_error(tmp_path, fake_s3):
    root = _write_tree(tmp_path / "raw")
    (root / "nanwan_bay" / "labelmap.txt").write_text("labelid,name\n1,Sand\n2,Coral\n")

    with pytest.raises(ValueError, match="nanwan_bay"):
        plc.run(_args(root), fake_s3)


def test_pa4_unknown_archived_id_drops_row_and_unknown_visitor_id_is_na(tmp_path, fake_s3, caplog):
    root = _write_tree(tmp_path / "raw")
    ann = EVAL_ANN + "1,1,1,99,,,,,,\n1,2,2,1,,99,,,,\n"
    _write_subset(root, "heron_reef", "evaluation", annotations=ann)

    with caplog.at_level("WARNING"):
        plc.run(_args(root), fake_s3)

    df = _parquet(fake_s3)
    kept = df[(df["site"] == "heron_reef") & (df["subset"] == "evaluation")]
    assert kept[["row", "col"]].to_numpy().tolist() == [[2, 2]]
    assert kept.iloc[0]["archived"] == "Sand"
    assert pd.isna(kept.iloc[0]["visitor1"])
    ingestion = json.loads(fake_s3.objects[(B, f"{P}/manifest.json")])["ingestion"]
    assert ingestion["annotations_dropped_unknown_archived"] == 1
    assert ingestion["annotations_dropped_unmapped_imageid"] == 0
    assert "unknown archived" in caplog.text


def test_pa5_image_column_is_accepted_and_neither_column_raises_key_error(tmp_path, fake_s3):
    root = _write_tree(tmp_path / "raw")
    _write_subset(root, "heron_reef", "reference", imagemap="imageid,image\n1,img1.JPG\n")
    plc.run(_args(root), fake_s3)
    assert f"{P}/images/heron_reef/reference/img1.JPG" in {k for (_, k) in fake_s3.objects}

    root2 = _write_tree(tmp_path / "raw2")
    (root2 / "heron_reef" / "reference" / "imagemap.txt").write_text("imageid,other\n1,img1.JPG\n")
    with pytest.raises(KeyError):
        plc.run(_args(root2), type(fake_s3)())


@pytest.mark.parametrize("missing", ["heron_reef", "heron_reef/evaluation"])
def test_pa6_missing_site_or_subset_directory_raises(tmp_path, fake_s3, missing):
    import shutil

    root = _write_tree(tmp_path / "raw")
    shutil.rmtree(root / missing)

    with pytest.raises(FileNotFoundError):
        plc.run(_args(root), fake_s3)


def test_pa7_annotation_for_unmapped_imageid_is_skipped(tmp_path, fake_s3, caplog):
    root = _write_tree(tmp_path / "raw")
    ann = REF_ANN + "1,10,20,1\n5,1,1,1\n"
    _write_subset(root, "line_islands", "reference", annotations=ann)

    with caplog.at_level("WARNING"):
        plc.run(_args(root), fake_s3)

    df = _parquet(fake_s3)
    sub = df[(df["site"] == "line_islands") & (df["subset"] == "reference")]
    assert sub["raw_imageid"].tolist() == [1]
    ingestion = json.loads(fake_s3.objects[(B, f"{P}/manifest.json")])["ingestion"]
    assert ingestion["annotations_dropped_unmapped_imageid"] == 1
    assert ingestion["annotations_dropped_unknown_archived"] == 0
    assert "unmapped imageid" in caplog.text


def test_existing_manifest_without_overwrite_raises_before_any_write(tmp_path, fake_s3):
    root = _write_tree(tmp_path / "raw")
    fake_s3.put(B, f"{P}/manifest.json", b"{}")

    with pytest.raises(FileExistsError):
        plc.run(_args(root), fake_s3)

    assert list(fake_s3.objects) == [(B, f"{P}/manifest.json")]
