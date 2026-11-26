from __future__ import annotations

import argparse
import io
import json
from pathlib import Path

import pandas as pd
import pyarrow.parquet as pq
import pytest

from segdata_etl.ingest import moorea_labeled_corals

from .raw_trees import add_moorea_image, jpg_bytes

B = "bkt"
P = "ds"


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
    body = fake_s3.objects[(B, f"{P}/moorea_labeled_corals_annotations.parquet")]
    return pq.read_table(io.BytesIO(body)).to_pandas()


def _json(fake_s3, name: str):
    return json.loads(fake_s3.objects[(B, f"{P}/{name}")])


def _image_keys(fake_s3) -> list[str]:
    return sorted(k for (_, k) in fake_s3.objects if k.startswith(f"{P}/images/"))


def test_mo1_two_years_publish_parquet_and_image_keys(tmp_path, fake_s3):
    root = tmp_path / "raw"
    add_moorea_image(root, "2008", "a.jpg", ["10;20;Sand", "30;40;Coral"])
    add_moorea_image(root, "2009", "b.png", ["5;6;Sand"])
    fake_s3.put(B, f"{P}/images/2008/a.jpg", b"already-there")

    assert moorea_labeled_corals.run(_args(root), fake_s3) == 0

    df = _parquet(fake_s3)
    assert list(df.columns) == [
        "year",
        "image_id",
        "image_ext",
        "row",
        "col",
        "source_label_name",
        "raw_label_name",
    ]
    assert df.to_dict("records") == [
        {
            "year": "2008",
            "image_id": "a",
            "image_ext": ".jpg",
            "row": 10,
            "col": 20,
            "source_label_name": "Sand",
            "raw_label_name": "Sand",
        },
        {
            "year": "2008",
            "image_id": "a",
            "image_ext": ".jpg",
            "row": 30,
            "col": 40,
            "source_label_name": "Coral",
            "raw_label_name": "Coral",
        },
        {
            "year": "2009",
            "image_id": "b",
            "image_ext": ".png",
            "row": 5,
            "col": 6,
            "source_label_name": "Sand",
            "raw_label_name": "Sand",
        },
    ]
    assert [str(t) for t in df.dtypes] == ["object"] * 3 + ["int64"] * 2 + ["object"] * 2
    assert _image_keys(fake_s3) == ["ds/images/2008/a.jpg", "ds/images/2009/b.png"]
    assert _json(fake_s3, "classes.json") == {"unlabeled": 0, "Coral": 1, "Sand": 2}
    manifest = _json(fake_s3, "manifest.json")
    assert manifest["source"] == {
        "citation": "Beijbom et al., 'Automated Annotation of Coral Reef Survey Images', "
        "IEEE CVPR, 2012.",
        "raw_data_url": "https://portal.edirepository.org/nis/mapbrowse"
        "?scope=knb-lter-mcr&identifier=5006",
        "license": "CC BY 4.0",
    }
    assert manifest["s3"]["image_key_template"] == "ds/images/{year}/{image_id}{image_ext}"
    assert manifest["ingestion"]["images_uploaded"] == 1
    assert manifest["ingestion"]["dropped"] == {"damaged_images": 0, "off_annotations": 0}


@pytest.mark.parametrize(
    ("raw", "canonical"),
    [
        (["ACROPORA", "Acropora"], {"ACROPORA": "Acropora", "Acropora": "Acropora"}),
        (["TURF"], {"TURF": "TURF"}),
        (["Macro", "macro"], {"Macro": "Macro", "macro": "Macro"}),
    ],
)
def test_mo2_case_variants_merge_to_preferred_spelling(tmp_path, fake_s3, raw, canonical):
    root = tmp_path / "raw"
    add_moorea_image(root, "2008", "a.jpg", [f"{i};{i};{label}" for i, label in enumerate(raw)])

    moorea_labeled_corals.run(_args(root), fake_s3)

    df = _parquet(fake_s3)
    assert dict(zip(df["raw_label_name"], df["source_label_name"], strict=True)) == canonical
    merges = _json(fake_s3, "manifest.json")["ingestion"]["details"]["case_merged_labels"]
    assert merges == {k: v for k, v in canonical.items() if k != v}


def test_mo3_off_points_are_dropped_and_counted(tmp_path, fake_s3):
    root = tmp_path / "raw"
    add_moorea_image(root, "2008", "a.jpg", ["1;1;Sand", "2;2;OFF", "3;3;Off", "4;4;off"])

    moorea_labeled_corals.run(_args(root), fake_s3)

    assert _parquet(fake_s3)["raw_label_name"].tolist() == ["Sand"]
    dropped = _json(fake_s3, "manifest.json")["ingestion"]["dropped"]
    assert dropped == {"damaged_images": 0, "off_annotations": 3}


def test_mo4_damaged_image_is_dropped_recorded_and_not_uploaded(tmp_path, fake_s3):
    root = tmp_path / "raw"
    jpg = jpg_bytes()
    add_moorea_image(root, "2008", "good.jpg", ["1;1;Sand"], jpg)
    add_moorea_image(root, "2008", "bad.jpg", ["2;2;Coral"], jpg[: len(jpg) // 2])

    moorea_labeled_corals.run(_args(root), fake_s3)

    assert _parquet(fake_s3)["image_id"].tolist() == ["good"]
    ingestion = _json(fake_s3, "manifest.json")["ingestion"]
    assert ingestion["dropped"] == {"damaged_images": 1, "off_annotations": 0}
    assert ingestion["details"]["damaged_images"] == ["2008/bad.jpg"]
    assert _image_keys(fake_s3) == ["ds/images/2008/good.jpg"]
    assert _json(fake_s3, "classes.json") == {"unlabeled": 0, "Sand": 1}


def test_mo5_image_without_annotation_file_raises(tmp_path, fake_s3):
    root = tmp_path / "raw"
    add_moorea_image(root, "2008", "a.jpg", ["1;1;Sand"])
    (root / "2008" / "a.jpg.txt").unlink()

    with pytest.raises(FileNotFoundError, match=r"a\.jpg"):
        moorea_labeled_corals.run(_args(root), fake_s3)


def test_mo5_no_year_dirs_raises(tmp_path, fake_s3):
    root = tmp_path / "raw"
    root.mkdir()

    with pytest.raises(FileNotFoundError, match="No <year>/<image> pairs"):
        moorea_labeled_corals.run(_args(root), fake_s3)


def test_mo6_malformed_and_comment_lines_are_skipped(tmp_path, fake_s3):
    root = tmp_path / "raw"
    add_moorea_image(root, "2008", "a.jpg", ["# header;1;2", "x;2;Coral", "1;2;Sand", "3;4"])

    moorea_labeled_corals.run(_args(root), fake_s3)

    df = _parquet(fake_s3)
    assert df[["row", "col", "raw_label_name"]].to_dict("records") == [
        {"row": 1, "col": 2, "raw_label_name": "Sand"}
    ]


def test_existing_manifest_without_overwrite_raises_before_any_write(tmp_path, fake_s3):
    root = tmp_path / "raw"
    add_moorea_image(root, "2008", "a.jpg", ["1;1;Sand"])
    fake_s3.put(B, f"{P}/manifest.json", b"{}")

    with pytest.raises(FileExistsError):
        moorea_labeled_corals.run(_args(root), fake_s3)

    assert list(fake_s3.objects) == [(B, f"{P}/manifest.json")]
