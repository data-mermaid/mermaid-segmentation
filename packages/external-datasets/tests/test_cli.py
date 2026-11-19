from __future__ import annotations

import io
from pathlib import Path

import pytest
from PIL import Image

from external_datasets import cli

CSV = "quadratid,x,y,label_name\n1,1,1,Sand\n"


def _tiny_tree(root: Path) -> Path:
    (root / "tabular-data").mkdir(parents=True)
    (root / "tabular-data" / "annotations_ATL.csv").write_text(CSV)
    (root / "ATL").mkdir()
    buf = io.BytesIO()
    Image.new("RGB", (8, 8), (10, 20, 30)).save(buf, format="JPEG")
    (root / "ATL" / "1.jpg").write_bytes(buf.getvalue())
    return root


def test_c1_dry_run_exits_zero_and_writes_nothing(tmp_path, fake_s3, monkeypatch):
    monkeypatch.setattr(cli, "make_s3_client", lambda: fake_s3)
    tree = _tiny_tree(tmp_path / "raw")

    code = cli.main(["catlin_seaview", "--data-dir", str(tree), "--dry-run"])

    assert code == 0
    assert fake_s3.objects == {}


def test_c2_second_run_without_overwrite_fails_and_keeps_parquet(tmp_path, fake_s3, monkeypatch):
    monkeypatch.setattr(cli, "make_s3_client", lambda: fake_s3)
    tree = _tiny_tree(tmp_path / "raw")
    key = (
        cli.DEFAULT_BUCKET,
        "external_validation_datasets/catlin_seaview/catlin_seaview_annotations.parquet",
    )

    assert cli.main(["catlin_seaview", "--data-dir", str(tree)]) == 0
    first = fake_s3.objects[key]

    with pytest.raises(FileExistsError, match="--overwrite"):
        cli.main(["catlin_seaview", "--data-dir", str(tree)])
    assert fake_s3.objects[key] == first
