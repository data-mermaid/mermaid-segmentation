"""Tests for ``segdata build coralscapes_v2`` and the CLI wiring of both Hugging Face builds."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from segdata_etl import cli
from segdata_etl.datasets import coralscapes_v2, ucsd_mosaics
from segdata_spec import DATASET_BUCKET

from .conftest import crc64nvme_b64, jpeg_bytes, png_bytes
from .hf_helpers import image_col, install_hub, mask_png

SHA = "8a4567b2fbec14882e09aebff63a7a1ca5e24dca"
CSV = "label_id,name,mermaid_benthic_attribute\n1,alpha,hard coral\n2,beta,\n"
ID2LABEL = {"1": "alpha", "2": "beta", "7": "gamma"}
# 4x3 mask: label 0 x5, label 2 x4, label 7 x3.
PIXELS = [0, 0, 0, 0, 0, 2, 2, 2, 2, 7, 7, 7]

IMG_A = jpeg_bytes(8, 6)
IMG_B = png_bytes(5, 4)
MASK = mask_png(PIXELS, 4, 3)
MASK_B = mask_png([1] * 6, 3, 2)


def shard(stems: list[str], images: list[bytes], masks: list[bytes], site: str = "s1"):
    return {
        "image": image_col(images),
        "label": image_col(masks),
        "instance_ids": image_col([b"INSTANCE-" + s.encode() for s in stems]),
        "stem": pa.array(stems),
        "site": pa.array([site] * len(stems)),
    }


def default_shards() -> dict[str, dict[str, Any]]:
    return {
        "instances/train-00000-of-00001.parquet": shard(["a1"], [IMG_A], [MASK]),
        "instances/validation-00000-of-00001.parquet": shard(["b1"], [IMG_B], [MASK_B], "s2"),
        "data/train-00000-of-00001.parquet": shard(["ignored"], [IMG_A], [MASK]),
    }


def build(monkeypatch, tmp_path, *, shards=None, csv=CSV, extra=()) -> Path:
    hub = install_hub(
        monkeypatch,
        tmp_path,
        vocabulary=("id2label.json", ID2LABEL),
        curated_csv=("coralscapes_v2", csv),
        shards=shards or default_shards(),
    )
    stage = tmp_path / "stage"
    args = cli.build_parser().parse_args(
        ["build", "coralscapes_v2", "--stage-dir", str(stage), *extra]
    )
    assert args.run(args, object()) == 0
    assert {r for _, _, r in hub.downloaded} == {SHA}
    (version,) = (stage / DATASET_BUCKET / "datasets" / "coralscapes_v2").iterdir()
    return version


def table(version: Path, name: str) -> list[dict[str, Any]]:
    return pq.read_table(version / name).to_pylist()


def images_root(version: Path) -> Path:
    return version.parents[2] / "images" / "coralscapes_v2"


def test_images_are_staged_byte_identical_with_row_fields_from_the_bytes(monkeypatch, tmp_path):
    version = build(monkeypatch, tmp_path)

    rows = {r["image_id"]: r for r in table(version, "images.parquet")}
    a = rows["a1"]
    assert (images_root(version) / "train" / "a1.jpg").read_bytes() == IMG_A
    assert (images_root(version) / "train" / "a1_label.png").read_bytes() == MASK
    assert a["key"] == "train/a1.jpg"
    assert a["mask_key"] == "train/a1_label.png"
    assert a["group"] == "train"
    assert a["source_name"] == "a1"
    assert json.loads(a["attrs"]) == {"hf_split": "train", "site": "s1"}
    assert (a["media_type"], a["width"], a["height"]) == ("image/jpeg", 8, 6)
    assert a["etag"] == hashlib.md5(IMG_A).hexdigest()  # noqa: S324
    assert (a["size"], a["crc64nvme"]) == (len(IMG_A), crc64nvme_b64(IMG_A))
    assert a["mask_etag"] == hashlib.md5(MASK).hexdigest()  # noqa: S324
    assert (a["mask_size"], a["mask_crc64nvme"]) == (len(MASK), crc64nvme_b64(MASK))
    assert a["source_fingerprint"] == hashlib.sha256(IMG_A).hexdigest()
    assert rows["b1"]["group"] == "validation"
    assert rows["b1"]["media_type"] == "image/png"
    assert set(rows) == {"a1", "b1"}
    manifest = json.loads((version / "manifest.json").read_bytes())
    assert manifest["kind"] == "dense"
    assert manifest["source"] == {
        "type": "huggingface",
        "uri": "hf://datasets/josauder/coralscapesV2",
        "revision": SHA,
        "citation": manifest["source"]["citation"],
        "license": "apache-2.0",
    }
    assert manifest["image_store"]["prefix"] == "images/coralscapes_v2/"
    assert manifest["etl"]["command"] == (
        f"segdata build coralscapes_v2 --stage-dir {version.parents[3]} --revision {SHA}"
    )


def test_instance_ids_are_not_staged(monkeypatch, tmp_path):
    version = build(monkeypatch, tmp_path)

    staged = sorted(p.name for p in images_root(version).rglob("*") if p.is_file())
    assert staged == ["a1.jpg", "a1_label.png", "b1.png", "b1_label.png"]


def test_labels_join_the_curated_map_and_leave_absent_ids_unmapped(monkeypatch, tmp_path):
    version = build(monkeypatch, tmp_path)

    labels = {r["label_id"]: r for r in table(version, "labels.parquet")}
    assert labels[1] == {
        "label_id": 1,
        "name": "alpha",
        "mermaid_benthic_attribute_id": None,
        "mermaid_benthic_attribute": "hard coral",
        "mermaid_growth_form": None,
        "mapping_source": "curated",
    }
    assert labels[2]["mermaid_benthic_attribute"] is None
    assert labels[7]["mermaid_benthic_attribute"] is None
    assert labels[7]["name"] == "gamma"
    mapping = json.loads((version / "manifest.json").read_bytes())["label_mapping"]
    assert {k: mapping[k] for k in ("source", "sha256", "n_labels", "n_mapped")} == {
        "source": "curated",
        "sha256": hashlib.sha256(CSV.encode()).hexdigest(),
        "n_labels": 3,
        "n_mapped": 1,
    }


def test_curated_name_disagreeing_with_the_vocabulary_fails_the_build(monkeypatch, tmp_path):
    with pytest.raises(ValueError, match="alpha"):
        build(monkeypatch, tmp_path, csv="label_id,name,mermaid_benthic_attribute\n1,wrong,x\n")

    assert not (tmp_path / "stage" / DATASET_BUCKET / "datasets").exists()


def test_label_counts_are_mask_pixel_counts_without_label_zero(monkeypatch, tmp_path):
    version = build(monkeypatch, tmp_path)

    counts = sorted(
        (c["image_id"], c["label_id"], c["n"]) for c in table(version, "stats/label_counts.parquet")
    )
    assert counts == [("a1", 2, 4), ("a1", 7, 3), ("b1", 1, 6)]


def test_gif_image_is_dropped_with_its_mask_and_its_id_logged(monkeypatch, tmp_path, caplog):
    shards = default_shards()
    shards["instances/train-00000-of-00001.parquet"] = shard(
        ["a1", "gifimg"], [IMG_A, b"GIF89a-not-supported"], [MASK, MASK]
    )

    with caplog.at_level("WARNING"):
        version = build(monkeypatch, tmp_path, shards=shards)

    assert {r["image_id"] for r in table(version, "images.parquet")} == {"a1", "b1"}
    assert not (images_root(version) / "train" / "gifimg_label.png").exists()
    dropped = json.loads((version / "manifest.json").read_bytes())["stats"]["dropped"]
    assert dropped["image_unreadable_header"] == 1
    assert "gifimg" in caplog.text


def test_mask_that_is_not_png_drops_the_image(monkeypatch, tmp_path):
    shards = default_shards()
    shards["instances/train-00000-of-00001.parquet"] = shard(
        ["a1", "badmask"], [IMG_A, IMG_A], [MASK, jpeg_bytes(8, 6)]
    )

    version = build(monkeypatch, tmp_path, shards=shards)

    assert {r["image_id"] for r in table(version, "images.parquet")} == {"a1", "b1"}
    assert not (images_root(version) / "train" / "badmask.jpg").exists()
    dropped = json.loads((version / "manifest.json").read_bytes())["stats"]["dropped"]
    assert dropped["image_unreadable_header"] == 1


def test_duplicate_stems_fail_the_build(monkeypatch, tmp_path):
    shards = default_shards()
    shards["instances/validation-00000-of-00001.parquet"] = shard(["a1"], [IMG_B], [MASK_B])

    with pytest.raises(ValueError, match="a1"):
        build(monkeypatch, tmp_path, shards=shards)


def test_uploads_jsonl_lists_each_staged_file_with_its_destination(monkeypatch, tmp_path):
    version = build(monkeypatch, tmp_path)

    lines = (version / "uploads.jsonl").read_text().splitlines()
    uploads = {u["dst_key"]: u for u in map(json.loads, lines)}
    assert sorted(uploads) == [
        "images/coralscapes_v2/train/a1.jpg",
        "images/coralscapes_v2/train/a1_label.png",
        "images/coralscapes_v2/validation/b1.png",
        "images/coralscapes_v2/validation/b1_label.png",
    ]
    assert uploads["images/coralscapes_v2/train/a1.jpg"] == {
        "local_path": f"{DATASET_BUCKET}/images/coralscapes_v2/train/a1.jpg",
        "dst_bucket": DATASET_BUCKET,
        "dst_key": "images/coralscapes_v2/train/a1.jpg",
        "size": len(IMG_A),
        "crc64nvme": crc64nvme_b64(IMG_A),
        "content_type": "image/jpeg",
    }
    assert uploads["images/coralscapes_v2/train/a1_label.png"]["content_type"] == "image/png"


@pytest.mark.parametrize("module", [coralscapes_v2, ucsd_mosaics])
def test_stage_dir_is_required_and_the_subcommand_dispatches_to_run(monkeypatch, tmp_path, module):
    name = module.__name__.rsplit(".", 1)[-1]
    with pytest.raises(SystemExit):
        cli.build_parser().parse_args(["build", name])
    seen: dict[str, Any] = {}

    def fake_run(args: Any, client: Any) -> int:
        seen["args"] = args
        return 7

    monkeypatch.setattr(module, "run", fake_run)
    monkeypatch.setattr(cli, "make_s3_client", lambda: object())

    code = cli.main(["build", name, "--stage-dir", str(tmp_path)])

    assert code == 7
    assert seen["args"].stage_dir == tmp_path
    assert seen["args"].revision == module.SOURCE.revision
