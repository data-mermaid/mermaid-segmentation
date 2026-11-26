"""``segdata ingest <name>`` followed by ``segdata build <name>`` on the same prefix.

The build reads the manifest the ingest wrote, so the staged manifest's licence, citation and drop
counts are the ones the ingest recorded for the planted damaged or dropped items.
"""

from __future__ import annotations

from pathlib import Path
from types import ModuleType

import numpy as np
import pytest
from ingest.raw_trees import (
    CATLIN_CSV_HEADER,
    PACIFIC_EVAL_ANN,
    PACIFIC_LABELMAP,
    PACIFIC_SITES,
    add_moorea_image,
    jpg_bytes,
    write_benthos_raw,
    write_catlin_tree,
    write_pacific_subset,
)
from s3_fake import FakeS3

from segdata_etl import cli
from segdata_etl.datasets import (
    benthos_yuval,
    catlin_seaview,
    moorea_labeled_corals,
    pacific_labeled_corals,
)
from segdata_spec.manifest import Manifest

from .conftest import Staged, make_repo, stage_version


def _ingest_then_build(
    name: str,
    builder: ModuleType,
    raw: Path,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
    *workdir_flags: str,
) -> Manifest:
    """Ingest ``raw`` through the CLI parser's defaults, build from that prefix, return the manifest."""
    s3 = FakeS3()
    argv = ["ingest", name, "--data-dir", str(raw), "--workers", "2", *workdir_flags]
    args = cli.build_parser().parse_args(argv)
    assert args.run(args, s3) == 0
    repo = make_repo(tmp_path, monkeypatch)
    staged: Staged = stage_version(builder, s3, repo, tmp_path, capsys)
    return staged.version.manifest


def test_catlin_build_records_the_ingest_source_and_damaged_image_count(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    jpg = jpg_bytes(64)
    raw = write_catlin_tree(
        tmp_path / "raw",
        {
            "ATL": f"{CATLIN_CSV_HEADER}\n1,10,3,Sand\n2,5,6,Sand\n",
            "PAC_AUS": f"{CATLIN_CSV_HEADER}\n7,1,2,Coral\n8,4,4,Coral\n",
        },
        # ATL/2 is truncated and PAC_AUS/8 has no file.
        {"ATL/1": jpg, "ATL/2": jpg[: len(jpg) // 2], "PAC_AUS/7": jpg},
    )

    manifest = _ingest_then_build(
        "catlin_seaview",
        catlin_seaview,
        raw,
        tmp_path,
        monkeypatch,
        capsys,
        "--workdir",
        str(tmp_path / "work"),
    )

    assert manifest.stats.dropped == {"damaged_images": 2}
    assert manifest.source.license == "CC BY-NC-SA 4.0"
    assert manifest.source.citation == (
        "Seaview Survey Photo-quadrat and Image Classification Dataset "
        "(Gonzalez-Rivero et al., 2019)"
    )


def test_moorea_build_records_the_ingest_source_damaged_images_and_off_points(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    raw = tmp_path / "raw"
    add_moorea_image(raw, "2008", "good.jpg", ["1;1;Sand", "2;2;OFF", "3;3;Off"])
    bad = jpg_bytes()
    add_moorea_image(raw, "2008", "bad.jpg", ["2;2;Coral"], bad[: len(bad) // 2])

    manifest = _ingest_then_build(
        "moorea_labeled_corals", moorea_labeled_corals, raw, tmp_path, monkeypatch, capsys
    )

    assert manifest.stats.dropped == {"damaged_images": 1, "off_annotations": 2}
    assert manifest.source.license == "CC BY 4.0"
    assert manifest.source.citation == (
        "Beijbom et al., 'Automated Annotation of Coral Reef Survey Images', IEEE CVPR, 2012."
    )


def test_pacific_build_records_the_ingest_source_and_both_annotation_drop_counts(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    raw = tmp_path / "raw"
    # One heron_reef evaluation row has labelid 99 (in no labelmap), another imageid 5 (in no
    # imagemap).
    planted = PACIFIC_EVAL_ANN + "1,10,20,1,2,3,,,,\n" + "1,2,2,99,,,,,,\n" + "5,1,1,1,,,,,,\n"
    for site in PACIFIC_SITES:
        (raw / site).mkdir(parents=True)
        (raw / site / "labelmap.txt").write_text(PACIFIC_LABELMAP)
        for subset in ("reference", "evaluation"):
            write_pacific_subset(
                raw,
                site,
                subset,
                imagemap=f"imageid,imagefile\n1,{site}_{subset}.JPG\n",
                annotations=planted if (site, subset) == ("heron_reef", "evaluation") else None,
            )

    manifest = _ingest_then_build(
        "pacific_labeled_corals", pacific_labeled_corals, raw, tmp_path, monkeypatch, capsys
    )

    assert manifest.stats.dropped == {
        "annotations_unknown_archived": 1,
        "annotations_unmapped_imageid": 1,
    }
    assert manifest.source.license == "CC0 1.0 (Dryad)"
    assert manifest.source.citation == (
        "Beijbom, O., Edmunds, P. J., Roelfsema, C., Smith, J., Kline, D. I., "
        "Neal, B. P., Dunlap, M. J., Moriarty, V., Fan, T.-Y., Tan, C.-J., "
        "Chan, S., Treibitz, T., Gamst, A., Mitchell, B. G., Kriegman, D. "
        "(2015). Towards Automated Annotation of Benthic Survey Images: "
        "Variability of Human Experts and Operational Modes of Automation. "
        "PLOS ONE, 10(7): e0130312."
    )


def test_benthos_build_records_the_ingest_source_and_no_drops(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    raw = write_benthos_raw(
        tmp_path / "raw", rs24_mask=np.array([[1, 2, 2, 1], [1, 1, 2, 2]], dtype=np.uint8)
    )

    manifest = _ingest_then_build(
        "benthos_yuval",
        benthos_yuval,
        raw,
        tmp_path,
        monkeypatch,
        capsys,
        "--workdir",
        str(tmp_path / "work"),
    )

    assert manifest.stats.dropped == {}
    assert manifest.source.license == "CC0 1.0 (Dryad)"
    assert manifest.source.citation == (
        "Yuval, M., Alonso, I., Murillo, A. C., Treibitz, T. (2021). "
        "Repeatable semantic reef-mapping through photogrammetry and "
        "label-augmentation. Dryad. doi:10.5061/dryad.8cz8w9gm3"
    )
