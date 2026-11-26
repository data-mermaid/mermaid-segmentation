"""training.concept_source_mask and training.concept_removal: loading, datasets and the train step."""

from __future__ import annotations

from typing import Any

import pandas as pd
import pytest
import torch
import yaml
from torch.utils.data import DataLoader

from mermaidseg.data import PointDataset
from mermaidseg.dataset_reconciliation.artifact_meta import file_sha1, meta_path
from mermaidseg.dataset_reconciliation.concept_removal import CoverageRule, concept_coverage
from mermaidseg.dataset_reconciliation.concept_schema import ConceptSchema
from mermaidseg.dataset_reconciliation.concept_source_mask import (
    attach_source_rows,
    load_concept_source_mask,
)
from mermaidseg.dataset_reconciliation.concepts import apply_concept_removal
from mermaidseg.model.concept_masking import apply_concept_source_mask
from tests.mermaidseg.conftest import concept_table
from tests.mermaidseg.data.conftest import ImageSpec, open_written
from tests.mermaidseg.dataset_reconciliation.test_concept_source_mask import CHANNELS, _schema

BLEACHED = CHANNELS.index("bleached")


def _write_mask(
    tmp_path,
    rows: list[tuple[str, str]],
    concepts_csv=None,
    source_ids: list[str] | None = None,
    omit_source_ids: bool = False,
) -> Any:
    """Write a mask CSV and sidecar; ``source_ids`` (the counted sources) default to the rows'."""
    path = tmp_path / "mask.csv"
    pd.DataFrame(rows, columns=["source_id", "concept"]).assign(
        n_true=0, n_false=1, n_not_given=0
    ).to_csv(path, index=False)
    meta: dict[str, Any] = {"dataset": "coralnet", "rule": "only_false"}
    if not omit_source_ids:
        meta["source_ids"] = (
            source_ids if source_ids is not None else sorted({source for source, _ in rows})
        )
    if concepts_csv is not None:
        meta["concepts_csv_sha1"] = file_sha1(concepts_csv)
    meta_path(path).write_text(yaml.safe_dump(meta))
    return path


# ---------- loading the mask ----------


def test_keep_table_rows_and_columns(tmp_path):
    mask = load_concept_source_mask(
        _write_mask(tmp_path, [("12", "bleached"), ("7", "branching")]), _schema()
    )
    assert mask.source_ids == ("7", "12")  # numeric order
    assert mask.num_masked == 2
    keep = mask.keep
    assert keep.shape == (3, len(CHANNELS)) and keep[0].all()  # row 0 keeps everything
    assert not keep[1, CHANNELS.index("branching")] and keep[1].sum() == len(CHANNELS) - 1
    assert not keep[2, BLEACHED] and keep[2].sum() == len(CHANNELS) - 1
    assert mask.rows_for([12, "7", 99]).tolist() == [2, 1, 0]  # unknown source -> row 0


def test_mask_refuses_taxonomic_or_unknown_concepts(tmp_path):
    for concept in ("genus__acropora", "not_a_concept"):
        with pytest.raises(ValueError, match="not binary channels"):
            load_concept_source_mask(_write_mask(tmp_path, [("1", concept)]), _schema())


def test_mask_without_source_ids_in_the_sidecar_is_refused(tmp_path):
    path = _write_mask(tmp_path, [("1", "bleached")], omit_source_ids=True)

    with pytest.raises(ValueError, match=r"source_ids.*build_concept_source_mask"):
        load_concept_source_mask(path, _schema())


def test_mask_skips_removed_concepts(tmp_path):
    path = _write_mask(tmp_path, [("1", "tubular"), ("1", "bleached")])
    with pytest.raises(ValueError, match="tubular"):
        load_concept_source_mask(path, _schema())
    mask = load_concept_source_mask(path, _schema(), removed_concepts=["tubular"])
    assert mask.num_masked == 1


def test_mask_checks_the_concept_table_sha1(tmp_path):
    table = tmp_path / "concepts.csv"
    table.write_text("a,b\n1,2\n")
    path = _write_mask(tmp_path, [("1", "bleached")], concepts_csv=table)
    load_concept_source_mask(path, _schema(), concepts_csv=table)
    table.write_text("a,b\n1,3\n")
    with pytest.raises(ValueError, match="sha1"):
        load_concept_source_mask(path, _schema(), concepts_csv=table)


def test_apply_zeroes_only_masked_channels():
    targets = torch.full((2, len(CHANNELS), 3, 3), 2.0)
    keep = torch.ones(2, len(CHANNELS), dtype=torch.bool)
    keep[1, BLEACHED] = False
    out = apply_concept_source_mask(targets, keep, torch.tensor([1, 0]))
    assert (out[0, BLEACHED] == 0).all()
    assert (out[0, [i for i in range(len(CHANNELS)) if i != BLEACHED]] == 2).all()
    assert torch.equal(out[1], targets[1])


# ---------- datasets ----------


def _crop_top_left(image: Any, mask: Any) -> dict[str, Any]:
    """Keep the top-left 4x4 corner, as a random crop that misses a point would."""
    return {"image": image[:4, :4], "mask": mask[:4, :4]}


def _reader(tmp_path, name: str, groups: list[str], **kwargs: Any) -> PointDataset:
    """A train reader with image ``i<k>`` in ``groups[k]``; image ``i0``'s point is cropped out."""
    images = [ImageSpec(f"i{k}", group=group) for k, group in enumerate(groups)]
    points = [("i0", 6, 6, 1)] + [(f"i{k}", 1, 1, 1) for k in range(1, len(groups))]
    version = open_written(
        tmp_path / name,
        kind="points",
        images=images,
        labels={1: "Coral"},
        points=points,
        name=name,
    )
    return PointDataset(version, split="train", global_offset=0, **kwargs)


def test_coralnet_rows_come_from_the_image_group_and_follow_the_retried_image(tmp_path):
    ds = _reader(tmp_path, "coralnet", ["s5", "s123"], transform=_crop_top_left)
    image, labels = ds[1]  # unchanged without a mask
    mask = load_concept_source_mask(
        _write_mask(tmp_path, [("5", "bleached"), ("123", "bleached")]), _schema()
    )

    (attached,) = attach_source_rows([ds], mask)

    assert attached is ds
    assert ds[1][2] == mask.row_by_source()["123"]
    # i0 has no labels after the crop, so the train split retries on i1 (source 123).
    assert ds[0][2] == mask.row_by_source()["123"]


def test_counted_source_without_masked_pairs_gets_row_zero(tmp_path):
    ds = _reader(tmp_path, "coralnet", ["s5", "s7"])
    mask = load_concept_source_mask(
        _write_mask(tmp_path, [("5", "bleached")], source_ids=["5", "7"]), _schema()
    )
    attach_source_rows([ds], mask)
    assert ds[1][2] == 0  # source 7 was counted and has nothing to mask


def test_source_the_mask_never_counted_is_refused(tmp_path):
    ds = _reader(tmp_path, "coralnet", ["s5", "s7"])
    mask = load_concept_source_mask(_write_mask(tmp_path, [("5", "bleached")]), _schema())
    with pytest.raises(ValueError, match=r"1 coralnet source.*\b7\b"):
        attach_source_rows([ds], mask)


def test_an_image_group_that_is_not_s_and_a_source_id_is_refused(tmp_path):
    ds = _reader(tmp_path, "coralnet", ["s5", "x12"])
    mask = load_concept_source_mask(_write_mask(tmp_path, [("5", "bleached")]), _schema())
    with pytest.raises(ValueError, match=r"not s<source id>.*x12"):
        attach_source_rows([ds], mask)


def test_other_datasets_get_row_zero_and_a_repeated_dataset_stays_one_object(tmp_path):
    mask = load_concept_source_mask(_write_mask(tmp_path, [("5", "bleached")]), _schema())
    other = _reader(tmp_path, "coralscapes_v2", ["s5", "train"])

    out = attach_source_rows([other, other], mask)

    assert out[0] is other and out[1] is other
    assert other[1][2] == 0
    batch = next(iter(DataLoader(other, batch_size=1)))
    assert len(batch) == 3 and batch[2].tolist() == [0]


# ---------- train step ----------


# ---------- concept removal ----------


def test_schema_removal_drops_exactly_the_listed_channels(table_csv, cbm_schema):
    full = cbm_schema
    reduced = ConceptSchema.from_csv(
        table_csv, sources={"coralnet"}, removed_concepts=["genus__porites", "tubular"]
    )
    assert set(full.channel_names) - set(reduced.channel_names) == {"genus__porites", "tubular"}
    assert len(reduced.channel_names) == len(full.channel_names) - 2
    porites = dict(zip(reduced.channel_names, reduced.row_for("coralnet", "porites"), strict=True))
    # A removed genus is not_given at genus, its family stays.
    assert porites["family__poritidae"] == 2
    assert all(v == 0 for k, v in porites.items() if k.startswith("genus__"))
    acropora = dict(
        zip(reduced.channel_names, reduced.row_for("coralnet", "acropora"), strict=True)
    )
    assert acropora["genus__acropora"] == 2


def test_schema_removal_rejects_bad_names(table_csv):
    with pytest.raises(ValueError, match="not channels"):
        ConceptSchema.from_csv(table_csv, sources={"coralnet"}, removed_concepts=["genus__nope"])
    with pytest.raises(ValueError, match="none"):
        apply_concept_removal(concept_table(), ["genus__none"])


def test_coverage_rules():
    base = _schema()
    schema = ConceptSchema(
        channel_names=base.channel_names,
        concept_value2id=base.concept_value2id,
        num_channels=base.num_channels,
        row_by_key={
            **base.row_by_key,
            ("coralscapes_v2", "porites"): base.row_for("coralnet", "porites"),
        },
    )
    counts = []
    # acropora: 600 points over 120 images in 3 CoralNet sources -> kept.
    for i in range(120):
        counts.append(("coralnet", f"a{i}", "acropora", 5, str(i % 3)))
    # porites: 2 dense masks (50 each) + 450 points in 1 CoralNet source, 1 non-CoralNet dataset.
    counts += [
        ("coralscapes_v2", "m0", "porites", 9999, ""),
        ("coralscapes_v2", "m1", "porites", 1, ""),
    ]
    for i in range(90):
        counts.append(("coralnet", f"p{i}", "porites", 5, "9"))
    # bleached acropora: 600 points but 10 images in 2 CoralNet sources -> fails images and sources.
    for i in range(10):
        counts.append(("coralnet", f"b{i}", "bleached acropora", 60, str(i % 2)))
    df = pd.DataFrame(counts, columns=["dataset", "image_id", "label", "n", "source"])
    cov = concept_coverage(df, schema, CoverageRule())
    assert cov.loc["genus__acropora", "n_annotations"] == 600 + 600
    assert not cov.loc["genus__acropora", "removed"]
    assert cov.loc["genus__porites", "n_annotations"] == 450 + 2 * 50  # each mask counts 50
    assert cov.loc["genus__porites", "n_images"] == 92
    assert cov.loc["genus__porites", "fails_images"]  # 92 < 100 and only 2 non-CoralNet images
    assert cov.loc["genus__porites", "removed"]
    assert not cov.loc["genus__porites", "fails_sources"]  # a non-CoralNet dataset has it
    bleached = cov.loc["bleached"]
    assert (
        bleached["n_coralnet_sources"] == 2
        and bleached["fails_sources"]
        and bleached["fails_images"]
    )
    assert cov.loc["branching", "n_images"] == 130  # acropora + bleached acropora images


@pytest.mark.parametrize(("n_points", "removed"), [(499, True), (500, False)])
def test_annotation_rule_removes_a_concept_below_the_point_threshold(n_points, removed):
    # One image in one CoralNet source passes the lowered image and source rules.
    rule = CoverageRule(
        min_images=1, min_noncoralnet_images=1, min_annotations=500, min_coralnet_sources=1
    )
    counts = pd.DataFrame(
        [("coralnet", "a0", "acropora", n_points, "1")],
        columns=["dataset", "image_id", "label", "n", "source"],
    )

    row = concept_coverage(counts, _schema(), rule).loc["genus__acropora"]

    assert row["n_annotations"] == n_points
    assert not row["fails_images"] and not row["fails_sources"]
    assert row["fails_annotations"] == removed
    assert row["removed"] == removed
