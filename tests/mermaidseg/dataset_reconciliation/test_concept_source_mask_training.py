"""training.concept_source_mask and training.concept_removal: loading, datasets and the train step."""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
import pytest
import torch
import yaml
from torch.utils.data import DataLoader, Dataset

from mermaidseg.dataset_reconciliation.concept_removal import (
    CoverageRule,
    apply_concept_removal,
    concept_coverage,
)
from mermaidseg.dataset_reconciliation.concept_schema import ConceptSchema
from mermaidseg.dataset_reconciliation.concept_source_mask import (
    apply_concept_source_mask,
    attach_source_rows,
    file_sha1,
    load_concept_source_mask,
    meta_path,
)
from mermaidseg.datasets.base_dataset import BaseCoralDataset
from tests.mermaidseg.conftest import concept_table
from tests.mermaidseg.dataset_reconciliation.test_concept_source_mask import CHANNELS, _schema

BLEACHED = CHANNELS.index("bleached")


def _write_mask(tmp_path, rows: list[tuple[str, str]], concepts_csv=None) -> Any:
    path = tmp_path / "mask.csv"
    pd.DataFrame(rows, columns=["source_id", "concept"]).assign(
        n_true=0, n_false=1, n_not_given=0
    ).to_csv(path, index=False)
    meta = {"dataset": "coralnet", "rule": "only_false"}
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


class _CropOutCoralNet(BaseCoralDataset):
    """Two images in two sources; image 0 loses its labels, so asking for it returns image 1."""

    SOURCE_NAME = "coralnet"

    def read_image(self, **row_kwargs) -> np.ndarray:
        return np.zeros((20, 20, 3), dtype=np.uint8)

    def _load_item(self, idx: int) -> tuple[Any, Any]:
        image, mask = super()._load_item(idx)
        if idx == 0:
            mask = np.zeros_like(mask)
        return image, mask


def _coralnet_dataset() -> _CropOutCoralNet:
    df_annotations = pd.DataFrame(
        {
            "image_id": ["img0", "img1"],
            "source_id": [5, 7],
            "source_label_name": ["Coral", "Coral"],
            "row": [5, 5],
            "col": [5, 5],
        }
    )
    df_images = df_annotations[["image_id", "source_id"]].reset_index(drop=True)
    return _CropOutCoralNet(df_annotations=df_annotations, df_images=df_images, split="train")


class _PlainDataset(Dataset):
    SOURCE_NAME = "coralscapes_v2"

    def __len__(self) -> int:
        return 3

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        return torch.zeros(3, 4, 4), torch.ones(4, 4, dtype=torch.long)


def test_source_row_follows_the_retried_image(tmp_path):
    ds = _coralnet_dataset()
    image, labels = ds[1]  # unchanged without a mask
    mask = load_concept_source_mask(
        _write_mask(tmp_path, [("5", "bleached"), ("7", "bleached")]), _schema()
    )
    (wrapped,) = attach_source_rows([ds], mask)
    assert wrapped is ds
    assert ds[1][2] == mask.row_by_source()["7"]
    # idx 0 has no labels after the crop, so the train split retries on image 1 (source 7).
    assert ds[0][2] == mask.row_by_source()["7"]


def test_other_datasets_get_row_zero(tmp_path):
    mask = load_concept_source_mask(_write_mask(tmp_path, [("5", "bleached")]), _schema())
    plain = _PlainDataset()
    out = attach_source_rows([plain, plain], mask)
    assert out[0] is out[1]  # a repeated dataset is wrapped once
    assert len(out[0]) == 3 and out[0].SOURCE_NAME == "coralscapes_v2"
    assert out[0][0][2] == 0
    batch = next(iter(DataLoader(out[0], batch_size=2)))
    assert len(batch) == 3 and batch[2].tolist() == [0, 0]


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
    assert not cov.loc["genus__porites", "fails_sources"]  # a non-CoralNet dataset has it
    bleached = cov.loc["bleached"]
    assert (
        bleached["n_coralnet_sources"] == 2
        and bleached["fails_sources"]
        and bleached["fails_images"]
    )
    assert cov.loc["branching", "n_images"] == 130  # acropora + bleached acropora images
