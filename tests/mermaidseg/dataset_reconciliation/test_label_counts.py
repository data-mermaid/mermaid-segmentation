"""Per-image train label counts over stub datasets."""

from __future__ import annotations

from types import SimpleNamespace
from typing import Any

import numpy as np
import pandas as pd
import pytest

from mermaidseg.dataset_reconciliation.label_counts import count_train_labels
from mermaidseg.datasets.coralscapes_v2.coralscapes_v2_dataset import CORALSCAPES_V2_ID2NAME

COLUMNS = ["dataset", "image_id", "label", "n"]


class _Rows:
    """HF-style table: ``select_columns`` returns an object indexed to ``{"label": mask}``."""

    def __init__(self, masks: list[np.ndarray]) -> None:
        self._masks = masks

    def select_columns(self, columns: list[str]) -> _Rows:
        assert columns == ["label"]
        return self

    def __len__(self) -> int:
        return len(self._masks)

    def __getitem__(self, idx: int) -> dict[str, np.ndarray]:
        return {"label": self._masks[idx]}


class _HFDense:
    """ucsd_mosaics / coralscapes_v2 stand-in: ``len`` counts images, ``.dataset`` yields masks."""

    def __init__(
        self,
        masks: list[np.ndarray],
        class_table: list[dict[str, Any]] | None = None,
        n_images: int | None = None,
    ) -> None:
        self.dataset = _Rows(masks)
        self.class_table = class_table
        self._n_images = len(masks) if n_images is None else n_images

    def __len__(self) -> int:
        return self._n_images


def _benthos(
    masks: dict[str, np.ndarray],
    classes: dict[str, int],
    image_ids: list[str],
    sites: list[str] | None = None,
) -> Any:
    sites = ["s"] * len(image_ids) if sites is None else sites
    df_images = pd.DataFrame({"image_id": image_ids, "site": sites})
    return SimpleNamespace(
        _classes_global=classes,
        df_images=df_images,
        read_label=lambda image_id, site: masks[image_id],
    )


def _sparse(rows: list[tuple[str, str]]) -> Any:
    df = pd.DataFrame(rows, columns=["image_id", "source_label_name"])
    return SimpleNamespace(df_annotations=df)


def _triples(df: pd.DataFrame) -> list[tuple[str, str, int]]:
    return sorted(zip(df.image_id, df.label, df.n, strict=True))


def test_sparse_counts_points_per_image_and_lowercased_stripped_label():
    ds = _sparse([("a", "Acropora "), ("a", "acropora"), ("a", "Porites"), ("b", "Porites")])

    out = count_train_labels({("mermaid", "train"): ds})

    assert _triples(out) == [("a", "acropora", 2), ("a", "porites", 1), ("b", "porites", 1)]
    assert set(out.dataset) == {"mermaid"}


@pytest.mark.parametrize("ignored", ["Unlabeled", " unlabeled / unidentified "])
def test_sparse_drops_ignore_labels(ignored):
    ds = _sparse([("a", ignored), ("a", "Sand")])

    out = count_train_labels({("mermaid", "train"): ds})

    assert _triples(out) == [("a", "sand", 1)]


def test_dense_skips_mask_id_zero():
    mask = np.zeros((2, 2), dtype=np.uint8)
    mask[0, 0] = 1
    ds = _HFDense([mask], [{"id": 1, "name": "Sand"}])

    out = count_train_labels({("ucsd_mosaics", "train"): ds})

    assert _triples(out) == [("0", "sand", 1)]


def test_dense_drops_positive_id_named_unlabeled():
    mask = np.array([[1, 2], [2, 2]], dtype=np.uint8)
    ds = _HFDense([mask], [{"id": 1, "name": "Sand"}, {"id": 2, "name": "Unlabeled"}])

    out = count_train_labels({("ucsd_mosaics", "train"): ds})

    assert _triples(out) == [("0", "sand", 1)]


def test_coralscapes_v2_uses_the_real_id_table():
    mask = np.array([[2, 2, 2], [2, 11, 11], [11, 11, 11]], dtype=np.uint8)
    assert CORALSCAPES_V2_ID2NAME[2] == "acropora alive"
    assert CORALSCAPES_V2_ID2NAME[11] == "algae covered substrate"

    out = count_train_labels({("coralscapes_v2", "train"): _HFDense([mask])})

    assert _triples(out) == [("0", "acropora alive", 4), ("0", "algae covered substrate", 5)]


def test_hf_image_ids_are_row_positions():
    mask = np.array([[1]], dtype=np.uint8)
    ds = _HFDense([mask, mask], [{"id": 1, "name": "Sand"}])

    out = count_train_labels({("ucsd_mosaics", "train"): ds})

    assert _triples(out) == [("0", "sand", 1), ("1", "sand", 1)]


def test_benthos_image_ids_are_site_prefixed_tile_ids():
    mask = np.array([[1]], dtype=np.uint8)
    ds = _benthos({"x7": mask, "x9": mask}, {"Sand": 1}, ["x7", "x9"])

    out = count_train_labels({("benthos_yuval", "train"): ds})

    assert _triples(out) == [("s/x7", "sand", 1), ("s/x9", "sand", 1)]


def test_benthos_tiles_with_the_same_id_at_two_sites_get_distinct_image_ids():
    mask = np.array([[1]], dtype=np.uint8)
    ds = _benthos({"0_0": mask}, {"Sand": 1}, ["0_0", "0_0"], ["RS24", "CR_DoubleWreck"])

    out = count_train_labels({("benthos_yuval", "train"): ds})

    assert _triples(out) == [("CR_DoubleWreck/0_0", "sand", 1), ("RS24/0_0", "sand", 1)]


def test_unknown_positive_mask_id_raises_naming_dataset_image_and_id():
    mask = np.array([[1, 99]], dtype=np.uint8)
    ds = _HFDense([mask], [{"id": 1, "name": "Sand"}])

    with pytest.raises(ValueError, match=r"ucsd_mosaics.*image '0'.*99"):
        count_train_labels({("ucsd_mosaics", "train"): ds})


@pytest.mark.parametrize("n_masks, n_images", [(2, 3), (2, 1), (1, 0)])
def test_mask_count_differing_from_image_count_raises(n_masks, n_images):
    mask = np.array([[1]], dtype=np.uint8)
    ds = _HFDense([mask] * n_masks, [{"id": 1, "name": "Sand"}], n_images=n_images)

    with pytest.raises(RuntimeError, match=f"{n_masks} masks, expected {n_images} images"):
        count_train_labels({("ucsd_mosaics", "train"): ds})


def test_dense_dataset_with_no_masks_returns_empty_frame():
    out = count_train_labels({("ucsd_mosaics", "train"): _HFDense([], [])})

    assert list(out.columns) == COLUMNS
    assert len(out) == 0


def test_empty_built_raises():
    with pytest.raises(ValueError, match="no train splits to count"):
        count_train_labels({})


def test_mixed_datasets_share_one_frame_and_unknown_keys_count_as_sparse():
    mask = np.array([[1]], dtype=np.uint8)
    built = {
        ("mermaid", "train"): _sparse([("m1", "Sand")]),
        ("ucsd_mosaics", "train"): _HFDense([mask], [{"id": 1, "name": "Sand"}]),
        ("new_source", "train"): _sparse([("n1", "Coral"), ("n1", "Coral")]),
    }

    out = count_train_labels(built)

    assert list(out.columns) == COLUMNS
    assert all(isinstance(v, str) for v in out.image_id)
    assert sorted(zip(out.dataset, out.image_id, out.label, out.n, strict=True)) == [
        ("mermaid", "m1", "sand", 1),
        ("new_source", "n1", "coral", 2),
        ("ucsd_mosaics", "0", "sand", 1),
    ]
