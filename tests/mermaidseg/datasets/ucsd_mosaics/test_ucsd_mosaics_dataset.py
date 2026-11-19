"""UCSDMosaicsDataset on a one-row stand-in for the HuggingFace mirror."""

from __future__ import annotations

import json

import numpy as np
import pytest
from PIL import Image

from mermaidseg.datasets.ucsd_mosaics import UCSDMosaicsDataset

CLASS_TABLE = [
    {"id": 1, "name": "acropora"},
    {"id": 2, "name": "sand"},
    {"id": 3, "name": "porites"},
]
# One 2x3 (height x width) sample; 0 is the unlabeled ignore label.
LABELS = np.array([[0, 1, 2], [3, 2, 0]], dtype=np.uint8)


@pytest.fixture
def repo_dir(tmp_path, monkeypatch):
    """A local 'repo' with classes.json, and a single-site stand-in for the HF loaders."""
    (tmp_path / "classes.json").write_text(json.dumps(CLASS_TABLE))
    rows = [
        {
            "image": Image.new("RGB", (3, 2), (5, 6, 7)),
            "label": Image.fromarray(LABELS, mode="L"),
            "original_split": "train",
        }
    ]
    monkeypatch.setattr("datasets.get_dataset_split_names", lambda repo: ["fr3"])
    monkeypatch.setattr("datasets.load_dataset", lambda repo, split: rows)
    return str(tmp_path)


def test_item_mask_is_the_native_class_ids_by_default(repo_dir):
    ds = UCSDMosaicsDataset(hf_repo_id=repo_dir)

    image, mask = ds[0]

    assert len(ds) == 1
    assert ds.source_id2name == {1: "acropora", 2: "sand", 3: "porites"}
    assert image.shape == (2, 3, 3) and np.all(image == (5, 6, 7))
    np.testing.assert_array_equal(mask, LABELS.astype(np.int32))
    assert mask.dtype == np.int32


def test_class_subset_renumbers_kept_classes_and_sends_the_rest_to_background(repo_dir):
    ds = UCSDMosaicsDataset(hf_repo_id=repo_dir, class_subset=["sand", "porites"])

    _, mask = ds[0]

    # acropora (native 1) is outside the subset; sand -> 1, porites -> 2.
    np.testing.assert_array_equal(mask, np.array([[0, 0, 1], [2, 1, 0]], dtype=np.int32))


def test_global_offset_shifts_foreground_ids_only(repo_dir):
    ds = UCSDMosaicsDataset(hf_repo_id=repo_dir)
    ds.set_global_offset(100)

    _, mask = ds[0]

    np.testing.assert_array_equal(mask, np.array([[0, 101, 102], [103, 102, 0]], dtype=np.int32))


def test_unknown_whitelist_site_raises(repo_dir):
    with pytest.raises(ValueError, match="unknown site"):
        UCSDMosaicsDataset(hf_repo_id=repo_dir, whitelist_sites=["nope"])


def test_an_unreadable_item_is_skipped_as_none_none(repo_dir):
    ds = UCSDMosaicsDataset(hf_repo_id=repo_dir)

    assert ds[7] == (None, None)
