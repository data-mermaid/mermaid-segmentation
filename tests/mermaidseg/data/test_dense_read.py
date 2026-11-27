from pathlib import Path

import numpy as np
import pytest
from PIL import Image

from mermaidseg.data.dense_read import read_label_mask


def _write_mask(path: Path, ids: list[list[int]]) -> Path:
    Image.fromarray(np.array(ids, dtype=np.uint16)).save(path)
    return path


def test_a_mask_id_beyond_uint8_raises_instead_of_wrapping(tmp_path):
    path = _write_mask(tmp_path / "mask.png", [[0, 300]])
    with pytest.raises(ValueError, match=r"mask\.png.*300"):
        read_label_mask(path, 96)


def test_a_wide_id_inside_num_ids_is_kept_unwrapped(tmp_path):
    path = _write_mask(tmp_path / "mask.png", [[0, 300]])
    assert read_label_mask(path, 301).tolist() == [[0, 300]]


@pytest.mark.parametrize(("max_id", "raises"), [(95, False), (96, True)])
def test_max_id_must_be_below_num_ids(tmp_path, max_id, raises):
    path = _write_mask(tmp_path / "mask.png", [[0, max_id]])
    if raises:
        with pytest.raises(ValueError, match=str(max_id)):
            read_label_mask(path, 96)
    else:
        assert int(read_label_mask(path, 96).max()) == max_id
