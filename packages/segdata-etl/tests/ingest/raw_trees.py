"""Tiny raw source trees, as each ``segdata ingest`` module reads them from ``--data-dir``."""

from __future__ import annotations

import io
from pathlib import Path

import numpy as np
from PIL import Image

CATLIN_CSV_HEADER = "quadratid,x,y,label_name"

PACIFIC_SITES = ("heron_reef", "line_islands", "nanwan_bay")
PACIFIC_LABELMAP = "labelid,name\n1,Sand\n2,Acropora\n3,Turf\n"
PACIFIC_EVAL_ANN = "imageid,row,col,archived,host,visitor1,visitor2,visitor3,visitor4,visitor5\n"
PACIFIC_REF_ANN = "imageid,row,col,archived\n"

# RS24 uses the default dictionary, CR_DoubleWreck the alt one; ids differ per variant.
BENTHOS_DEFAULT_DICT = "{'Coral': 1, 'Sand': 2}"
BENTHOS_ALT_DICT = "{'Coral': 5, 'Algae': 6}"


def jpg_bytes(size: int = 32) -> bytes:
    """Return a decodable ``size`` x ``size`` JPEG."""
    buf = io.BytesIO()
    Image.effect_noise((size, size), 80).convert("RGB").save(buf, format="JPEG")
    return buf.getvalue()


def write_catlin_tree(root: Path, csvs: dict[str, str], images: dict[str, bytes]) -> Path:
    """``csvs`` maps region -> CSV text; ``images`` maps ``region/id`` -> jpg bytes."""
    (root / "tabular-data").mkdir(parents=True)
    for region, text in csvs.items():
        (root / "tabular-data" / f"annotations_{region}.csv").write_text(text)
    for rel, data in images.items():
        path = root / f"{rel}.jpg"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(data)
    return root


def add_moorea_image(
    root: Path, year: str, name: str, lines: list[str], data: bytes | None = None
) -> None:
    """Write ``<root>/<year>/<name>`` and its ``<name>.txt`` annotation file."""
    year_dir = root / year
    year_dir.mkdir(parents=True, exist_ok=True)
    (year_dir / name).write_bytes(jpg_bytes() if data is None else data)
    (year_dir / f"{name}.txt").write_text("\n".join(lines) + "\n")


def write_jpg(path: Path, *, valid: bool = True) -> None:
    """Write a decodable JPEG to ``path``, or bytes that are not one when ``valid`` is false."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(jpg_bytes() if valid else b"not a jpeg")


def write_pacific_subset(
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
    default = (
        PACIFIC_REF_ANN + "1,10,20,1\n"
        if subset == "reference"
        else PACIFIC_EVAL_ANN + "1,10,20,1,2,3,,,,\n"
    )
    (subset_dir / "annotations.txt").write_text(annotations or default)
    for line in imagemap.splitlines()[1:]:
        write_jpg(subset_dir / "imgs" / line.split(",")[1], valid=valid)


def write_pacific_tree(root: Path, **kwargs) -> Path:
    for site in PACIFIC_SITES:
        (root / site).mkdir(parents=True, exist_ok=True)
        (root / site / "labelmap.txt").write_text(PACIFIC_LABELMAP)
        for subset in ("reference", "evaluation"):
            write_pacific_subset(root, site, subset, **kwargs)
    return root


def rgb(h: int, w: int) -> np.ndarray:
    return np.full((h, w, 3), 7, dtype=np.uint8)


def save_png(path: Path, arr: np.ndarray) -> None:
    Image.fromarray(arr).save(path)


def write_benthos_raw(
    root: Path,
    *,
    rs24_mask: np.ndarray,
    rs24_image: np.ndarray | None = None,
    dw_mask: np.ndarray | None = None,
    default_dict: str = BENTHOS_DEFAULT_DICT,
) -> Path:
    root.mkdir(parents=True, exist_ok=True)
    h, w = rs24_mask.shape
    save_png(root / "RS24.png", rs24_image if rs24_image is not None else rgb(h, w))
    save_png(root / "ManualRS24.png", rs24_mask)
    dw = dw_mask if dw_mask is not None else np.full((2, 2), 5, dtype=np.uint8)
    save_png(root / "CR_DoubleWreck.png", rgb(*dw.shape))
    save_png(root / "ManualDW.png", dw)
    (root / "dictionary_labels.txt").write_text(default_dict)
    (root / "dictionary_labelsDW.txt").write_text(BENTHOS_ALT_DICT)
    return root
