"""CoralscapesV2 linear-probe evaluation.

Trains a linear SVM (sklearn ``LinearSVC``) on activated concept features sampled
at a fixed probe point set from the train split (see
``scripts/build_coralscapes_v2_probe_points.py``), then applies that one SVM over
each requested CoralscapesV2 split (``val`` and/or ``test``) and reports accuracy
and mIoU in a label space: ``CORALSCAPES_V2`` (the 95 native classes) or
``CORALSCAPES39`` (the published 95-to-39 collapse). Each probe file is one
pixels-per-class budget k; several budgets share one forward pass and each gets its
own SVM.

Every 1024x2048 image is predicted with three windows, each resized to the model
input size:

- left:   rows ``0:1024``, cols ``0:1024``
- center: rows ``0:1024``, cols ``512:1536``
- right:  rows ``0:1024``, cols ``1024:2048``

The three concept-logit maps are bilinearly upsampled back onto those rectangles
and averaged, on the model device, where the windows overlap. The model's
``concept_outputs_activation`` (per-rank softmax, sigmoid on the binary tail) is
applied to the averaged logits. Probe pixels and the dense SVM both read that
activated map, so train and test use the same features.

When concepts are given, the binary concepts in ``REPORTED_BINARY_CONCEPTS`` (``bleached``,
``anthropogenic``) are scored from the same activated map, not from the SVM
class prediction. ``TRUE`` (code 2) is the positive class, ``FALSE`` (code 1) is
the negative class, ``not_given`` (code 0) is excluded, and mask id 0
(unannotated) is excluded separately. A probability of exactly 0.5 counts as a
negative prediction. Hierarchical metrics (consistency, mistake severity, ECE)
are computed from the per-rank softmax of the same map against the taxonomy of
each pixel's class in ``class_to_concepts.csv``.

Probe and eval images are read from the synced ``coralscapes_v2`` dataset version: a probe
image is the ``train`` group image whose ``image_id`` is the probe entry's ``stem`` (its label
md5 is verified), and the ``val`` and ``test`` splits are the ``validation`` and ``test`` groups.
"""

from __future__ import annotations

import hashlib
import json
import logging
import time
from collections import defaultdict
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, NamedTuple

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from numpy.typing import NDArray
from tqdm import tqdm

from mermaidseg.data.dense_read import read_label_mask, read_rgb
from mermaidseg.dataset_reconciliation.concepts import parse_concept_cell, read_class_to_concepts
from mermaidseg.evaluation.gt_mapping import build_concept_lookup
from mermaidseg.evaluation.hierarchical import HierarchicalScore
from mermaidseg.evaluation.metrics import (
    REPORTED_BINARY_CONCEPTS,
    BinaryConceptStats,
    ClassConfusion,
    concept_block_from_binary,
    require_reported_concepts,
)
from mermaidseg.evaluation.reporting import fmt, write_json
from mermaidseg.inference.predictor import CBMPredictor
from segdata_spec.read import DatasetVersion

logger = logging.getLogger(__name__)

NUM_CLASSES = 95
NATIVE_IDS = NUM_CLASSES + 1  # native Coralscapes v2 mask ids 0..95; 0 is unannotated
CORALSCAPES39_NUM_CLASSES = 39
CORALSCAPES_HEIGHT = 1024
CORALSCAPES_WIDTH = 2048
SPLITS = ("val", "test")
_SPLIT_GROUPS = {"val": "validation", "test": "test"}
PROBE_GROUP = "train"
_TILE = 1024
_SVM_STRIP_ROWS = 128
_CODE_OF_CELL = {"not_given": 0, "false": 1, "true": 2}


# Native CoralscapesV2 class ids; equal to the coralscapes_v2 version's labels.parquet ids.
CORALSCAPES_V2_ID2NAME: dict[int, str] = {
    1: "acanthaster planci",
    2: "acropora alive",
    3: "acropora bleached",
    4: "acropora dead",
    5: "algae covered brain coral",
    6: "algae covered branching coral",
    7: "algae covered lobophylliidae",
    8: "algae covered massive coral",
    9: "algae covered pocillopora",
    10: "algae covered porites",
    11: "algae covered substrate",
    12: "algae covered table acropora",
    13: "anemone",
    14: "background",
    15: "brain coral alive",
    16: "brain coral bleached",
    17: "brain coral dead",
    18: "branching coral alive",
    19: "branching coral bleached",
    20: "branching coral dead",
    21: "cirrhipathes",
    22: "dark",
    23: "dead giant clam",
    24: "feather worm",
    25: "fish",
    26: "fungiidae alive",
    27: "fungiidae bleached",
    28: "fungiidae dead",
    29: "galaxea alive",
    30: "galaxea bleached",
    31: "galaxea dead",
    32: "giant clam",
    33: "goniopora alive",
    34: "goniopora dead",
    35: "hard coral alive",
    36: "hard coral bleached",
    37: "hard coral dead",
    38: "hard substrate",
    39: "human",
    40: "human-made structure",
    41: "leather coral alive",
    42: "lobophylliidae alive",
    43: "lobophylliidae bleached",
    44: "lobophylliidae dead",
    45: "massive alive",
    46: "massive bleached",
    47: "massive dead",
    48: "meandering coral alive",
    49: "meandering coral bleached",
    50: "meandering coral dead",
    51: "millepora",
    52: "millepora bleached",
    53: "millepora dead",
    54: "nephtheidae alive",
    55: "other animal",
    56: "other coral alive",
    57: "other coral bleached",
    58: "other coral dead",
    59: "pavona alive",
    60: "pavona bleached",
    61: "pavona dead",
    62: "pocillopora alive",
    63: "pocillopora bleached",
    64: "pocillopora dead",
    65: "porites alive",
    66: "porites bleached",
    67: "porites dead",
    68: "rubble",
    69: "sand",
    70: "sea cucumber",
    71: "sea urchin",
    72: "seagrass",
    73: "seriatopora alive",
    74: "seriatopora bleached",
    75: "seriatopora dead",
    76: "soft coral (malacalcyonacea) alive",
    77: "soft coral (malacalcyonacea) bleached",
    78: "sponge",
    79: "stylophora alive",
    80: "stylophora bleached",
    81: "stylophora dead",
    82: "table acropora alive",
    83: "table acropora bleached",
    84: "table acropora dead",
    85: "thin plate/encrusting coral alive",
    86: "thin plate/encrusting coral bleached",
    87: "thin plate/encrusting coral dead",
    88: "transect line",
    89: "transect tools",
    90: "trash",
    91: "turbinaria",
    92: "turbinaria dead",
    93: "water column",
    94: "water surface",
    95: "xeniidae alive",
}

# Legacy column of the CoralscapesV2 dataset card (the paper's 95 -> 39 map).
# The card calls V1 class 12 "hard substrate"; this repo's Coralscapes ids name
# that class "unknown hard substrate".
CORALSCAPES_V2_TO_LEGACY_NAME: dict[str, str] = {
    "acanthaster planci": "crown of thorn",
    "acropora alive": "acropora alive",
    "acropora bleached": "branching bleached",
    "acropora dead": "branching dead",
    "algae covered brain coral": "algae covered substrate",
    "algae covered branching coral": "algae covered substrate",
    "algae covered lobophylliidae": "algae covered substrate",
    "algae covered massive coral": "algae covered substrate",
    "algae covered pocillopora": "algae covered substrate",
    "algae covered porites": "algae covered substrate",
    "algae covered substrate": "algae covered substrate",
    "algae covered table acropora": "algae covered substrate",
    "anemone": "anemone",
    "background": "background",
    "brain coral alive": "meandering alive",
    "brain coral bleached": "meandering bleached",
    "brain coral dead": "meandering dead",
    "branching coral alive": "branching alive",
    "branching coral bleached": "branching bleached",
    "branching coral dead": "branching dead",
    "cirrhipathes": "other coral alive",
    "dark": "dark",
    "dead giant clam": "dead clam",
    "feather worm": "other animal",
    "fish": "fish",
    "fungiidae alive": "other coral alive",
    "fungiidae bleached": "other coral bleached",
    "fungiidae dead": "other coral dead",
    "galaxea alive": "other coral alive",
    "galaxea bleached": "other coral bleached",
    "galaxea dead": "massive/meandering dead",
    "giant clam": "clam",
    "goniopora alive": "other coral alive",
    "goniopora dead": "other coral dead",
    "hard coral alive": "other coral alive",
    "hard coral bleached": "other coral bleached",
    "hard coral dead": "other coral dead",
    "hard substrate": "unknown hard substrate",
    "human": "human",
    "human-made structure": "unknown hard substrate",
    "leather coral alive": "other coral alive",
    "lobophylliidae alive": "meandering alive",
    "lobophylliidae bleached": "meandering bleached",
    "lobophylliidae dead": "meandering dead",
    "massive alive": "massive/meandering alive",
    "massive bleached": "massive/meandering bleached",
    "massive dead": "massive/meandering dead",
    "meandering coral alive": "meandering alive",
    "meandering coral bleached": "meandering bleached",
    "meandering coral dead": "meandering dead",
    "millepora": "millepora",
    "millepora bleached": "branching bleached",
    "millepora dead": "branching dead",
    "nephtheidae alive": "other coral alive",
    "other animal": "other animal",
    "other coral alive": "other coral alive",
    "other coral bleached": "other coral bleached",
    "other coral dead": "other coral dead",
    "pavona alive": "other coral alive",
    "pavona bleached": "other coral bleached",
    "pavona dead": "other coral dead",
    "pocillopora alive": "pocillopora alive",
    "pocillopora bleached": "branching bleached",
    "pocillopora dead": "branching dead",
    "porites alive": "massive/meandering alive",
    "porites bleached": "massive/meandering bleached",
    "porites dead": "massive/meandering dead",
    "rubble": "rubble",
    "sand": "sand",
    "sea cucumber": "sea cucumber",
    "sea urchin": "sea urchin",
    "seagrass": "seagrass",
    "seriatopora alive": "branching alive",
    "seriatopora bleached": "branching bleached",
    "seriatopora dead": "branching dead",
    "soft coral (malacalcyonacea) alive": "other coral alive",
    "soft coral (malacalcyonacea) bleached": "other coral bleached",
    "sponge": "sponge",
    "stylophora alive": "stylophora alive",
    "stylophora bleached": "branching bleached",
    "stylophora dead": "branching dead",
    "table acropora alive": "table acropora alive",
    "table acropora bleached": "branching bleached",
    "table acropora dead": "table acropora dead",
    "thin plate/encrusting coral alive": "other coral alive",
    "thin plate/encrusting coral bleached": "other coral bleached",
    "thin plate/encrusting coral dead": "other coral dead",
    "transect line": "transect line",
    "transect tools": "transect tools",
    "trash": "trash",
    "turbinaria": "turbinaria",
    "turbinaria dead": "other coral dead",
    "water column": "background",
    "water surface": "background",
    "xeniidae alive": "other coral alive",
}


# Original Coralscapes (v1) 39 class ids; the target of the 95 -> 39 legacy map.
CORALSCAPES_ID2NAME: dict[int, str] = {
    1: "seagrass",
    2: "trash",
    3: "other coral dead",
    4: "other coral bleached",
    5: "sand",
    6: "other coral alive",
    7: "human",
    8: "transect tools",
    9: "fish",
    10: "algae covered substrate",
    11: "other animal",
    12: "unknown hard substrate",
    13: "background",
    14: "dark",
    15: "transect line",
    16: "massive/meandering bleached",
    17: "massive/meandering alive",
    18: "rubble",
    19: "branching bleached",
    20: "branching dead",
    21: "millepora",
    22: "branching alive",
    23: "massive/meandering dead",
    24: "clam",
    25: "acropora alive",
    26: "sea cucumber",
    27: "turbinaria",
    28: "table acropora alive",
    29: "sponge",
    30: "anemone",
    31: "pocillopora alive",
    32: "table acropora dead",
    33: "meandering bleached",
    34: "stylophora alive",
    35: "sea urchin",
    36: "meandering alive",
    37: "meandering dead",
    38: "crown of thorn",
    39: "dead clam",
}


def coralscapes_v2_to_legacy_lut() -> NDArray[np.uint8]:
    """Map a native CoralscapesV2 mask onto the original 39-class ids.

    Index 0 stays 0 (unannotated). Indexes 1..95 are the V1 ids of
    :data:`CORALSCAPES_V2_TO_LEGACY_NAME`.
    """
    legacy_ids = {name: class_id for class_id, name in CORALSCAPES_ID2NAME.items()}
    fine_names = set(CORALSCAPES_V2_ID2NAME.values())
    if set(CORALSCAPES_V2_TO_LEGACY_NAME) != fine_names:
        missing = sorted(fine_names - set(CORALSCAPES_V2_TO_LEGACY_NAME))
        extra = sorted(set(CORALSCAPES_V2_TO_LEGACY_NAME) - fine_names)
        raise RuntimeError(
            f"95-to-39 map does not match the V2 names: missing={missing} extra={extra}"
        )
    unknown = sorted(set(CORALSCAPES_V2_TO_LEGACY_NAME.values()) - set(legacy_ids))
    if unknown:
        raise RuntimeError(
            f"95-to-39 map uses names that are not Coralscapes V1 classes: {unknown}"
        )
    lut = np.zeros(len(CORALSCAPES_V2_ID2NAME) + 1, dtype=np.uint8)
    for class_id, name in CORALSCAPES_V2_ID2NAME.items():
        lut[class_id] = legacy_ids[CORALSCAPES_V2_TO_LEGACY_NAME[name]]
    return lut


@dataclass(frozen=True)
class LabelSpace:
    """A class space the probe is fitted and scored in, reached from native masks.

    Attributes:
        name: The ``label_space`` value of a probe JSON in this space.
        eval_name: Prefix of the summary keys and output directories.
        num_classes: Class ids run ``1..num_classes``; 0 is unannotated.
        id2name: Class name of each id.
        lut: Maps a native Coralscapes v2 id (the index) to this space's id.
    """

    name: str
    eval_name: str
    num_classes: int
    id2name: dict[int, str]
    lut: np.ndarray


CORALSCAPES_V2 = LabelSpace(
    name="coralscapes_v2",
    eval_name="coralscapes",
    num_classes=NUM_CLASSES,
    id2name=CORALSCAPES_V2_ID2NAME,
    lut=np.arange(NATIVE_IDS, dtype=np.uint8),
)
CORALSCAPES39 = LabelSpace(
    name="coralscapes39",
    eval_name="coralscapes39",
    num_classes=CORALSCAPES39_NUM_CLASSES,
    id2name=CORALSCAPES_ID2NAME,
    lut=coralscapes_v2_to_legacy_lut(),
)
_LABEL_SPACES = {space.name: space for space in (CORALSCAPES_V2, CORALSCAPES39)}


def label_space_from_name(name: str) -> LabelSpace:
    """Return the label space a probe JSON's ``label_space`` names.

    Raises:
        ValueError: No label space has that name.
    """
    if name not in _LABEL_SPACES:
        raise ValueError(f"unknown label space {name!r}; expected one of {sorted(_LABEL_SPACES)}")
    return _LABEL_SPACES[name]


@dataclass(frozen=True)
class ProbeConcepts:
    """The concept channels and taxonomy the probe scores concepts and hierarchy with.

    Attributes:
        concept_names: The model's concept channel names, in channel order.
        taxonomy_csv: ``class_to_concepts.csv`` of the bundle's run.
        removed_concepts: Concepts taken out of the taxonomy tree (``training.concept_removal``).
    """

    concept_names: list[str]
    taxonomy_csv: str | Path
    removed_concepts: Sequence[str] = ()


def label_md5(arr: NDArray[np.integer]) -> str:
    """Return the md5 hex digest of a label mask as uint8 bytes."""
    return hashlib.md5(np.ascontiguousarray(arr, dtype=np.uint8).tobytes()).hexdigest()


def coralscapes_windows() -> tuple[tuple[str, int, int, int, int], ...]:
    """Half-open windows ``(name, row0, col0, row1, col1)`` on a 1024x2048 image.

    Left is columns 0:1024, center is 512:1536, right is 1024:2048. Every column
    is covered: 0:512 by left only, 512:1024 by left and center, 1024:1536 by
    center and right, and 1536:2048 by right only.
    """
    center0 = (CORALSCAPES_WIDTH - _TILE) // 2
    return (
        ("left", 0, 0, _TILE, _TILE),
        ("center", 0, center0, _TILE, center0 + _TILE),
        ("right", 0, CORALSCAPES_WIDTH - _TILE, _TILE, CORALSCAPES_WIDTH),
    )


def _require_rgb(image: np.ndarray, what: str) -> None:
    if image.ndim != 3 or image.shape[2] != 3:
        raise ValueError(f"{what} must be HxWx3 RGB, got shape {tuple(image.shape)}")
    height, width = int(image.shape[0]), int(image.shape[1])
    if (height, width) != (CORALSCAPES_HEIGHT, CORALSCAPES_WIDTH):
        raise ValueError(
            f"{what} must be {CORALSCAPES_HEIGHT}x{CORALSCAPES_WIDTH} for strided "
            f"Coralscapes eval, got {height}x{width}"
        )


def _require_label(label: np.ndarray, what: str) -> None:
    if label.ndim != 2 or tuple(int(s) for s in label.shape) != (
        CORALSCAPES_HEIGHT,
        CORALSCAPES_WIDTH,
    ):
        raise ValueError(
            f"{what} label must be {CORALSCAPES_HEIGHT}x{CORALSCAPES_WIDTH}, got shape {tuple(label.shape)}"
        )


def average_window_logits(
    window_logits: torch.Tensor,
    windows: Sequence[tuple[int, int, int, int]],
    height: int,
    width: int,
) -> torch.Tensor:
    """Upsample each window's logits onto the full canvas and average the overlap.

    ``window_logits`` has shape ``(N, K, h, w)`` at model resolution. Window ``i``
    is bilinearly resized to the source rectangle ``windows[i] = (row0, col0, row1, col1)``
    and added on ``window_logits.device``. Overlapping pixels are an unweighted
    mean. Raises if any output pixel is covered by zero windows.
    """
    if window_logits.dim() != 4:
        raise ValueError(f"window_logits must be (N, K, h, w); got {tuple(window_logits.shape)}")
    if window_logits.shape[0] != len(windows):
        raise ValueError(f"got {window_logits.shape[0]} logit maps for {len(windows)} windows")
    canvas = torch.zeros(
        (window_logits.shape[1], height, width),
        device=window_logits.device,
        dtype=torch.float32,
    )
    weight = torch.zeros((1, height, width), device=window_logits.device, dtype=torch.float32)
    for index, (row0, col0, row1, col1) in enumerate(windows):
        if row0 < 0 or col0 < 0 or row1 > height or col1 > width or row1 <= row0 or col1 <= col0:
            raise ValueError(f"window {(row0, col0, row1, col1)} does not fit in {height}x{width}")
        upsampled = F.interpolate(
            window_logits[index : index + 1].float(),
            size=(row1 - row0, col1 - col0),
            mode="bilinear",
            align_corners=False,
        )[0]
        canvas[:, row0:row1, col0:col1] += upsampled
        weight[:, row0:row1, col0:col1] += 1
    if int(weight.min().item()) < 1:
        raise RuntimeError("strided reassembly left pixels with no window coverage")
    return canvas / weight


@torch.inference_mode()
def reassemble_concept_logits(
    predictor: CBMPredictor, image_rgb: NDArray[np.uint8]
) -> torch.Tensor:
    """Run the three Coralscapes windows; return averaged concept logits ``(K, 1024, 2048)``."""
    _require_rgb(image_rgb, "Coralscapes image")
    crops: list[torch.Tensor] = []
    boxes: list[tuple[int, int, int, int]] = []
    for name, row0, col0, row1, col1 in coralscapes_windows():
        crop = np.ascontiguousarray(image_rgb[row0:row1, col0:col1])
        if crop.shape[0] != _TILE or crop.shape[1] != _TILE:
            raise RuntimeError(
                f"window {name} crop is {crop.shape[0]}x{crop.shape[1]}, expected {_TILE}x{_TILE}"
            )
        crops.append(predictor.preprocess(crop))
        boxes.append((row0, col0, row1, col1))
    logits = predictor.forward_concept_logits(torch.stack(crops, dim=0))
    return average_window_logits(logits, boxes, CORALSCAPES_HEIGHT, CORALSCAPES_WIDTH)


def build_coralscapes_concept_codes(csv_path: str | Path, concept: str) -> NDArray[np.int8]:
    """Map CoralscapesV2 class id -> binary concept code (0 not_given, 1 FALSE, 2 TRUE).

    Index 0 stays 0. That slot is unannotated mask pixels, not a class lookup,
    and it is not the class named ``background``. A native class missing from the
    CSV raises: a missing row is not the same thing as ``not_given``. Any cell
    other than TRUE, FALSE, or not_given raises.
    """
    path = Path(csv_path)
    frame = read_class_to_concepts(path)
    source_rows = frame[frame["source_dataset_source"] == "coralscapes_v2"]

    codes = np.zeros(NUM_CLASSES + 1, dtype=np.int8)
    n_true = 0
    n_false = 0
    n_not_given = 0
    for class_id, name in CORALSCAPES_V2_ID2NAME.items():
        key = str(name).lower()
        matched = source_rows[source_rows["source_label_class_name"] == key]
        if matched.empty:
            raise RuntimeError(
                f"CoralscapesV2 class {class_id} {name!r} is missing from {path}. "
                f"A missing row is not {concept}=not_given."
            )
        try:
            cells = {parse_concept_cell(value, rank=False) for value in matched[concept]}
        except ValueError as error:
            raise RuntimeError(f"CoralscapesV2 class {class_id} {name!r}: {error}") from error
        if len(cells) != 1:
            raise RuntimeError(
                f"CoralscapesV2 class {class_id} {name!r} has conflicting {concept} cells: "
                f"{sorted(cells)}"
            )
        cell = next(iter(cells))
        codes[class_id] = _CODE_OF_CELL[cell]
        n_true += cell == "true"
        n_false += cell == "false"
        n_not_given += cell == "not_given"
    if int(codes[0]) != 0:
        raise RuntimeError(f"label id 0 must stay {concept} code 0 (unannotated, not a class)")
    logger.info(
        "CoralscapesV2 %s codes: %d TRUE, %d FALSE, %d not_given (label 0 is unannotated)",
        concept,
        n_true,
        n_false,
        n_not_given,
    )
    return codes


# --------------------------------------------------------------------------- #
# Image source.
# --------------------------------------------------------------------------- #
def _dense_images(version: DatasetVersion) -> pd.DataFrame:
    """Return the images of a dense ``coralscapes_v2`` version sorted by ``image_id``.

    Raises:
        ValueError: The version is not dense.
    """
    if version.manifest.kind != "dense":
        raise ValueError(
            f"Coralscapes probes read a dense version; {version.manifest.dataset} "
            f"{version.manifest.version} is {version.manifest.kind!r}"
        )
    return version.images.to_pandas().sort_values("image_id").reset_index(drop=True)


def split_group(split: str) -> str:
    """Return the ``coralscapes_v2`` group that the probes score as ``split``."""
    return _SPLIT_GROUPS[split]


def _split_rows(images: pd.DataFrame, split: str) -> list[dict[str, Any]]:
    """Return the rows of a split's group, in ``image_id`` order."""
    return images[images["group"] == split_group(split)].to_dict("records")


def _load_row(version: DatasetVersion, row: Mapping[str, Any]) -> tuple[NDArray, NDArray]:
    """Load the RGB image and native label mask of an image row."""
    return (
        read_rgb(version.image_path(row)),
        read_label_mask(version.mask_path(row), NATIVE_IDS),
    )


def _load_probe_entry(
    version: DatasetVersion, frame: pd.DataFrame, entry: dict
) -> tuple[NDArray, NDArray]:
    """Load the train image and label of a probe-set entry.

    Args:
        version: The opened dense ``coralscapes_v2`` version.
        frame: The version's images, from :func:`_dense_images`.
        entry: Probe image entry with ``site``, ``stem`` and ``label_md5``.

    Returns:
        The RGB image and the native label map.

    Raises:
        KeyError: No ``train`` group image has the entry's ``stem`` as ``image_id``.
        ValueError: The version's label md5 differs from the entry's ``label_md5``.
    """
    match = frame[(frame["group"] == PROBE_GROUP) & (frame["image_id"] == entry["stem"])]
    if match.empty:
        raise KeyError(
            f"Probe image {entry['site']}/{entry['stem']} is not a {PROBE_GROUP!r} image of "
            f"{version.manifest.dataset} {version.manifest.version}"
        )
    img, lab = _load_row(version, match.iloc[0].to_dict())
    actual = label_md5(lab)
    if actual != entry["label_md5"]:
        raise ValueError(
            f"Label md5 mismatch for {entry['site']}/{entry['stem']}: the probe set has "
            f"{entry['label_md5']}, the version's mask has {actual}"
        )
    return img, lab


# --------------------------------------------------------------------------- #
# Probe sets.
# --------------------------------------------------------------------------- #
class ProbeSets(NamedTuple):
    """Probe JSON files of one label space, keyed by ``points_per_class_target``.

    Attributes:
        label_space: The label space every file is in.
        by_k: Each file's payload, keyed by its ``points_per_class_target``.
        images: One record per train image, its points grouped by k under ``points_by_k``.
    """

    label_space: LabelSpace
    by_k: dict[int, dict]
    images: list[dict]


def load_probes(paths: Sequence[Path]) -> ProbeSets:
    """Load probe JSON files of one label space, keyed by ``points_per_class_target``.

    Raises:
        ValueError: A file has no ``label_space`` field or names an unknown one.
        RuntimeError: No file is given, the files mix label spaces, two files share a
            k, or a file's ``num_classes`` differs from its label space.
    """
    probes: dict[int, dict] = {}
    space: LabelSpace | None = None
    for path in paths:
        probe_path = Path(path)
        with probe_path.open() as handle:
            probe = json.load(handle)
        if "label_space" not in probe:
            raise ValueError(f"{probe_path} has no label_space field")
        file_space = label_space_from_name(probe["label_space"])
        if space is not None and file_space is not space:
            raise RuntimeError(
                f"probe sets mix label spaces {space.name} and {file_space.name} ({probe_path})"
            )
        space = file_space
        k = int(probe["points_per_class_target"])
        if k in probes:
            raise RuntimeError(f"two probe sets target k={k}")
        if int(probe.get("num_classes", 0)) != space.num_classes:
            raise RuntimeError(
                f"{probe_path} num_classes is {probe.get('num_classes')}, "
                f"expected {space.num_classes}"
            )
        probes[k] = probe
    if space is None:
        raise RuntimeError("no probe sets were given")
    return ProbeSets(space, probes, _union_probe_images(probes))


def _union_probe_images(probes: Mapping[int, dict]) -> list[dict]:
    """One record per train image, with that image's points grouped by k.

    The same ``(site, stem)`` across budgets is forwarded once. A disagreeing
    ``label_md5`` raises.
    """
    merged: dict[tuple[str, str], dict] = {}
    for k, probe in probes.items():
        for image in probe["images"]:
            key = (str(image["site"]), str(image["stem"]))
            slot = merged.get(key)
            if slot is None:
                slot = {
                    "site": image["site"],
                    "stem": image["stem"],
                    "label_md5": image["label_md5"],
                    "height": image["height"],
                    "width": image["width"],
                    "points_by_k": {},
                }
                merged[key] = slot
            elif slot["label_md5"] != image["label_md5"]:
                raise RuntimeError(
                    f"{key[0]}/{key[1]} has label md5 {slot['label_md5']} at one k and "
                    f"{image['label_md5']} at k={k}"
                )
            if k in slot["points_by_k"]:
                raise RuntimeError(f"duplicate k={k} points for {key[0]}/{key[1]}")
            slot["points_by_k"][k] = list(image["points"])
    return sorted(merged.values(), key=lambda item: (str(item["site"]), str(item["stem"])))


# --------------------------------------------------------------------------- #
# Probe feature extraction + SVM.
# --------------------------------------------------------------------------- #
def _activated_concepts(predictor: CBMPredictor, image_rgb: NDArray[np.uint8]) -> torch.Tensor:
    """Strided concept logits, averaged, then activated. Shape ``(K, 1024, 2048)``."""
    logits = reassemble_concept_logits(predictor, image_rgb)
    return predictor.activate_concept_logits(logits)


def _sample_activated(
    activated: torch.Tensor, rows: NDArray[np.int64], cols: NDArray[np.int64]
) -> torch.Tensor:
    """Index a source-resolution ``(K, H, W)`` map. Returns ``(K, N)``."""
    if rows.shape != cols.shape:
        raise ValueError(f"row/col shape mismatch: {rows.shape} vs {cols.shape}")
    height, width = int(activated.shape[1]), int(activated.shape[2])
    if np.any(rows < 0) or np.any(cols < 0) or np.any(rows >= height) or np.any(cols >= width):
        raise ValueError(
            f"probe point outside the {height}x{width} activated map "
            f"(rows {int(rows.min())}..{int(rows.max())}, cols {int(cols.min())}..{int(cols.max())})"
        )
    row_t = torch.as_tensor(rows, dtype=torch.long, device=activated.device)
    col_t = torch.as_tensor(cols, dtype=torch.long, device=activated.device)
    return activated[:, row_t, col_t]


def _extract_probe_features(
    predictor: CBMPredictor, version: DatasetVersion, frame: pd.DataFrame, images: list[dict]
) -> dict[int, tuple[NDArray[np.float32], NDArray[np.int64]]]:
    """Sample every k's pixels from one activated map per train image."""
    feats: dict[int, list[NDArray[np.float32]]] = defaultdict(list)
    labels: dict[int, list[int]] = defaultdict(list)
    for entry in tqdm(images, desc="probe features"):
        image_np, label_np = _load_probe_entry(version, frame, entry)
        what = f"{entry.get('site')}/{entry.get('stem')}"
        _require_rgb(image_np, what)
        _require_label(label_np, what)
        activated = _activated_concepts(predictor, image_np)
        for k, pts in entry["points_by_k"].items():
            if not pts:
                continue
            rows = np.array([point["row"] for point in pts], dtype=np.int64)
            cols = np.array([point["col"] for point in pts], dtype=np.int64)
            sampled = _sample_activated(activated, rows, cols)
            feats[k].append(sampled.transpose(0, 1).cpu().numpy().astype(np.float32))
            labels[k].extend(int(point["class_id"]) for point in pts)
    if not feats:
        raise RuntimeError("No probe features extracted (empty probe set?)")
    return {
        k: (np.concatenate(feats[k], axis=0), np.asarray(labels[k], dtype=np.int64))
        for k in sorted(feats)
    }


def _fit_linear_svm(
    X: NDArray[np.float32], y: NDArray[np.int64], svm_c: float
) -> tuple[Any, Any, float]:
    from sklearn.preprocessing import StandardScaler
    from sklearn.svm import LinearSVC

    scaler = StandardScaler().fit(X)
    Xs = scaler.transform(X)
    svm = LinearSVC(C=svm_c, max_iter=20000, dual="auto")
    svm.fit(Xs, y)
    train_acc = float(svm.score(Xs, y))
    return scaler, svm, train_acc


def _folded_linear(
    scaler: Any, svm: Any, device: torch.device
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Fold StandardScaler into the SVM linear layer: scores = X @ Weff.T + beff."""
    coef = torch.tensor(svm.coef_, dtype=torch.float32, device=device)  # (ncls, D)
    intercept = torch.tensor(svm.intercept_, dtype=torch.float32, device=device)  # (ncls,)
    mean = torch.tensor(scaler.mean_, dtype=torch.float32, device=device)  # (D,)
    scale = torch.tensor(scaler.scale_, dtype=torch.float32, device=device)  # (D,)
    weff = coef / scale[None, :]
    beff = intercept - (mean / scale) @ coef.t()
    classes = torch.tensor(svm.classes_, dtype=torch.long, device=device)
    return weff, beff, classes


def _probe_info(
    k: int, probe: dict, X: NDArray[np.float32], y: NDArray[np.int64], train_acc: float
) -> dict:
    classes_present = sorted({int(class_id) for class_id in y})
    return {
        "k": k,
        "num_points": int(X.shape[0]),
        "feature_dim": int(X.shape[1]),
        "num_classes_present": len(classes_present),
        "classes_present": classes_present,
        "train_accuracy": train_acc,
        "points_per_class_target": int(probe["points_per_class_target"]),
        "points_per_class": {
            str(class_id): int(count)
            for class_id, count in probe.get("points_per_class", {}).items()
        },
    }


def _fit_heads(
    label_space: LabelSpace,
    probes: Mapping[int, dict],
    features: Mapping[int, tuple[NDArray[np.float32], NDArray[np.int64]]],
    svm_c: float,
    device: torch.device,
) -> tuple[dict[int, tuple[torch.Tensor, torch.Tensor, torch.Tensor]], dict[int, dict]]:
    """Fit one folded linear SVM per k; return the heads and their probe info."""
    heads: dict[int, tuple[torch.Tensor, torch.Tensor, torch.Tensor]] = {}
    probe_infos: dict[int, dict] = {}
    for k in sorted(probes):
        X, y = features[k]
        if int(y.min()) < 1 or int(y.max()) > label_space.num_classes:
            raise RuntimeError(
                f"k={k} probe labels must be {label_space.name} ids "
                f"1..{label_space.num_classes}, got {int(y.min())}..{int(y.max())}"
            )
        scaler, svm, train_acc = _fit_linear_svm(X, y, svm_c)
        logger.info(
            "[%s/k=%d] SVM train accuracy = %s (%d pixels)",
            label_space.eval_name,
            k,
            fmt(train_acc),
            X.shape[0],
        )
        heads[k] = _folded_linear(scaler, svm, device)
        probe_infos[k] = _probe_info(k, probes[k], X, y, train_acc)
    return heads, probe_infos


@torch.no_grad()
def _predict_classes_from_activated(
    activated: torch.Tensor,
    weff: torch.Tensor,
    beff: torch.Tensor,
    classes: torch.Tensor,
    strip_rows: int = _SVM_STRIP_ROWS,
) -> torch.Tensor:
    """Apply the folded SVM on ``(K, H, W)`` features, in row strips. Returns class ids."""
    if activated.dim() != 3:
        raise ValueError(f"activated concepts must be (K, H, W); got {tuple(activated.shape)}")
    n_channels, height, width = activated.shape
    if weff.shape[1] != n_channels:
        raise RuntimeError(
            f"SVM feature dim {int(weff.shape[1])} does not match concept channels {int(n_channels)}"
        )
    if strip_rows < 1:
        raise ValueError(f"strip_rows must be >= 1, got {strip_rows}")
    pred = torch.empty((height, width), dtype=torch.long, device=activated.device)
    for row0 in range(0, height, strip_rows):
        row1 = min(height, row0 + strip_rows)
        slab = activated[:, row0:row1, :]
        flat = slab.permute(1, 2, 0).reshape(-1, n_channels)
        scores = flat @ weff.t() + beff
        # A two-class LinearSVC has one score column: positive selects classes[1], else classes[0].
        winner = (scores[:, 0] > 0).long() if scores.shape[1] == 1 else scores.argmax(dim=1)
        pred[row0:row1] = classes[winner].reshape(row1 - row0, width)
    return pred


def legacy_label(label_v2: NDArray[np.integer], lut: NDArray[np.uint8]) -> NDArray[np.uint8]:
    """Map a native CoralscapesV2 mask through ``lut``. Index 0 stays unannotated."""
    label = np.asarray(label_v2)
    if label.size == 0:
        return np.asarray(label, dtype=np.uint8)
    return np.asarray(lut, dtype=np.uint8)[label]


@torch.no_grad()
def _score_classes(
    activated: torch.Tensor,
    label_v2: NDArray[np.integer],
    label_space: LabelSpace,
    heads: Mapping[int, tuple[torch.Tensor, torch.Tensor, torch.Tensor]],
    confusions: Mapping[int, ClassConfusion],
) -> None:
    """Apply every fitted head to one activated map and update its confusion matrix.

    ``label_v2`` is the native mask; ``label_space.lut`` maps it onto the space's ids.
    Ground-truth id 0 stays ignored. The activated tensor is shared by every k.
    """
    if set(heads) != set(confusions):
        raise ValueError(
            f"heads and confusions cover different budgets: {sorted(heads)} vs {sorted(confusions)}"
        )
    mapped = legacy_label(label_v2, label_space.lut)
    if mapped.size and int(mapped.max()) > label_space.num_classes:
        raise RuntimeError(
            f"{label_space.name} label id {int(mapped.max())} is outside "
            f"0..{label_space.num_classes}"
        )
    gt = np.asarray(mapped).reshape(-1)
    for k, (weff, beff, classes) in heads.items():
        pred = _predict_classes_from_activated(activated, weff, beff, classes)
        confusions[k].update(gt, pred.detach().cpu().numpy().reshape(-1))


# --------------------------------------------------------------------------- #
# Concept and hierarchy scoring.
# --------------------------------------------------------------------------- #
def _class_taxonomy_targets(
    hier: HierarchicalScore,
    concept_names: list[str],
    taxonomy_csv: str | Path,
    device: torch.device,
) -> torch.Tensor:
    """Dense taxonomy ids per CoralscapesV2 class, shape ``(R, NUM_CLASSES + 1)``.

    Column 0 is unannotated mask pixels and is not_given at every rank.
    """
    names = [CORALSCAPES_V2_ID2NAME[i] for i in range(1, NUM_CLASSES + 1)]
    lookup = build_concept_lookup("coralscapes_v2", names, concept_names, taxonomy_csv)
    rows = np.stack([np.zeros(len(concept_names), dtype=np.int8), *(lookup.row(n) for n in names)])
    return hier.dense_targets(torch.from_numpy(rows).to(device))


class _ConceptScoring:
    """Binary-concept and hierarchy scores of one split, read off the native label mask."""

    def __init__(
        self, concepts: ProbeConcepts, codes: NDArray[np.int8], device: torch.device
    ) -> None:
        self.concept_names = list(concepts.concept_names)
        self.codes = codes
        self.channels = [self.concept_names.index(c) for c in REPORTED_BINARY_CONCEPTS]
        self.binary = BinaryConceptStats(
            list(REPORTED_BINARY_CONCEPTS), list(range(len(REPORTED_BINARY_CONCEPTS)))
        )
        self.hier = HierarchicalScore(
            self.concept_names, concepts.taxonomy_csv, concepts.removed_concepts
        )
        self.class_targets = _class_taxonomy_targets(
            self.hier, self.concept_names, concepts.taxonomy_csv, device
        )

    def update(self, activated: torch.Tensor, label_v2: NDArray[np.integer]) -> None:
        """Score one image's activated map against its native label mask."""
        if int(activated.shape[0]) != len(self.concept_names):
            raise RuntimeError(
                f"activated concept map has {int(activated.shape[0])} channels, "
                f"but concept_names has {len(self.concept_names)}"
            )
        flat = np.asarray(label_v2).reshape(-1)
        probs = activated[self.channels].reshape(len(self.channels), -1).t().cpu().numpy()
        self.binary.update(self.codes[flat], probs, unannotated=flat == 0)
        labels = torch.from_numpy(flat).to(activated.device).long()
        self.hier.update(activated.reshape(activated.shape[0], -1), self.class_targets[:, labels])

    def write(self, results: dict, status: str, processed: int) -> None:
        """Put the concept blocks, and the hierarchy once ``done``, into ``results``."""
        binary = self.binary.to_dict()
        for concept in REPORTED_BINARY_CONCEPTS:
            block = concept_block_from_binary(binary, concept, self.binary.threshold)
            results[concept] = {**block, "status": status, "num_images": processed}
        if status == "done":
            results["hierarchical"] = self.hier.to_dict()


def _log_concepts(eval_name: str, split: str, results: dict) -> None:
    for concept in REPORTED_BINARY_CONCEPTS:
        block = results[concept]
        logger.info(
            "[%s_%s/%s] acc=%s precision=%s recall=%s f1=%s "
            "(n_true=%d n_false=%d n_not_given=%d n_unannotated=%d)",
            eval_name,
            split,
            concept,
            fmt(block["accuracy"]),
            fmt(block["precision"]),
            fmt(block["recall"]),
            fmt(block["f1"]),
            block["n_true"],
            block["n_false"],
            block["n_not_given"],
            block["n_unannotated"],
        )
    h = results["hierarchical"]
    logger.info(
        "[%s_%s/hierarchical] consistency=%s severity(mean norm)=%s ECE(mean)=%s",
        eval_name,
        split,
        fmt(h.get("consistency/overall")),
        fmt(h.get("severity/mean_normalized")),
        fmt(h.get("ece/mean")),
    )


# --------------------------------------------------------------------------- #
# Split scoring.
# --------------------------------------------------------------------------- #
def _evaluate_split(
    *,
    split: str,
    predictor: CBMPredictor,
    version: DatasetVersion,
    frame: pd.DataFrame,
    label_space: LabelSpace,
    heads: dict[int, tuple[torch.Tensor, torch.Tensor, torch.Tensor]],
    probe_infos: dict[int, dict],
    scoring: _ConceptScoring | None,
    out_dir: Path,
    max_images: int | None,
) -> dict:
    """Score one split with every fitted SVM, and the concepts when given. One forward per image."""
    n_classes = label_space.num_classes
    eval_class_ids = list(range(1, n_classes + 1))
    confusions = {k: ClassConfusion(n_classes + 1, ignore_index=0) for k in heads}
    results: dict = {
        "eval": label_space.eval_name,
        "split": split,
        "label_space": label_space.name,
        "num_classes": n_classes,
        "by_k": {str(k): {"probe": probe_infos[k]} for k in sorted(heads)},
    }

    def _dump(status: str, processed: int) -> None:
        for k, confusion in confusions.items():
            detail = confusion.to_dict(id2name=label_space.id2name, class_ids=eval_class_ids)
            scores: dict = {
                "status": status,
                "num_images": processed,
                "accuracy": detail["accuracy"],
                "miou": detail["miou"],
            }
            if status == "done":
                scores["per_class"] = detail["per_class"]
            results["by_k"][str(k)][split] = scores
        if scoring is not None:
            scoring.write(results, status, processed)
        write_json(out_dir / "metrics.json", results)

    rows = _split_rows(frame, split)
    n_images = len(rows) if max_images is None else min(len(rows), max_images)
    t0 = time.time()
    processed = 0
    for row in tqdm(rows, total=n_images, desc=f"dense {split} ({label_space.eval_name})"):
        if max_images is not None and processed >= max_images:
            break
        image_np, label_np = _load_row(version, row)
        what = f"{split} image {processed}"
        _require_rgb(image_np, what)
        _require_label(label_np, what)
        activated = _activated_concepts(predictor, image_np)
        _score_classes(activated, label_np, label_space, heads, confusions)
        if scoring is not None:
            scoring.update(activated, label_np)
        processed += 1
        if processed % 25 == 0:
            _dump("running", processed)

    _dump("done", processed)
    logger.info(
        "[%s_%s] DONE in %.1fs over %d images",
        label_space.eval_name,
        split,
        time.time() - t0,
        processed,
    )
    for k in sorted(heads):
        scores = results["by_k"][str(k)][split]
        logger.info(
            "[%s_%s/k=%d] acc=%s miou=%s (probe train acc %s, %d pixels)",
            label_space.eval_name,
            split,
            k,
            fmt(scores["accuracy"]),
            fmt(scores["miou"]),
            fmt(probe_infos[k]["train_accuracy"]),
            probe_infos[k]["num_points"],
        )
    if scoring is not None:
        _log_concepts(label_space.eval_name, split, results)
    return results


def evaluate_coralscapes_probe(
    label_space: LabelSpace,
    probe_jsons: Sequence[Path],
    version: DatasetVersion,
    *,
    predictor: CBMPredictor,
    output_dir: str | Path,
    concepts: ProbeConcepts | None = None,
    splits: Sequence[str] = ("test",),
    svm_c: float = 1.0,
    max_test_images: int | None = None,
) -> dict[str, dict]:
    """Fit one SVM per probe set, then score every split in ``splits``.

    Train images are the union of the probe sets and each image is forwarded once.
    Each split image is forwarded once and scored by every SVM, and by the binary
    concepts and the hierarchy when ``concepts`` is given.

    Args:
        label_space: The space the probe sets are in and the splits are scored in.
        probe_jsons: Probe JSON files, one per ``points_per_class_target``.
        version: The opened dense ``coralscapes_v2`` version.
        predictor: The model under evaluation.
        output_dir: Each split's metrics go to ``<output_dir>/<eval_name>_<split>/metrics.json``.
        concepts: Concept channels and taxonomy to score; None scores classes only.
        splits: Subset of ``("val", "test")``.
        svm_c: ``LinearSVC`` regularisation.
        max_test_images: Caps the images of every split.

    Returns:
        ``{f"{label_space.eval_name}_{split}": metrics}``.

    Raises:
        ValueError: A split is unknown.
        RuntimeError: ``concepts`` lacks a reported binary concept, the probe files are
            in another label space, or a probe set yields no pixels.
    """
    unknown = [split for split in splits if split not in SPLITS]
    if unknown:
        raise ValueError(f"unknown Coralscapes splits {unknown}; expected a subset of {SPLITS}")
    codes = None
    if concepts is not None:
        require_reported_concepts(concepts.concept_names)
        codes = np.stack(
            [
                build_coralscapes_concept_codes(concepts.taxonomy_csv, concept)
                for concept in REPORTED_BINARY_CONCEPTS
            ],
            axis=1,
        )

    probes = load_probes(probe_jsons)
    if probes.label_space is not label_space:
        raise RuntimeError(f"the probe sets are {probes.label_space.name}, not {label_space.name}")
    logger.info(
        "Loaded %d %s probe sets: k=%s",
        len(probes.by_k),
        label_space.name,
        ", ".join(str(k) for k in sorted(probes.by_k)),
    )

    frame = _dense_images(version)
    features = _extract_probe_features(predictor, version, frame, probes.images)
    missing_k = sorted(set(probes.by_k) - set(features))
    if missing_k:
        raise RuntimeError(f"probe budgets with no extracted pixels: {missing_k}")
    heads, probe_infos = _fit_heads(label_space, probes.by_k, features, svm_c, predictor.device)

    results: dict[str, dict] = {}
    for split in splits:
        scoring = None
        if concepts is not None and codes is not None:
            scoring = _ConceptScoring(concepts, codes, predictor.device)
        results[f"{label_space.eval_name}_{split}"] = _evaluate_split(
            split=split,
            predictor=predictor,
            version=version,
            frame=frame,
            label_space=label_space,
            heads=heads,
            probe_infos=probe_infos,
            scoring=scoring,
            out_dir=Path(output_dir) / f"{label_space.eval_name}_{split}",
            max_images=max_test_images,
        )
    return results
