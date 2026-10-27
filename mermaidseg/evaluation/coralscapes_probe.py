"""CoralscapesV2 linear-probe evaluation.

Trains a linear SVM (sklearn ``LinearSVC``) on activated concept features sampled
at a fixed probe point set from the train split (see
``scripts/build_coralscapes_v2_probe_points.py``), then applies that one SVM over
each requested CoralscapesV2 split (``val`` and/or ``test``) and reports accuracy
and mIoU over the 95 classes.

Every 1024x2048 image is predicted with three windows, each resized to the model
input size:

- left:   rows ``0:1024``, cols ``0:1024``
- center: rows ``0:1024``, cols ``512:1536``
- right:  rows ``0:1024``, cols ``1024:2048``

The three concept-logit maps are bilinearly upsampled back onto those rectangles
and averaged, on the model device, where the windows overlap. The model's
``concept_outputs_activation`` (per-rank softmax, sigmoid on the binary tail) is
applied to the averaged logits. Probe pixels and the dense SVM both read that
activated map, so train and test use the same features. Only the ``concepts``
feature kind is supported.

The binary concepts in ``REPORTED_BINARY_CONCEPTS`` (``bleached``,
``anthropogenic``) are scored from the same activated map, not from the SVM
class prediction. ``TRUE`` (code 2) is the positive class, ``FALSE`` (code 1) is
the negative class, ``not_given`` (code 0) is excluded, and mask id 0
(unannotated) is excluded separately. A probability of exactly 0.5 counts as a
negative prediction. Hierarchical metrics (consistency, mistake severity, ECE)
are computed from the per-rank softmax of the same map against the taxonomy of
each pixel's class in ``class_to_concepts.csv``.

Probe/eval images are resolved either from a local Cityscapes-style mirror
(``--coralscapes-root``, matched by stem, md5-verified) or from the HuggingFace
dataset ``josauder/coralscapesV2`` (matched by label md5).
"""

from __future__ import annotations

import hashlib
import logging
import time
from collections.abc import Sequence
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from numpy.typing import NDArray
from tqdm import tqdm

from mermaidseg.datasets.coralscapes_v2.coralscapes_v2_dataset import CORALSCAPES_V2_ID2NAME
from mermaidseg.evaluation.gt_mapping import build_concept_lookup
from mermaidseg.evaluation.hierarchical import HierarchicalScore
from mermaidseg.evaluation.metrics import (
    REPORTED_BINARY_CONCEPTS,
    BinaryConceptScore,
    ClassConfusion,
)
from mermaidseg.evaluation.predictor import CBMPredictor
from mermaidseg.evaluation.reporting import fmt, write_json

logger = logging.getLogger(__name__)

NUM_CLASSES = 95
CORALSCAPES_HEIGHT = 1024
CORALSCAPES_WIDTH = 2048
SPLITS = ("val", "test")
_HF_SPLIT_NAMES = {"val": "validation", "test": "test"}
_TILE = 1024
_SVM_STRIP_ROWS = 128
_ALLOWED_CONCEPT_CELLS = ("true", "false", "not_given")


def _to_label(arr_like) -> NDArray[np.uint8]:
    arr = np.asarray(arr_like)
    if arr.ndim == 3:
        arr = arr[..., 0]
    return arr.astype(np.uint8, copy=False)


def _label_md5(arr: NDArray[np.uint8]) -> str:
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
    if label.ndim != 2 or tuple(int(s) for s in label.shape) != (CORALSCAPES_HEIGHT, CORALSCAPES_WIDTH):
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
        raise ValueError(
            f"got {window_logits.shape[0]} logit maps for {len(windows)} windows"
        )
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


@torch.no_grad()
def reassemble_concept_logits(predictor: CBMPredictor, image_rgb: NDArray[np.uint8]) -> torch.Tensor:
    """Run the three Coralscapes windows and return averaged concept logits ``(K, 1024, 2048)``."""
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


def _raw_concept_cell(value: object) -> str:
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return "nan"
    return str(value).strip().lower()


def build_coralscapes_concept_codes(csv_path: str | Path, concept: str) -> NDArray[np.int8]:
    """Map CoralscapesV2 class id -> binary concept code (0 not_given, 1 FALSE, 2 TRUE).

    Index 0 stays 0. That slot is unannotated mask pixels, not a class lookup,
    and it is not the class named ``background``. A native class missing from the
    CSV raises: a missing row is not the same thing as ``not_given``. Any cell
    other than TRUE, FALSE, or not_given raises.
    """
    import pandas as pd

    path = Path(csv_path)
    frame = pd.read_csv(path)
    frame["source_label_class_name"] = frame["source_label_class_name"].astype(str).str.lower()
    frame["source_dataset_source"] = frame["source_dataset_source"].astype(str).str.lower()
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
        cells = {_raw_concept_cell(value) for value in matched[concept].tolist()}
        if len(cells) != 1:
            raise RuntimeError(
                f"CoralscapesV2 class {class_id} {name!r} has conflicting {concept} cells: "
                f"{sorted(cells)}"
            )
        cell = next(iter(cells))
        if cell == "true":
            codes[class_id] = 2
            n_true += 1
        elif cell == "false":
            codes[class_id] = 1
            n_false += 1
        elif cell == "not_given":
            codes[class_id] = 0
            n_not_given += 1
        else:
            raise RuntimeError(
                f"CoralscapesV2 class {class_id} {name!r} has {concept} cell {cell!r}. "
                f"Expected one of {list(_ALLOWED_CONCEPT_CELLS)}."
            )
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
# Image sources.
# --------------------------------------------------------------------------- #
class LocalMirrorSource:
    """Cityscapes-style mirror (leftImg8bit + gtFine/*_labelIds.png)."""

    def __init__(self, root: str | Path):
        self.root = Path(root)

    def train_image_label(self, entry: dict) -> tuple[NDArray[np.uint8], NDArray[np.uint8]]:
        site, stem = entry["site"], entry["stem"]
        from PIL import Image

        img = np.asarray(
            Image.open(self.root / "leftImg8bit" / "train" / site / f"{stem}_leftImg8bit.png").convert("RGB"),
            dtype=np.uint8,
        )
        lab = _to_label(
            Image.open(self.root / "gtFine" / "train" / site / f"{stem}_gtFine_labelIds.png")
        )
        md5 = _label_md5(lab)
        if md5 != entry["label_md5"]:
            logger.warning(
                "Label md5 mismatch for %s/%s (probe set may not match this mirror).", site, stem
            )
        return img, lab

    def _label_paths(self, split: str) -> list[Path]:
        return sorted((self.root / "gtFine" / split).glob("*/*_gtFine_labelIds.png"))

    def iter_split(self, split: str):
        from PIL import Image

        for p in self._label_paths(split):
            site = p.parent.name
            stem = p.name.replace("_gtFine_labelIds.png", "")
            img = np.asarray(
                Image.open(self.root / "leftImg8bit" / split / site / f"{stem}_leftImg8bit.png").convert("RGB"),
                dtype=np.uint8,
            )
            yield f"{site}/{stem}", img, _to_label(Image.open(p))

    def num_split(self, split: str) -> int:
        return len(self._label_paths(split))


class HFSource:
    """HuggingFace ``josauder/coralscapesV2`` dataset (default config)."""

    def __init__(self, repo: str = "josauder/coralscapesV2"):
        from datasets import load_dataset

        logger.info("Loading HF dataset %s ...", repo)
        self.ds = load_dataset(repo)
        self._md5_index: dict[str, int] | None = None

    def _build_index(self) -> None:
        logger.info("Indexing HF train labels by md5 (%d images) ...", len(self.ds["train"]))
        idx: dict[str, int] = {}
        train = self.ds["train"]
        for i in tqdm(range(len(train)), desc="md5 index"):
            lab = _to_label(train[i]["label"])
            idx[_label_md5(lab)] = i
        self._md5_index = idx

    def train_image_label(self, entry: dict) -> tuple[NDArray[np.uint8], NDArray[np.uint8]]:
        if self._md5_index is None:
            self._build_index()
        i = self._md5_index.get(entry["label_md5"])
        if i is None:
            raise KeyError(
                f"Probe image (site={entry['site']} stem={entry['stem']} md5={entry['label_md5'][:10]}) "
                "not found in HF train split by md5. The dataset revision may differ from the probe set."
            )
        item = self.ds["train"][i]
        return np.asarray(item["image"].convert("RGB"), dtype=np.uint8), _to_label(item["label"])

    def iter_split(self, split: str):
        ds = self.ds[_HF_SPLIT_NAMES[split]]
        for i in range(len(ds)):
            item = ds[i]
            yield str(i), np.asarray(item["image"].convert("RGB"), dtype=np.uint8), _to_label(
                item["label"]
            )

    def num_split(self, split: str) -> int:
        return len(self.ds[_HF_SPLIT_NAMES[split]])


def build_source(coralscapes_root: str | Path | None, hf_repo: str):
    if coralscapes_root is not None:
        logger.info("Using local CoralscapesV2 mirror at %s", coralscapes_root)
        return LocalMirrorSource(coralscapes_root)
    return HFSource(hf_repo)


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
    predictor: CBMPredictor, source, images: list[dict]
) -> tuple[NDArray[np.float32], NDArray[np.int64]]:
    feats: list[NDArray[np.float32]] = []
    labels: list[int] = []
    for entry in tqdm(images, desc="probe features (concepts)"):
        pts = entry["points"]
        if not pts:
            continue
        image_np, label_np = source.train_image_label(entry)
        what = f"{entry.get('site')}/{entry.get('stem')}"
        _require_rgb(image_np, what)
        _require_label(label_np, what)
        activated = _activated_concepts(predictor, image_np)
        rows = np.array([p["row"] for p in pts], dtype=np.int64)
        cols = np.array([p["col"] for p in pts], dtype=np.int64)
        sampled = _sample_activated(activated, rows, cols)  # (K, N)
        feats.append(sampled.transpose(0, 1).cpu().numpy().astype(np.float32))
        labels.extend(int(p["class_id"]) for p in pts)
    if not feats:
        raise RuntimeError("No probe features extracted (empty probe set?)")
    return np.concatenate(feats, axis=0), np.asarray(labels, dtype=np.int64)


def _fit_linear_svm(X: NDArray[np.float32], y: NDArray[np.int64], svm_c: float):
    from sklearn.preprocessing import StandardScaler
    from sklearn.svm import LinearSVC

    scaler = StandardScaler().fit(X)
    Xs = scaler.transform(X)
    svm = LinearSVC(C=svm_c, max_iter=20000, dual="auto")
    svm.fit(Xs, y)
    train_acc = float(svm.score(Xs, y))
    return scaler, svm, train_acc


def _folded_linear(scaler, svm, device: torch.device) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Fold StandardScaler into the SVM linear layer: scores = X @ Weff.T + beff."""
    coef = torch.tensor(svm.coef_, dtype=torch.float32, device=device)  # (ncls, D)
    intercept = torch.tensor(svm.intercept_, dtype=torch.float32, device=device)  # (ncls,)
    mean = torch.tensor(scaler.mean_, dtype=torch.float32, device=device)  # (D,)
    scale = torch.tensor(scaler.scale_, dtype=torch.float32, device=device)  # (D,)
    weff = coef / scale[None, :]
    beff = intercept - (mean / scale) @ coef.t()
    classes = torch.tensor(svm.classes_, dtype=torch.long, device=device)
    return weff, beff, classes


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
        pred[row0:row1] = classes[scores.argmax(dim=1)].reshape(row1 - row0, width)
    return pred


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


def _evaluate_split(
    *,
    split: str,
    predictor: CBMPredictor,
    source,
    weff: torch.Tensor,
    beff: torch.Tensor,
    classes: torch.Tensor,
    probe_info: dict,
    concept_names: list[str],
    concept_codes: dict[str, NDArray[np.int8]],
    taxonomy_csv: str | Path,
    out_dir: Path,
    max_images: int | None,
) -> dict:
    """Score one split with the fitted SVM, the binary concepts, and the hierarchy."""
    kind = "concepts"
    eval_class_ids = list(range(1, NUM_CLASSES + 1))
    conf = ClassConfusion(NUM_CLASSES + 1, ignore_index=0)  # 0..95, 0 = ignore
    binary = {concept: BinaryConceptScore(threshold=0.5) for concept in concept_codes}
    hier = HierarchicalScore(concept_names, taxonomy_csv)
    class_targets = _class_taxonomy_targets(hier, concept_names, taxonomy_csv, predictor.device)
    results: dict = {
        "eval": "coralscapes",
        "split": split,
        "feature_kinds": {kind: {"probe": probe_info}},
    }

    def _dump(status: str, processed: int) -> dict:
        d = conf.to_dict(id2name=CORALSCAPES_V2_ID2NAME, class_ids=eval_class_ids)
        scores = {"status": status, "num_images": processed, "accuracy": d["accuracy"], "miou": d["miou"]}
        if status == "done":
            scores["per_class"] = d["per_class"]
            results["hierarchical"] = hier.to_dict()
        results["feature_kinds"][kind][split] = scores
        for concept, score in binary.items():
            results[concept] = {**score.to_dict(), "status": status, "num_images": processed}
        write_json(out_dir / "metrics.json", results)
        return scores

    n_images = source.num_split(split)
    if max_images is not None:
        n_images = min(n_images, max_images)
    t0 = time.time()
    processed = 0
    for _key, image_np, label_np in tqdm(
        source.iter_split(split), total=n_images, desc=f"dense {split} (concepts)"
    ):
        if max_images is not None and processed >= max_images:
            break
        what = f"{split} image {processed}"
        _require_rgb(image_np, what)
        _require_label(label_np, what)
        if int(label_np.min()) < 0 or int(label_np.max()) > NUM_CLASSES:
            raise RuntimeError(
                f"{what} has label ids outside 0..{NUM_CLASSES} "
                f"(min={int(label_np.min())}, max={int(label_np.max())})"
            )
        activated = _activated_concepts(predictor, image_np)
        if int(activated.shape[0]) != len(concept_names):
            raise RuntimeError(
                f"activated concept map has {int(activated.shape[0])} channels, "
                f"but concept_names has {len(concept_names)}"
            )
        pred = _predict_classes_from_activated(activated, weff, beff, classes)
        conf.update(label_np.reshape(-1), pred.cpu().numpy().reshape(-1))
        unannotated = label_np == 0
        for concept, score in binary.items():
            prob = activated[concept_names.index(concept)].cpu().numpy()
            score.update(concept_codes[concept][label_np], prob, unannotated=unannotated)
        labels = torch.from_numpy(label_np).to(activated.device).long().reshape(-1)
        hier.update(activated.reshape(activated.shape[0], -1), class_targets[:, labels])
        processed += 1
        if processed % 25 == 0:
            _dump("running", processed)

    scores = _dump("done", processed)
    logger.info(
        "[coralscapes_%s/%s] DONE in %.1fs: acc=%s miou=%s over %d images",
        split,
        kind,
        time.time() - t0,
        fmt(scores["accuracy"]),
        fmt(scores["miou"]),
        processed,
    )
    for concept in binary:
        block = results[concept]
        logger.info(
            "[coralscapes_%s/%s] acc=%s precision=%s recall=%s f1=%s "
            "(n_true=%d n_false=%d n_not_given=%d n_unannotated=%d)",
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
        "[coralscapes_%s/hierarchical] consistency=%s severity(mean norm)=%s ECE(mean)=%s",
        split,
        fmt(h.get("consistency/overall")),
        fmt(h.get("severity/mean_normalized")),
        fmt(h.get("ece/mean")),
    )
    return results


def evaluate_coralscapes_probe(
    *,
    predictor: CBMPredictor,
    probe_json: str | Path,
    feature_kinds: list[str],
    coralscapes_root: str | Path | None,
    output_dir: str | Path,
    concept_names: list[str],
    taxonomy_csv: str | Path,
    splits: Sequence[str] = ("test",),
    hf_repo: str = "josauder/coralscapesV2",
    svm_c: float = 1.0,
    max_test_images: int | None = None,
) -> dict[str, dict]:
    """Fit the probe once on the train probe points, then score every split in ``splits``.

    Returns ``{split: metrics}``; each split's metrics also go to
    ``<output_dir>/coralscapes_<split>/metrics.json``. ``max_test_images`` caps every split.
    """
    import json

    with Path(probe_json).open() as f:
        probe = json.load(f)
    images = probe["images"]
    logger.info(
        "Loaded probe set: %d images, %d points, target %s pts/class",
        probe.get("num_images", len(images)),
        probe.get("total_points", sum(len(im["points"]) for im in images)),
        probe.get("points_per_class_target"),
    )

    if feature_kinds != ["concepts"]:
        raise RuntimeError(
            "Strided Coralscapes eval only supports feature kind 'concepts'. "
            "Concept logits are averaged across the three windows, then activated. "
            f"Got {feature_kinds!r}."
        )
    unknown = [split for split in splits if split not in SPLITS]
    if unknown:
        raise ValueError(f"unknown Coralscapes splits {unknown}; expected a subset of {SPLITS}")
    missing = [c for c in REPORTED_BINARY_CONCEPTS if c not in concept_names]
    if missing:
        raise RuntimeError(
            f"Coralscapes eval requires concept channels named exactly {missing}, "
            f"which concept_names ({len(concept_names)} entries) does not have."
        )
    concept_codes = {
        concept: build_coralscapes_concept_codes(taxonomy_csv, concept)
        for concept in REPORTED_BINARY_CONCEPTS
    }

    source = build_source(coralscapes_root, hf_repo)
    kind = "concepts"
    logger.info("=== CoralscapesV2 linear probe: feature kind = %s ===", kind)
    X, y = _extract_probe_features(predictor, source, images)
    classes_present = sorted(set(int(c) for c in y))
    logger.info(
        "[coralscapes/%s] probe features: X=%s, %d classes present",
        kind,
        X.shape,
        len(classes_present),
    )
    scaler, svm, train_acc = _fit_linear_svm(X, y, svm_c)
    logger.info("[coralscapes/%s] SVM train accuracy = %s", kind, fmt(train_acc))
    weff, beff, classes = _folded_linear(scaler, svm, predictor.device)
    probe_info = {
        "num_points": int(X.shape[0]),
        "feature_dim": int(X.shape[1]),
        "num_classes_present": len(classes_present),
        "classes_present": classes_present,
        "train_accuracy": train_acc,
    }

    return {
        split: _evaluate_split(
            split=split,
            predictor=predictor,
            source=source,
            weff=weff,
            beff=beff,
            classes=classes,
            probe_info=probe_info,
            concept_names=concept_names,
            concept_codes=concept_codes,
            taxonomy_csv=taxonomy_csv,
            out_dir=Path(output_dir) / f"coralscapes_{split}",
            max_images=max_test_images,
        )
        for split in splits
    }
