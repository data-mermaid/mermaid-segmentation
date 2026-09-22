"""CoralscapesV2 linear-probe evaluation.

Trains a simple linear SVM (sklearn ``LinearSVC``) on per-pixel model embeddings
sampled at a fixed probe point set (5 pixels/class from different connected
components, see ``scripts/build_coralscapes_v2_probe_points.py``), then applies
it densely over the CoralscapesV2 test split and reports accuracy + mIoU over the
95 classes.

Feature kinds (``--probe-features``):
- ``concepts`` : activated concept probabilities (the bottleneck), default.
- ``dpt``      : 256-d DPT features before the concept projection.
- ``backbone`` : raw DINOv3 patch tokens (bilinearly upsampled).

Probe/test images are resolved either from a local Cityscapes-style mirror
(``--coralscapes-root``, matched by stem, md5-verified) or from the HuggingFace
dataset ``josauder/coralscapesV2`` (matched by label md5).
"""

from __future__ import annotations

import hashlib
import logging
import time
from pathlib import Path

import numpy as np
import torch
from numpy.typing import NDArray
from tqdm import tqdm

from mermaidseg.datasets.coralscapes_v2.coralscapes_v2_dataset import CORALSCAPES_V2_ID2NAME
from mermaidseg.evaluation.metrics import ClassConfusion
from mermaidseg.evaluation.predictor import CBMPredictor
from mermaidseg.evaluation.reporting import fmt, write_json

logger = logging.getLogger(__name__)

NUM_CLASSES = 95


def _to_label(arr_like) -> NDArray[np.uint8]:
    arr = np.asarray(arr_like)
    if arr.ndim == 3:
        arr = arr[..., 0]
    return arr.astype(np.uint8, copy=False)


def _label_md5(arr: NDArray[np.uint8]) -> str:
    return hashlib.md5(np.ascontiguousarray(arr, dtype=np.uint8).tobytes()).hexdigest()


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

    def iter_test(self):
        from PIL import Image

        paths = sorted((self.root / "gtFine" / "test").glob("*/*_gtFine_labelIds.png"))
        for p in paths:
            site = p.parent.name
            stem = p.name.replace("_gtFine_labelIds.png", "")
            img = np.asarray(
                Image.open(self.root / "leftImg8bit" / "test" / site / f"{stem}_leftImg8bit.png").convert("RGB"),
                dtype=np.uint8,
            )
            yield f"{site}/{stem}", img, _to_label(Image.open(p))

    def num_test(self) -> int:
        return len(list((self.root / "gtFine" / "test").glob("*/*_gtFine_labelIds.png")))


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

    def iter_test(self):
        test = self.ds["test"]
        for i in range(len(test)):
            item = test[i]
            yield str(i), np.asarray(item["image"].convert("RGB"), dtype=np.uint8), _to_label(
                item["label"]
            )

    def num_test(self) -> int:
        return len(self.ds["test"])


def build_source(coralscapes_root: str | Path | None, hf_repo: str):
    if coralscapes_root is not None:
        logger.info("Using local CoralscapesV2 mirror at %s", coralscapes_root)
        return LocalMirrorSource(coralscapes_root)
    return HFSource(hf_repo)


# --------------------------------------------------------------------------- #
# Probe feature extraction + SVM.
# --------------------------------------------------------------------------- #
def _extract_probe_features(
    predictor: CBMPredictor, source, images: list[dict], kind: str
) -> tuple[NDArray[np.float32], NDArray[np.int64]]:
    feats: list[NDArray[np.float32]] = []
    labels: list[int] = []
    for entry in tqdm(images, desc=f"probe features ({kind})"):
        pts = entry["points"]
        if not pts:
            continue
        img, lab = source.train_image_label(entry)
        h, w = lab.shape[:2]
        img_tensor = predictor.preprocess(img.astype(np.uint8)).unsqueeze(0)
        feat = predictor.forward_features(img_tensor, kind=kind)[0]  # (D, hh, ww)
        rows = np.array([p["row"] for p in pts], dtype=np.int64)
        cols = np.array([p["col"] for p in pts], dtype=np.int64)
        sampled = predictor.sample_at_points(feat, rows, cols, h, w)  # (D, N)
        feats.append(sampled.transpose(0, 1).cpu().numpy().astype(np.float32))  # (N, D)
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
def _predict_dense(
    predictor: CBMPredictor,
    image_np: NDArray[np.uint8],
    kind: str,
    weff: torch.Tensor,
    beff: torch.Tensor,
    classes: torch.Tensor,
    height: int,
    width: int,
) -> NDArray[np.int64]:
    img_tensor = predictor.preprocess(image_np).unsqueeze(0)
    feat = predictor.forward_features(img_tensor, kind=kind)[0]  # (D, hh, ww)
    d, hh, ww = feat.shape
    flat = feat.permute(1, 2, 0).reshape(-1, d)  # (hh*ww, D)
    scores = flat @ weff.t() + beff  # (hh*ww, ncls)
    scores = scores.reshape(hh, ww, -1).permute(2, 0, 1)  # (ncls, hh, ww)
    up = predictor.upsample_scores(scores, height, width)  # (ncls, H, W)
    idx = up.argmax(dim=0)  # (H, W)
    pred = classes[idx]  # map column index -> class label
    return pred.cpu().numpy().astype(np.int64)


def evaluate_coralscapes_probe(
    *,
    predictor: CBMPredictor,
    probe_json: str | Path,
    feature_kinds: list[str],
    coralscapes_root: str | Path | None,
    output_dir: str | Path,
    hf_repo: str = "josauder/coralscapesV2",
    svm_c: float = 1.0,
    max_test_images: int | None = None,
) -> dict:
    import json

    out_dir = Path(output_dir) / "coralscapes"
    out_dir.mkdir(parents=True, exist_ok=True)

    with Path(probe_json).open() as f:
        probe = json.load(f)
    images = probe["images"]
    logger.info(
        "Loaded probe set: %d images, %d points, target %s pts/class",
        probe.get("num_images", len(images)),
        probe.get("total_points", sum(len(im["points"]) for im in images)),
        probe.get("points_per_class_target"),
    )

    source = build_source(coralscapes_root, hf_repo)
    id2name = {int(k): v for k, v in CORALSCAPES_V2_ID2NAME.items()}
    num_conf_classes = NUM_CLASSES + 1  # 0..95, 0 = ignore
    eval_class_ids = list(range(1, NUM_CLASSES + 1))

    results: dict[str, dict] = {"eval": "coralscapes", "feature_kinds": {}}

    def _dump() -> None:
        write_json(out_dir / "metrics.json", results)

    for kind in feature_kinds:
        logger.info("=== CoralscapesV2 linear probe: feature kind = %s ===", kind)
        t0 = time.time()
        X, y = _extract_probe_features(predictor, source, images, kind)
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

        conf = ClassConfusion(num_conf_classes, ignore_index=0)
        n_test = source.num_test() if max_test_images is None else min(source.num_test(), max_test_images)
        results["feature_kinds"][kind] = {
            "probe": {
                "num_points": int(X.shape[0]),
                "feature_dim": int(X.shape[1]),
                "num_classes_present": len(classes_present),
                "classes_present": classes_present,
                "train_accuracy": train_acc,
            },
            "test": {"status": "running", "num_images": 0},
        }
        _dump()

        processed = 0
        for _key, image_np, label_np in tqdm(
            source.iter_test(), total=n_test, desc=f"dense test ({kind})"
        ):
            if max_test_images is not None and processed >= max_test_images:
                break
            h, w = label_np.shape[:2]
            pred = _predict_dense(predictor, image_np, kind, weff, beff, classes, h, w)
            conf.update(label_np.reshape(-1), pred.reshape(-1))
            processed += 1
            if processed % 25 == 0:
                d = conf.to_dict(id2name=id2name, class_ids=eval_class_ids)
                results["feature_kinds"][kind]["test"] = {
                    "status": "running",
                    "num_images": processed,
                    "accuracy": d["accuracy"],
                    "miou": d["miou"],
                }
                _dump()

        d = conf.to_dict(id2name=id2name, class_ids=eval_class_ids)
        results["feature_kinds"][kind]["test"] = {
            "status": "done",
            "num_images": processed,
            "accuracy": d["accuracy"],
            "miou": d["miou"],
            "per_class": d["per_class"],
        }
        _dump()
        logger.info(
            "[coralscapes/%s] DONE in %.1fs: test acc=%s miou=%s over %d images",
            kind,
            time.time() - t0,
            fmt(d["accuracy"]),
            fmt(d["miou"]),
            processed,
        )

    return results
