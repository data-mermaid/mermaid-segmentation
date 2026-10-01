"""Point-annotation evaluation datasets and the shared per-unit runner.

Wraps a point-annotated source dataset (CoralNet / Pacific Labeled Corals) and
emits, per image, the full-resolution image (resized to the model input size)
together with the annotation ``(row, col)`` coordinates in *source* resolution,
their mapped MERMAID class ids, and their concept rows. The runner batches
images, runs the CBM, samples predictions exactly at the annotated pixels, and
accumulates per-unit + pooled + macro metrics, printing each unit as soon as its
last image has been consumed.
"""

from __future__ import annotations

import logging
import time
from collections import defaultdict
from collections.abc import Callable
from pathlib import Path
from typing import Any

import numpy as np
import torch
from numpy.typing import NDArray
from torch.utils.data import DataLoader, Dataset

from mermaidseg.datasets.local_cache import LocalS3Cache
from mermaidseg.evaluation.gt_mapping import ClassLookup, ConceptLookup
from mermaidseg.evaluation.metrics import ConceptLayout, MetricBundle, bleached_block_from_binary
from mermaidseg.evaluation.predictor import CBMPredictor, make_eval_transform
from mermaidseg.evaluation.reporting import fmt, write_csv, write_json

logger = logging.getLogger(__name__)


def _worker_init(_worker_id: int) -> None:
    """Re-configure the local S3 cache from env in DataLoader worker processes."""
    LocalS3Cache.configure_from_env()


class PointEvalDataset(Dataset):
    """Yields ``(image_tensor, points, class_ids, concept_rows, H, W, unit)`` per image."""

    def __init__(
        self,
        base_ds: Any,
        unit_col: str,
        key_cols: list[str],
        input_size: tuple[int, int],
        class_lookup: ClassLookup,
        concept_lookup: ConceptLookup,
    ):
        self.base = base_ds
        self.unit_col = unit_col
        self.key_cols = list(key_cols)
        self.class_lookup = class_lookup
        self.concept_lookup = concept_lookup
        self.input_hw = (int(input_size[0]), int(input_size[1]))
        self.transform = make_eval_transform(input_size)

        # Sort images by unit (then key) so units are contiguous -> timely prints.
        df_images = base_ds.df_images.copy()
        df_images = df_images.sort_values([unit_col, *key_cols]).reset_index(drop=True)
        self.df_images = df_images

        # Pre-group annotations by the per-image key for O(1) lookup in workers.
        ann = base_ds.df_annotations
        self._groups: dict[tuple, tuple[NDArray, NDArray, list[str]]] = {}
        grouped = ann.groupby(self.key_cols)
        for key, g in grouped:
            key_t = key if isinstance(key, tuple) else (key,)
            rows = g["row"].to_numpy()
            cols = g["col"].to_numpy()
            names = g["source_label_name"].astype(str).str.lower().tolist()
            self._groups[key_t] = (rows, cols, names)

    def __len__(self) -> int:
        return len(self.df_images)

    def unit_image_counts(self) -> dict[str, int]:
        counts = self.df_images[self.unit_col].astype(str).value_counts().to_dict()
        return {str(k): int(v) for k, v in counts.items()}

    def __getitem__(self, idx: int) -> dict[str, Any]:
        row = self.df_images.iloc[idx]
        unit = str(row[self.unit_col])
        key_t = tuple(row[c] for c in self.key_cols)
        try:
            image = self.base.read_image(**row.to_dict())
            image = np.asarray(image)
            if image.ndim == 2:
                image = np.stack([image] * 3, axis=-1)
            h, w = image.shape[:2]
            img_tensor = torch.from_numpy(
                self.transform(image=image)["image"]
            ).permute(2, 0, 1).contiguous().float()
        except Exception as e:  # noqa: BLE001
            logger.warning("Failed to load image (unit=%s key=%s): %s", unit, key_t, e)
            th, tw = self.input_hw
            return {
                "ok": False,
                "image": torch.zeros(3, th, tw, dtype=torch.float32),
                "rows": np.zeros(0, np.int64),
                "cols": np.zeros(0, np.int64),
                "class_ids": np.zeros(0, np.int64),
                "concept_rows": np.zeros((0, self.concept_lookup.num_channels), np.int8),
                "H": 1,
                "W": 1,
                "unit": unit,
            }

        rows_a, cols_a, names = self._groups.get(key_t, (np.zeros(0), np.zeros(0), []))
        rows_a = np.asarray(rows_a).astype(np.int64)
        cols_a = np.asarray(cols_a).astype(np.int64)
        in_bounds = (rows_a >= 0) & (rows_a < h) & (cols_a >= 0) & (cols_a < w)
        if not in_bounds.all():
            rows_a = rows_a[in_bounds]
            cols_a = cols_a[in_bounds]
            names = [n for n, keep in zip(names, in_bounds, strict=True) if keep]

        class_ids = np.array([self.class_lookup.class_id(n) for n in names], dtype=np.int64)
        if names:
            concept_rows = np.stack([self.concept_lookup.row(n) for n in names]).astype(np.int8)
        else:
            concept_rows = np.zeros((0, self.concept_lookup.num_channels), np.int8)

        return {
            "ok": True,
            "image": img_tensor,
            "rows": rows_a,
            "cols": cols_a,
            "class_ids": class_ids,
            "concept_rows": concept_rows,
            "H": int(h),
            "W": int(w),
            "unit": unit,
        }


def _collate(batch: list[dict[str, Any]]) -> dict[str, Any]:
    images = torch.stack([b["image"] for b in batch])
    return {
        "images": images,
        "rows": [b["rows"] for b in batch],
        "cols": [b["cols"] for b in batch],
        "class_ids": [b["class_ids"] for b in batch],
        "concept_rows": [b["concept_rows"] for b in batch],
        "H": [b["H"] for b in batch],
        "W": [b["W"] for b in batch],
        "unit": [b["unit"] for b in batch],
        "ok": [b["ok"] for b in batch],
    }


def _unit_row_for_csv(unit: str, d: dict) -> dict:
    row = {
        "unit": unit,
        "num_points": d["num_points"],
        "class_accuracy": d["class_accuracy"],
        "class_miou": d["class_miou"],
        "binary_macro_accuracy": d["binary"].get("macro_accuracy") if d["binary"] else None,
        "binary_macro_f1": d["binary"].get("macro_f1") if d["binary"] else None,
    }
    for rank, rd in d["taxonomic"].items():
        row[f"tax_{rank}_acc_all"] = rd["acc_all"]
        row[f"tax_{rank}_acc_living"] = rd["acc_living"]
    return row


def _log_unit(logger_: logging.Logger, unit: str, d: dict) -> None:
    tax = " ".join(
        f"{r}={fmt(v['acc_all'])}/{fmt(v['acc_living'])}" for r, v in d["taxonomic"].items()
    )
    binm = d["binary"] or {}
    logger_.info(
        "[unit %s] points=%d class_acc=%s | tax(all/living) %s | binary macro acc=%s f1=%s",
        unit,
        d["num_points"],
        fmt(d["class_accuracy"]),
        tax,
        fmt(binm.get("macro_accuracy")),
        fmt(binm.get("macro_f1")),
    )


def run_point_evaluation(
    *,
    eval_name: str,
    base_ds: Any,
    unit_col: str,
    key_cols: list[str],
    predictor: CBMPredictor,
    class_lookup: ClassLookup,
    concept_lookup: ConceptLookup,
    concept_names: list[str],
    id2label: dict[int, str],
    output_dir: str | Path,
    batch_size: int = 8,
    num_workers: int = 8,
    max_images_per_unit: int | None = None,
    progress_cb: Callable[[int, int], None] | None = None,
) -> dict:
    """Run a point-annotation evaluation with per-unit streaming reports."""
    if eval_name == "coralnet" and "bleached" not in concept_names:
        raise RuntimeError(
            "CoralNet eval requires a concept channel named exactly 'bleached'. "
            f"concept_names has {len(concept_names)} entries and no 'bleached'."
        )
    out_dir = Path(output_dir) / eval_name
    out_dir.mkdir(parents=True, exist_ok=True)

    dataset = PointEvalDataset(
        base_ds=base_ds,
        unit_col=unit_col,
        key_cols=key_cols,
        input_size=predictor.input_size,
        class_lookup=class_lookup,
        concept_lookup=concept_lookup,
    )

    if max_images_per_unit is not None:
        limited = (
            dataset.df_images.groupby(unit_col, sort=False)
            .head(max_images_per_unit)
            .reset_index(drop=True)
        )
        dataset.df_images = limited
        logger.info("[%s] DEBUG: limited to <=%d images per unit", eval_name, max_images_per_unit)

    num_classes = max(id2label.keys()) + 1
    layout = ConceptLayout.from_concept_names(concept_names)

    # Compute unmapped-label reports in the main process (worker copies of the
    # lookups would not propagate their runtime `.unmapped` sets back here).
    all_names = sorted(
        base_ds.df_annotations["source_label_name"].astype(str).str.lower().unique().tolist()
    )
    unmapped_class = sorted(
        n for n in all_names if class_lookup.name_to_class_id.get(n, 0) == 0
    )
    unmapped_concept = sorted(n for n in all_names if n not in concept_lookup.name_to_row)
    if unmapped_class:
        logger.info(
            "[%s] %d/%d source labels have no MERMAID class (ignored for class acc): %s",
            eval_name,
            len(unmapped_class),
            len(all_names),
            ", ".join(unmapped_class[:20]) + (" ..." if len(unmapped_class) > 20 else ""),
        )
    if unmapped_concept:
        logger.info(
            "[%s] %d/%d source labels missing from concept CSV (ignored for concepts): %s",
            eval_name,
            len(unmapped_concept),
            len(all_names),
            ", ".join(unmapped_concept[:20]) + (" ..." if len(unmapped_concept) > 20 else ""),
        )

    remaining = dataset.unit_image_counts()
    total_images = len(dataset)
    logger.info(
        "[%s] %d images across %d units; num_classes=%d, num_concepts=%d",
        eval_name,
        total_images,
        len(remaining),
        num_classes,
        len(concept_names),
    )

    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        collate_fn=_collate,
        pin_memory=(predictor.device.type == "cuda"),
        worker_init_fn=_worker_init if num_workers > 0 else None,
        persistent_workers=num_workers > 0,
    )

    pending: dict[str, MetricBundle] = {}
    pooled = MetricBundle.create(num_classes, layout)
    unit_results: dict[str, dict] = {}
    macro: dict[str, list[float]] = defaultdict(list)
    csv_rows: list[dict] = []
    n_failed = 0
    processed = 0
    t0 = time.time()

    def finalize_unit(unit: str) -> None:
        bundle = pending.pop(unit)
        d = bundle.to_dict(class_id2name=id2label, include_class_detail=True)
        unit_results[unit] = d
        _log_unit(logger, unit, d)
        pooled.merge(bundle)
        # collect macro scalars
        macro["class_accuracy"].append(d["class_accuracy"])
        macro["class_miou"].append(d["class_miou"])
        if d["binary"]:
            macro["binary_macro_accuracy"].append(d["binary"]["macro_accuracy"])
            macro["binary_macro_f1"].append(d["binary"]["macro_f1"])
        for rank, rd in d["taxonomic"].items():
            macro[f"tax_{rank}_acc_all"].append(rd["acc_all"])
            macro[f"tax_{rank}_acc_living"].append(rd["acc_living"])
        csv_rows.append(_unit_row_for_csv(unit, d))
        _flush(final=False)

    def _macro_dict() -> dict:
        return {
            k: (float(np.nanmean(v)) if len(v) else float("nan")) for k, v in macro.items()
        }

    def _pooled_dict() -> dict:
        pooled_d = pooled.to_dict(class_id2name=id2label, include_class_detail=True)
        if eval_name == "coralnet":
            pooled_d["bleached"] = bleached_block_from_binary(pooled_d.get("binary") or {})
        return pooled_d

    def _flush(final: bool) -> None:
        pooled_d = _pooled_dict()
        payload = {
            "eval": eval_name,
            "num_units_done": len(unit_results),
            "num_units_total": len(remaining) if not max_images_per_unit else len(unit_results) + len(pending),
            "images_processed": processed,
            "images_failed": n_failed,
            "pooled": pooled_d,
            "macro": _macro_dict(),
            "units": unit_results,
            "unmapped_class_labels": unmapped_class,
            "unmapped_concept_labels": unmapped_concept,
        }
        write_json(out_dir / "metrics.json", payload)
        write_csv(out_dir / "per_unit.csv", csv_rows)

    for batch in loader:
        class_probs, concept_probs = predictor.forward_probs(batch["images"])
        bsz = class_probs.shape[0]
        for b in range(bsz):
            unit = batch["unit"][b]
            processed += 1
            bundle = pending.get(unit)
            if bundle is None:
                bundle = MetricBundle.create(num_classes, layout)
                pending[unit] = bundle

            if not batch["ok"][b]:
                n_failed += 1
            else:
                rows_a = batch["rows"][b]
                cols_a = batch["cols"][b]
                if rows_a.shape[0] > 0:
                    cs = predictor.sample_at_points(
                        class_probs[b], rows_a, cols_a, batch["H"][b], batch["W"][b]
                    )  # (C, N)
                    ks = predictor.sample_at_points(
                        concept_probs[b], rows_a, cols_a, batch["H"][b], batch["W"][b]
                    )  # (K, N)
                    pred_class = cs.argmax(dim=0).cpu().numpy()
                    pred_probs = ks.transpose(0, 1).cpu().numpy()  # (N, K)
                    bundle.update(
                        batch["class_ids"][b], pred_class, batch["concept_rows"][b], pred_probs
                    )

            remaining[unit] = remaining.get(unit, 0) - 1
            if remaining.get(unit, 0) <= 0:
                finalize_unit(unit)

        if progress_cb is not None:
            progress_cb(processed, total_images)

    # Finalize any stragglers (e.g. all-failed units or count mismatches).
    for unit in list(pending.keys()):
        finalize_unit(unit)

    _flush(final=True)
    dt = time.time() - t0
    pooled_d = _pooled_dict()
    logger.info(
        "[%s] DONE in %.1fs: %d images (%d failed), %d units. POOLED class_acc=%s miou=%s "
        "binary macro acc=%s f1=%s",
        eval_name,
        dt,
        processed,
        n_failed,
        len(unit_results),
        fmt(pooled_d["class_accuracy"]),
        fmt(pooled_d["class_miou"]),
        fmt((pooled_d["binary"] or {}).get("macro_accuracy")),
        fmt((pooled_d["binary"] or {}).get("macro_f1")),
    )
    for rank, rd in pooled_d["taxonomic"].items():
        logger.info(
            "[%s] POOLED tax %s: acc_all=%s (n=%d) acc_living=%s (n=%d)",
            eval_name,
            rank,
            fmt(rd["acc_all"]),
            rd["n_all"],
            fmt(rd["acc_living"]),
            rd["n_living"],
        )
    if eval_name == "coralnet":
        bleached = pooled_d["bleached"]
        logger.info(
            "[coralnet/bleached] acc=%s precision=%s recall=%s f1=%s "
            "(n_true=%s n_false=%s n_not_given=%s)",
            fmt(bleached["accuracy"]),
            fmt(bleached["precision"]),
            fmt(bleached["recall"]),
            fmt(bleached["f1"]),
            bleached["n_true"],
            bleached["n_false"],
            bleached["n_not_given"],
        )

    return {
        "eval": eval_name,
        "pooled": pooled_d,
        "macro": _macro_dict(),
        "units": unit_results,
        "images_processed": processed,
        "images_failed": n_failed,
        "unmapped_class_labels": unmapped_class,
        "unmapped_concept_labels": unmapped_concept,
    }
