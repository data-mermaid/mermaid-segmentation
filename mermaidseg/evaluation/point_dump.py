"""Per-point prediction dumps for point-annotated evals, and grouped metrics on them.

``evaluate.py`` reduces predictions to count accumulators as it goes, so any breakdown it
does not compute (per label, per concept per source, by image size, ...) needs a new GPU
pass. This module instead writes *one row per annotated point* once and computes the same
metrics afterwards, grouped by any column(s), with pandas.

Dump layout (one directory)::

    points.parquet          one row per scored point (see ``point_records``)
    images.parquet          one row per image: load ok?, H, W, points kept / out of bounds
    concept_probs.npy       (N, K) concept probabilities at the points (float16), row = point_id
    class_probs.npy         (N, C) class softmax at the points (float16), row = point_id
    label_concept_rows.npy  (L, K) int8 GT concept row of each label in meta["labels"]
    meta.json               concept names, id2label, rank layout, run arguments

``points.parquet`` columns (``<r>`` is each taxonomic rank, ``<c>`` each binary concept):

- identity: ``point_id``, the unit column (``source_id``), key columns (``image_id``),
  ``image_key``, ``ann_idx`` (index into the dataset's ``df_annotations``), ``row``, ``col``,
  ``H``, ``W``, ``edge_dist`` (pixels to the nearest image border / min(H, W))
- label: ``source_label_name``, ``label_idx`` (row of ``label_concept_rows.npy``),
  ``concept_mapped`` (label is in the concept table), ``gt_finest_rank``
- MERMAID class head (named ``mermaid`` because ``class`` is a taxonomic rank):
  ``gt_mermaid_id`` (0 = no MERMAID class, not scored), ``gt_mermaid``,
  ``pred_mermaid_id``, ``pred_mermaid``, ``pred_mermaid_prob``, ``gt_mermaid_prob``
- taxonomic concepts: ``gt_<r>`` (taxon name, ``None`` if not given at this rank),
  ``pred_<r>`` (argmax over the rank's channels), ``pred_<r>_prob``, ``gt_<r>_prob``
- binary concepts: ``gt__<c>`` (0 not given, 1 False, 2 True) and ``p__<c>`` (probability)

Predictions (argmax / threshold) are taken from fp32 probabilities before the float16 cast,
so the metrics below reproduce ``metrics.json`` of ``evaluate.py`` exactly.
"""

from __future__ import annotations

import json
import logging
import time
from collections.abc import Sequence
from dataclasses import dataclass, field
from functools import cached_property
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from numpy.typing import NDArray
from torch.utils.data import DataLoader

from mermaidseg.dataset_reconciliation.concepts import TAXONOMIC_CONCEPTS
from mermaidseg.evaluation.gt_mapping import ClassLookup, ConceptLookup
from mermaidseg.evaluation.metrics import ConceptLayout
from mermaidseg.evaluation.point_datasets import PointEvalDataset, _collate, _worker_init
from mermaidseg.evaluation.predictor import CBMPredictor

logger = logging.getLogger(__name__)

DUMP_FORMAT_VERSION = 1
BINARY_THRESHOLD = 0.5


# --------------------------------------------------------------------------------------
# Building per-point records (pure numpy; unit-tested without a model)
# --------------------------------------------------------------------------------------


def point_records(
    class_probs: NDArray[np.floating],
    concept_probs: NDArray[np.floating],
    gt_class: NDArray[np.integer],
    concept_rows: NDArray[np.integer],
    layout: ConceptLayout,
    id2label: dict[int, str],
) -> dict[str, NDArray]:
    """Per-point prediction/GT columns for ``N`` points.

    ``class_probs`` is ``(N, C)``, ``concept_probs`` ``(N, K)`` (fp32), ``gt_mermaid`` ``(N,)``
    MERMAID class ids, ``concept_rows`` ``(N, K)`` GT codes (0 not given, 1 False, 2 True).
    """
    n = class_probs.shape[0]
    ar = np.arange(n)
    gt_class = np.asarray(gt_class).astype(np.int64)
    pred_class = class_probs.argmax(axis=1)
    out: dict[str, NDArray] = {
        "gt_mermaid_id": gt_class,
        "gt_mermaid": np.array([id2label.get(int(c)) if c != 0 else None for c in gt_class], dtype=object),
        "pred_mermaid_id": pred_class.astype(np.int64),
        "pred_mermaid": np.array([id2label.get(int(c)) for c in pred_class], dtype=object),
        "pred_mermaid_prob": class_probs[ar, pred_class].astype(np.float32),
        "gt_mermaid_prob": np.where(
            gt_class != 0, class_probs[ar, np.clip(gt_class, 0, class_probs.shape[1] - 1)], np.nan
        ).astype(np.float32),
    }
    finest = np.full(n, None, dtype=object)
    for rank, idx, _none_local in layout.taxo:
        taxa = np.array([layout.concept_names[i].split("__", 1)[1] for i in idx], dtype=object)
        p = concept_probs[:, idx]
        g = np.asarray(concept_rows)[:, idx] == 2
        given = g.any(axis=1)
        gt_local = g.argmax(axis=1)
        pred_local = p.argmax(axis=1)
        out[f"gt_{rank}"] = np.where(given, taxa[gt_local], None)
        out[f"pred_{rank}"] = taxa[pred_local]
        out[f"pred_{rank}_prob"] = p[ar, pred_local].astype(np.float32)
        out[f"gt_{rank}_prob"] = np.where(given, p[ar, gt_local], np.nan).astype(np.float32)
        # Ranks are ordered coarse -> fine; a real (non-"none") taxon refines gt_finest_rank.
        finest = np.where(given & (taxa[gt_local] != "none"), rank, finest)
    out["gt_finest_rank"] = finest
    for name, i in zip(layout.binary_names, layout.binary_indices, strict=True):
        out[f"gt__{name}"] = np.asarray(concept_rows)[:, i].astype(np.int8)
        out[f"p__{name}"] = concept_probs[:, i].astype(np.float32)
    return out


# --------------------------------------------------------------------------------------
# Running the model and writing a dump
# --------------------------------------------------------------------------------------


def dump_point_predictions(
    *,
    predictor: CBMPredictor,
    base_ds: Any,
    unit_col: str,
    key_cols: list[str],
    class_lookup: ClassLookup,
    concept_lookup: ConceptLookup,
    concept_names: list[str],
    id2label: dict[int, str],
    output_dir: str | Path,
    batch_size: int = 8,
    num_workers: int = 8,
    max_images_per_unit: int | None = None,
    save_concept_probs: bool = True,
    save_class_probs: bool = True,
    extra_meta: dict | None = None,
) -> Path:
    """Run ``predictor`` over every image of ``base_ds`` and write a per-point dump."""
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    layout = ConceptLayout.from_concept_names(concept_names)

    dataset = PointEvalDataset(
        base_ds=base_ds,
        unit_col=unit_col,
        key_cols=key_cols,
        input_size=predictor.input_size,
        class_lookup=class_lookup,
        concept_lookup=concept_lookup,
    )
    if max_images_per_unit is not None:
        dataset.df_images = (
            dataset.df_images.groupby(unit_col, sort=False).head(max_images_per_unit).reset_index(drop=True)
        )
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        collate_fn=_collate,
        pin_memory=(predictor.device.type == "cuda"),
        worker_init_fn=_worker_init if num_workers > 0 else None,
    )

    ann = base_ds.df_annotations
    labels = sorted(ann["source_label_name"].astype(str).str.lower().unique().tolist())
    label_to_idx = {n: i for i, n in enumerate(labels)}
    label_rows = np.stack([concept_lookup.row(n) for n in labels]).astype(np.int8)

    image_cols = [c for c in dict.fromkeys([unit_col, *key_cols, "image_key"]) if c in dataset.df_images]
    parts: list[pd.DataFrame] = []
    concept_parts: list[NDArray] = []
    class_parts: list[NDArray] = []
    image_rows: list[dict] = []
    n_points = 0
    img_idx = 0
    t0 = time.time()
    total = len(dataset)

    for batch in loader:
        class_probs, concept_probs = predictor.forward_probs(batch["images"])
        for b in range(class_probs.shape[0]):
            meta = dataset.df_images.iloc[img_idx]
            img_idx += 1
            n = int(batch["rows"][b].shape[0]) if batch["ok"][b] else 0
            image_rows.append(
                {
                    **{c: meta[c] for c in image_cols},
                    "ok": bool(batch["ok"][b]),
                    "H": int(batch["H"][b]) if batch["ok"][b] else None,
                    "W": int(batch["W"][b]) if batch["ok"][b] else None,
                    "n_points": n,
                    "n_out_of_bounds": int(batch["n_out_of_bounds"][b]),
                }
            )
            if n == 0:
                continue
            rows_a, cols_a, h, w = batch["rows"][b], batch["cols"][b], batch["H"][b], batch["W"][b]
            cs = predictor.sample_at_points(class_probs[b], rows_a, cols_a, h, w).T.cpu().numpy()
            ks = predictor.sample_at_points(concept_probs[b], rows_a, cols_a, h, w).T.cpu().numpy()
            rec = point_records(cs, ks, batch["class_ids"][b], batch["concept_rows"][b], layout, id2label)
            names = batch["names"][b]
            edge = np.minimum.reduce([rows_a, cols_a, h - 1 - rows_a, w - 1 - cols_a]) / min(h, w)
            part = pd.DataFrame(
                {
                    **{c: [meta[c]] * n for c in image_cols},
                    "ann_idx": batch["ann_idx"][b],
                    "row": rows_a,
                    "col": cols_a,
                    "H": h,
                    "W": w,
                    "edge_dist": edge.astype(np.float32),
                    "source_label_name": names,
                    "label_idx": np.array([label_to_idx[x] for x in names], dtype=np.int32),
                    "concept_mapped": np.array([x in concept_lookup.name_to_row for x in names]),
                    **rec,
                }
            )
            parts.append(part)
            if save_concept_probs:
                concept_parts.append(ks.astype(np.float16))
            if save_class_probs:
                class_parts.append(cs.astype(np.float16))
            n_points += n
        if len(image_rows) % (50 * batch_size) < batch_size:
            dt = time.time() - t0
            logger.info("[dump] %d/%d images, %d points, %.1f img/s", img_idx, total, n_points, img_idx / max(dt, 1e-6))

    df = pd.concat(parts, ignore_index=True).copy()
    df.insert(0, "point_id", np.arange(len(df), dtype=np.int64))
    for c in ["source_label_name", "gt_mermaid", "pred_mermaid", "gt_finest_rank"] + [
        f"{p}_{rank}" for rank, _, _ in layout.taxo for p in ("gt", "pred")
    ]:
        df[c] = df[c].astype("category")
    df.to_parquet(out_dir / "points.parquet", index=False)
    pd.DataFrame(image_rows).to_parquet(out_dir / "images.parquet", index=False)
    np.save(out_dir / "label_concept_rows.npy", label_rows)
    if save_concept_probs:
        np.save(out_dir / "concept_probs.npy", np.concatenate(concept_parts))
    if save_class_probs:
        np.save(out_dir / "class_probs.npy", np.concatenate(class_parts))

    all_names = set(labels)
    meta_out = {
        "format_version": DUMP_FORMAT_VERSION,
        "unit_col": unit_col,
        "key_cols": key_cols,
        "num_points": int(len(df)),
        "num_images": len(image_rows),
        "num_images_failed": int(sum(not r["ok"] for r in image_rows)),
        "concept_names": list(concept_names),
        "id2label": {str(k): v for k, v in id2label.items()},
        "ranks": [rank for rank, _, _ in layout.taxo],
        "binary_concepts": layout.binary_names,
        "labels": labels,
        "unmapped_class_labels": sorted(n for n in all_names if class_lookup.name_to_class_id.get(n, 0) == 0),
        "unmapped_concept_labels": sorted(n for n in all_names if n not in concept_lookup.name_to_row),
        "binary_threshold": BINARY_THRESHOLD,
        "seconds": time.time() - t0,
        **(extra_meta or {}),
    }
    (out_dir / "meta.json").write_text(json.dumps(meta_out, indent=2, default=str))
    logger.info("[dump] wrote %d points from %d images to %s", len(df), len(image_rows), out_dir)
    return out_dir


# --------------------------------------------------------------------------------------
# Loading
# --------------------------------------------------------------------------------------


@dataclass
class PointDump:
    """A loaded dump. ``df`` is ``points.parquet``; the arrays are memory-mapped lazily."""

    path: Path
    df: pd.DataFrame
    images: pd.DataFrame
    meta: dict = field(repr=False)

    @classmethod
    def load(cls, path: str | Path) -> "PointDump":
        path = Path(path)
        return cls(
            path=path,
            df=pd.read_parquet(path / "points.parquet"),
            images=pd.read_parquet(path / "images.parquet"),
            meta=json.loads((path / "meta.json").read_text()),
        )

    @property
    def concept_names(self) -> list[str]:
        return self.meta["concept_names"]

    @property
    def ranks(self) -> list[str]:
        return self.meta["ranks"]

    @property
    def binary_concepts(self) -> list[str]:
        return self.meta["binary_concepts"]

    @cached_property
    def concept_probs(self) -> np.ndarray:
        """``(N, K)`` float16, row ``i`` = ``point_id`` ``i``. Memory-mapped."""
        return np.load(self.path / "concept_probs.npy", mmap_mode="r")

    @cached_property
    def class_probs(self) -> np.ndarray:
        """``(N, C)`` float16, row ``i`` = ``point_id`` ``i``. Memory-mapped."""
        return np.load(self.path / "class_probs.npy", mmap_mode="r")

    @cached_property
    def label_concept_rows(self) -> np.ndarray:
        return np.load(self.path / "label_concept_rows.npy")

    def concept_gt(self, df: pd.DataFrame | None = None) -> np.ndarray:
        """``(n, K)`` int8 GT concept codes for the rows of ``df`` (default: all points)."""
        df = self.df if df is None else df
        return self.label_concept_rows[df["label_idx"].to_numpy()]

    def rank_probs(self, rank: str, df: pd.DataFrame | None = None) -> pd.DataFrame:
        """Probabilities over one rank's taxa for the rows of ``df`` (columns = taxon names)."""
        df = self.df if df is None else df
        idx = [i for i, n in enumerate(self.concept_names) if n.startswith(f"{rank}__")]
        probs = np.asarray(self.concept_probs[df["point_id"].to_numpy()][:, idx], dtype=np.float32)
        return pd.DataFrame(probs, index=df.index, columns=[self.concept_names[i].split("__", 1)[1] for i in idx])


# --------------------------------------------------------------------------------------
# Grouped metrics (definitions identical to mermaidseg.evaluation.metrics)
# --------------------------------------------------------------------------------------

_ALL = "__all__"


def _by(df: pd.DataFrame, by: str | Sequence[str] | None) -> tuple[pd.DataFrame, list[str]]:
    if by is None:
        return df.assign(**{_ALL: "all"}), [_ALL]
    keys = [by] if isinstance(by, str) else list(by)
    return df, keys


def _finish(out: pd.DataFrame, keys: list[str]) -> pd.DataFrame:
    out = out.reset_index()
    return out.drop(columns=[_ALL]) if keys == [_ALL] else out


def _div(a, b):
    a = np.asarray(a, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)
    with np.errstate(invalid="ignore", divide="ignore"):
        return np.where(b > 0, a / b, np.nan)


def _infer_ranks(df: pd.DataFrame) -> list[str]:
    return [r for r in TAXONOMIC_CONCEPTS if f"gt_{r}" in df.columns and f"pred_{r}" in df.columns]


def _infer_binary(df: pd.DataFrame) -> list[str]:
    return [c[4:] for c in df.columns if c.startswith("gt__")]


def class_metrics(df: pd.DataFrame, by: str | Sequence[str] | None = None) -> pd.DataFrame:
    """Per (group, MERMAID class): ``n_gt``, ``n_pred``, ``tp``, recall, precision, IoU.

    Only points with a MERMAID class (``gt_mermaid_id != 0``) are scored, as in ``ClassConfusion``.
    """
    df, keys = _by(df[df["gt_mermaid_id"] != 0], by)
    gt = df.groupby([*keys, "gt_mermaid"], observed=True).size().rename("n_gt")
    pred = df[df["pred_mermaid_id"] != 0].groupby([*keys, "pred_mermaid"], observed=True).size().rename("n_pred")
    pred.index = pred.index.set_names([*keys, "gt_mermaid"])
    tp = (
        df[df["gt_mermaid_id"] == df["pred_mermaid_id"]].groupby([*keys, "gt_mermaid"], observed=True).size().rename("tp")
    )
    out = pd.concat([gt, pred, tp], axis=1).fillna(0).astype(np.int64)
    out.index = out.index.set_names([*keys, "mermaid_class"])
    out["recall"] = _div(out["tp"], out["n_gt"])
    out["precision"] = _div(out["tp"], out["n_pred"])
    out["iou"] = _div(out["tp"], out["n_gt"] + out["n_pred"] - out["tp"])
    return _finish(out, keys)


def taxon_metrics(df: pd.DataFrame, rank: str, by: str | Sequence[str] | None = None) -> pd.DataFrame:
    """Per (group, taxon at ``rank``): ``n_gt``, ``n_pred``, ``tp``, recall, precision, F1.

    Only points whose GT is given at ``rank`` are scored (as in ``TaxonomicRankAccuracy``);
    ``n_pred`` counts predictions among those points.
    """
    g, p = f"gt_{rank}", f"pred_{rank}"
    df, keys = _by(df[df[g].notna()], by)
    gt = df.groupby([*keys, g], observed=True).size().rename("n_gt")
    pred = df.groupby([*keys, p], observed=True).size().rename("n_pred")
    pred.index = pred.index.set_names([*keys, g])
    tp = df[df[g].astype(str) == df[p].astype(str)].groupby([*keys, g], observed=True).size().rename("tp")
    out = pd.concat([gt, pred, tp], axis=1).fillna(0).astype(np.int64)
    out.index = out.index.set_names([*keys, "taxon"])
    out["recall"] = _div(out["tp"], out["n_gt"])
    out["precision"] = _div(out["tp"], out["n_pred"])
    out["f1"] = _div(2 * out["tp"], out["n_gt"] + out["n_pred"])
    return _finish(out, keys)


def binary_metrics(
    df: pd.DataFrame,
    by: str | Sequence[str] | None = None,
    concepts: Sequence[str] | None = None,
    threshold: float = BINARY_THRESHOLD,
) -> pd.DataFrame:
    """Per (group, binary concept): tp/fp/fn/tn, n_not_given, accuracy, precision, recall, F1."""
    concepts = list(concepts) if concepts is not None else _infer_binary(df)
    df, keys = _by(df, by)
    frames = []
    for c in concepts:
        g = df[f"gt__{c}"].to_numpy()
        pos = df[f"p__{c}"].to_numpy() > threshold
        t = df[keys].assign(
            tp=(g == 2) & pos, fp=(g == 1) & pos, fn=(g == 2) & ~pos, tn=(g == 1) & ~pos, n_not_given=g == 0
        )
        frames.append(t.groupby(keys, observed=True).sum().astype(np.int64).assign(concept=c))
    out = pd.concat(frames).set_index("concept", append=True)
    out["n_true"] = out["tp"] + out["fn"]
    out["n_false"] = out["fp"] + out["tn"]
    out["n_valid"] = out["n_true"] + out["n_false"]
    out["accuracy"] = _div(out["tp"] + out["tn"], out["n_valid"])
    out["precision"] = _div(out["tp"], out["tp"] + out["fp"])
    out["recall"] = _div(out["tp"], out["n_true"])
    out["f1"] = _div(2 * out["tp"], 2 * out["tp"] + out["fp"] + out["fn"])
    return _finish(out, keys)


def summary_by(df: pd.DataFrame, by: str | Sequence[str] | None = None) -> pd.DataFrame:
    """One row per group with the headline metrics of ``metrics.json``.

    Columns: ``num_points``, ``class_accuracy``, ``class_miou``, ``class_points_scored``,
    ``<rank>_acc_all`` / ``<rank>_acc_living`` / ``<rank>_n_all`` / ``<rank>_n_living`` per rank,
    and ``binary_macro_accuracy`` / ``binary_macro_f1``. With ``by="source_id"`` this equals
    ``per_unit.csv``; with ``by=None`` the pooled block.
    """
    d, keys = _by(df, by)
    grp = d.groupby(keys, observed=True)
    out = pd.DataFrame({"num_points": grp.size()})
    scored = d["gt_mermaid_id"] != 0
    out["class_points_scored"] = scored.groupby([d[k] for k in keys], observed=True).sum()
    correct = scored & (d["gt_mermaid_id"] == d["pred_mermaid_id"])
    out["class_accuracy"] = _div(correct.groupby([d[k] for k in keys], observed=True).sum(), out["class_points_scored"])
    cm = class_metrics(df, by)
    ck = keys if keys != [_ALL] else None
    miou = cm.groupby(ck, observed=True)["iou"].mean() if ck else pd.Series([cm["iou"].mean()], index=out.index)
    out["class_miou"] = miou.reindex(out.index)
    for rank in _infer_ranks(df):
        g = d[f"gt_{rank}"]
        given = g.notna()
        living = given & (g.astype(str) != "none")
        ok = given & (g.astype(str) == d[f"pred_{rank}"].astype(str))
        gb = [d[k] for k in keys]
        out[f"{rank}_n_all"] = given.groupby(gb, observed=True).sum()
        out[f"{rank}_n_living"] = living.groupby(gb, observed=True).sum()
        out[f"{rank}_acc_all"] = _div(ok.groupby(gb, observed=True).sum(), out[f"{rank}_n_all"])
        out[f"{rank}_acc_living"] = _div((ok & living).groupby(gb, observed=True).sum(), out[f"{rank}_n_living"])
    if _infer_binary(df):
        bm = binary_metrics(df, by)
        bk = keys if keys != [_ALL] else None
        acc = bm[bm["n_valid"] > 0]
        f1 = bm[(2 * bm["tp"] + bm["fp"] + bm["fn"]) > 0]
        if bk:
            out["binary_macro_accuracy"] = acc.groupby(bk, observed=True)["accuracy"].mean().reindex(out.index)
            out["binary_macro_f1"] = f1.groupby(bk, observed=True)["f1"].mean().reindex(out.index)
        else:
            out["binary_macro_accuracy"] = acc["accuracy"].mean()
            out["binary_macro_f1"] = f1["f1"].mean()
    return _finish(out, keys)


def confusions(
    df: pd.DataFrame, target: str = "mermaid", by: str | Sequence[str] | None = None, errors_only: bool = True
) -> pd.DataFrame:
    """Counts of ``(gt, pred)`` pairs for ``target`` (``"mermaid"`` for the class head, or a rank), most frequent first.

    ``share_of_gt`` is the fraction of that GT value's points (within the group) predicted as
    ``pred``.
    """
    g, p = f"gt_{target}", f"pred_{target}"
    d = df[df["gt_mermaid_id"] != 0] if target == "mermaid" else df[df[g].notna()]
    d, keys = _by(d, by)
    pairs = d.groupby([*keys, g, p], observed=True).size().rename("n").reset_index()
    pairs["share_of_gt"] = pairs["n"] / pairs.groupby([*keys, g], observed=True)["n"].transform("sum")
    if errors_only:
        pairs = pairs[pairs[g].astype(str) != pairs[p].astype(str)]
    pairs = pairs.rename(columns={g: "gt", p: "pred"}).sort_values("n", ascending=False).reset_index(drop=True)
    return pairs.drop(columns=[_ALL]) if keys == [_ALL] else pairs
