"""Benthos zero-shot benthic segmentation evaluation (per orthomosaic + pooled).

Concepts are mapped to a small set of final segmentation classes via an editable
spec (``configs/eval/benthos_zero_shot.yaml``). The model runs at its input size
on each stored tile; the per-class score maps are bilinearly upsampled back to
the tile's source resolution and argmaxed. Nodata (transparent border, RGB == 0)
and unlabeled / out-of-spec GT pixels are ignored. Accuracy, mIoU and per-class
IoU are reported per site (RS24, CR_DoubleWreck) and pooled ("benthos").
"""

from __future__ import annotations

import logging
import time
from pathlib import Path

import numpy as np
import torch
import yaml
from tqdm import tqdm

from mermaidseg.datasets.benthos_yuval.benthos_yuval_corals_dataset import (
    BenthosYuvalCoralsDataset,
)
from mermaidseg.evaluation.metrics import ClassConfusion
from mermaidseg.evaluation.predictor import CBMPredictor
from mermaidseg.evaluation.reporting import fmt, write_json
from mermaidseg.model.concept_expr import ConceptResolver, evaluate as eval_expr, tokenize

logger = logging.getLogger(__name__)

SOURCE_NAME = "benthos_yuval"


def load_spec(spec_path: str | Path) -> list[dict]:
    with Path(spec_path).open() as f:
        spec = yaml.safe_load(f)
    classes = spec["classes"] if isinstance(spec, dict) else spec
    if not classes:
        raise ValueError(f"Empty benthos spec at {spec_path}")
    return classes


def _validate_spec(classes: list[dict], resolver: ConceptResolver) -> None:
    missing: set[str] = set()
    for entry in classes:
        tokens = tokenize(entry["expr"])
        for i, tok in enumerate(tokens):
            if tok.kind.name != "IDENT":
                continue
            # Function names (``max(...)``) are not concept channels.
            if i + 1 < len(tokens) and tokens[i + 1].value == "(":
                continue
            try:
                resolver.resolve(tok.value)
            except Exception:  # noqa: BLE001
                missing.add(tok.value)
    if missing:
        raise ValueError(
            "Benthos spec references concept channels not present in this model: "
            f"{sorted(missing)}. Update the spec or use a checkpoint with these concepts."
        )


def _build_gt_name_to_spec(classes: list[dict]) -> dict[str, int]:
    gt_to_spec: dict[str, int] = {}
    for spec_id, entry in enumerate(classes):
        for gt_name in entry.get("gt", []):
            key = str(gt_name).lower()
            if key in gt_to_spec:
                raise ValueError(f"GT name {gt_name!r} listed under multiple spec classes")
            gt_to_spec[key] = spec_id
    return gt_to_spec


def evaluate_benthos(
    *,
    predictor: CBMPredictor,
    whitelist_sites: list[str] | None,
    concept_names: list[str],
    spec_path: str | Path,
    output_dir: str | Path,
    max_tiles_per_site: int | None = None,
) -> dict:
    out_dir = Path(output_dir) / "benthos"
    out_dir.mkdir(parents=True, exist_ok=True)

    classes = load_spec(spec_path)
    class_names = [str(e["name"]) for e in classes]
    num_spec = len(classes)
    resolver = ConceptResolver(concept_names)
    _validate_spec(classes, resolver)
    gt_to_spec = _build_gt_name_to_spec(classes)
    spec_id2name = {i: n for i, n in enumerate(class_names)}

    logger.info("Building Benthos dataset (whitelist_sites=%s) ...", whitelist_sites)
    base = BenthosYuvalCoralsDataset(whitelist_sites=whitelist_sites, transform=None)
    classes_global = base._classes_global  # name -> classes.json id
    gid2name = {int(v): str(k).lower() for k, v in classes_global.items()}
    logger.info(
        "Benthos: %d tiles across sites %s; classes.json=%s",
        len(base.df_images),
        sorted(base.df_images["site"].unique().tolist()),
        classes_global,
    )

    sites = sorted(base.df_images["site"].unique().tolist())
    site_conf = {s: ClassConfusion(num_spec, ignore_index=None) for s in sites}
    site_stats = {
        s: {"tiles": 0, "nodata_px": 0, "unlabeled_or_oos_px": 0, "valid_px": 0} for s in sites
    }

    df_images = base.df_images.sort_values(["site", "image_id"]).reset_index(drop=True)
    if max_tiles_per_site is not None:
        df_images = (
            df_images.groupby("site", sort=False).head(max_tiles_per_site).reset_index(drop=True)
        )
        logger.info("[benthos] DEBUG: limited to <=%d tiles/site", max_tiles_per_site)

    def _dump(final: bool) -> None:
        payload = {
            "eval": "benthos",
            "spec": [{"name": e["name"], "expr": e["expr"], "gt": e.get("gt", [])} for e in classes],
            "per_site": {},
            "benthos": {},
        }
        pooled = ClassConfusion(num_spec, ignore_index=None)
        for s in sites:
            payload["per_site"][s] = {
                **site_conf[s].to_dict(id2name=spec_id2name),
                **site_stats[s],
            }
            pooled.merge(site_conf[s])
        stat_tot = {k: sum(site_stats[s][k] for s in sites) for k in site_stats[sites[0]]}
        payload["benthos"] = {**pooled.to_dict(id2name=spec_id2name), **stat_tot}
        write_json(out_dir / "metrics.json", payload)

    t0 = time.time()
    for _, row in tqdm(df_images.iterrows(), total=len(df_images), desc="Benthos tiles"):
        site = str(row["site"])
        image_id = str(row["image_id"])
        try:
            rgb = np.asarray(base.read_image(image_id=image_id, site=site))
            mask = np.asarray(base.read_label(image_id=image_id, site=site))
        except Exception as e:  # noqa: BLE001
            logger.warning("[benthos] failed tile site=%s id=%s: %s", site, image_id, e)
            continue
        h, w = mask.shape[:2]
        nodata = (rgb == 0).all(axis=-1) if rgb.ndim == 3 else (rgb == 0)

        # GT spec ids (-1 = ignore).
        gt_spec = np.full((h, w), -1, dtype=np.int64)
        present_ids = np.unique(mask)
        for gid in present_ids.tolist():
            name = gid2name.get(int(gid))
            if name is None or name == "unlabeled":
                continue
            spec_id = gt_to_spec.get(name)
            if spec_id is None:
                continue
            gt_spec[mask == gid] = spec_id

        # Predict concept probs at input res, evaluate spec, upsample, argmax.
        img_tensor = predictor.preprocess(rgb.astype(np.uint8)).unsqueeze(0)
        concept_probs = predictor.forward_features(img_tensor, kind="concepts")[0]  # (K, hh, ww)
        cp_np = concept_probs.cpu().numpy()
        score_maps = np.stack(
            [eval_expr(e["expr"], cp_np, resolver) for e in classes], axis=0
        )  # (num_spec, hh, ww)
        scores_up = predictor.upsample_scores(
            torch.from_numpy(score_maps).to(predictor.device), h, w
        )
        pred = scores_up.argmax(dim=0).cpu().numpy().astype(np.int64)  # (h, w)

        valid = (gt_spec >= 0) & (~nodata)
        site_conf[site].update(gt_spec[valid], pred[valid])
        st = site_stats[site]
        st["tiles"] += 1
        st["nodata_px"] += int(nodata.sum())
        st["unlabeled_or_oos_px"] += int(((gt_spec < 0) & (~nodata)).sum())
        st["valid_px"] += int(valid.sum())

        if st["tiles"] % 10 == 0:
            _dump(final=False)

    _dump(final=True)

    pooled = ClassConfusion(num_spec, ignore_index=None)
    for s in sites:
        d = site_conf[s].to_dict(id2name=spec_id2name)
        logger.info(
            "[benthos] site %s: acc=%s miou=%s (valid_px=%d nodata_px=%d oos_px=%d)",
            s,
            fmt(d["accuracy"]),
            fmt(d["miou"]),
            site_stats[s]["valid_px"],
            site_stats[s]["nodata_px"],
            site_stats[s]["unlabeled_or_oos_px"],
        )
        pooled.merge(site_conf[s])
    pooled_d = pooled.to_dict(id2name=spec_id2name)
    logger.info(
        "[benthos] DONE in %.1fs POOLED acc=%s miou=%s",
        time.time() - t0,
        fmt(pooled_d["accuracy"]),
        fmt(pooled_d["miou"]),
    )
    for name, cd in pooled_d["per_class"].items():
        logger.info("[benthos] pooled IoU %-18s = %s", name, fmt(cd["iou"]))

    return {
        "eval": "benthos",
        "per_site": {s: site_conf[s].to_dict(id2name=spec_id2name) for s in sites},
        "benthos": pooled_d,
        "site_stats": site_stats,
    }
