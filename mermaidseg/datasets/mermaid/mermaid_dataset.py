"""MERMAID PyTorch dataset.

Reads MERMAID confirmed annotations from a parquet file on S3 and emits
``(image, source_labels)`` tuples where ``source_labels`` is in the MERMAID
benthic attribute label space — the canonical target space for this project.
"""

from __future__ import annotations

import functools
from typing import Any, Literal

import boto3
import numpy as np
import pandas as pd
from numpy.typing import NDArray

from mermaidseg.datasets.base_dataset import BaseCoralDataset
from mermaidseg.datasets.utils import get_image_s3_candidates, s3_training_config

RARE_IMAGE_THRESHOLD = 10
DEFAULT_HOLDOUT_FRACTION = 0.1
DEFAULT_HOLDOUT_SEED = 42
DEFAULT_HOLDOUT_ROLE: Literal["train", "val"] = "train"
_HOLDOUT_MISSING = "__missing__"
_STRATIFY_COLS = ("benthic_attribute_name", "benthic_attribute_id", "growth_form_name")
_INVALID_GROWTH_FORMS = frozenset({"", "none", "nan", "<na>", "null"})


@functools.lru_cache(maxsize=8)
def _read_annotations_cached(annotations_path: str) -> pd.DataFrame:
    """Read the MERMAID annotations parquet once per path and share it.

    Train- and val-role ``MermaidDataset`` instances in the same run both need to split
    the *identical* annotation set to avoid drifting apart; caching the read (rather
    than the pure split function) means both roles see the same object without either
    mutating it in place.
    """
    return pd.read_parquet(annotations_path)


def mermaid_region_key_series(df_annotations: pd.DataFrame) -> pd.Series:
    """Return a string region key per annotation row (name with id fallback)."""
    if "region_name" in df_annotations.columns:
        names = df_annotations["region_name"].astype(str)
        if "region_id" in df_annotations.columns:
            missing = names.isna() | (names.str.lower().isin(["nan", "none", ""]))
            ids = df_annotations["region_id"].astype(str)
            names = names.where(~missing, ids)
        return names
    if "region_id" in df_annotations.columns:
        return df_annotations["region_id"].astype(str)
    raise ValueError("MERMAID holdout requires region_name or region_id columns.")


def _prepare_holdout_df(df_annotations: pd.DataFrame) -> pd.DataFrame:
    df = df_annotations.copy()
    if "benthic_attribute_name" not in df.columns and "source_label_name" in df.columns:
        df = df.rename(columns={"source_label_name": "benthic_attribute_name"})
    df["image_id"] = df["image_id"].astype(str)
    df["region_name"] = mermaid_region_key_series(df).astype(str)
    df["benthic_attribute_name"] = df["benthic_attribute_name"].astype(str)
    if "growth_form_name" in df.columns:
        df["growth_form_name"] = df["growth_form_name"].astype(str)
    return df


def _is_valid_growth_form(value: object) -> bool:
    if pd.isna(value):
        return False
    return str(value).strip().lower() not in _INVALID_GROWTH_FORMS


def _image_label_sets(df_annotations: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    growth_col = "growth_form_name" if "growth_form_name" in df_annotations.columns else None
    for image_id, group in df_annotations.groupby("image_id", sort=False):
        growth: set[str] = set()
        if growth_col is not None:
            growth = {str(value) for value in group[growth_col] if _is_valid_growth_form(value)}
        rows.append(
            {
                "image_id": str(image_id),
                "region_name": str(group["region_name"].iloc[0]),
                "benthic_attribute_names": frozenset(group["benthic_attribute_name"].astype(str)),
                "growth_form_names": frozenset(growth),
            }
        )
    # Canonical row order regardless of input row order — see _fill_regional_quota,
    # which iterates images/regions in an order derived from this DataFrame while
    # consuming a shared RNG, so its output would otherwise depend on annotation
    # row order rather than only on the seed.
    return pd.DataFrame(rows).sort_values("image_id").reset_index(drop=True)


def _pair_image_index(
    image_sets: pd.DataFrame,
    labels_col: Literal["benthic_attribute_names", "growth_form_names"],
) -> dict[tuple[str, str], list[str]]:
    index: dict[tuple[str, str], list[str]] = {}
    for row in image_sets.itertuples(index=False):
        region = row.region_name
        for label in getattr(row, labels_col):
            key = (region, label)
            index.setdefault(key, []).append(row.image_id)
    return {key: sorted(set(ids)) for key, ids in index.items()}


def _pick_coverage_image(
    candidate_ids: list[str],
    val_ids: set[str],
    val_load: dict[str, int],
    rng: np.random.Generator,
) -> str:
    pool = [image_id for image_id in candidate_ids if image_id not in val_ids] or list(
        candidate_ids
    )
    min_load = min(val_load.get(image_id, 0) for image_id in pool)
    tied = sorted(image_id for image_id in pool if val_load.get(image_id, 0) == min_load)
    return str(rng.choice(tied))


def _rare_pair_keys(
    index: dict[tuple[str, str], list[str]],
) -> list[tuple[int, tuple[str, str], list[str]]]:
    pairs = [
        (len(ids), key, ids) for key, ids in index.items() if 2 <= len(ids) < RARE_IMAGE_THRESHOLD
    ]
    pairs.sort(key=lambda item: (item[0], item[1][0], item[1][1]))
    return pairs


def _pair_covered(
    val_ids: set[str],
    image_lookup: dict[str, Any],
    region: str,
    label: str,
    labels_col: Literal["benthic_attribute_names", "growth_form_names"],
) -> bool:
    for image_id in val_ids:
        row = image_lookup[image_id]
        if row.region_name == region and label in getattr(row, labels_col):
            return True
    return False


def _assign_rare_coverage(
    index: dict[tuple[str, str], list[str]],
    labels_col: Literal["benthic_attribute_names", "growth_form_names"],
    image_lookup: dict[str, Any],
    val_ids: set[str],
    val_load: dict[str, int],
    rng: np.random.Generator,
) -> None:
    for _, key, candidate_ids in _rare_pair_keys(index):
        region, label = key
        if _pair_covered(val_ids, image_lookup, region, label, labels_col):
            continue
        chosen = _pick_coverage_image(candidate_ids, val_ids, val_load, rng)
        val_ids.add(chosen)
        val_load[chosen] = val_load.get(chosen, 0) + 1


def _dominant_benthic_name(group: pd.DataFrame) -> str:
    return str(group["benthic_attribute_name"].value_counts().index[0])


def _fill_regional_quota(
    df_annotations: pd.DataFrame,
    image_sets: pd.DataFrame,
    val_ids: set[str],
    holdout_fraction: float,
    rng: np.random.Generator,
) -> None:
    dominant = (
        df_annotations.groupby("image_id", sort=True)
        .apply(_dominant_benthic_name, include_groups=False)
        .rename("dominant_benthic_attribute_name")
        .reset_index()
    )
    image_sets = image_sets.merge(dominant, on="image_id", how="left")

    # sort=True is load-bearing: each region below consumes draws from one shared
    # `rng`, so visiting regions in a canonical (not input-row-order) sequence is
    # what makes the resulting split depend only on `holdout_seed`.
    for _region, region_df in image_sets.groupby("region_name", sort=True):
        region_ids = sorted(region_df["image_id"].astype(str).unique())
        n = len(region_ids)
        if n <= 1:
            continue
        target = max(1, int(round(n * holdout_fraction)))
        target = min(target, n - 1)
        need = target - sum(1 for image_id in region_ids if image_id in val_ids)
        if need <= 0:
            continue

        pool = region_df.loc[~region_df["image_id"].isin(val_ids)].copy()
        if pool.empty:
            continue

        chosen: list[str] = []
        by_class = pool.groupby("dominant_benthic_attribute_name", sort=True)
        class_ids = {
            str(label): sorted(map(str, ids))
            for label, ids in by_class["image_id"].apply(list).items()
        }
        remaining = need
        active = {label: ids for label, ids in class_ids.items() if ids}
        while remaining > 0 and active:
            for label in sorted(active):
                if remaining <= 0:
                    break
                ids = active[label]
                if not ids:
                    continue
                pick = str(rng.choice(ids))
                ids.remove(pick)
                chosen.append(pick)
                remaining -= 1
                if not ids:
                    del active[label]
            active = {label: ids for label, ids in active.items() if ids}

        val_ids.update(chosen)


def select_stratified_holdout_image_ids(
    df_annotations: pd.DataFrame,
    *,
    holdout_fraction: float,
    holdout_seed: int,
) -> set[str]:
    """Rare-aware region-blocked holdout using full per-image label sets."""
    if not 0.0 < holdout_fraction < 1.0:
        raise ValueError(f"holdout_fraction must be in (0, 1); got {holdout_fraction}")

    df = _prepare_holdout_df(df_annotations)
    if df.empty:
        return set()

    image_sets = _image_label_sets(df)
    if image_sets.empty:
        return set()

    image_lookup = {row.image_id: row for row in image_sets.itertuples(index=False)}
    benthic_index = _pair_image_index(image_sets, "benthic_attribute_names")
    growth_index = _pair_image_index(image_sets, "growth_form_names")

    rng = np.random.default_rng(holdout_seed)
    val_ids: set[str] = set()
    val_load: dict[str, int] = {}

    _assign_rare_coverage(
        benthic_index, "benthic_attribute_names", image_lookup, val_ids, val_load, rng
    )
    _assign_rare_coverage(growth_index, "growth_form_names", image_lookup, val_ids, val_load, rng)
    _fill_regional_quota(df, image_sets, val_ids, holdout_fraction, rng)
    return val_ids


def _mode_or_missing(series: pd.Series) -> str:
    values = series.dropna()
    if values.empty:
        return _HOLDOUT_MISSING
    mode = values.astype(str).mode()
    return str(mode.iloc[0]) if not mode.empty else _HOLDOUT_MISSING


def build_image_holdout_features(df_annotations: pd.DataFrame) -> pd.DataFrame:
    """One row per image with region and mode label features (legacy composite
    strata)."""
    df = _prepare_holdout_df(df_annotations)
    for col in _STRATIFY_COLS:
        if col not in df.columns:
            df[col] = _HOLDOUT_MISSING

    rows: list[dict[str, str]] = []
    for image_id, group in df.groupby("image_id", sort=False):
        rows.append(
            {
                "image_id": str(image_id),
                "region_name": str(group["region_name"].iloc[0]),
                "benthic_attribute_name": _mode_or_missing(group["benthic_attribute_name"]),
                "benthic_attribute_id": _mode_or_missing(group["benthic_attribute_id"]),
                "growth_form_name": _mode_or_missing(group["growth_form_name"]),
            }
        )
    return pd.DataFrame(rows)


def select_composite_holdout_image_ids(
    df_annotations: pd.DataFrame,
    *,
    holdout_fraction: float,
    holdout_seed: int,
) -> set[str]:
    """Legacy mode-based composite-stratum holdout (for before/after comparisons)."""
    if not 0.0 < holdout_fraction < 1.0:
        raise ValueError(f"holdout_fraction must be in (0, 1); got {holdout_fraction}")
    features = build_image_holdout_features(df_annotations)
    if features.empty:
        return set()

    rng = np.random.default_rng(holdout_seed)
    holdout_ids: set[str] = set()
    group_cols = ["region_name", *_STRATIFY_COLS]
    for _, stratum in features.groupby(group_cols, sort=False):
        ids = sorted(stratum["image_id"].astype(str).tolist())
        n = len(ids)
        if n <= 1:
            continue
        n_holdout = max(1, int(round(n * holdout_fraction)))
        n_holdout = min(n_holdout, n - 1)
        chosen = rng.choice(ids, size=n_holdout, replace=False)
        holdout_ids.update(str(x) for x in chosen)
    return holdout_ids


def compute_split_quality(
    df_annotations: pd.DataFrame,
    val_ids: set[str],
    *,
    holdout_fraction: float,
    rare_image_threshold: int = RARE_IMAGE_THRESHOLD,
) -> dict[str, Any]:
    """Summarize rare-class coverage and regional val balance for a holdout split."""
    df = _prepare_holdout_df(df_annotations)
    image_sets = _image_label_sets(df)
    image_lookup = {row.image_id: row for row in image_sets.itertuples(index=False)}

    benthic_index = _pair_image_index(image_sets, "benthic_attribute_names")
    growth_index = _pair_image_index(image_sets, "growth_form_names")

    def rare_coverage(index: dict[tuple[str, str], list[str]], labels_col: str) -> float:
        rare_keys = [key for key, ids in index.items() if 2 <= len(ids) < rare_image_threshold]
        if not rare_keys:
            return 1.0
        covered = sum(
            1
            for key in rare_keys
            if _pair_covered(val_ids, image_lookup, key[0], key[1], labels_col)
        )
        return covered / len(rare_keys)

    region_stats: list[dict[str, Any]] = []
    for region, region_df in image_sets.groupby("region_name", sort=False):
        region_ids = sorted(region_df["image_id"].astype(str).unique())
        n = len(region_ids)
        val_n = sum(1 for image_id in region_ids if image_id in val_ids)
        region_stats.append(
            {
                "region_name": region,
                "images": n,
                "val_images": val_n,
                "val_fraction": val_n / n if n else 0.0,
                "target_fraction": holdout_fraction,
            }
        )

    all_images = set(image_sets["image_id"].astype(str))
    train_ids = all_images - val_ids

    def split_label_sets(split_ids: set[str], labels_col: str) -> set[str]:
        labels: set[str] = set()
        for image_id in split_ids:
            labels.update(getattr(image_lookup[image_id], labels_col))
        return labels

    val_benthic = split_label_sets(val_ids, "benthic_attribute_names")
    train_benthic = split_label_sets(train_ids, "benthic_attribute_names")

    missing_rare_rows: list[dict[str, str]] = []
    for key, ids in benthic_index.items():
        if not (2 <= len(ids) < rare_image_threshold):
            continue
        if not _pair_covered(val_ids, image_lookup, key[0], key[1], "benthic_attribute_names"):
            missing_rare_rows.append(
                {
                    "region_name": key[0],
                    "benthic_attribute_name": key[1],
                    "images": str(len(ids)),
                }
            )

    return {
        "rare_class_val_coverage": rare_coverage(benthic_index, "benthic_attribute_names"),
        "rare_growth_form_val_coverage": rare_coverage(growth_index, "growth_form_names"),
        "val_share_by_region": pd.DataFrame(region_stats),
        "classes_val_only": sorted(val_benthic - train_benthic),
        "classes_train_only": sorted(train_benthic - val_benthic),
        "missing_rare_class_pairs": pd.DataFrame(missing_rare_rows),
    }


class MermaidDataset(BaseCoralDataset):
    """A PyTorch Dataset for loading MERMAID annotated coral reef images from a Parquet
    file stored on S3.

    Each item returned is a tuple ``(image, source_labels)`` where
    ``source_labels`` is an integer mask in the MERMAID benthic-attribute label
    space (or in the joint global space, if the dataset has been registered
    with a :class:`SourceLabelRegistry`).

    Image object keys are resolved across ``.png``, ``.jpg``, and ``.jpeg``.
    Pillow detects the encoded format from the object bytes, so the loader also
    supports the current upstream objects whose ``.png`` keys contain JPEG data.

    Split controls:
        - ``holdout_fraction`` + ``holdout_role``: rare-aware region-blocked holdout
          (default 10% val, seed 42, train role). Pass ``holdout_fraction=None`` and
          ``holdout_role=None`` to load the full corpus without splitting.
    """

    SOURCE_NAME = "mermaid"

    annotations_path: str
    source_bucket: str
    s3: boto3.client
    holdout_fraction: float | None
    holdout_seed: int
    holdout_role: Literal["train", "val"] | None

    def __init__(
        self,
        annotations_path: str = "s3://coral-reef-training/mermaid/mermaid_confirmed_annotations.parquet",
        source_bucket: str = "coral-reef-training",
        holdout_fraction: float | None = DEFAULT_HOLDOUT_FRACTION,
        holdout_seed: int = DEFAULT_HOLDOUT_SEED,
        holdout_role: Literal["train", "val"] | None = DEFAULT_HOLDOUT_ROLE,
        **base_kwargs: Any,
    ):
        if holdout_fraction is not None and holdout_role not in ("train", "val"):
            raise ValueError("holdout_role must be 'train' or 'val' when holdout_fraction is set.")
        if holdout_role is not None and holdout_fraction is None:
            raise ValueError("holdout_fraction is required when holdout_role is set.")

        self.annotations_path = annotations_path
        self.source_bucket = source_bucket
        self.s3 = boto3.client("s3", config=s3_training_config())
        self.holdout_fraction = holdout_fraction
        self.holdout_seed = holdout_seed
        self.holdout_role = holdout_role

        df_annotations, df_images = self.load_annotations(self.annotations_path)
        super().__init__(
            df_annotations=df_annotations,
            df_images=df_images,
            split=holdout_role,
            **base_kwargs,
        )

    def load_annotations(self, annotations_path: str) -> tuple[pd.DataFrame, pd.DataFrame]:
        """Load annotations from a Parquet file on S3 and apply split filters."""
        df_annotations = _read_annotations_cached(annotations_path)
        df_annotations = self._apply_image_holdout(df_annotations)
        df_annotations = df_annotations.rename(
            columns={"benthic_attribute_name": "source_label_name"}
        )
        if df_annotations.empty:
            raise ValueError(
                "MermaidDataset has zero annotations after applying split filters "
                f"(holdout_fraction={self.holdout_fraction}, holdout_role={self.holdout_role})."
            )
        df_images = self._derive_df_images_from_annotations(df_annotations)
        return df_annotations, df_images

    def _apply_image_holdout(self, df_annotations: pd.DataFrame) -> pd.DataFrame:
        if self.holdout_fraction is None or self.holdout_role is None:
            return df_annotations
        holdout_ids = select_stratified_holdout_image_ids(
            df_annotations,
            holdout_fraction=self.holdout_fraction,
            holdout_seed=self.holdout_seed,
        )
        image_ids = df_annotations["image_id"].astype(str)
        if self.holdout_role == "val":
            mask = image_ids.isin(holdout_ids)
        else:
            mask = ~image_ids.isin(holdout_ids)
        return df_annotations.loc[mask].reset_index(drop=True)

    def _derive_df_images_from_annotations(self, df_annotations: pd.DataFrame) -> pd.DataFrame:
        cols = [c for c in ("image_id", "region_id", "region_name") if c in df_annotations.columns]
        return df_annotations[cols].drop_duplicates(subset=["image_id"]).reset_index(drop=True)

    def read_image(self, image_id: str, **row_kwargs: Any) -> NDArray[Any]:
        keys = [f"mermaid/{image_id}{extension}" for extension in (".png", ".jpg", ".jpeg")]
        image = get_image_s3_candidates(
            s3=self.s3,
            bucket=self.source_bucket,
            keys=keys,
        )
        return np.array(image.convert("RGB"))
