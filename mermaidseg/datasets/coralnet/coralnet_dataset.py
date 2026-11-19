"""CoralNet PyTorch dataset.

Reads CoralNet point annotations from a self-contained training manifest
Parquet file on S3 and emits ``(image, source_labels)`` tuples where
``source_labels`` is in the **CoralNet provider label space** — i.e. the
integer IDs used by CoralNet's own label catalogue. Mapping into the MERMAID
benthic attribute target space is performed externally via
:mod:`mermaidseg.dataset_reconciliation.label_mapping`.

The training manifest (``coralnet_training_manifest_<run>.parquet``) carries
per-annotation rows with ``(row, col, coralnet_id)`` in the **resized**
coordinate space (longest edge clamped to 2048; resizing
lives in ``coralnet_dataset.resize``) plus the resolved
``image_s3_key`` for each image (resized copy when the image was resized,
otherwise the original). The manifest comes from the data config's
``coralnet.<split>.manifest_path``.

The ``coralnet-dataset`` package (``packages/coralnet-dataset``) produces both
the manifest and the per-run label-name file ``coralnet_id2name_<run>.json``.
The JSON sits beside the manifest, in a run directory named ``<run>``, so the
configured ``manifest_path`` needs its sibling id2name JSON in that same directory.
Images with no ``image_s3_key`` are skipped (never loaded at the wrong
resolution); the dataset recovers by serving a different image.

Duplicate images (the same photo uploaded to several sources) are removed with
``image_blacklist`` and their merged annotations applied with
``annotation_overrides``; both are versioned in ``configs/coralnet/``.
"""

from __future__ import annotations

import hashlib
import io
import logging
import posixpath
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from numpy.typing import NDArray

from mermaidseg.datasets.base_dataset import BaseCoralDataset
from mermaidseg.datasets.local_cache import LocalS3Cache
from mermaidseg.datasets.utils import get_image_s3

logger = logging.getLogger(__name__)

_REPO_ROOT = Path(__file__).resolve().parents[3]


def _id2name_key(manifest_path: str) -> str:
    """Return the S3 key of the id2name JSON that sits beside a manifest.

    The key is ``<dir>/coralnet_id2name_<basename(dir)>.json``.

    Raises:
        ValueError: If ``manifest_path`` has no directory component.
    """
    run_dir = posixpath.dirname(manifest_path)
    if not run_dir:
        raise ValueError(
            f"manifest_path {manifest_path!r} must sit in a run directory "
            "(<run>/<manifest>.parquet) so its id2name JSON can be located."
        )
    return posixpath.join(run_dir, f"coralnet_id2name_{posixpath.basename(run_dir)}.json")


def _as_int_ids(values: pd.Series, column: str) -> pd.Series:
    """Return ``values`` as integers (int64 unless already an integer dtype), raising on bad values.

    A float column (one null is enough to force float64) would stringify as ``"82.0"`` and
    miss the id2name lookup, turning the label into ``"none"``.
    """
    if pd.api.types.is_integer_dtype(values) and not values.isna().any():
        return values  # already integral; keeps the manifest's compact dtype
    numeric = pd.to_numeric(values, errors="coerce")
    bad = numeric.isna() | (numeric != numeric.round())
    if bad.any():
        examples = values[bad].unique()[:5].tolist()
        raise ValueError(
            f"CoralNet {column} must be non-null integers; {int(bad.sum())} invalid rows "
            f"(e.g. {examples})."
        )
    return numeric.astype(np.int64)


def _as_int_id_set(ids: list[int | str], name: str) -> set[int]:
    """Coerce configured source ids (YAML may quote them) to ints."""
    try:
        return {int(i) for i in ids}
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} must contain integer source ids, got {ids!r}") from exc


def _warn_unnamed_ids(coralnet_ids: pd.Series, names: pd.Series) -> None:
    """Warn about ids that the id2name JSON could not name."""
    unnamed = coralnet_ids[names.isna()]
    if unnamed.empty:
        return
    logger.warning(
        "%d annotations have a coralnet_id with no name in the id2name JSON "
        "(missing or null; e.g. %s); they get a placeholder label that maps to no target.",
        len(unnamed),
        sorted(unnamed.unique().tolist())[:5],
    )


class CoralNetDataset(BaseCoralDataset):
    """Load CoralNet annotated coral reef images from a Parquet file on S3.

    Each item returned is a tuple ``(image, source_labels)`` where
    ``source_labels`` is an integer mask in CoralNet's own provider-ID label
    space (or in the joint global space, if the dataset has been registered
    with a :class:`SourceLabelRegistry`).

    Attributes:
        manifest_path (str): Key (relative to ``source_bucket``) of the self-contained
            training manifest parquet.
        manifest_sha256 (str): SHA-256 hex digest of the manifest bytes parsed by
            ``load_annotations``.
        id2name_key (str): Key of the ``coralnet_id2name_<run>.json`` beside the manifest.
        source_bucket (str): S3 bucket name containing the dataset files.
        source_s3_prefix (str): S3 prefix under which the per-source CoralNet image folders live.

    Args:
        manifest_path (str): Key (relative to ``source_bucket``) of the training manifest
            parquet; the data config's ``coralnet.<split>.manifest_path``. Its id2name JSON must
            sit beside it.
        source_bucket (str, optional): S3 bucket name containing the dataset files.
        source_s3_prefix (str, optional): S3 prefix containing per-source CoralNet image folders.
        whitelist_sources / blacklist_sources: Optional CoralNet source-id allowlist/denylist.
        image_blacklist (str | list, optional): CoralNet image ids to drop, or a parquet reference
            with an ``image_id`` column. Applied after the source filters. A reference is a path
            relative to the repo root if that file exists (e.g.
            ``configs/coralnet/image_blacklist.parquet``), otherwise an absolute local path, an S3
            URI or a key relative to ``source_bucket``.
        annotation_overrides (str, optional): Parquet reference (as for ``image_blacklist``) with
            columns ``image_id``, ``row``, ``col``, ``coralnet_id``. Every image listed there has
            all its manifest points replaced by the listed ones.
        **base_kwargs: Forwarded to :class:`BaseCoralDataset`.
    """

    SOURCE_NAME = "coralnet"

    manifest_path: str
    manifest_sha256: str
    id2name_key: str
    source_bucket: str
    source_s3_prefix: str
    whitelist_sources: list[int | str] | None
    blacklist_sources: list[int | str] | None
    image_blacklist: str | list[int | str] | None
    annotation_overrides: str | None

    def __init__(
        self,
        manifest_path: str,
        source_bucket: str = "dev-datamermaid-sm-sources",
        source_s3_prefix: str = "coralnet-public-images",
        whitelist_sources: list[int | str] | None = None,
        blacklist_sources: list[int | str] | None = None,
        image_blacklist: str | list[int | str] | None = None,
        annotation_overrides: str | None = None,
        **base_kwargs: Any,
    ) -> None:
        """Initialize the dataset; arguments are described in the class docstring."""
        self.manifest_path = manifest_path
        self.id2name_key = _id2name_key(self.manifest_path)
        self.source_bucket = source_bucket
        self.source_s3_prefix = source_s3_prefix
        self.whitelist_sources = whitelist_sources
        self.blacklist_sources = blacklist_sources
        self.image_blacklist = image_blacklist
        self.annotation_overrides = annotation_overrides
        if self.whitelist_sources is not None and self.blacklist_sources is not None:
            raise ValueError("Cannot specify both whitelist and blacklist sources.")

        self.split = base_kwargs.get("split")
        df_annotations, df_images = self.load_annotations()
        if self.annotation_overrides is not None:
            df_annotations = self._apply_annotation_overrides(df_annotations)

        present = set(df_annotations["source_id"].unique())
        if self.whitelist_sources is not None:
            wanted = _as_int_id_set(self.whitelist_sources, "whitelist_sources")
            if wanted - present:
                raise ValueError(
                    f"whitelist_sources {sorted(wanted - present)} are not in the manifest "
                    f"{self.manifest_path!r}."
                )
            df_annotations = self._drop_annotations(
                df_annotations, df_annotations["source_id"].isin(wanted), "whitelist_sources"
            )
        if self.blacklist_sources is not None:
            unwanted = _as_int_id_set(self.blacklist_sources, "blacklist_sources")
            if unwanted - present:
                logger.warning(
                    "blacklist_sources %s are not in the manifest %r.",
                    sorted(unwanted - present),
                    self.manifest_path,
                )
            df_annotations = self._drop_annotations(
                df_annotations, ~df_annotations["source_id"].isin(unwanted), "blacklist_sources"
            )
        if self.image_blacklist is not None:
            df_annotations = self._drop_annotations(
                df_annotations,
                ~df_annotations["image_id"].astype(str).isin(self._load_image_blacklist()),
                "image_blacklist",
            )
        if (
            self.whitelist_sources is not None
            or self.blacklist_sources is not None
            or self.image_blacklist is not None
            or self.annotation_overrides is not None
        ):
            df_annotations = df_annotations.reset_index(drop=True)
            df_images = self._derive_df_images_from_annotations(df_annotations)

        super().__init__(df_annotations=df_annotations, df_images=df_images, **base_kwargs)

    def _log_dropped(self, name: str, dropped: int, total: int) -> None:
        logger.info(
            "CoralNetDataset(split=%r): %s dropped %d of %d annotations",
            self.split,
            name,
            dropped,
            total,
        )

    def _drop_annotations(
        self, df_annotations: pd.DataFrame, keep: pd.Series, name: str
    ) -> pd.DataFrame:
        """Keep the rows where ``keep`` is true and log how many the filter ``name`` dropped."""
        self._log_dropped(name, int((~keep).sum()), len(df_annotations))
        return df_annotations[keep]

    def load_annotations(self) -> tuple[pd.DataFrame, pd.DataFrame]:
        """Load CoralNet annotations from the training manifest parquet on S3.

        The manifest carries per-annotation rows plus the resolved ``image_s3_key``
        for each image. ``source_label_name`` is derived from ``coralnet_id`` via
        the ``coralnet_id2name_<run>.json`` beside the manifest (the manifest's ``source_label_name`` column
        holds numeric IDs and is overwritten). No CoralNet -> MERMAID translation
        happens here; that mapping is owned by
        :mod:`mermaidseg.dataset_reconciliation.label_mapping` and is applied
        at training time on the GPU via a long-tensor lookup.
        """
        data = LocalS3Cache.get().read_bytes(self.source_bucket, self.manifest_path)
        self.manifest_sha256 = hashlib.sha256(data).hexdigest()
        df = pd.read_parquet(io.BytesIO(data))
        df["source_id"] = _as_int_ids(df["source_id"], "source_id")
        df["coralnet_id"] = _as_int_ids(df["coralnet_id"], "coralnet_id")
        coralnet_id2name = LocalS3Cache.get().read_json(self.source_bucket, self.id2name_key)

        df["coralnet_name"] = df["coralnet_id"].map(lambda x: coralnet_id2name.get(str(x)))
        _warn_unnamed_ids(df["coralnet_id"], df["coralnet_name"])
        df["source_label_name"] = df["coralnet_name"].astype(str).str.lower()
        df = df.rename(columns={"image_s3_key": "image_key"})

        df_annotations = df[
            [
                "source_id",
                "image_id",
                "row",
                "col",
                "coralnet_id",
                "coralnet_name",
                "source_label_name",
                "image_key",
            ]
        ]

        df_images = self._derive_df_images_from_annotations(df_annotations)
        return df_annotations, df_images

    def _read_parquet_ref(self, ref: str) -> pd.DataFrame:
        """Read a dedup parquet from a repo path if it exists, else a storage ref."""
        repo_path = _REPO_ROOT / ref
        if not Path(ref).is_absolute() and repo_path.is_file():
            return pd.read_parquet(repo_path)
        return LocalS3Cache.get().read_parquet_ref(ref, self.source_bucket)

    def _load_image_blacklist(self) -> set[str]:
        if isinstance(self.image_blacklist, str):
            df = self._read_parquet_ref(self.image_blacklist)
            return set(df["image_id"].astype(str))
        assert self.image_blacklist is not None  # callers check before loading
        return {str(x) for x in self.image_blacklist}

    def _apply_annotation_overrides(self, df_annotations: pd.DataFrame) -> pd.DataFrame:
        """Replace each image's points with those in ``annotation_overrides``.

        ``image_key`` and ``source_id`` are taken from the image's manifest rows, so overrides
        for images missing from the manifest are dropped.
        """
        assert self.annotation_overrides is not None  # callers check before applying
        overrides = self._read_parquet_ref(self.annotation_overrides)[
            ["image_id", "row", "col", "coralnet_id"]
        ]
        overrides["image_id"] = overrides["image_id"].astype(str)
        overrides["coralnet_id"] = _as_int_ids(overrides["coralnet_id"], "coralnet_id")
        image_ids = df_annotations["image_id"].astype(str)
        images = df_annotations.assign(image_id=image_ids)[
            ["source_id", "image_id", "image_key"]
        ].drop_duplicates("image_id")
        n_missing = len(set(overrides["image_id"]) - set(images["image_id"]))
        if n_missing:
            logger.warning("annotation_overrides: %d images are not in the manifest", n_missing)
        overrides = overrides.merge(images, on="image_id")

        coralnet_id2name = LocalS3Cache.get().read_json(self.source_bucket, self.id2name_key)
        overrides["coralnet_name"] = overrides["coralnet_id"].map(
            lambda x: coralnet_id2name.get(str(x))
        )
        overrides["source_label_name"] = overrides["coralnet_name"].astype(str).str.lower()
        key_cols = ("source_id", "image_id", "row", "col", "coralnet_id")
        overrides = overrides.astype({c: df_annotations[c].dtype for c in key_cols})
        kept = df_annotations[~image_ids.isin(set(overrides["image_id"].astype(str)))]
        self._log_dropped(
            "annotation_overrides", len(df_annotations) - len(kept), len(df_annotations)
        )
        return pd.concat([kept, overrides[df_annotations.columns]], ignore_index=True)

    def _derive_df_images_from_annotations(self, df_annotations: pd.DataFrame) -> pd.DataFrame:
        return (
            df_annotations[["source_id", "image_id", "image_key"]]
            .drop_duplicates(subset=["source_id", "image_id"])
            .reset_index(drop=True)
        )

    def read_image(  # type: ignore[override]  # row columns are required named parameters
        self,
        image_id: str,
        source_id: str,
        image_key: str | None = None,
        **row_kwargs: Any,
    ) -> NDArray[Any]:
        """Read the source image for one row of ``df_images`` as an RGB array."""
        # ``image_key`` (resized vs original) comes from the training manifest via df_images.
        # When the manifest has no entry we deliberately do NOT fall back to the original
        # full-resolution image: its pixels would be misaligned with the 2048-resized annotation
        # coordinates. Raising here lets BaseCoralDataset warn and recover with a different image.
        if not (isinstance(image_key, str) and image_key):
            raise KeyError(
                f"No manifest entry for source_id={source_id} image_id={image_id}; "
                f"refusing to load the original full-resolution image (its pixels would be "
                f"misaligned with the 2048-resized annotation coordinates)."
            )
        return np.array(get_image_s3(bucket=self.source_bucket, key=image_key).convert("RGB"))
