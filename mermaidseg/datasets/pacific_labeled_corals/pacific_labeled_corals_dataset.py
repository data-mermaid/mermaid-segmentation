"""Pacific Labeled Corals PyTorch dataset.

Reads Pacific Labeled Corals point annotations from the Parquet produced by
``verify_raw_data_and_add_to_s3.py`` and emits ``(image, source_labels)``
tuples in the per-site ``labelmap.txt`` name space (Beijbom et al., 2015,
Dryad ``doi:10.5061/dryad.m5pr3``). Mapping into the MERMAID benthic-attribute
target space is performed externally via
:mod:`mermaidseg.dataset_reconciliation.label_mapping`.

Each evaluation point carries 7 alternative labels (``archived``, ``host``,
``visitor1..visitor5``); ``annotator_column`` (default ``"host"``) picks one
to expose as ``source_label_name``, with ``fallback_annotator`` (default
``"archived"``) covering reference-subset rows where the chosen column is
null. The actually-used column is recorded per row in the new
``annotator_used`` column on ``df_annotations``.
"""

from __future__ import annotations

from typing import Any

import boto3
import numpy as np
import pandas as pd
from numpy.typing import NDArray

from mermaidseg.datasets.base_dataset import BaseCoralDataset
from mermaidseg.datasets.utils import get_image_s3

VALID_ANNOTATOR_COLUMNS: tuple[str, ...] = (
    "archived",
    "host",
    "visitor1",
    "visitor2",
    "visitor3",
    "visitor4",
    "visitor5",
)


class PacificLabeledCoralsDataset(BaseCoralDataset):
    """PyTorch Dataset for Pacific Labeled Corals point annotations on S3.

    Each item is ``(image, source_labels)`` where ``source_labels`` is an
    integer mask in the Pacific Labeled Corals source label space (or shifted
    into the joint global space once registered with a
    :class:`SourceLabelRegistry`).

    Args:
        annotations_path: Key (relative to ``source_bucket``) of the
            annotations Parquet produced by ``verify_raw_data_and_add_to_s3.py``.
        source_bucket: S3 bucket name.
        source_s3_prefix: S3 prefix containing the
            ``images/<site>/<subset>/<image_id><image_ext>`` tree.
        whitelist_sites / blacklist_sites: Optional Pacific site
            allowlist/denylist (e.g. ``["heron_reef", "line_islands"]``).
        whitelist_subsets / blacklist_subsets: Optional subset
            allowlist/denylist (subset is one of ``"reference"`` /
            ``"evaluation"``).
        annotator_column: Annotator column to expose as ``source_label_name``;
            one of :data:`VALID_ANNOTATOR_COLUMNS`. Defaults to ``"host"``
            (typically the cleanest single-annotator stream available for the
            evaluation subsets).
        fallback_annotator: Annotator column to fall back on when
            ``annotator_column`` is null. Defaults to ``"archived"`` so
            reference subsets (which only carry ``archived``) are still
            included; pass ``None`` to drop those rows instead.
        **base_kwargs: Forwarded to :class:`BaseCoralDataset`.
    """

    SOURCE_NAME = "pacific_labeled_corals"

    REQUIRED_COLUMNS: tuple[str, ...] = (
        "site",
        "subset",
        "image_id",
        "image_ext",
        "row",
        "col",
        "archived",
    )

    def __init__(
        self,
        annotations_path: str = (
            "external_validation_datasets/pacific_labeled_corals/"
            "pacific_labeled_corals_annotations.parquet"
        ),
        source_bucket: str = "dev-datamermaid-sm-sources",
        source_s3_prefix: str = "external_validation_datasets/pacific_labeled_corals",
        whitelist_sites: list[str] | None = None,
        blacklist_sites: list[str] | None = None,
        whitelist_subsets: list[str] | None = None,
        blacklist_subsets: list[str] | None = None,
        annotator_column: str = "host",
        fallback_annotator: str | None = "archived",
        **base_kwargs: Any,
    ):
        if whitelist_sites is not None and blacklist_sites is not None:
            raise ValueError("Cannot specify both whitelist_sites and blacklist_sites.")
        if whitelist_subsets is not None and blacklist_subsets is not None:
            raise ValueError(
                "Cannot specify both whitelist_subsets and blacklist_subsets."
            )
        if annotator_column not in VALID_ANNOTATOR_COLUMNS:
            raise ValueError(
                f"annotator_column={annotator_column!r} not in "
                f"{VALID_ANNOTATOR_COLUMNS}"
            )
        if (
            fallback_annotator is not None
            and fallback_annotator not in VALID_ANNOTATOR_COLUMNS
        ):
            raise ValueError(
                f"fallback_annotator={fallback_annotator!r} not in "
                f"{VALID_ANNOTATOR_COLUMNS} (or None)"
            )

        self.annotations_path = annotations_path
        self.source_bucket = source_bucket
        self.source_s3_prefix = source_s3_prefix.rstrip("/")
        self.s3 = boto3.client("s3")
        self.whitelist_sites = whitelist_sites
        self.blacklist_sites = blacklist_sites
        self.whitelist_subsets = whitelist_subsets
        self.blacklist_subsets = blacklist_subsets
        self.annotator_column = annotator_column
        self.fallback_annotator = fallback_annotator

        df_annotations, df_images = self.load_annotations()
        super().__init__(df_annotations=df_annotations, df_images=df_images, **base_kwargs)

    def load_annotations(self) -> tuple[pd.DataFrame, pd.DataFrame]:
        """Load annotations from S3, apply site/subset filters, pick the active annotator."""
        annotations_uri = f"s3://{self.source_bucket}/{self.annotations_path}"
        df = pd.read_parquet(annotations_uri)

        missing = set(self.REQUIRED_COLUMNS) - set(df.columns)
        if missing:
            raise ValueError(
                f"Pacific Labeled Corals annotations parquet at {annotations_uri} is "
                f"missing required columns: {sorted(missing)}"
            )

        if self.whitelist_sites is not None:
            df = df[df["site"].isin(self.whitelist_sites)].reset_index(drop=True)
        if self.blacklist_sites is not None:
            df = df[~df["site"].isin(self.blacklist_sites)].reset_index(drop=True)
        if self.whitelist_subsets is not None:
            df = df[df["subset"].isin(self.whitelist_subsets)].reset_index(drop=True)
        if self.blacklist_subsets is not None:
            df = df[~df["subset"].isin(self.blacklist_subsets)].reset_index(drop=True)

        primary = df[self.annotator_column]
        if self.fallback_annotator is None:
            df["source_label_name"] = primary
            df["annotator_used"] = self.annotator_column
        else:
            fallback = df[self.fallback_annotator]
            used_primary = primary.notna()
            df["source_label_name"] = primary.where(used_primary, fallback)
            df["annotator_used"] = np.where(
                used_primary, self.annotator_column, self.fallback_annotator
            )

        df = df.dropna(subset=["source_label_name"]).reset_index(drop=True)
        df["source_label_name"] = df["source_label_name"].astype(str)
        df["annotator_used"] = df["annotator_used"].astype(str)

        df_images = self._derive_df_images_from_annotations(df)
        return df, df_images

    def _derive_df_images_from_annotations(self, df_annotations: pd.DataFrame) -> pd.DataFrame:
        return (
            df_annotations[["image_id", "site", "subset", "image_ext"]]
            .drop_duplicates(subset=["site", "subset", "image_id"])
            .reset_index(drop=True)
        )

    def read_image(
        self,
        image_id: str,
        site: str,
        subset: str,
        image_ext: str,
        **row_kwargs: Any,
    ) -> NDArray[Any]:
        key = f"{self.source_s3_prefix}/images/{site}/{subset}/{image_id}{image_ext}"
        return np.array(get_image_s3(s3=self.s3, bucket=self.source_bucket, key=key).convert("RGB"))
