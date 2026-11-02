"""Benthos Yuval PyTorch dataset (dense segmentation tiles).

Reads tiles produced by ``verify_raw_data_and_add_to_s3.py`` and emits ``(image, source_labels)``
pairs in the Benthos Yuval source-label space. See ``README.md`` for dataset details.
"""

from __future__ import annotations

import json
import logging
from typing import Any

import boto3
import numpy as np
import pandas as pd
from numpy.typing import NDArray

from mermaidseg.datasets.base_dataset import BaseCoralDataset
from mermaidseg.datasets.utils import get_image_s3

logger = logging.getLogger(__name__)


class BenthosYuvalCoralsDataset(BaseCoralDataset):
    """PyTorch Dataset for the dense Benthos Yuval segmentation tiles on S3.

    Each item is ``(image, source_labels)`` where ``source_labels`` is an
    integer mask in the local source-id space (or shifted into the joint
    global space once registered with a :class:`SourceLabelRegistry`).

    Args:
        annotations_path: S3 key (relative to ``source_bucket``) of the
            annotations Parquet.
        source_bucket: S3 bucket name.
        source_s3_prefix: S3 prefix containing the
            ``{images,labels}/<site>/<image_id>.png`` tree plus
            ``classes.json``.
        whitelist_sites / blacklist_sites: Optional site allowlist/denylist
            (currently ``"RS24"`` and/or ``"CR_DoubleWreck"``).
        **base_kwargs: Forwarded to :class:`BaseCoralDataset`.
    """

    SOURCE_NAME = "benthos_yuval"

    annotations_path: str
    source_bucket: str
    source_s3_prefix: str
    s3: boto3.client
    whitelist_sites: list[str] | None
    blacklist_sites: list[str] | None

    REQUIRED_COLUMNS: tuple[str, ...] = (
        "site",
        "image_id",
        "source_label_name",
    )

    def __init__(
        self,
        annotations_path: str = (
            "external_validation_datasets/benthos_yuval/benthos_yuval_annotations.parquet"
        ),
        source_bucket: str = "dev-datamermaid-sm-sources",
        source_s3_prefix: str = "external_validation_datasets/benthos_yuval",
        whitelist_sites: list[str] | None = None,
        blacklist_sites: list[str] | None = None,
        **base_kwargs: Any,
    ):
        if whitelist_sites is not None and blacklist_sites is not None:
            raise ValueError("Cannot specify both whitelist_sites and blacklist_sites.")

        self.annotations_path = annotations_path
        self.source_bucket = source_bucket
        self.source_s3_prefix = source_s3_prefix.rstrip("/")
        self.s3 = boto3.client("s3")
        self.whitelist_sites = whitelist_sites
        self.blacklist_sites = blacklist_sites

        df_annotations, df_images = self.load_annotations()
        super().__init__(df_annotations=df_annotations, df_images=df_images, **base_kwargs)

        # Mask PNGs are in the global ``classes.json`` ID space; remap to the
        # value-counts-ordered local space set up by ``BaseCoralDataset``.
        self._classes_global = self._load_classes_json()
        self._classes_global_to_local = self._build_classes_global_to_local()

    def load_annotations(self) -> tuple[pd.DataFrame, pd.DataFrame]:
        """Load the annotations Parquet from S3 and apply site filters."""
        annotations_uri = f"s3://{self.source_bucket}/{self.annotations_path}"
        df_annotations = pd.read_parquet(annotations_uri)

        missing = set(self.REQUIRED_COLUMNS) - set(df_annotations.columns)
        if missing:
            raise ValueError(
                f"Benthos Yuval annotations parquet at {annotations_uri} is missing required "
                f"columns: {sorted(missing)}"
            )

        df_annotations["site"] = df_annotations["site"].astype(str)
        df_annotations["image_id"] = df_annotations["image_id"].astype(str)
        df_annotations["source_label_name"] = df_annotations["source_label_name"].astype(str)

        if self.whitelist_sites is not None:
            df_annotations = df_annotations[
                df_annotations["site"].isin(self.whitelist_sites)
            ].reset_index(drop=True)
        if self.blacklist_sites is not None:
            df_annotations = df_annotations[
                ~df_annotations["site"].isin(self.blacklist_sites)
            ].reset_index(drop=True)

        df_images = self._derive_df_images_from_annotations(df_annotations)
        return df_annotations, df_images

    def _derive_df_images_from_annotations(self, df_annotations: pd.DataFrame) -> pd.DataFrame:
        return (
            df_annotations[["image_id", "site"]]
            .drop_duplicates(subset=["image_id", "site"])
            .reset_index(drop=True)
        )

    def _load_classes_json(self) -> dict[str, int]:
        """Fetch ``classes.json`` from S3 (used to interpret the dense mask PNGs)."""
        key = f"{self.source_s3_prefix}/classes.json"
        body = self.s3.get_object(Bucket=self.source_bucket, Key=key)["Body"].read()
        return {str(name): int(idx) for name, idx in json.loads(body).items()}

    def _build_classes_global_to_local(self) -> np.ndarray:
        """Lookup table from classes.json IDs to local source IDs.

        Background, classes filtered out by ``class_subset``, and any unknown class fall through to
        local ID 0.
        """
        max_global = max(self._classes_global.values()) + 1
        lookup = np.zeros(max_global, dtype=np.int64)
        for name, global_id in self._classes_global.items():
            local_id = self.source_name2id.get(name)
            if local_id is not None:
                lookup[int(global_id)] = int(local_id)
        return lookup

    def set_source_vocabulary(
        self,
        source_id2name: dict[int, str],
        source_name2id: dict[str, int],
        num_source_classes: int,
    ) -> None:
        """Replace local source maps and rebuild the classes.json lookup."""
        self.source_id2name = dict(source_id2name)
        self.source_name2id = dict(source_name2id)
        self.num_source_classes = int(num_source_classes)
        self._classes_global_to_local = self._build_classes_global_to_local()

    def set_global_offset(self, offset: int) -> None:
        super().set_global_offset(offset)
        self._classes_global_to_local = self._build_classes_global_to_local()

    def read_image(self, image_id: str, site: str, **row_kwargs: Any) -> NDArray[Any]:
        key = f"{self.source_s3_prefix}/images/{site}/{image_id}.png"
        return np.array(get_image_s3(s3=self.s3, bucket=self.source_bucket, key=key).convert("RGB"))

    def read_label(self, image_id: str, site: str, **row_kwargs: Any) -> NDArray[Any]:
        """Read the dense uint8 label PNG (in the ``classes.json`` ID space)."""
        key = f"{self.source_s3_prefix}/labels/{site}/{image_id}.png"
        arr = np.asarray(get_image_s3(s3=self.s3, bucket=self.source_bucket, key=key))
        if arr.ndim == 3:
            arr = arr[..., 0]
        return arr.astype(np.uint8, copy=False)

    def _load_item(self, idx: int) -> tuple[Any, Any]:
        image_id = self.df_images.loc[idx, "image_id"]
        site = self.df_images.loc[idx, "site"]

        image = self.read_image(image_id=image_id, site=site)
        raw_mask = self.read_label(image_id=image_id, site=site)

        local_mask = self._classes_global_to_local[raw_mask]

        if self._global_offset:
            local_mask = np.where(
                local_mask > 0, local_mask + self._global_offset, local_mask
            ).astype(local_mask.dtype, copy=False)

        if self.transform:
            transformed = self.transform(image=image, mask=local_mask)
            image = transformed["image"].transpose(2, 0, 1)
            local_mask = transformed["mask"]

        return image, local_mask
