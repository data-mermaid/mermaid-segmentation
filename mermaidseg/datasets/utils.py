import contextlib
import io
import logging
import os
import sys
import tempfile
from pathlib import Path

import boto3
import numpy as np
import pandas as pd
import torch
from botocore.config import Config as BotoConfig
from botocore.exceptions import ClientError
from PIL import Image, UnidentifiedImageError
from torch.utils.data import Dataset, default_collate
from tqdm import tqdm

logger = logging.getLogger(__name__)


def emit_dataset_warning(message: str) -> None:
    """Emit a dataset-load warning via the logger AND raw stdout/stderr.

    PyTorch DataLoader worker processes often have their ``logging`` handlers
    unconfigured, which means ``logger.warning`` is silently dropped. To make sure the
    user sees skip-and-recover messages no matter where they're triggered (main process,
    worker, notebook, terminal), we additionally ``print`` to both ``sys.stdout`` and
    ``sys.stderr`` with ``flush=True``.
    """
    logger.warning(message)
    full = f"WARNING: {message}"
    print(full, file=sys.stderr, flush=True)
    print(full, file=sys.stdout, flush=True)


class DataLoadError(Exception):
    """Raised when an image cannot be loaded from S3 or decoded."""


_IMAGE_CACHE_DIR: str | None = os.environ.get("MERMAIDSEG_IMAGE_CACHE_DIR")


def s3_training_config(max_pool_connections: int = 20) -> BotoConfig:
    """Botocore config for training-path S3 clients: adaptive retries + timeouts."""
    return BotoConfig(
        retries={"max_attempts": 10, "mode": "adaptive"},
        connect_timeout=10,
        read_timeout=30,
        max_pool_connections=max_pool_connections,
    )


def _cache_path(bucket: str, key: str) -> Path | None:
    if _IMAGE_CACHE_DIR is None:
        return None
    return Path(_IMAGE_CACHE_DIR) / bucket / key


def get_image_s3(
    s3: boto3.client,
    bucket: str,
    key: str,
    thumbnail: bool = False,
):
    """Fetch an image from S3, serving from disk cache when
    ``MERMAIDSEG_IMAGE_CACHE_DIR`` is set."""
    if thumbnail:
        key = key.replace(".png", "_thumbnail.png")

    cached = _cache_path(bucket, key)
    if cached is not None and cached.exists():
        try:
            return Image.open(cached)
        except (UnidentifiedImageError, OSError):
            cached.unlink(missing_ok=True)

    try:
        response = s3.get_object(Bucket=bucket, Key=key)
        try:
            image_data = response["Body"].read()
        finally:
            response["Body"].close()
    except ClientError as e:
        error_code = e.response["Error"]["Code"]
        logger.warning(
            "S3 error loading image (bucket=%s, key=%s): %s %s", bucket, key, error_code, e
        )
        raise DataLoadError(f"S3 ClientError for s3://{bucket}/{key}: {error_code}") from e

    if cached is not None:
        try:
            cached.parent.mkdir(parents=True, exist_ok=True)
            fd, tmp = tempfile.mkstemp(dir=cached.parent)
            closed = False
            try:
                os.write(fd, image_data)
                os.close(fd)
                closed = True
                os.rename(tmp, cached)
            except BaseException:
                if not closed:
                    with contextlib.suppress(OSError):
                        os.close(fd)
                Path(tmp).unlink(missing_ok=True)
                raise
        except OSError:
            pass

    try:
        image = Image.open(io.BytesIO(image_data))
    except (UnidentifiedImageError, OSError) as e:
        logger.warning("Corrupted image (bucket=%s, key=%s): %s", bucket, key, e)
        if cached is not None:
            cached.unlink(missing_ok=True)
        raise DataLoadError(f"PIL cannot open image at s3://{bucket}/{key}") from e

    return image


def create_annotation_mask(
    annotations: pd.DataFrame,
    shape: tuple[int, int],
    source_name2id: dict[str, int],
    padding: int | None = None,
) -> np.ndarray:
    """Creates a source-space annotation mask for a given image.

    Args:
        annotations (pd.DataFrame): DataFrame with 'row', 'col', and
            'source_label_name' columns. ``source_label_name`` holds labels in
            the source dataset's own label space.
        shape (tuple[int, int]): Output mask shape (height, width).
        source_name2id (dict[str, int]): Mapping from source-space label names
            to integer class IDs (1..N; 0 is reserved for background).
        padding (int | None, optional): Half-size of a square pad region around each point annotation.
            If None or 0, only the exact annotation pixel is set. Defaults to None.
    Returns:
        np.ndarray: Integer annotation mask with shape (height, width). Values
        are 0 (background) or local source-class IDs (1..N).
    """
    # TODO: Make padding percentage-based so it scales with image resolution
    mask = np.zeros(shape[:2], dtype=np.int64)

    if annotations.empty:
        return mask

    valid = annotations[annotations["source_label_name"].notna()].copy()

    unknown = set(valid["source_label_name"]) - set(source_name2id.keys())
    if unknown:
        logger.warning(
            "create_annotation_mask: skipping %d unknown label(s): %s",
            len(unknown),
            sorted(unknown),
        )
        valid = valid[valid["source_label_name"].isin(source_name2id)]

    if valid.empty:
        return mask

    rows = valid["row"].to_numpy(dtype=np.intp)
    cols = valid["col"].to_numpy(dtype=np.intp)
    label_ids = valid["source_label_name"].map(source_name2id).to_numpy(dtype=np.int64)

    if padding is not None and padding > 0:
        h, w = shape[:2]
        # Vectorized bounds computation
        r0 = np.maximum(0, rows - padding)
        r1 = np.minimum(h, rows + padding)
        c0 = np.maximum(0, cols - padding)
        c1 = np.minimum(w, cols + padding)

        # Apply padding regions (later annotations overwrite earlier ones)
        for r0i, r1i, c0i, c1i, lid in zip(r0, r1, c0, c1, label_ids, strict=False):
            mask[r0i:r1i, c0i:c1i] = lid
    else:
        mask[rows, cols] = label_ids

    return mask


def calculate_weights(dataset: Dataset, const: int = 2000000) -> torch.Tensor:
    """Calculate class weights for a given dataset.

    This function computes the weights for each class in the dataset based on
    the frequency of each class label. The weights are inversely proportional
    to the square root of the class frequency, adjusted by a constant value.
    Args:
        dataset (Dataset): The dataset object which contains images and labels.
                            It should have an attribute `N_classes` indicating
                            the number of classes.
        const (int, optional): A constant value added to the class frequency
                                to avoid division by zero and to smooth the weights.
                                Default is 2000000.
    Returns:
        torch.Tensor: A tensor of weights for each class, normalized by the mean weight.
    """
    label_counts = dict.fromkeys(range(dataset.N_classes), 0)
    for i in tqdm(range(len(dataset))):
        _, label, _ = dataset[i]
        unique_labels = np.unique(label, return_counts=True)
        for label_id, count in zip(*unique_labels, strict=False):
            label_counts[label_id] += int(count)

    weights = np.zeros(dataset.N_classes)
    for index, count in label_counts.items():
        weights[index] = 1 / np.sqrt(count + const)
    weight = torch.tensor(weights).float()
    weight /= weight.mean()

    return weight


def _joint_collate(batch: list) -> tuple[torch.Tensor, torch.Tensor]:
    """Collate function to combine a list of samples into a batch."""
    images, labels = zip(*batch, strict=False)
    images = default_collate(images)
    labels = default_collate(labels)
    return images, labels


def get_coralnet_sources():
    """Discover and validate CoralNet source folders stored in the S3 bucket "dev-
    datamermaid-sm- sources".

    Returns:
        whitelist: A list of all valid CoralNet source folder names that contain both
              'annotations.csv' and 'image_list.csv' files.
    """
    s3 = boto3.client("s3")
    bucket_name = "dev-datamermaid-sm-sources"

    response = s3.list_objects_v2(Bucket=bucket_name, Delimiter="/")
    if "CommonPrefixes" in response:
        folders_new = [prefix["Prefix"] for prefix in response["CommonPrefixes"]]
        folder = "coralnet-public-images/"
        sub_response = s3.list_objects_v2(Bucket=bucket_name, Prefix=folder, Delimiter="/")
        if "CommonPrefixes" in sub_response:
            logger.warning("Subfolders in coralnet-public-images/:")
            folders_new = [prefix["Prefix"] for prefix in sub_response["CommonPrefixes"]]
            folders_new = [folder.replace("coralnet-public-images/", "") for folder in folders_new]
        else:
            logger.warning("No subfolders found in coralnet-public-images/")
    else:
        logger.warning("No folders found in the bucket")

    whitelist_sources = []
    for source in tqdm(folders_new):
        if not source.startswith("s"):
            logger.warning("Unexpected source folder name (no 's' prefix): %s", source)

        file_key = f"coralnet-public-images/{source}annotations.csv"
        try:
            s3.head_object(Bucket=bucket_name, Key=file_key)
        except ClientError as e:
            error_code = e.response["Error"]["Code"]
            if error_code == "404":
                logger.warning(
                    "get_coralnet_sources: annotations.csv not found for source %s", source
                )
            else:
                logger.warning(
                    "get_coralnet_sources: unexpected S3 error for %s (code=%s): %s",
                    file_key,
                    error_code,
                    e,
                )
            continue

        file_key = f"coralnet-public-images/{source}image_list.csv"
        try:
            s3.head_object(Bucket=bucket_name, Key=file_key)
        except ClientError as e:
            error_code = e.response["Error"]["Code"]
            if error_code == "404":
                logger.warning(
                    "get_coralnet_sources: image_list.csv not found for source %s", source
                )
            else:
                logger.warning(
                    "get_coralnet_sources: unexpected S3 error for %s (code=%s): %s",
                    file_key,
                    error_code,
                    e,
                )
            continue

        whitelist_sources.append(source)
    return whitelist_sources
