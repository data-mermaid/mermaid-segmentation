import logging
import sys

import boto3
import numpy as np
import pandas as pd
import torch
from botocore.exceptions import ClientError
from torch.utils.data import Dataset, default_collate
from tqdm import tqdm

from mermaidseg.datasets.local_cache import LocalS3Cache

logger = logging.getLogger(__name__)


def emit_cache_stats(message: str) -> None:
    """Emit per-epoch cache statistics to logger, stdout, and stderr."""
    logger.info(message)
    full = f"CACHE: {message}"
    print(full, file=sys.stderr, flush=True)
    print(full, file=sys.stdout, flush=True)


def emit_load_stage_stats(prefix: str, stats: object) -> None:
    """Emit per-epoch data-loading stage timings (from ``LocalS3Cache.snapshot_stats``).

    Reports the cumulative seconds spent in each ``BaseCoralDataset._load_item`` stage (with each
    stage's share of the total point-load time), the mean point-load time per sample, and the
    empty/failed-load retry count and rate. No-op when no stage timing was recorded (e.g. a run of
    only dense HuggingFace datasets, or stats disabled).
    """
    stage_sec: dict[str, float] = dict(getattr(stats, "stage_sec", {}) or {})
    samples = int(getattr(stats, "samples", 0) or 0)
    retries = int(getattr(stats, "retries", 0) or 0)
    total = sum(stage_sec.values())

    if total <= 0 and samples == 0:
        return

    parts = [
        f"{name}={sec:.1f}s ({(sec / total * 100) if total else 0:.0f}%)"
        for name, sec in stage_sec.items()
    ]
    per_sample_ms = (total / samples * 1000.0) if samples else 0.0
    retry_rate = (retries / samples) if samples else 0.0
    emit_cache_stats(
        f"{prefix} load stages: [{', '.join(parts)}]; "
        f"{samples} samples, {per_sample_ms:.1f} ms/sample point-load, "
        f"{retries} retries ({retry_rate:.2f}/sample)"
    )


def emit_dataset_warning(message: str) -> None:
    """Emit a dataset-load warning via the logger AND raw stdout/stderr.

    PyTorch DataLoader worker processes often have their ``logging`` handlers unconfigured, which
    means ``logger.warning`` is silently dropped. To make sure the user sees skip-and-recover
    messages no matter where they're triggered (main process, worker, notebook, terminal), we
    additionally ``print`` to both ``sys.stdout`` and ``sys.stderr`` with ``flush=True``.
    """
    logger.warning(message)
    full = f"WARNING: {message}"
    print(full, file=sys.stderr, flush=True)
    print(full, file=sys.stdout, flush=True)


def get_image_s3(
    s3: boto3.client,
    bucket: str,
    key: str,
    thumbnail: bool = False,
):
    """Fetches an image from local cache or S3 and returns it as a PIL Image.

    Args:
        s3 (boto3.client): Optional Boto3 S3 client (injected into the cache when provided).
        bucket (str): The name of the S3 bucket.
        key (str): The key (path) of the image in the S3 bucket.
        thumbnail (bool, optional): If True, fetches the thumbnail version of the image by modifying the key. Defaults to False.
    Returns:
        PIL.Image.Image: The image loaded from cache or S3 as a PIL Image object.
    """
    cache = LocalS3Cache.get()
    if s3 is not None:
        cache.set_s3_client(s3)
    return cache.read_pil_image(bucket, key, thumbnail=thumbnail)


def create_annotation_mask(
    annotations: pd.DataFrame,
    shape: tuple[int, int],
    source_name2id: dict[str, int],
    padding: int | None = None,
    offset: int = 0,
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
        offset (int, optional): Global source-label offset added to every foreground
            label id (background stays 0). Folding it in here is equivalent to a
            ``np.where(mask > 0, mask + offset, mask)`` pass but avoids a second full-resolution
            scan. Defaults to 0.
    Returns:
        np.ndarray: Integer annotation mask with shape (height, width). Values
        are 0 (background) or (offset-shifted) local source-class IDs (1..N).
    """
    # TODO: Make padding percentage-based so it scales with image resolution
    # int32 keeps the emitted mask small (albumentations casts masks to int32 for cv2 anyway, and
    # the training loop calls ``.long()`` downstream); global ids are far below the int32 range.
    mask = np.zeros(shape[:2], dtype=np.int32)

    if annotations.empty:
        return mask

    valid = annotations[annotations["source_label_name"].notna()]

    # Map to ids once; unknown labels become NaN. Only materialise the unknown-label set (and warn)
    # when something actually failed to map, instead of building a set every call.
    mapped = valid["source_label_name"].map(source_name2id)
    if mapped.isna().any():
        unmapped = mapped.isna()
        unknown = sorted(set(valid["source_label_name"][unmapped]))
        logger.warning(
            "create_annotation_mask: skipping %d unknown label(s): %s",
            len(unknown),
            unknown,
        )
        keep = ~unmapped
        valid = valid[keep]
        mapped = mapped[keep]

    if valid.empty:
        return mask

    rows = valid["row"].to_numpy(dtype=np.intp)
    cols = valid["col"].to_numpy(dtype=np.intp)
    label_ids = mapped.to_numpy(dtype=np.int32)
    if offset:
        label_ids = label_ids + np.int32(offset)

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
    """Discover and validate CoralNet source folders stored in the S3 bucket "dev-datamermaid-sm-
    sources".

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
