import logging
import sys

import numpy as np
import pandas as pd

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
    bucket: str,
    key: str,
    thumbnail: bool = False,
):
    """Fetches an image from local cache or S3 and returns it as a PIL Image.

    Args:
        bucket (str): The name of the S3 bucket.
        key (str): The key (path) of the image in the S3 bucket.
        thumbnail (bool, optional): If True, fetches the thumbnail version of the image by modifying the key. Defaults to False.

    Returns:
        PIL.Image.Image: The image loaded from cache or S3 as a PIL Image object.
    """
    cache = LocalS3Cache.get()
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
