# CoralNet Image Resizing Preprocessing — Design Doc

**Date:** 2026-05-25
**Issue:** #103
**Related:** #79 (coralnet-etl), #100, #60

---

## Overview

A standalone CLI script that takes images already downloaded by the CoralNet ETL, identifies which ones exceed the size threshold (longest edge > 2048px), and creates resized versions on S3. The script runs in two phases: **Phase 1** scans what needs resizing; **Phase 2** downloads, resizes, and uploads concurrently with checkpointing.

Images are resized by scaling the longest edge down to the threshold while maintaining aspect ratio.

---

## Architecture

```
scripts/preprocess_coralnet_images.py  ← entry point
  └─ mermaidseg/datasets/coralnet/preprocessing/
      ├── __init__.py
      ├── resize.py           ← Phase 1 (scan) + Phase 2 (resize) logic
      └── manifest.py         ← manifest parquet creation
```

The script operates independently of the main ETL pipeline but consumes its output (images parquet).

---

## Phase 1: Scan & Build Todo List

**Input:** Images parquet from ETL
Path: `s3://{bucket}/{output_prefix}/images.parquet`
Columns (pre-populated): `source_id, image_id, width, height, needs_resize`

**Assumption:** The parquet already contains `width`, `height`, and `needs_resize` columns, computed during the ETL image scan phase. We assume this data is accurate and complete.

**Process:**
1. Read images parquet
2. For each image marked `needs_resize=True`:
   - Query S3 concurrently: does the resized version already exist?
   - Output path: `s3://{bucket}/{output_prefix}/resized/{threshold}/s{source_id}/images/{image_id}.jpg`
   - If not found → add to "todo" dataframe
3. Write `todo.parquet` checkpoint file with columns: `[source_id, image_id, width, height, original_s3_key, output_s3_key]`

**Concurrency:** ThreadPoolExecutor with ~32 workers for S3 object-existence checks (following `images.py` pattern)

**Idempotency:** Any image already found on S3 is skipped; only missing images are queued for resize.

---

## Phase 2: Resize & Upload with Checkpointing

**Input:** `todo.parquet` from Phase 1

**Checkpoint Structure:**
Maintain a local checkpoint parquet at `{temp_dir}/checkpoint.parquet` with columns:
- `source_id` (int32)
- `image_id` (string)
- `status` ∈ `{pending, completed, failed}` (string)
- `resize_timestamp` (datetime, filled on completion)
- `error_message` (string, filled if status='failed')

**Process:**
1. On start, read checkpoint (if exists); skip rows marked `completed`
2. For each pending image:
   - Download original from S3 to memory
   - Validate: PIL can fully decode the image
   - Resize using PIL: longest edge → threshold, maintain aspect ratio
   - Upload resized JPEG to output S3 key
   - Mark in checkpoint as `completed` with timestamp
3. Every 500 images (configurable via `--checkpoint-every`): flush checkpoint to disk
4. On exception: log error message, mark as `failed`, continue processing

**Worker Pool:** ThreadPoolExecutor with configurable concurrency (default 16, to avoid overwhelming S3)

**Restart Semantics:** Checkpoint ensures only failed or pending images are reprocessed; completed images are skipped.

---

## Phase 2b: Build & Upload Manifest

Once all images are processed:

**Manifest Parquet Schema:**
```
- source_id (int32)
- image_id (string)
- original_width (int32)
- original_height (int32)
- resized_width (int32)
- resized_height (int32)
- output_s3_key (string)
- resize_timestamp (datetime)
- status (string)  ← 'completed' or 'failed'
```

**Output Path:**
`s3://{bucket}/{output_prefix}/resized/{threshold}/manifest.parquet`

The manifest enables downstream code (dataset loaders, validation scripts) to quickly locate resized images by source_id/image_id without scanning S3.

---

## CLI Interface

```bash
python -m mermaidseg.datasets.coralnet.preprocessing resize \
  --images-parquet s3://dev-datamermaid-sm-sources/etl-outputs/coralnet/images.parquet \
  --bucket dev-datamermaid-sm-sources \
  --output-prefix etl-outputs/coralnet \
  --threshold 2048 \
  --workers 16 \
  --checkpoint-every 500 \
  --temp-dir /tmp/coralnet-resize-checkpoint
```

**Arguments:**
- `--images-parquet` (required): S3 path to images parquet from ETL
- `--bucket` (required): S3 bucket name
- `--output-prefix` (required): S3 prefix for resized images and manifest
- `--threshold` (int, default 2048): resize target for longest edge
- `--workers` (int, default 16): ThreadPoolExecutor concurrency for Phase 2
- `--checkpoint-every` (int, default 500): flush checkpoint after N images
- `--temp-dir` (str, default `/tmp/coralnet-resize-{random}`): local checkpoint storage

---

## Error Handling

**Phase 1 (S3 checks):**
- Worker exceptions are caught; count non-existent objects as "need resize"
- Transient failures (e.g., network timeout) are retried within the worker pool's default retry logic

**Phase 2 (download/resize/upload):**
- Worker catches all exceptions, logs `(source_id, image_id, error_type, message)` to checkpoint
- Marks image as `failed`, continues processing
- No hard failures; process all images in todo list
- Summary logged at end: "N resized, M skipped (already on S3), K failed"

**Checkpoint Corruption:**
- If checkpoint parquet is corrupted, delete it; script restarts from Phase 1 (conservative, safe)
- Partially uploaded images on S3 are detected in Phase 1 (doesn't exist → reprocess)

---

## Data Quality

**Input Validation:**
- Images parquet must contain columns: `source_id, image_id, width, height, needs_resize`
- Throw `ValueError` if missing
- Assume `width`, `height`, `needs_resize` are pre-populated and accurate (trust ETL output)

**Resize Validation:**
- PIL must successfully open and decode original image before attempting resize
- On decode failure: mark as `failed`, log error, skip
- Verify longest_edge > threshold before resizing (skip if not needed, idempotency)

**Output Validation:**
- After upload, verify resized image exists on S3 (HEAD request) before marking checkpoint complete
- On verification failure: mark as `failed`, log error

---

## Testing

**Unit tests** (`tests/datasets/coralnet/test_preprocessing.py`):
- Resize logic: PIL resize with aspect ratio preservation
- Manifest schema validation
- Checkpoint read/write (parquet I/O)

**Integration tests** (with live S3, marked `@pytest.mark.live`):
- Phase 1 scan against test bucket
- Phase 2 download/resize/upload against test bucket
- Checkpoint restart (kill job mid-way, verify resume works)
- Idempotency (run twice, verify no re-processing)

---

## Dependencies

- **Pillow** (PIL): image resizing — already in `pyproject.toml`
- **pandas**: checkpoint/manifest parquet I/O — already in `pyproject.toml`
- **boto3**: S3 access — already in `pyproject.toml`
- **tqdm**: progress bars (optional, nice-to-have)

No new dependencies required.

---

## Notes

- **Annotation scaling** is handled separately (future ticket). This design focuses only on image resizing.
- **Threshold is configurable** but defaults to 2048 (matching ETL's `needs_resize` flag).
- **S3 costs:** Phase 1 uses HEAD requests (minimal cost); Phase 2 uses GET + PUT (standard data transfer costs).
- **Restart strategy:** Checkpoint + S3 idempotency means the script is safe to kill and resume at any time.
