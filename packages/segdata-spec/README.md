# segdata-spec

The versioned dataset contract between the ETL that publishes segmentation datasets and the trainer that reads them. Import package: `segdata_spec`.

## Scope

- **Owns:** the `manifest.json` and parquet table schemas, the S3 key layout, the local cache layout, publishing a version, syncing pinned versions and their image files into the cache (resizing them there when given a resize spec), opening synced versions, the benthic-attribute `roll_up`, and the split-file format with its coverage check.
- **Does not own:** extracting or converting source data, choosing versions or pins, choosing the train/val/test split (the package reads and checks a split file; the split is built elsewhere), or publishing and owning the original image files. The ETL puts the originals in S3; the manifest's `image_store` names the bucket and prefix.
- **Boundary:** the package does not import `mermaidseg`, `coralnet_scraper` or `segdata_etl`, and `mermaidseg/inference` does not import it. The root `tests/test_package_boundaries.py` enforces this. `mermaidseg` imports it (`mermaidseg/data/`).

## Callers

| Caller | Calls |
|---|---|
| ETL | `publish.publish_dataset` and `publish.publish_taxonomy`. Each returns the manifest and its sha256, which becomes a `manifest.Pin`. |
| Trainer | `sync.sync(client, pins, cache_dir, bucket=..., resize=..., max_workers=...)`, which needs S3 access, then `read.open_version(..., resize=...)` and `read.open_taxonomy`, which read only the cache. Also `resize` (`ResizeSpec`, `scaled_size`, `resize_image`, `resize_mask`, `scale_points`, `scale_images`). `scale_images` returns the images table with the resized `width` and `height` and `media_type` `image/jpeg`; `open_version(resize=)` uses it. |
| Both | `taxonomy.roll_up` and `taxonomy.hierarchy_from_table`. |
| Split builder and trainer | `splits.read_split_file`. The trainer also passes `splits.coverage_checker` to `sync`. |

`resize` copies the ETL's decode-and-resize rather than being called by it. For JPEG and PNG images whose scaled edges are at least 1 px, `test_segdata_spec_parity.py` keeps the two byte-identical. Otherwise they diverge: the spec refuses formats other than JPEG and PNG, which the ETL decodes, and clamps a zero edge to 1 px, where the ETL fails.

`resize` on `sync` maps a dataset name to a `ResizeSpec` or `None`. A dataset that is absent or maps to `None` is cached as is. `DatasetSyncResult.resize` records the spec that was applied, for model lineage. Pass `open_version` the spec that `sync` applied (`DatasetSyncResult.resize`). Nothing checks this: `image_path` and `mask_path` do not check that the file exists, so a mismatch surfaces only when the image is opened. A `ResizeSpec` holds `max_edge`, `filter` (`lanczos`, `bicubic` or `bilinear`) and `jpeg_quality`; its `id` is the first 12 hex characters of a hash of the three fields.

`sync` runs on `max_workers` threads (default `min(8, os.cpu_count())`). Create the boto3 client with `max_pool_connections` of at least `max_workers`.

## S3 layout

```text
datasets/<name>/<version>/manifest.json
datasets/<name>/<version>/<table>.parquet      images, labels, points (points datasets only), stats/label_counts
taxonomy/<name>/<version>/manifest.json
taxonomy/<name>/<version>/benthic_attributes.parquet
taxonomy/<name>/<version>/growth_forms.parquet
```

Images and masks are not under these prefixes. Their keys are `image_store.prefix` plus the `key` and `mask_key` columns of `images.parquet`.

`images.parquet` also has a nullable `source_name`: the image's name at its source. It is data only and is never part of a key. Publishers must supply the column (nulls allowed); `publish_dataset` refuses a table without it. `sync` and `open_version` accept a version published without the column; its `images` table then has no `source_name`.

## Cache layout

Without a resize spec, `sync` writes every object to `<cache_dir>/<bucket>/<key>`, the same key as on S3, so `read` finds a file from the manifest alone. Manifests and tables always use this layout.

With a spec, `sync` writes each image and mask to `<cache_dir>/resized/<spec id>/<bucket>/<key>` and does not cache the original. Each output has an index file at `<cache_dir>/resized/<spec id>/.index/<bucket>/<key>.json` with the source ETag, the output size, the output sha256 and the output's pixel width and height. Two specs share one cache without interfering.

A resized file keeps the source key name, so a key ending `.png` may hold JPEG bytes. Read the format from `media_type` in the images table that `open_version(resize=...)` returns, not from the file extension.

`DatasetVersion.image_path` and `mask_path` return the cached path for an `images` row, in the layout of the version's `resize`. A key that would leave the cache is refused.

## Resize rule

- Images are decoded with JPEG draft mode, then scaled with Lanczos (or the spec's filter) to `int(edge * (max_edge / longest))` on each edge. This is the operation order of the legacy CoralNet resize stage, so a few sizes come out one pixel short, for example 2215 to 2047 at `max_edge=2048`.
- Originals are decoded only as JPEG or PNG, and masks only as PNG. Any other format fails with `ImageUndecodable`.
- Each scaled edge is at least 1 px: a 10000x3 image at `max_edge=2048` becomes 2048x1.
- A JPEG in RGB or L mode within `max_edge` is cached byte for byte. Any other image is re-encoded as JPEG.
- Masks are scaled to their image's new size with nearest-neighbour and stay PNG.
- `open_version(resize=)` reports the resized `width` and `height` and `media_type` `image/jpeg`. It scales `points` from the recorded original size, with floor and then a clamp to the image. It reads no images. `size`, `etag`, the CRC columns and their mask counterparts still describe the S3 original.

## Splits

`splits.py` defines the split file: a CSV that assigns each dataset group to `train`, `val`, `test` or `excluded`. It does not choose the assignment.

- **Columns:** `group` and `split` are required. `source_id` and `split_group` are optional, and other columns are allowed. A blank `split_group` means the row has none, and the straddling check skips it.
- `read_split_file(path)` returns the file as a table of string columns. It raises `ValueError` if a required column is missing, a `split` value is not `train`, `val`, `test` or `excluded`, a group appears twice, or a `split_group` spans more than one split.
- `check_coverage(split_file, dataset, groups)` raises `SplitCoverageError` if `groups` holds a group the file does not list. A group listed as `excluded` counts as listed, so a version may hold it. Groups the file lists but `groups` lacks are allowed.
- `excluded` marks a group that is kept in the version but not used for training. A data config's `use:` accepts only `train`, `val` and `test`, so a `use:` selection never picks an `excluded` group. A data config block with `groups: all` or named groups selects by name, bypasses the split file, and includes excluded groups.
- `coverage_checker(split_files)` takes a mapping from dataset name to split file and returns the `check_split` callback for `sync`. It checks the mapped datasets and skips the others.

```python
sync(client, pins, cache_dir, bucket=bucket,
     check_split=coverage_checker({"coralnet": Path("configs/splits/coralnet.csv")}))
```

`mermaidseg/data/sync.py` passes this callback in `run_sync`, so a sync fails before any image download when a pinned version has an unlisted group.

## Guarantees

Publish:

- Every put sends `If-None-Match: *`. An existing object with identical bytes counts as written; any other existing object raises `ObjectExists`.
- Table files are written first and `manifest.json` last. A prefix without a manifest is an incomplete publish.
- A version that already has a manifest is refused.
- A dirty tree (tracked files only), a `HEAD` that no remote branch contains, and a missing `uv.lock` are refused with `PublishRefused`. The manifest records the git SHA and the sha256 of `uv.lock`.
- The same tables always serialize to the same bytes.

Sync:

- Order of checks: `max_workers` (before any S3 call), every manifest against its pin, every table against its manifest sha256, the `selection` and `resize` checks, `check_split` for each dataset, then the images. A failure before the images downloads no image. `check_split` is a callable `(dataset, groups) -> None` that receives the groups in the dataset's `images.parquet`. Callers pass `coverage_checker(split_files)` for it (see [Splits](#splits)).
- `on_missing_images` is `"fail"` (default) or `"drop"` for each dataset. `drop` skips an image or mask that is absent and reports its ID in `DatasetSyncResult.dropped_image_ids`. A changed object always fails.
- Images and masks are fetched with `IfMatch` on the recorded ETag.
- With a resize spec, the original is verified (`IfMatch`, size, MD5, CRC64NVME) before it is resized, and only the result is written. The index entry is written before the output, so the output file is the commit point.
- After resizing, the output's pixel size is checked against the size computed from the row's `width` and `height`. A mismatch raises `ChecksumMismatch` and nothing is cached.
- On a cache hit for a resized output, `sync` checks the output against its index entry, and the entry's source ETag against `images.parquet`. The entry's recorded pixel size is also checked against the expected size; a missing or different size raises `ChecksumMismatch`. A mismatch, or a missing or malformed entry, raises `ChecksumMismatch`; the original is not downloaded again.
- The worker pool starts only after every check above. Results are taken in row order, so the error raised and `dropped_image_ids` match a serial run.
- An original above twice Pillow's `Image.MAX_IMAGE_PIXELS` fails the sync with `DecompressionBombError`; one between the limit and twice it only warns.
- Cached files are verified locally and never downloaded again on a mismatch. Manifests and tables are checked by sha256. Images and masks are checked by size and, for a single-part ETag, by MD5; a multipart ETag gets no MD5 check. When `images.parquet` records `crc64nvme` or `mask_crc64nvme`, sync also checks it with awscrt, on a cache hit and on a download. A row without a recorded CRC gets only the size and single-part MD5 checks.

## Errors

All derive from `errors.SegdataError`.

| Class | Raised when |
|---|---|
| `PublishRefused` | The tree is dirty or unpushed, `uv.lock` is missing, or the version exists. |
| `ObjectExists` | A write-once key exists with different bytes. |
| `SchemaValidationError` | A table fails its schema, or the cross-table rules. Also a `ValueError`. |
| `PinMismatch` | A manifest's sha256 differs from its pin. |
| `ChecksumMismatch` | A table, image or mask differs from its recorded checksum or size, or a cached resized output has no valid index entry or differs from it. |
| `ImageChanged` | An image no longer matches its recorded ETag. |
| `ImageMissing` | An image is absent and the dataset's mode is `fail`. |
| `SplitCoverageError` | A dataset version has a group that its split file does not list. |
| `ImageUndecodable` | An image or mask matches its checksums but cannot be decoded (for example, a format other than JPEG or PNG). Names the S3 key. |

`sync` also raises `ValueError` for a `selection` that names an unpinned dataset or an absent group, a `resize` that names an unpinned dataset, or `max_workers` below 1. `read` raises `FileNotFoundError` for a manifest or table that is not cached. `read` and `open_version` raise `ValueError` when a point names an `image_id` absent from `images` (resize set).

## Tests

```bash
uv run --package segdata-spec --group tests pytest packages/segdata-spec/tests
```

The tests use an in-memory S3 fake. `pytest-socket` blocks every host except localhost.

`packages/segdata-etl/tests/test_segdata_spec_parity.py` checks that `resize_image` output is byte-identical to the ETL's decode and resize, and that `scale_points` matches the `row` and `col` the ETL training manifest computes.
