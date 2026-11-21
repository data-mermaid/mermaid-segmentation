# segdata-spec

The versioned dataset contract between the ETL that publishes segmentation datasets and the trainer that reads them. Import package: `segdata_spec`.

## Scope

- **Owns:** the `manifest.json` and parquet table schemas, the S3 key layout, the local cache layout, publishing a version, syncing pinned versions into the cache, opening synced versions, the benthic-attribute `roll_up`, and the split-file format with its coverage check.
- **Does not own:** extracting or converting source data, choosing versions or pins, choosing the train/val/test split (the package reads and checks a split file; the split is built elsewhere), or the image files themselves. Images stay where the ETL put them; the manifest's `image_store` names the bucket and prefix.
- **Boundary:** the package does not import `mermaidseg`, `coralnet_scraper` or `segdata_etl`, and `mermaidseg/inference` does not import it. The root `tests/test_package_boundaries.py` enforces this. `mermaidseg` imports it (`mermaidseg/data/`).

## Callers

| Caller | Calls |
|---|---|
| ETL | `publish.publish_dataset` and `publish.publish_taxonomy`. Each returns the manifest and its sha256, which becomes a `manifest.Pin`. |
| Trainer | `sync.sync(client, pins, cache_dir, bucket=...)`, which needs S3 access, then `read.open_version` and `read.open_taxonomy`, which read only the cache. |
| Both | `taxonomy.roll_up` and `taxonomy.hierarchy_from_table`. |
| Split builder and trainer | `splits.read_split_file`. The trainer also passes `splits.coverage_checker` to `sync`. |

## S3 layout

```text
datasets/<name>/<version>/manifest.json
datasets/<name>/<version>/<table>.parquet      images, labels, points (points datasets only), stats/label_counts
taxonomy/<name>/<version>/manifest.json
taxonomy/<name>/<version>/benthic_attributes.parquet
taxonomy/<name>/<version>/growth_forms.parquet
```

Images and masks are not under these prefixes. Their keys are `image_store.prefix` plus the `key` and `mask_key` columns of `images.parquet`.

## Cache layout

`sync` writes every object to `<cache_dir>/<bucket>/<key>`, the same key as on S3, so `read` finds a file from the manifest alone. `DatasetVersion.image_path` and `mask_path` return the cached path for an `images` row. A key that would leave the cache is refused.

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

- Order of checks: every manifest against its pin, then every table against its manifest sha256, then the `selection` check, then `check_split` for each dataset, then the images. A failure before the images downloads no image. `check_split` is a callable `(dataset, groups) -> None` that receives the groups in the dataset's `images.parquet`. Callers pass `coverage_checker(split_files)` for it (see [Splits](#splits)).
- `on_missing_images` is `"fail"` (default) or `"drop"` for each dataset. `drop` skips an image or mask that is absent and reports its ID in `DatasetSyncResult.dropped_image_ids`. A changed object always fails.
- Images and masks are fetched with `IfMatch` on the recorded ETag.
- Cached files are verified locally and never downloaded again on a mismatch. Manifests and tables are checked by sha256. Images and masks are checked by size and, for a single-part ETag, by MD5; a multipart ETag gets no MD5 check. When `images.parquet` records `crc64nvme` or `mask_crc64nvme`, sync also checks it with awscrt, on a cache hit and on a download. A row without a recorded CRC gets only the size and single-part MD5 checks.

## Errors

All derive from `errors.SegdataError`.

| Class | Raised when |
|---|---|
| `PublishRefused` | The tree is dirty or unpushed, `uv.lock` is missing, or the version exists. |
| `ObjectExists` | A write-once key exists with different bytes. |
| `SchemaValidationError` | A table fails its schema, or the cross-table rules. Also a `ValueError`. |
| `PinMismatch` | A manifest's sha256 differs from its pin. |
| `ChecksumMismatch` | A table, image or mask differs from its recorded checksum or size. |
| `ImageChanged` | An image no longer matches its recorded ETag. |
| `ImageMissing` | An image is absent and the dataset's mode is `fail`. |
| `SplitCoverageError` | A dataset version has a group that its split file does not list. |

`sync` also raises `ValueError` for a `selection` that names an unpinned dataset or an absent group. `read` raises `FileNotFoundError` for a file that is not cached.

## Tests

```bash
uv run --package segdata-spec --group tests pytest packages/segdata-spec/tests
```

The tests use an in-memory S3 fake. `pytest-socket` blocks every host except localhost.
