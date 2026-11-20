# segdata-spec

The versioned dataset contract between the ETL that publishes segmentation datasets and the trainer that reads them. Import package: `segdata_spec`.

## Scope

- **Owns:** the `manifest.json` and parquet table schemas, the S3 key layout, the local cache layout, publishing a version, syncing pinned versions into the cache, opening synced versions, and the benthic-attribute `roll_up`.
- **Does not own:** extracting or converting source data, choosing versions or pins, train/validation splits (the caller passes a `check_split` callback), or the image files themselves. Images stay where the ETL put them; the manifest's `image_store` names the bucket and prefix.
- **Boundary:** the package does not import `mermaidseg`, `coralnet_scraper` or `segdata_etl`, and `mermaidseg/inference` does not import it. The root `tests/test_package_boundaries.py` enforces this. `mermaidseg` declares it as a dependency but does not import it.

## Callers

| Caller | Calls |
|---|---|
| ETL | `publish.publish_dataset` and `publish.publish_taxonomy`. Each returns the manifest and its sha256, which becomes a `manifest.Pin`. |
| Trainer | `sync.sync(client, pins, cache_dir, bucket=...)`, which needs S3 access, then `read.open_version` and `read.open_taxonomy`, which read only the cache. |
| Both | `taxonomy.roll_up` and `taxonomy.hierarchy_from_table`. |

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

## Guarantees

Publish:

- Every put sends `If-None-Match: *`. An existing object with identical bytes counts as written; any other existing object raises `ObjectExists`.
- Table files are written first and `manifest.json` last. A prefix without a manifest is an incomplete publish.
- A version that already has a manifest is refused.
- A dirty tree (tracked files only), a `HEAD` that no remote branch contains, and a missing `uv.lock` are refused with `PublishRefused`. The manifest records the git SHA and the sha256 of `uv.lock`.
- The same tables always serialize to the same bytes.

Sync:

- Order of checks: every manifest against its pin, then every table against its manifest sha256, then the `selection` check, then `check_split` for each dataset, then the images. A failure before the images downloads no image.
- `on_missing_images` is `"fail"` (default) or `"drop"` for each dataset. `drop` skips an image or mask that is absent and reports its ID in `DatasetSyncResult.dropped_image_ids`. A changed object always fails.
- Images and masks are fetched with `IfMatch` on the recorded ETag.
- Cached files are verified locally and never downloaded again on a mismatch. Manifests and tables are checked by sha256. Images and masks are checked by size and MD5 against the ETag; a multipart ETag is not an MD5, so only the size is checked.

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

`sync` also raises `ValueError` for a `selection` that names an unpinned dataset or an absent group. `read` raises `FileNotFoundError` for a file that is not cached.

## Tests

```bash
uv run --package segdata-spec --group tests pytest packages/segdata-spec/tests
```

The tests use an in-memory S3 fake. `pytest-socket` blocks every host except localhost.
