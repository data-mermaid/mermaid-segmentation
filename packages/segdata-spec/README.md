# segdata-spec

The versioned dataset contract between the ETL that publishes segmentation datasets and the trainer that reads them. Import package: `segdata_spec`.

## Scope

- **Owns:** the `manifest.json` and parquet table schemas, the S3 key layout, the local cache layout, publishing a version, syncing pinned versions and their image files into the cache (resizing them there when given a resize spec) or their tables alone, publishing and syncing feature sets, opening synced versions, the benthic-attribute `roll_up`, and the split-file format with its coverage check.
- **Does not own:** extracting or converting source data, computing features or writing feature files (the spec reads and verifies feature files and never writes them), choosing versions or pins, choosing the train/val/test split (the package reads and checks a split file; the split is built elsewhere), or publishing and owning the original image files. The ETL puts the originals in S3; the manifest's `image_store` names the bucket and prefix.
- **Boundary:** the package does not import `mermaidseg`, `coralnet_scraper` or `segdata_etl`, and `mermaidseg/inference` does not import it. The root `tests/test_package_boundaries.py` enforces this. `mermaidseg` imports it (`mermaidseg/data/`).

## Callers

| Caller | Calls |
|---|---|
| ETL | `publish.publish_dataset` and `publish.publish_taxonomy`. Each returns the manifest and its sha256, which becomes a `manifest.Pin`. `features.publish_feature_set` returns a `manifest.FeatureSetPin` (dataset `Pin`, family, feature-set manifest sha256). A feature writer also calls `features.claim_extractor` before it writes any feature file, and names files with `features.feature_file_key`. |
| Trainer | `sync.sync(client, pins, cache_dir, bucket=..., resize=..., max_workers=...)`, which needs S3 access; `sync.check_published_manifest(client, bucket, pin, taxonomy=...)`, which reads the manifest without caching it; then `read.open_version(..., resize=...)`, `read.open_tables` and `read.open_taxonomy`, which read only the cache. For tables without images: `sync.sync_tables` (needs S3 access) in place of `sync.sync`. For features: `sync.sync_feature_set` (needs S3 access) and `read.open_feature_set` (cache only), both taking a `manifest.FeatureSetPin`. Also `resize` (`ResizeSpec`, `scaled_size`, `resize_image`, `resize_mask`, `scale_points`, `scale_images`). `scale_images` returns the images table with the resized `width` and `height` and `media_type` `image/jpeg`; `open_version(resize=)` uses it. |
| Both | `taxonomy.roll_up` and `taxonomy.hierarchy_from_table`. |
| Split builder and trainer | `splits.read_split_file`. The trainer also passes `splits.coverage_checker` to `sync`. |

`resize` on `sync` maps a dataset name to a `ResizeSpec` or `None`. A dataset that is absent or maps to `None` is cached as is. `DatasetSyncResult.resize` records the spec that was applied, for model lineage. After each dataset version syncs, `sync` writes a marker (see Cache layout) that records the resize spec, the synced groups and the dropped image ids. `open_version(..., resize=...)` raises `NotSynced` when the version has no marker for that spec, and the message names the requested resize id and the ids that do have a marker. `DatasetVersion.synced_groups` holds the groups whose images are cached, or `None` when all of them are. Sync never deletes cached images, so `open_version` unions the groups of every marker for the version and spec: a later sync adds its groups, and a sync whose selection names no groups adds none. `ResizeSpec` is a training-cache step only: evaluation and serving resize any input to the model's `input_size`, so inference does not apply it. A `ResizeSpec` holds `max_edge`, `filter` (`lanczos`, `bicubic` or `bilinear`) and `jpeg_quality`; its `id` is the first 12 hex characters of a hash of the three fields.

`sync_tables(client, pins, cache_dir, *, bucket=DATASET_BUCKET, taxonomy=None, check_split=None)` fetches and verifies manifests and tables only. No image or mask is requested, and the marker goes to the `tables` namespace. It takes no `selection`, `resize` or `on_missing_images`. `check_split` still runs, before any marker is written.

| Function | Key parameters | Verifies |
|---|---|---|
| `read.open_tables(cache_dir, bucket, pin)` | | Raises `NotSynced` without a tables-only or any image marker. Re-verifies the manifest against `pin`. Returns `VersionTables` (`manifest`, `images`, `points`, `labels`, `label_counts`); `images` and `points` describe the originals. |
| `sync.sync_feature_set(client, pin, cache_dir, *, bucket=DATASET_BUCKET, image_ids=None, max_workers=None)` | `pin` is a `FeatureSetPin`; `image_ids` selects images, `None` syncs all | Set manifest against `pin.manifest_sha256`, its recorded dataset sha256 against `pin.dataset`, the index against its manifest entry, then each selected file against its index row: size, ETag, and CRC64NVME when recorded. Cached files are verified, not downloaded again. Returns `FeatureSyncResult` (`manifest`, `index`, `paths` by image id). |
| `read.open_feature_set(cache_dir, bucket, pin)` | `pin` is a `FeatureSetPin` | Raises `NotSynced` without a feature marker. Re-verifies the set manifest against `pin.manifest_sha256` and its dataset sha256 against `pin.dataset`. Returns `FeatureSet` (`manifest`, `index`, `path(image_id)`); `path` resolves any indexed image, and only the images `sync_feature_set` selected are cached. |

`sync` runs on `max_workers` threads (default `min(8, os.cpu_count())`). Create the boto3 client with `max_pool_connections` of at least `max_workers`.

## S3 layout

```text
datasets/<name>/<version>/manifest.json
datasets/<name>/<version>/<table>.parquet      images, labels, points (points datasets only), stats/label_counts
datasets/<name>/<version>/features/<family>/index.parquet
datasets/<name>/<version>/features/<family>/manifest.json
taxonomy/<name>/<version>/manifest.json
taxonomy/<name>/<version>/benthic_attributes.parquet
taxonomy/<name>/<version>/growth_forms.parquet
features/<store>/<family>/_extractor_spec.json
features/<store>/<family>/<image_id>[~<suffix>].featurevector
```

Images and masks are not under these prefixes. Their keys are `image_store.prefix` plus the `key` and `mask_key` columns of `images.parquet`.

`images.parquet` also has a nullable `source_name`: the image's name at its source. It is data only and is never part of a key. Publishers must supply the column (nulls allowed); `publish_dataset` refuses a table without it. `sync` and `open_version` accept a version published without the column; its `images` table then has no `source_name`.

## Feature sets

A feature set holds one pyspacer feature file per image, computed by one extractor, for one dataset version. The files and the per-version index are stored separately.

- **Files:** `features/<store>/<family>/<image_id>[~<suffix>].featurevector`. `feature_store_name(image_store, bucket)` returns `<store>`: the image store must be in the dataset bucket with prefix exactly `images/<store>/`, else `ValueError`. Versions that share an image store share feature files. `feature_file_key(image_store, bucket, family, image_id, suffix=None)` returns a file's key. The family's `_extractor_spec.json` sits beside the files. The writer puts the feature files; the spec never writes a `.featurevector`.
- **Writers:** call `claim_extractor(client, bucket, image_store, family, extractor)` before writing any `.featurevector` into a family, and put each feature file with `If-None-Match: *`. `claim_extractor` writes `_extractor_spec.json` if it is absent, or checks that the stored one has the same four identity values (see the `extractor` check below), else `ExtractorMismatch`. `publish_feature_set` calls it too.
- **Index:** `datasets/<d>/<v>/features/<family>/index.parquet` lists the files that belong to the version. `manifest.json` in the same prefix is written last. It records the dataset `Pin`, the family, the extractor spec, the index's sha256, row count and size, `stats` (`n_images`, `n_missing`) and the ETL provenance. Its sha256 pins the set. `FeatureSetManifest`, `FeatureSetStats` and `FeatureSetPin` are in `manifest.py`. A `FeatureSetPin` holds the dataset `Pin`, the family and the set manifest's sha256.
- **Names:** `store`, `family` and `suffix` match `[a-z0-9][a-z0-9._-]*` and contain no `..`. `image_id` is not checked against this pattern.

Immutability:

- A published version's manifest and tables never change.
- Feature sets are add-only. Each set is write-once and pinned by its own sha256, and publishing one does not touch the dataset manifest or any earlier set.
- A wrong or incomplete feature set cannot be rewritten. Publish the corrected one under a new family name.
- A reader fetches only what a manifest names.

`index.parquet` columns (`schema.FEATURE_INDEX_SCHEMA`):

| Column | Type | Null |
|---|---|---|
| `image_id` | string | no |
| `key` | string | no |
| `etag` | string | no |
| `size` | int64 | no |
| `crc64nvme` | string | yes |
| `n_points` | int32 | no |

`publish_feature_set(client, pin, family, index, extractor, *, bucket, etl, allow_missing=False)` returns a `FeatureSetPin`. All checks run before the first write:

- The dataset manifest is verified against `pin` and `images.parquet` against the manifest.
- `index` conforms to `FEATURE_INDEX_SCHEMA`, else `SchemaValidationError`.
- The dataset's image store is `images/<store>/` in `bucket`, else `ValueError`.
- Every `image_id` is an image of the version, and none repeats.
- Every `key` is that image's own file under `features/<store>/<family>/`: `<image_id>.featurevector` or `<image_id>~<suffix>.featurevector`. No two rows share a `key`.
- A version image the index leaves out is refused unless `allow_missing` is true. The count is recorded in `stats.n_missing`.
- `extractor` holds `class_path`, `crop_size`, `feature_dim` and `weights_sha256`, else `ValueError`. If `_extractor_spec.json` exists, it must have the same four values, else `ExtractorMismatch`; its other keys may differ and the stored file is kept. Otherwise it is written from `extractor`.
- `etl` is recorded in the manifest as given. No git check runs.
- `_extractor_spec.json`, `index.parquet` and then `manifest.json` are put with `If-None-Match: *`, as in Guarantees.

The publish does not look at the feature files: it does not check that they exist, that `etag` or `size` match them, or that `n_points` matches the image's points. Point coverage is the writer's job.

## Cache layout

Without a resize spec, `sync` writes every object to `<cache_dir>/<bucket>/<key>`, the same key as on S3, so `read` finds a file from the manifest alone. Manifests and tables always use this layout.

With a spec, `sync` writes each image and mask to `<cache_dir>/resized/<spec id>/<bucket>/<key>` and does not cache the original. Each output has an index file at `<cache_dir>/resized/<spec id>/.index/<bucket>/<key>.json` with the source ETag, the output size, the output sha256 and the output's pixel width and height. Two specs share one cache without interfering.

After a dataset version syncs, `sync` writes a new marker file at `<cache_dir>/synced/<spec id or original>/<bucket>/<dataset>/<version>/<uuid>.json`, with `original` for a dataset that has no spec. The file holds the resize spec (or null), the synced groups (`"all"` when the selection is `None`) and the dropped image ids of that sync. It is written atomically once that version's images verify, so an interrupted multi-dataset sync leaves no marker for the version it was syncing, while markers for versions that finished earlier remain, and `sync` never reads or changes an existing marker, so concurrent syncs of one version do not overwrite each other. `open_version` reads every marker file in the directory and requires at least one. Its groups are the union of the files' groups, where `"all"` absorbs the rest. A marker file that cannot be parsed raises `CorruptSyncMarker` naming its path.

`sync_tables` writes its marker to `<cache_dir>/synced/tables/<bucket>/<dataset>/<version>/<uuid>.json` instead, in the same format with no groups and no dropped ids. `open_version` never reads it, so it raises `NotSynced` for a version synced only that way; `open_tables` accepts it, or the marker of any resize id.

`sync_feature_set` writes `<cache_dir>/synced/features/<family>/<bucket>/<dataset>/<version>/<uuid>.json` after every selected feature file verifies. It holds `feature_set_sha256` and `image_ids` (the synced ids, or `"all"`). `open_feature_set` requires at least one such file for the family. `tables` and `features` are reserved marker namespaces, never resize ids.

A resized file keeps the source key name, so a key ending `.png` may hold JPEG bytes. Read the format from `media_type` in the images table that `open_version(resize=...)` returns, not from the file extension.

`DatasetVersion.image_path` and `mask_path` return the cached path for an `images` row, in the layout of the version's `resize`. A key that would leave the cache is refused.

## Resize rule

- Images are decoded with JPEG draft mode, then scaled with Lanczos (or the spec's filter) to `int(edge * (max_edge / longest))` on each edge. Because the float product is truncated, a few sizes come out one pixel short, for example 2215 to 2047 at `max_edge=2048`.
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
| `PinMismatch` | A manifest's sha256 differs from its pin, or a feature-set manifest differs from its sha256 or records another dataset manifest sha256. |
| `ManifestNotFound` | A pinned manifest is absent from the bucket. |
| `ChecksumMismatch` | A table, feature index, feature file, image or mask differs from its recorded checksum or size, or a cached resized output has no valid index entry or differs from it. |
| `ImageChanged` | An image or feature file no longer matches its recorded ETag. |
| `ImageMissing` | An image is absent and the dataset's mode is `fail`. |
| `SplitCoverageError` | A dataset version has a group that its split file does not list. |
| `ExtractorMismatch` | A family's stored `_extractor_spec.json` differs from the given extractor in `class_path`, `crop_size`, `feature_dim` or `weights_sha256`. |
| `FeatureIndexInvalid` | A feature index names an image outside the version, repeats one, has a `key` that is not its image's file, repeats a `key`, or leaves images out without `allow_missing`. Also `sync_feature_set` with an `image_ids` entry absent from the index. |
| `FeatureMissing` | A selected feature file is absent from the bucket. |
| `NotSynced` | `open_version` finds no marker for the resize spec (including a tables-only sync), `open_tables` finds neither a tables-only nor an image marker, or `open_feature_set` finds no feature marker. |
| `CorruptSyncMarker` | A marker file cannot be parsed. |
| `ImageUndecodable` | An image or mask matches its checksums but cannot be decoded (for example, a format other than JPEG or PNG). Names the S3 key. |

`sync` also raises `ValueError` for a `selection` that names an unpinned dataset or an absent group, a `resize` that names an unpinned dataset, or `max_workers` below 1. `read` raises `FileNotFoundError` for a manifest or table that is not cached. `read` and `open_version` raise `ValueError` when a point names an `image_id` absent from `images` (resize set).

## Tests

```bash
uv run --package segdata-spec --group tests pytest packages/segdata-spec/tests
```

The tests use an in-memory S3 fake. `pytest-socket` blocks every host except localhost.
