# CoralNet ETL

Reproducible build of the three CoralNet parquet artifacts the training stack
depends on:

| Artifact | Purpose | Approx rows |
|---|---|---|
| `coralnet_audit_<version>.parquet` | per-source S3 audit (presence + counts + completeness) | ~1,400 |
| `coralnet_annotations_<version>.parquet` | merged point annotations across all complete sources | ~21M |
| `coralnet_images_<version>.parquet` | per-image dimensions + `needs_resize` flag (groundwork for issue #103) | ~1.2M |

Lives at [`mermaidseg/datasets/coralnet/etl/`](../mermaidseg/datasets/coralnet/etl/).
Closes [#79](https://github.com/data-mermaid/mermaid-segmentation/issues/79);
lays groundwork for [#103](https://github.com/data-mermaid/mermaid-segmentation/issues/103).

The `mermaid_confirmed_annotations.parquet` used by `MermaidDataset` is owned
externally (MERMAID API) and is not rebuilt here. Snapshot discipline for that
artifact lives in the MERMAID API project.

## Quickstart

Assumes `uv` is set up and AWS SSO is signed in.

```sh
export AWS_PROFILE=mermaid-core

# Dev smoke test (5 sources, low worker count) — finishes in a minute or two.
uv run coralnet-etl all \
  --limit-sources 5 \
  --output-dir ./outputs/coralnet-etl-dev \
  --workers 4

# Full run + upload to the canonical ETL output prefix.
uv run coralnet-etl all \
  --workers 32 \
  --upload-to-s3
```

## Subcommands

| Command | What it does |
|---|---|
| `audit` | Walk `s3://<bucket>/<prefix>/`, validate per-source presence of `annotations.csv` + `image_list.csv`, count rows, write audit parquet. |
| `build-annotations --audit <path>` | For every `is_complete` source in the audit parquet, merge annotations and image_list CSVs, derive `image_id` from the `Image Page` URL, write annotations parquet. |
| `build-images --annotations <path>` | For every unique `(source_id, image_id)` in the annotations parquet, read the JPEG header via S3 ranged GET, extract width/height, compute `needs_resize`. Writes checkpoint parts every `--checkpoint-every` rows. |
| `all` | All three in sequence. |

All commands accept `--bucket`, `--prefix`, `--output-dir`, `--upload-to-s3`,
`--workers`, `--version-tag`. Run with `-h` for the full list.

## Environment variables

| Variable | Default | Purpose |
|---|---|---|
| `AWS_PROFILE` | (required) | Must be `mermaid-core`. |
| `MERMAID_CORALNET_BUCKET` | `dev-datamermaid-sm-sources` | Source bucket. |
| `MERMAID_CORALNET_PREFIX` | `coralnet-public-images` | Source prefix. |
| `MERMAID_CORALNET_OUTPUT_PREFIX` | `etl-outputs/coralnet` | S3 prefix for `--upload-to-s3`. |
| `MERMAID_CORALNET_ETL_WORKERS` | `16` | Default `--workers` value. |
| `MERMAID_CORALNET_VERSION_OVERRIDE` | (unset) | Force a specific version tag. |
| `MERMAID_CORALNET_ANNOTATIONS_PATH` | (unset) | Override CoralNetDataset default with an exact S3 key. |
| `MERMAID_CORALNET_ANNOTATIONS_VERSION` | (unset) | Override CoralNetDataset default with `coralnet_annotations_<version>.parquet`. |

## Output layout

Local runs write under `--output-dir/<version>/`:

```
outputs/coralnet-etl/20260515_a1b2c3d/
├── coralnet_audit_20260515_a1b2c3d.parquet
├── coralnet_annotations_20260515_a1b2c3d.parquet
└── coralnet_images_20260515_a1b2c3d.parquet
```

`--upload-to-s3` mirrors each parquet to:

```
s3://${MERMAID_CORALNET_BUCKET}/${MERMAID_CORALNET_OUTPUT_PREFIX}/<version>/<filename>
```

## Runtime estimates

Measured on a `ml.t3.large` SageMaker notebook, mermaid-core profile:

| Stage | Wall time |
|---|---|
| `audit` (16 workers, ~1.4k sources) | ~5 min |
| `build-annotations` (16 workers, ~1.4k sources) | ~5 min |
| `build-images` (32 workers, ~1.2M unique JPEG keys) | ~30–60 min |

`build-images` is the dominant cost. Ranged GETs (~64 KB per image, ~1 KB max
after SOF) keep bandwidth low; the wall time is dominated by per-key request
latency. Checkpoint part files are written every `--checkpoint-every` rows
(default `50_000`) so an interrupted run resumes by reading the existing
checkpoint parts.

## Running on SageMaker

Use SageMaker for full rebuilds and overnight runs — IAM credentials rotate every
~60 minutes, which would require manual intervention for long local runs.

### Full ETL rebuild (`coralnet-etl`)

The `coralnet-etl` task runs `audit → build-annotations → build-images` in one
ProcessingJob and uploads versioned parquets to the canonical S3 prefix:

```bash
export AWS_PROFILE=wcs-launcher
uv run --extra sagemaker python scripts/launch_processing.py \
    --run-config sagemaker/runs/issue_130_coralnet_etl_audit.yaml \
    --config-dir sagemaker/runs/
```

### Image-list refresh (`coralnet-refresh`)

After an audit, any source where `image_list_covers_annotations=False` has a
truncated `image_list.csv` (fix for [#130](https://github.com/data-mermaid/mermaid-segmentation/issues/130)).
The `coralnet-refresh` task re-downloads those CSVs by scraping the CoralNet browse
listing. It has been run against the full set of affected sources; see
[`docs/sagemaker.md` — Processing Jobs](../docs/sagemaker.md#run-a-processing-job)
for the full runbook including concurrency guidance, the pre-flight reachability
probe, and CloudWatch monitoring.

**Quick concurrency rule:** use 2–3 `workers`, not more. CoralNet's browse pages
are server-side rendered; many simultaneous requests cause thundering-herd timeouts.
The task ships with startup jitter and login retries, but low concurrency is the
primary lever.

## Determinism

All three writers route through `write_parquet_deterministic` in
[`etl/io.py`](../mermaidseg/datasets/coralnet/etl/io.py). They sort rows by
primary key, cast every column to the declared pyarrow type, and emit zstd-3
parquet without `created_by` metadata or page statistics. Running the same
inputs twice produces byte-identical parquets (verified by
`tests/datasets/coralnet/test_images_pipeline.py::test_build_images_deterministic_bytes`).

## Schemas

See [`etl/schemas.py`](../mermaidseg/datasets/coralnet/etl/schemas.py) for
the authoritative pyarrow definitions. Highlights:

- **audit** — `source_id`, `audit_timestamp`, presence flags for the four
  CoralNet CSVs, row counts, `is_complete`, `image_count_match`, error list.
- **annotations** — `source_id`, `image_id`, `row`, `col`, `coralnet_id`,
  `status` (Confirmed / Unconfirmed / Unclassified — new vs. the 30112025
  parquet). Sorted by `(source_id, image_id, row, col)`.
- **images** — `source_id`, `image_id`, `s3_key`, `width`, `height`,
  `longest_edge`, `file_size`, `needs_resize`, `header_status`,
  `error_message`. Sorted by `(source_id, image_id)`.

## CSV reads: ibis + DuckDB httpfs

`audit` and `build-annotations` read every per-source CSV through an ibis
connection backed by DuckDB's `httpfs` extension (configured with
`CREATE OR REPLACE SECRET s3 (TYPE S3, PROVIDER CREDENTIAL_CHAIN)`). Same
pattern PR #80 used; we kept it for two properties:

1. `null_padding=True` tolerates ragged CoralNet CSVs that pandas' default
   parser rejects outright.
2. DuckDB httpfs streams CSV bytes directly, avoiding an extra
   `get_object`→`BytesIO`→`pd.read_csv` hop per file.

Each worker thread gets its own DuckDB connection (the engine isn't
thread-safe). Tests inject a boto3-backed `CsvReader` against the `FakeS3`
fixture so they don't need a real DuckDB instance.

Non-CSV S3 operations (paginator, `head_object`, images-folder counts) still
go through boto3 with adaptive retries.

## Failure handling

- 404 / `NoSuchKey` per source → row flagged in audit, not raised.
- Malformed CSV → ibis reads with `null_padding=True` so ragged rows parse
  cleanly. If the read still fails, `*_csv_read_failed=True` plus an `errors`
  entry.
- Corrupt JPEG → `header_status="corrupt"`, dimensions NULL — not a hard
  failure.
- SageMaker IAM credentials rotate every ~60 min:
  - Each audit worker thread refreshes its own boto3 session on a wall-clock
    timer (`_CRED_REFRESH_INTERVAL_SECONDS = 3000`, i.e. every 50 min) so the
    rotation never races against the 60-min SageMaker boundary.
  - DuckDB httpfs does NOT auto-refresh credentials. The audit loop re-runs
    `CREATE OR REPLACE SECRET s3 (TYPE S3, PROVIDER CREDENTIAL_CHAIN)` every
    `--checkpoint-every` sources on each worker's thread-local ibis connection.
  - The `build-images` checkpoint-part design means an interrupted run
    resumes without re-reading processed keys.

## Pointing CoralNetDataset at a new build

After uploading a new parquet:

```sh
export MERMAID_CORALNET_ANNOTATIONS_VERSION=20260515_a1b2c3d
```

`CoralNetDataset()` will read
`s3://${MERMAID_CORALNET_BUCKET}/coralnet_annotations_20260515_a1b2c3d.parquet`.
Without that variable set, it falls back to the legacy
`coralnet_annotations_30112025.parquet` so existing training runs continue to
work until a new default is published as the canonical S3 key.

## See also

- [Issue #79](https://github.com/data-mermaid/mermaid-segmentation/issues/79) — reproducible scripts for MERMAID and CoralNet annotation parquets.
- [Issue #103](https://github.com/data-mermaid/mermaid-segmentation/issues/103) — image-size preprocessing for training speed.
- `mermaidseg/datasets/coralnet/nbs/CoralNet_Annotations.ipynb` — historical notebook the ETL supersedes.
