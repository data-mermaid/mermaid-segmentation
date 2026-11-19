# coralnet-scraper

Downloads public CoralNet sources (images, labelsets, annotations) to S3. Import package: `coralnet_scraper`; console script: `coralnet-scrape`.

## Scope

- **Owns:** recreating the raw CoralNet dataset in S3, one folder per public source.
- **Does not own:** turning a scrape into the training manifest that `CoralNetDataset` (`mermaidseg/datasets/coralnet/coralnet_dataset.py`) reads. That is [`coralnet-dataset`](../coralnet-dataset/README.md).
- **End to end:** to recreate or update the CoralNet data, scrape through training, follow the [CoralNet data runbook](../../docs/runbooks/coralnet-data.md).
- **Boundary:** `mermaidseg` does not depend on or import this package, and the package does not import `mermaidseg`. The root `tests/test_package_boundaries.py` enforces both (run with the root suite). A root `uv sync` does not install it.

## S3 layout

Each source is written under `s3://<bucket>/<prefix>/s<source_id>/`:

| Object | Content |
|---|---|
| `metadata.csv` | Classifier history from the source overview page. Absent when the page has none. |
| `labelset.csv` | `Label ID`, `Name`, `Short Code`. Absent when the source has no labels. |
| `annotations.csv` | CoralNet annotation export of the whole source. |
| `image_list.csv` | One row per image from the browse pages; the `Image URL` column is empty. |
| `images/<image_id>.jpg` | One object per image, keyed by the numeric CoralNet image id. |

Each run also writes one run-level object under `s3://<bucket>/<prefix>/_runs/`:

| Object | Content |
|---|---|
| `_runs/<UTC time>-<short id>.json` | The [run report](#run-report) of one run. |

`<bucket>` and `<prefix>` come from the environment variables below.

## Run

Run from the repository root. `uv run --package coralnet-scraper` installs the package and its dependencies on first use:

```bash
uv run --package coralnet-scraper coralnet-scrape
```

Flags:

| Flag | Default | Effect |
|---|---|---|
| `--force` | off | Process every source, including those already on S3. |
| `--delay-seconds` | `0.0` | Pause after each source not skipped by the resume check. |
| `--scraper-workers` | auto | Concurrent image download and upload workers. Overrides `MERMAID_CORALNET_SCRAPER_WORKERS`. |
| `--source-ids` | all public sources | Comma-separated CoralNet source ids. The run processes only these, in the given order, after source discovery. An id that is not in the public listing is still attempted. |
| `--requests-per-second` | no limit | Maximum request rate, a positive number. The limit applies to the first attempt of each CoralNet request and is shared by all threads. Retries wait for their backoff or the server's `Retry-After` and do not count against the limit. Image downloads from S3 pre-signed URLs are never limited. Overrides `MERMAID_CORALNET_REQUESTS_PER_SECOND`. |
| `--export-workers` | `3` | Concurrent page exports for sources with more than 5000 images. Overrides `MERMAID_CORALNET_EXPORT_WORKERS`. |
| `-v`, `--verbose` | off | Debug logging. |

The command logs in to CoralNet, discovers the public source ids from `https://coralnet.ucsd.edu/source/about/`, and downloads each source. A failed login ends the run with exit code 1.

An exception inside one source, in the skip check or the download, is logged and recorded under its exception class name. The run continues with the next source.

A source fails when:

- any of its images had no image URL (`images_missing_url`) or failed to upload (`images_upload_failed`);
- the metadata, labelset, annotations, or image list step failed, or the source could not be accessed;
- CoralNet login failed for the source (`login failed`);
- the source overview page could not be fetched (`overview fetch failed: <exception class>: <message>`);
- an exception was raised for it, as above.

At the end the run logs a summary at error level: the number of failed sources, then one line per error kind with the sorted source ids. Image kinds give the image count per source, for example `images_missing_url: 202 (14 images)`. A failed source with no error code is listed under `no_error_code`.

Exit code:

| Code | When |
|---|---|
| 0 | Every source downloads or is skipped. A source with zero Confirmed images, or with zero images, counts as skipped. |
| 1 | The CoralNet login fails, any source fails, the [circuit breaker](#retries-and-circuit-breaker) stops the run, the [run report](#run-report) cannot be written to S3 (after the transport figures and failure summary are logged, the run logs `Could not write run report s3://<bucket>/<key>`), or the run stops on an uncaught error (invalid configuration, source discovery failure). |

### Retries and circuit breaker

- **Retries.** A GET request is retried up to 5 times on timeouts, dropped connections, and HTTP 429, 500, 502, 503 and 504. The wait grows exponentially with jitter and honours the `Retry-After` header. A request that could not connect is retried whatever its method, because it was never sent.
- **Circuit breaker.** After 5 consecutive failed sources the run stops. The summary's first line is `Run stopped early: circuit_open` and the exit code is 1. A source skipped by the resume check neither counts toward the streak nor resets it. Running the same command again continues with the remaining sources.

### Run report

After the batch loop finishes or the circuit breaker stops it, the run writes a JSON report to `s3://<bucket>/<prefix>/_runs/<UTC time>-<short id>.json`. A failed login or failed source discovery ends the run before this point, so it writes no report.

| Field | Content |
|---|---|
| `settings` | `scraper_workers`, `export_workers`, `requests_per_second` (`null` = no limit), `source_ids` (`null` = every public source). |
| `started_at`, `finished_at` | Run start and end, ISO 8601 UTC. |
| `duration_s` | Run time in seconds. |
| `sources` | Per attempted source id: `{"ok": bool, "errors": [codes]}`. Sources skipped by the resume check are absent. |
| `stopped` | `null`, or why the run stopped early: `circuit_open`. |
| `transport` | `requests`: requests sent, retries included. `mean_rate_per_s`: `requests` divided by the run time. `retries`: retries by cause: `"429"`, each 5xx code, `"timeout"`, `"connection"`. `final_status`: final responses per status code. |

The run log prints the same transport figures at the end. To choose a request rate from these figures, follow [Tune the request rate](../../docs/runbooks/coralnet-data.md#tune-the-request-rate).

### Environment variables

Nothing loads `.env` for you: neither the package nor `uv run`. Copy the variables from `.env.example` into `.env`, then export them in your shell before the command:

```bash
set -a; source .env; set +a
```

| Variable | Required | Default | Meaning |
|---|---|---|---|
| `MERMAID_CORALNET_BUCKET` | yes | none | Destination bucket. An unset or empty value raises `ValueError`. |
| `MERMAID_CORALNET_PREFIX` | no | `coralnet-public-images` | Key prefix. Surrounding slashes are stripped; an empty prefix raises `ValueError`. |
| `MERMAID_CORALNET_SCRAPER_WORKERS` | no | `min(32, max(8, 4 * CPU count))` | Image worker count. Must be an integer of at least 1. |
| `MERMAID_CORALNET_REQUESTS_PER_SECOND` | no | no limit | Maximum request rate. Must be a positive number; 0, a negative number or a non-numeric value raises `ValueError`. |
| `MERMAID_CORALNET_EXPORT_WORKERS` | no | `3` | Page export worker count. Must be an integer of at least 1. |
| `MERMAID_CORALNET_USER_AGENT` | no | `mermaid-coralnet-scraper (+https://github.com/data-mermaid/mermaid-segmentation)` | `User-Agent` header of every request. |
| `CORALNET_USERNAME` | no | prompt | CoralNet login. Used only when `CORALNET_PASSWORD` is also set. |
| `CORALNET_PASSWORD` | no | prompt | CoralNet password. |

A flag overrides its environment variable.

Without both credential variables the command prompts for the username and password on the terminal.

S3 access uses the default boto3 credential chain, for example `AWS_PROFILE`.

### Resume and `--force`

- A source is skipped when `<prefix>/s<id>/annotations.csv` exists in S3. The check is made before any download.
- A source runs metadata, labelset, annotations, image list, then images. A failure in the metadata or labelset step does not stop the run, so `annotations.csv` is still uploaded. A source that failed with `annotations.csv` already uploaded is skipped on the next run unless `--force` is set. That covers `metadata_download_failed`, `labelset_failed`, `image_list_failed`, `images_missing_url`, `images_upload_failed`, and a source that stopped after the annotations upload. Find these sources in the failure summary logged at the end of the run. To retry one of them, delete its `annotations.csv` from S3. `--force` retries every source.
- Sources with zero Confirmed images, or zero images, on CoralNet are skipped, and nothing is written for them.
- `--force` downloads every source. Image objects already in S3 are not uploaded again.
- A source with more than 5000 images is exported in pages. It loads an existing `annotations.csv` and merges new rows into it, and uploads a partial file every 50 pages. A source whose export fails (`annotations_failed`) or is interrupted after the first checkpoint leaves that partial `annotations.csv`, which the resume rule skips until `--force`. If the existing file already covers as many images as CoralNet reports, the export is skipped, even with `--force`, so new annotations on existing images are not picked up.

## Tests

```bash
uv run --package coralnet-scraper --group tests pytest packages/coralnet-scraper/tests
```

The tests are offline. They use the HTML fixtures in `tests/fixtures/coralnet_html/` and no network.
