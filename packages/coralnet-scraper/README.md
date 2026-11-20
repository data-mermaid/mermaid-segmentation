# coralnet-scraper

Downloads public CoralNet sources (images, labelsets, annotations) to S3. Import package: `coralnet_scraper`; console script: `coralnet-scrape`.

## Scope

- **Owns:** recreating the raw CoralNet dataset in S3, one folder per public source.
- **Does not own:** turning a scrape into the training manifest that `CoralNetDataset` (`mermaidseg/datasets/coralnet/coralnet_dataset.py`) reads. That is [`segdata-etl`](../segdata-etl/README.md).
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
| `_complete.json` | The [completion marker](#resume). Written last, after verification. `segdata-etl` reads only the sources that have it. |
| `_partial/` | Progress of a source that is not complete. Deleted after the marker is written. |

Each run also writes run-level objects under `s3://<bucket>/<prefix>/`:

| Object | Content |
|---|---|
| `_runs/<UTC time>-<short id>.json` | The [run report](#run-report) of one run. |
| `_meta/discovered_source_ids.json` | The source ids listed by CoralNet in the previous run. Used by the [discovery guard](#discovery-guard). |

`<bucket>` and `<prefix>` come from the environment variables below.

## Run

Run from the repository root. `uv run --package coralnet-scraper` installs the package and its dependencies on first use:

```bash
uv run --package coralnet-scraper coralnet-scrape
```

Flags:

| Flag | Default | Effect |
|---|---|---|
| `--delay-seconds` | `0.0` | Pause after each source not skipped by the [resume](#resume) check. |
| `--scraper-workers` | auto | Concurrent image download and upload workers. Overrides `MERMAID_CORALNET_SCRAPER_WORKERS`. |
| `--source-ids` | all public sources | Comma-separated CoralNet source ids. The run processes only these, in the given order, after source discovery. An id that is not in the public listing is still attempted. |
| `--requests-per-second` | no limit | Maximum request rate, a finite positive number. The limit applies to the first attempt of each CoralNet request and is shared by all threads. Retries wait for their backoff or the server's `Retry-After` and do not count against the limit. Image downloads from S3 pre-signed URLs are never limited. Overrides `MERMAID_CORALNET_REQUESTS_PER_SECOND`. |
| `--export-workers` | `3` | Concurrent page exports for sources with more than 5000 images. Overrides `MERMAID_CORALNET_EXPORT_WORKERS`. |
| `--prune` | off | After a source verifies and before its marker is written, delete the image objects that are not in the source's image list. Sources that already have a marker are not touched. Deletes cannot be undone unless the bucket has S3 versioning enabled. Read the `orphan_image_objects:<n>` [warning](#verification) in the source's marker first. Default run only; a subcommand rejects it with exit code 2. |
| `--allow-source-drop` | off | Continue when CoralNet lists fewer than 95% of the sources of the previous run. See [Discovery guard](#discovery-guard). Default run only; a subcommand rejects it with exit code 2. |
| `-v`, `--verbose` | off | Debug logging. |

Every run flag is safe to repeat: running the same command again after a failure or a kill resumes the work. Operations that change state are subcommands:

| Command | Effect |
|---|---|
| `coralnet-scrape verify-existing [--source-ids <ids>]` | Verifies each source that has no marker against CoralNet's live image count, and writes the marker for each source that passes. Without `--source-ids`, it checks every discovered source that has no marker. Reports `overview_total_missing` for a source whose overview page shows no image count. Used to adopt a bucket scraped before markers existed. Reads no credentials and does not log in. |
| `coralnet-scrape invalidate --source-ids <ids>` | Deletes the marker and `_partial/` of the listed sources. |
| `coralnet-scrape invalidate --all` | Deletes the marker and `_partial/` of every source. The next plain run exports every source again. |
| `coralnet-scrape invalidate --changed` | Deletes the marker and `_partial/` of each marked source whose live CoralNet image count differs from the count in its marker. |

`invalidate` changes state only. Run the plain command afterwards to download. `invalidate` reads no credentials and does not log in. It deletes the marker and `_partial/`, which cannot be undone unless the bucket has S3 versioning enabled.

There is no `--force` flag.

A plain run on a bucket that has source folders but no marker in any of them logs a warning at the start. The warning points to `verify-existing`; without it every source downloads again. See [Adopt an existing bucket](../../docs/runbooks/coralnet-data.md#adopt-an-existing-bucket).

The run logs in to CoralNet, discovers the public source ids from `https://coralnet.ucsd.edu/source/about/`, and downloads each source. A failed login ends the run with exit code 1. CoralNet overview pages are public, so `verify-existing` and `invalidate` do not log in.

An exception inside one source, in the skip check or the download, is logged and recorded under its exception class name. The run continues with the next source.

A source fails when:

- any of its images had no image URL (`images_missing_url`) or failed to upload (`images_upload_failed`). These still fail the source and set exit code 1, but do not count toward the [circuit breaker](#retries-and-circuit-breaker);
- the metadata, labelset, annotations, or image list step failed;
- the last [browse walk attempt](#retries-and-circuit-breaker) found a different number of images than the overview reports (`walk_count_mismatch:<found>!=<total>`). Only the saved walk is discarded and the next run walks again. The export and the images are not attempted. This failure does not count toward the circuit breaker;
- the source is not accessible (`cannot access source <id>: <reason>`). The reason is CoralNet's page text, for example `Permission denied`, or `HTTP 403` or `HTTP 404` for a missing or private source. This failure does not count toward the [circuit breaker](#retries-and-circuit-breaker);
- verification of the source failed (`verification_failed`, with the [codes](#verification));
- the overview page shows no "Total images" count (`overview_total_missing`; `verify-existing` reports it too);
- CoralNet login failed for the source (`login failed`);
- the source overview page could not be fetched (`overview fetch failed: <exception class>: <message>`, for 5xx responses, timeouts and connection errors). This failure counts toward the circuit breaker;
- an exception was raised for it, as above.

At the end the run logs a summary at error level: the number of failed sources, then one line per error kind with the sorted source ids. Image kinds give the image count per source, for example `images_missing_url: 202 (14 images)`. A failed source with no error code is listed under `no_error_code`.

Exit code:

| Code | When |
|---|---|
| 0 | Every source downloads or is skipped. A source with a marker, with zero Confirmed images, or with zero images counts as skipped. |
| 1 | The CoralNet login fails, the source listing is empty, any source fails, the [circuit breaker](#retries-and-circuit-breaker) stops the run, the [run report](#run-report) cannot be written to S3, even when every source succeeded (after the transport figures and failure summary are logged, the run logs `Could not write run report s3://<bucket>/<key>`), the [discovery guard](#discovery-guard) stops the run (no run report is written), or the run stops on an uncaught error (invalid configuration, source discovery failure). |

Exit code of the subcommands:

| Command | Code | When |
|---|---|---|
| `verify-existing` | 0 | Every checked source verifies, or is skipped because it has a marker. |
| `verify-existing` | 1 | Any source fails verification, or verification raises for it. Each failure is logged as an error. |
| `invalidate` | 0 | The selected sources are invalidated. |
| `invalidate` | 1 | `--changed` could not read the live image count of some source. The sources that could be checked are still invalidated. |

### Response checks

The scraper accepts a response only when its content is valid. A status of 200 is not enough.

- **Annotation export.** The CSV is decoded as UTF-8, because CoralNet sends `text/csv` without a charset. An export must be CSV with a `Name` column, no null names, no HTML columns and no column whose name starts with `Unnamed`. It must contain only the requested images.
- **Browse walk.** The walk rejects a page that lists no images. It compares the thumbnail count of a page to page 1 only when the page has a Next link. It also checks repeated image ids, and the final image count against the source overview. The overview is read again after the walk.
- **Logged-out answers.** These answers log the session in again: a browse page without the export form, an `export_prep` answer of "must be signed in", and a serve request that lands on the login page.
- **Images.** An image response must have a 2xx status, an `image/*`, `binary/octet-stream` (S3's default) or `application/octet-stream` content type, and the full `Content-Length`. Pillow must open and fully decode it. Only JPEG and PNG are accepted; an MPO file opens through the JPEG reader and is accepted. The body may be at most 256 MiB, declared or read. The download has a 300 s deadline. The scraper never uploads an invalid body. The stored `ContentType` comes from the detected format.

### Retries and circuit breaker

- **Retries.** A GET request is retried up to 5 times on timeouts, dropped connections, and HTTP 429, 500, 502, 503 and 504 that occur before the response headers arrive. A connection dropped while the body is being read is not retried. The wait grows exponentially with jitter and honours the `Retry-After` header. A request that could not connect is retried whatever its method, because it was never sent.
- **Content-level retries.** A response that fails a [response check](#response-checks) is retried as a unit:

  | Unit | Attempts |
  |---|---|
  | Browse walk | 3 |
  | Export round trip, including the POST | 4 |
  | Image: resolve the pre-signed URL and download | 4 |
  | Login | 4 |

  The image unit repeats the resolve step, so an expired pre-signed URL is resolved again. Waits are 5 to 120 s with jitter. There is no cap on total retry time. A page-level walk failure resumes from the last walk checkpoint. A count mismatch redoes the walk. These retries sit on top of the transport retries above, so one image can make up to about 4 × 6 requests. The first bulk `export_prep` read timeout is sized from the estimated annotation rows: the overview's image count times its points per image (100 when the overview gives none), at 250 rows/s plus 120 s, within 300 to 7200 s. After an `export_prep` read timeout, the next attempt waits an extra 60 s, because CoralNet may still be building the export. Once one page export fails, the other pages make no further attempts.
- **Image-stage stop.** After 20 images in a row fail, the scraper skips the remaining queued images and the source gets `images_outage`.
- **Circuit breaker.** After 5 consecutive outage-type failures the run stops. Outage-type failures are a raised exception, a failure with no error code, `login failed`, `overview fetch failed …`, `annotations_failed`, `image_list_failed` and `images_outage`. Other failed sources, for example `cannot access source <id>: <reason>`, `verification_failed` or image failures, neither count toward the streak nor reset it. A source that is ok resets it. The summary's first line is `Run stopped early: circuit_open` and the exit code is 1. A source skipped because it is complete neither counts toward the streak nor resets it. Running the same command again continues with the remaining sources.

### Run report

After the batch loop finishes or the circuit breaker stops it, the run writes a JSON report to `s3://<bucket>/<prefix>/_runs/<UTC time>-<short id>.json`. A failed login, failed source discovery or a [discovery guard](#discovery-guard) stop ends the run before this point, so it writes no report.

| Field | Content |
|---|---|
| `settings` | `scraper_workers`, `export_workers`, `requests_per_second` (`null` = no limit), `source_ids` (`null` = every public source), `prune` and `allow_source_drop` (booleans). |
| `started_at`, `finished_at` | Run start and end, ISO 8601 UTC. |
| `duration_s` | Run time in seconds. |
| `sources` | Per attempted source id: `{"ok": bool, "errors": [codes]}`. Skipped sources are absent. |
| `skipped` | `{"complete": count, "no_confirmed": [ids], "no_images": [ids]}`. `complete` counts the sources skipped because they have a marker. `no_confirmed` and `no_images` list the sources with zero Confirmed images and with zero images. |
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
| `MERMAID_CORALNET_REQUESTS_PER_SECOND` | no | no limit | Maximum request rate. Must be a finite positive number; 0, a negative number, infinity or a non-numeric value raises `ValueError`. |
| `MERMAID_CORALNET_EXPORT_WORKERS` | no | `3` | Page export worker count. Must be an integer of at least 1. |
| `MERMAID_CORALNET_USER_AGENT` | no | `mermaid-coralnet-scraper (+https://github.com/data-mermaid/mermaid-segmentation)` | `User-Agent` header of every request. |
| `CORALNET_USERNAME` | no | prompt | CoralNet login. Used only when `CORALNET_PASSWORD` is also set. |
| `CORALNET_PASSWORD` | no | prompt | CoralNet password. |

A flag overrides its environment variable.

Without both credential variables the run prompts for the username and password on the terminal. `verify-existing` and `invalidate` read no credentials.

S3 access uses the default boto3 credential chain, for example `AWS_PROFILE`.

### Resume

- A source is complete only when `s<id>/_complete.json` exists. The marker is written last, after the source is verified against CoralNet's image count. A complete source is skipped without any download.
- Progress inside a source is kept under `s<id>/_partial/`:
  - `state.json`: the image count of the source overview at the start, the finished stages, and the update time (`total_images`, `stages`, `updated_at`).
  - `walk.json`: the browse walk, saved every 50 pages.
  - `page-<n>.csv`: one chunk for each exported page of a source with more than 5000 images.
- The next run reuses `_partial/` and skips the finished stages, the walked pages, and the exported pages. If CoralNet's image count for the source changed since `_partial/` was started, the next run discards `_partial/` and starts the source again.
- Recovery from any failure or kill is to run the same command again. Image objects already on S3 are skipped.
- A source with a failed stage, for example an image that could not be downloaded, keeps its progress and is not verified. The next run redoes the failed stage and every stage after it. Finished earlier stages, saved walk pages, saved export pages and images already on S3 are kept.
- Verification runs only when every stage succeeded. A source that fails verification has its `_partial/` discarded and no marker. The next run redoes the walk and the export. Image objects already on S3 are skipped.
- A source with zero Confirmed images, or zero images, gets no marker and nothing is written for it. The next run checks it again, and the run report lists it under `skipped`.
- To redo a complete source, run `coralnet-scrape invalidate` for it, then the plain command.

### Verification

Before the marker is written, the scraper reads the source back from S3 and compares it with CoralNet's image count. These failures block the marker and fail the source with `verification_failed`:

| Code | Meaning |
|---|---|
| `annotations_missing` | `annotations.csv` is absent. |
| `image_list_missing` | `image_list.csv` is absent. |
| `image_list_bad_ids` | An image id in the image list cannot be parsed, or occurs twice. |
| `image_list_count:N!=total` | The image list has N rows and CoralNet reports `total` images. |
| `annotations_bad_columns` | `annotations.csv` has no `Name` column, or has an HTML or `Unnamed` column. |
| `annotations_null_name` | An annotation row has no `Name`. |
| `annotations_name_not_in_list:N` | N image names in `annotations.csv` are not in the image list. |
| `image_objects_missing:N` | N listed images have no object on S3, or a 0-byte object. |

These warnings are recorded in the marker and do not block it:

| Code | Meaning |
|---|---|
| `confirmed_without_annotations:N` | N Confirmed images have no annotation rows. |
| `orphan_image_objects:N` | N image objects on S3 are not in the image list. `--prune` deletes them. |
| `duplicate_names:N` | N image names occur more than once. |

The marker holds the scraper version, the completion time, the counts `total_images`, `annotation_rows`, `annotated_images` and `image_objects`, and the warnings.

### Discovery guard

Each run compares the source listing from CoralNet with `<prefix>/_meta/discovered_source_ids.json`, the listing of the previous run. If CoralNet lists fewer than 95% of the previous count, the run stops unless `--allow-source-drop` is set. A guard stop exits 1 and writes no run report. The run logs the sources that are no longer listed. The file is overwritten on each run, after the guard passes.

## Tests

```bash
uv run --package coralnet-scraper --group tests pytest packages/coralnet-scraper/tests
```

The tests are offline. They use the HTML fixtures in `tests/fixtures/coralnet_html/` and no network.
