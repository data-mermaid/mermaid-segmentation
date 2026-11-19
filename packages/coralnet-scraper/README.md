# coralnet-scraper

Downloads public CoralNet sources (images, labelsets, annotations) to S3. Import package: `coralnet_scraper`; console script: `coralnet-scrape`.

## Scope

- **Owns:** recreating the raw CoralNet dataset in S3, one folder per public source.
- **Does not own:** turning a scrape into the training manifest that `CoralNetDataset` reads. That step is not part of this package.
- **Boundary:** `mermaidseg` does not depend on or import this package, and the package does not import `mermaidseg`. the root `tests/test_package_boundaries.py` enforces both (run with the root suite). A root `uv sync` does not install it.

## S3 layout

Everything is written under `s3://<bucket>/<prefix>/s<source_id>/`:

| Object | Content |
|---|---|
| `metadata.csv` | Classifier history from the source overview page. Absent when the page has none. |
| `labelset.csv` | `Label ID`, `Name`, `Short Code`. Absent when the source has no labels. |
| `annotations.csv` | CoralNet annotation export of the whole source. |
| `image_list.csv` | One row per image from the browse pages; the `Image URL` column is empty. |
| `images/<image_id>.jpg` | One object per image, keyed by the numeric CoralNet image id. |

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
| `-v`, `--verbose` | off | Debug logging. |

The command logs in to CoralNet, discovers the public source ids from `https://coralnet.ucsd.edu/source/about/`, and downloads each source. A failed login ends the run with exit code 1.

An exception inside one source, in the skip check or the download, is logged and recorded under its exception class name. The run continues with the next source.

A source fails when:

- any of its images had no image URL (`images_missing_url`) or failed to upload (`images_upload_failed`);
- the metadata, labelset, annotations, or image list step failed, or the source could not be accessed;
- an exception was raised for it, as above.

At the end the run logs a summary at error level: the number of failed sources, then one line per error kind with the sorted source ids. Image kinds give the image count per source, for example `images_missing_url: 202 (14 images)`. A failed source with no error code is listed under `no_error_code`.

Exit code:

| Code | When |
|---|---|
| 0 | Every source downloads or is skipped. A source with zero Confirmed images, or with zero images, counts as skipped. |
| 1 | The CoralNet login fails, any source fails, or the run stops on an uncaught error (invalid configuration, source discovery failure). |

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
| `CORALNET_USERNAME` | no | prompt | CoralNet login. Used only when `CORALNET_PASSWORD` is also set. |
| `CORALNET_PASSWORD` | no | prompt | CoralNet password. |

Without both credential variables the command prompts for the username and password on the terminal.

S3 access uses the default boto3 credential chain, for example `AWS_PROFILE`.

### Resume and `--force`

- A source is skipped when `<prefix>/s<id>/annotations.csv` exists in S3. The check is made before any download.
- A source runs metadata, labelset, annotations, image list, then images. A failure in the metadata or labelset step does not stop the run, so `annotations.csv` is still uploaded. A source that failed with `annotations.csv` already uploaded is skipped on the next run unless `--force` is set. That covers `metadata_download_failed`, `labelset_failed`, `image_list_failed`, `images_missing_url`, `images_upload_failed`, and a source that stopped after the annotations upload. Find these sources in the failure summary logged at the end of the run. `--force` is the only way to retry them, and it reprocesses every source.
- Sources with zero Confirmed images, or zero images, on CoralNet are skipped, and nothing is written for them.
- `--force` downloads every source. Image objects already in S3 are not uploaded again.
- A source with more than 5000 images is exported in pages. It loads an existing `annotations.csv` and merges new rows into it, and uploads a partial file every 50 pages. A source whose export fails (`annotations_failed`) or is interrupted after the first checkpoint leaves that partial `annotations.csv`, which the resume rule skips until `--force`.

## Tests

```bash
uv run --package coralnet-scraper --group tests pytest packages/coralnet-scraper/tests
```

The tests are offline. They use the HTML fixtures in `tests/fixtures/coralnet_html/` and no network.
