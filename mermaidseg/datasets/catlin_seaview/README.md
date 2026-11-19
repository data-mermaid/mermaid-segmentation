# Catlin Seaview

Point-annotated photo-quadrats from nine reef regions, read by `CatlinSeaviewDataset` in `catlin_seaview_dataset.py`.

## Source and citation

Seaview Survey Photo-quadrat and Image Classification Dataset (González-Rivero et al., 2019). The raw data is at [data.qld.edu.au/public/Q1281](http://data.qld.edu.au/public/Q1281/).

| Region code | Place | Images | Annotations | Label classes present | Damaged JPG files |
|---|---|---|---|---|---|
| ATL | Atlantic | 1407 | 92900 | 67 | 0 |
| IND_CHA | Indian Ocean, Chagos Archipelago | 686 | 52450 | 42 | 0 |
| IND_MDV | Indian Ocean, Maldives | 1612 | 144100 | 38 | 0 |
| PAC_AUS | Pacific Ocean, Australia | 2657 | 186420 | 29 | 0 |
| PAC_IDN_PHL | Pacific Ocean, Indonesia and Philippines | 1608 | 118600 | 67 | 30 |
| PAC_SLB | Pacific Ocean, Solomon Islands | 732 | 59200 | 55 | 0 |
| PAC_TLS | Pacific Ocean, Timor-Leste | 864 | 71600 | 59 | 0 |
| PAC_TWN | Pacific Ocean, Taiwan | 638 | 50000 | 54 | 0 |
| PAC_USA | Pacific Ocean, USA (Hawaii) | 1153 | 83100 | 39 | 0 |
| **Total** | | **11357** | **858370** | **194** | **30** |

The release has 11,387 images. The per-region image counts above sum to 11,357.

## Licence

Creative Commons Attribution-NonCommercial-ShareAlike 4.0 International (CC BY-NC-SA 4.0). See the UQ eSpace record UQ:734799 (doi:10.14264/uql.2019.930).

## Where the class reads from

Bucket `dev-datamermaid-sm-sources`, prefix `external_validation_datasets/catlin_seaview/`:

- `catlin_seaview_annotations.parquet`. The class reads the columns `region`, `image_id`, `row`, `col` and `source_label_name`.
- `images/<region>/<image_id>.jpg`.

The prefix also holds `classes.json`, `colors.json` and `manifest.json`. `manifest.json` lists the damaged images that the run dropped, by region.

## How to rebuild

The `external-datasets` package downloads the release and writes it to S3. See the [package README](../../../packages/external-datasets/README.md) for the output contract and every flag.

```bash
uv run --package external-datasets external-datasets catlin_seaview
```

Without `--data-dir`, the run downloads `tabular-data.zip` and one zip for each region from `http://data.qld.edu.au/public/Q1281/` into `--workdir` (default `outputs/external-datasets/catlin_seaview/`), and extracts them there. A rerun uses the zips already on disk.

To use a raw tree that you downloaded before, pass `--data-dir`. The layout is:

```text
<data-dir>/tabular-data/annotations_<region>.csv
<data-dir>/<region>/<quadrat_id>.jpg
```

Pass `--dry-run` to check a run without writing to S3. If the parquet or a JSON file is already on S3, the run stops. Pass `--overwrite` to replace them.
