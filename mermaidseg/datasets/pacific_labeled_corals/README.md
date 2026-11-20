# Pacific Labeled Corals

Point-annotated photo-quadrats from three Pacific reef sites, with up to seven annotators for each point, read by `PacificLabeledCoralsDataset` in `pacific_labeled_corals_dataset.py`.

## Source and citation

Beijbom, O., Edmunds, P. J., Roelfsema, C., Smith, J., Kline, D. I., Neal, B. P., Dunlap, M. J., Moriarty, V., Fan, T.-Y., Tan, C.-J., Chan, S., Treibitz, T., Gamst, A., Mitchell, B. G., Kriegman, D. (2015). Towards Automated Annotation of Benthic Survey Images: Variability of Human Experts and Operational Modes of Automation. PLOS ONE 10(7): e0130312. [doi:10.1371/journal.pone.0130312](https://doi.org/10.1371/journal.pone.0130312)

The data is on Dryad at [doi:10.5061/dryad.m5pr3](https://datadryad.org/dataset/doi:10.5061/dryad.m5pr3). It has four sites that share one 20-class label set. The `moorea` site is not loaded, because the same images are in [Moorea Labeled Corals](../moorea_labeled_corals/README.md).

| Site | Images | Annotations (points) | Label classes present |
|---|---|---|---|
| `heron_reef` | 2,797 | 64,328 | 5 |
| `line_islands` | 732 | 55,200 | 20 |
| `nanwan_bay` | 890 | 36,260 | 20 |
| **Total** | 4,419 | 155,788 | 20 |

Heron Reef has 5 classes because, in Beijbom et al.: *"Corals were not resolved to genus level in the original Heron Reef label-set and all `Archived` coral annotations were therefore mapped to the generic 'other scleractinians' label for this location."*

Each site has two subsets:

- `reference`: the survey images, with one label for each point, `archived`, from the original ecological survey.
- `evaluation`: 200 images for each site, 10 points for each image. Each point has seven labels: `archived`, `host` (the same local expert, annotating again for the study), and `visitor1` to `visitor5` (five visiting experts).

All images in the Dryad release decode.

## Licence

CC0 1.0 (Dryad).

## Where the class reads from

Bucket `dev-datamermaid-sm-sources`, prefix `external_validation_datasets/pacific_labeled_corals/`:

- `pacific_labeled_corals_annotations.parquet`. The class requires the columns `site`, `subset`, `image_id`, `image_ext`, `row`, `col` and `archived`. It also reads the annotator columns `host` and `visitor1` to `visitor5`, which are empty outside the `evaluation` subsets. `annotator_column` (default `host`) selects the label, and `fallback_annotator` (default `archived`) fills the points where that column is empty.
- `images/<site>/<subset>/<image_id><image_ext>`.

## How to rebuild

Dryad has no stable download API, so download the package by hand from the link above. Unzip it so that the layout is:

```text
<data-dir>/<site>/labelmap.txt                    # labelid,name
<data-dir>/<site>/<subset>/imagemap.txt           # imageid,imagefile
<data-dir>/<site>/<subset>/annotations.txt        # imageid,row,col,archived[,host,visitor1..visitor5]
<data-dir>/<site>/<subset>/imgs/<file>.JPG
```

Then run:

```bash
uv run --package segdata-etl segdata build pacific_labeled_corals --data-dir <data-dir>
```

The run stops before it writes to S3 if the three `labelmap.txt` files differ, or if an image does not decode. See the [package README](../../../packages/segdata-etl/README.md) for the output contract and every flag. Pass `--dry-run` to check a run without writing to S3. If the parquet or a JSON file is already on S3, the run stops. Pass `--overwrite` to replace them.
