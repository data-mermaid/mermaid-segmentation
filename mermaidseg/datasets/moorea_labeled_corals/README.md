# Moorea Labeled Corals

Point-annotated photo-quadrats from the Moorea Coral Reef LTER, 2008 to 2010, read by `MooreaLabeledCoralsDataset` in `moorea_labeled_corals_dataset.py`.

## Source and citation

Beijbom O., Edmunds P.J., Kline D.I., Mitchell G.B., Kriegman D., "Automated Annotation of Coral Reef Survey Images", IEEE Conference on Computer Vision and Pattern Recognition (CVPR), 2012.

The data is the EDI package [knb-lter-mcr.5006](https://portal.edirepository.org/nis/mapbrowse?scope=knb-lter-mcr&identifier=5006), a subset of the MCR LTER packaged for computer vision research. It has 2055 images from three habitats, with random point annotations (row, col, label). Nine labels account for 96% of the points: Crustose Coralline Algae (CCA), Turf algae, Macroalgae, Sand, Acropora, Pavona, Montipora, Pocillopora and Porites. Less abundant genera are also present.

| Year | Images | Annotations (9 classes) | Annotations (20 classes) | Label classes present | Damaged or missing images |
|---|---|---|---|---|---|
| 2008 | 671 | 131260 | 133045 | 26 | 0 |
| 2009 | 695 | 132112 | 133072 | 21 | 0 |
| 2010 | 689 | 113256 | 120324 | 26 | 53 |
| **Total** | **2055** | **376628** | **386441** | **29** | **53** |

## Licence

CC BY 4.0 (Creative Commons Attribution 4.0 International), as stated by the EDI data package.

## Where the class reads from

Bucket `dev-datamermaid-sm-sources`, prefix `external_validation_datasets/moorea_labeled_corals/`:

- `moorea_labeled_corals_annotations.parquet`. The class reads the columns `year`, `image_id`, `image_ext`, `row`, `col` and `source_label_name`.
- `images/<year>/<image_id><image_ext>`.

`source_label_name` is the label after case merging: spellings that differ only in case become one label, and the mixed-case spelling is kept. The parquet also has `raw_label_name`, the spelling in the raw file. Points labelled `OFF` (in any case) are not in the parquet. `manifest.json` records the damaged images, the merged spellings and the number of `OFF` points.

## How to rebuild

The EDI package needs a login, so download it by hand from the link above. Unzip it so that the layout is:

```text
<data-dir>/<year>/<image>.<jpg|jpeg|png|tif|tiff>
<data-dir>/<year>/<image>.<ext>.txt      # one "row;col;label" line per point
```

Then run:

```bash
uv run --package segdata-etl segdata ingest moorea_labeled_corals --data-dir <data-dir>
```

The run drops each damaged image and its annotations. See the [package README](../../../packages/segdata-etl/README.md) for the output contract and every flag. Pass `--dry-run` to check a run without writing to S3. If the parquet or a JSON file is already on S3, the run stops. Pass `--overwrite` to replace them.
