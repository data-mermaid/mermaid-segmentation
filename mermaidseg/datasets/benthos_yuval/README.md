# Benthos

Dense label masks over two reef photomosaics, cut into tiles and read by `BenthosYuvalCoralsDataset` in `benthos_yuval_corals_dataset.py`.

## Source and citation

Yuval, M., Alonso, I., Eyal, G., Tchernov, D., Loya, Y., Murillo, A. C., Treibitz, T. (2021). Repeatable Semantic Reef-Mapping through Photogrammetry and Label-Augmentation. Remote Sensing 13(4): 659. [doi:10.3390/rs13040659](https://doi.org/10.3390/rs13040659)

The data is on Dryad at [doi:10.5061/dryad.8cz8w9gm3](https://datadryad.org/dataset/doi:10.5061/dryad.8cz8w9gm3). It has fully labelled photomosaics of three sites, over eight classes (`Algae`, `Calcareous Algae`, `Coral`, `Other`, `Rock`, `Sand`, `Soft Coral`, `Sponge`). Pixels outside the labelled area are `unlabeled`.

| Site | Region | Depth (m) | Mosaic size (px) | Loaded |
|---|---|---|---|---|
| `RS24` | IUI Eilat reef, Gulf of Aqaba, Red Sea | 24–28 | 11070x11786 | Yes |
| `CR_DoubleWreck` | Double-Wreck reef, St. Eustatius, Caribbean | 20 | 24192x3982 | Yes |
| `MD_spartan` | Spartan Reef, offshore Haifa, Mediterranean | 20 | — | No |

`MD_spartan` is excluded because it is a rocky Mediterranean site with no scleractinian coral community.

## Licence

CC0 1.0 (Dryad).

## Where the class reads from

Bucket `dev-datamermaid-sm-sources`, prefix `external_validation_datasets/benthos_yuval/`:

- `benthos_yuval_annotations.parquet`, one row for each class present in a tile. The class reads the columns `site`, `image_id` and `source_label_name`. The parquet also has `tile_height` and `tile_width`.
- `images/<site>/<image_id>.png`: RGB tiles.
- `labels/<site>/<image_id>.png`: single-channel masks, with pixel values from `classes.json`.
- `classes.json`: the class reads it to convert mask values into its own label IDs.

Each mosaic is cut into 2048x2048 tiles with no overlap. `image_id` is `<y0>_<x0>`, the pixel position of the tile's top-left corner in the mosaic. The tiles on the bottom and right edges are smaller than 2048 and are not padded.

## How to rebuild

The Dryad download needs a login, so download these six files by hand from the link above into one directory:

| File | Content |
|---|---|
| `RS24.png` | RGB mosaic of `RS24`. |
| `ManualRS24.png` | Label mask of `RS24`. |
| `CR_DoubleWreck.png` | RGB mosaic of `CR_DoubleWreck`. |
| `ManualDW.png` | Label mask of `CR_DoubleWreck`. |
| `dictionary_labels.txt` | `{class_name: mask_value}` for `ManualRS24.png`. |
| `dictionary_labelsDW.txt` | `{class_name: mask_value}` for `ManualDW.png`. |

Then run:

```bash
uv run --package external-datasets external-datasets benthos_yuval --data-dir <data-dir>
```

The run writes the tiles to `--workdir` (default `outputs/external-datasets/benthos_yuval/`) before it uploads them. See the [package README](../../../packages/external-datasets/README.md) for the output contract and every flag. Pass `--dry-run` to check a run without writing to S3. If the parquet or a JSON file is already on S3, the run stops. Pass `--overwrite` to replace them.
