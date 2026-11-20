# CoralNet

Point-annotated images from the public sources on CoralNet, read by `CoralNetDataset` in `coralnet_dataset.py`. The labels are CoralNet Label IDs.

## Source and citation

[CoralNet](https://coralnet.ucsd.edu/), an online repository of coral reef images with expert point annotations. A source groups the images of one project, survey, site or contributor.

## Licence

Not stated.

## Where the class reads from

Bucket `dev-datamermaid-sm-sources`:

- The training manifest that the data config names in `coralnet.<split>.manifest_path`, under `etl-outputs/coralnet/<run>/`. The class reads the columns `source_id`, `image_id`, `row`, `col`, `coralnet_id` and `image_s3_key`.
- `coralnet_id2name_<run>.json` beside the manifest, for the label names.
- The images at the keys in `image_s3_key`: the resized copy if the build resized the image, else the original.

`image_blacklist` and `annotation_overrides`, in `configs/coralnet/`, remove duplicate images. See [Use a build in training](../../../packages/segdata-etl/README.md#use-a-build-in-training).

## How to rebuild

Two workspace packages build the data: [`coralnet-scraper`](../../../packages/coralnet-scraper/README.md) downloads the sources to S3, and [`segdata-etl`](../../../packages/segdata-etl/README.md) builds the manifest, the resized images and the label names. For the order of the steps, see [Recreate and update the CoralNet data](../../../docs/runbooks/coralnet-data.md).
