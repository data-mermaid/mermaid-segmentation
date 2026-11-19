# MERMAID

Point-annotated images confirmed by users of the MERMAID platform, read by `MermaidDataset` in `mermaid_dataset.py`. The labels are MERMAID benthic attributes, the target label space of the project.

## Source and citation

The [MERMAID platform](https://datamermaid.org/). Users upload images and confirm annotations, so the dataset grows over time. Each image has 25 points on a 5x5 grid.

## Licence

Not stated.

## Where the class reads from

Bucket `coral-reef-training`:

- `mermaid/mermaid_confirmed_annotations.parquet`. The class reads the columns `image_id`, `row`, `col`, `benthic_attribute_name` (renamed to `source_label_name`), `updated_on`, `region_id` and `region_name`.
- `mermaid/<image_id>.png`.

All image keys end in `.png`, but some objects hold JPEG data. A check on 2026-07-24 found 14,258 JPEG and 2,423 PNG objects among the 16,681 that the parquet references.

The parquet changes in place. `annotation_date_cutoff` (required) freezes the training data: the class uses only annotations with `updated_on` on or before that date, in UTC. `configs/data_config.yaml` sets it under `mermaid.train`.

## How to rebuild

This repository does not write the parquet or the images. The MERMAID platform exports them, outside this repository. To train on newer annotations, change `annotation_date_cutoff`. See [Update the MERMAID data](../../../docs/runbooks/training-datasets.md#update-the-mermaid-data).
