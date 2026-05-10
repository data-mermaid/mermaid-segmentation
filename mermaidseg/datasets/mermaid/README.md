# MERMAID dataset

PyTorch dataset and supporting code for the MERMAID source dataset.

The dataset class ([`MermaidDataset`](mermaid_dataset.py)) emits annotation
masks in the **MERMAID benthic attribute label space**. The MERMAID label
space is the canonical target space used by
[`SourceLabelRegistry`](../../dataset_reconciliation/registry.py); the
`source -> target` mapping for MERMAID is therefore the identity (modulo the
global offset assigned by the registry).

The dataset class reads the annotations from the a parquet file (s3://coral-reef-training/mermaid/mermaid_confirmed_annotations.parquet), from which the images are also derived. 

## Layout

- `mermaid_dataset.py` — `MermaidDataset` class.
- `nbs/` — exploration notebooks for MERMAID-specific data analysis.

## Description 

The MERMAID dataset is a continuously growing dataset with images uploaded by users of the MERMAID platform (https://datamermaid.org/). 
- The annotations of each image are done in a systematic approach, such that 25 points are taken in a 5x5 grid across the image. The annotations dataframe, contains the label for a specific row & column of a specific image, as well as the MEOW region it belongs to. 
- We apply padding to the annotations, with the assumption that for a specific point (pixel being) assigned to a class, the neighbouring pixels are very likely to also be in that class as these labels come either from a image classification approach that makes a prediction based on a image crop around the point, or a manual annotation, both of which are most likely not precise to a pixel level.

## Note
As the mermaid_confirmed_annotations.parquet is continuously being updated, each run currently might have slightly different results due to changes in the (number of) images. As a solution to this, we can potentially save occasional copies of the file (e.g. at the end of every month).
