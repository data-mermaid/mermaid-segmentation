# CoralNet dataset

<<<<<<< HEAD
PyTorch dataset and supporting code for the CoralNet source dataset.

The dataset class ([`CoralNetDataset`](coralnet_dataset.py)) emits annotation
masks in the **CoralNet provider label space** (i.e. CoralNet `Label ID` /
`provider_id` values). Mapping into the unified target MERMAID benthic
attribute space is done at training time by the
[`SourceLabelRegistry`](../../dataset_reconciliation/registry.py) and
[`fetch_coralnet_to_mermaid`](../../dataset_reconciliation/label_mapping.py).

The dataset class reads the annotations from the a parquet file (currently: s3://dev-datamermaid-sm-sources/coralnet_annotations_30112025.parquet), from which the images are also derived.

## Layout

- `coralnet_dataset.py` — `CoralNetDataset` class.
- `scraper/` — scraping utilities and CSV artifacts originally located under
  the repo-root `scraping/` directory.
- `nbs/` — exploration and annotation-aggregation notebooks for the CoralNet
  sources.

## Description

CoralNet (https://coralnet.ucsd.edu/) is an online repository of annotated coral reef imagery and expert annotations. It provides a large-scale source of coral reef images with point annotations for benthic attributes structured around sources. A source typically corresponds to a single research project, survey, institution, field site, camera system, or contributor that supplied imagery and metadata to CoralNet. Sources group images that share common collection conditions (location, time period, photographer or instrument, and sampling protocol). Specific sources also contain metadata including the location, datetime and depth.

- Annotations in CoralNet are usually acquired through random sampling or through human labelling across the image (which also result in randomly spaced annotations). The annotations dataframe, contains the label for a specific row & column of a specific image from a specific source. The CoralNet classes are mapped to MERMAID classes using the LabelMapping API endpoint.
- We apply padding to the annotations, with the assumption that for a specific point (pixel being) assigned to a class, the neighbouring pixels are very likely to also be in that class as these labels come either from a image classification approach that makes a prediction based on a image crop around the point, or a manual annotation, both of which are most likely not precise to a pixel level.

## Note
The parquet file currently only contains a subset of the sources and annotations and needs to be updated.
=======
The CoralNet Dataset can be used through the CoralNetDataset class. This class reads the annotations from the coralnet_annotations_30112025.parquet file found in the dev-datamermaid-sm-sources S3 Bucket which can be found as the df_annotations attribute of the class, and from which the df_images attribute is also derived. Unlike the systematic annotation of MERMAID, annotations in CoralNet are usually acquired through random sampling or just human labels across the image (which result in random annotations). From the annotations dataframe, which contains the label for a specific row & column in the image, we generate annotation masks which equal 0 (for background) and a class id for each annotated point (based on the id2label attribute). The CoralNet classes are mapped to MERMAID classes using the LabelMapping API endpoint.

We apply padding to the annotations, with the assumption that for a specific point (pixel being) assigned to a class, the neighbouring pixels are very likely to also be in that class as these labels come either from a image classification approach that makes a prediction based on a image crop around the point, or a manual annotation, both of which are most likely not precise to a pixel level.
The parquet file currently only contains a subset of the sources in order to speed up data loading, and can be updated using the mermaidseg/nbs/datasets/CoralNet_Annotations.ipynb notebook.
>>>>>>> origin/main
